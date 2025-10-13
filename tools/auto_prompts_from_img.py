# tools/auto_prompts_from_img.py
from __future__ import annotations
import os, math, json, argparse
import numpy as np
import torch
import open_clip

def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# 候选属性词表（可自行扩展）
ATTR = {
    "upper_color": ["black", "white", "red", "blue", "green", "yellow", "gray", "brown", "purple", "pink"],
    "upper_type":  ["t-shirt", "shirt", "jacket", "coat", "hoodie", "sweater"],
    "lower_type":  ["jeans", "long pants", "shorts", "skirt"],
    "shoes":       ["sneakers", "boots", "dress shoes"],
    "bag":         ["carrying a backpack", "carrying a shoulder bag", "carrying a handbag", "no bag"],
    "head":        ["wearing a hat", "wearing a cap", "no headwear"],
}

def build_phrases() -> dict[str, list[str]]:
    phrases = {}
    phrases["upper_color"] = [f"wearing a {c} top" for c in ATTR["upper_color"]]
    phrases["upper_type"]  = [f"wearing a {t}" for t in ATTR["upper_type"]]
    phrases["lower_type"]  = [f"wearing {t}" for t in ATTR["lower_type"]]
    phrases["shoes"]       = [f"wearing {t}" for t in ATTR["shoes"]]
    phrases["bag"]         = [t for t in ATTR["bag"]]
    phrases["head"]        = [t for t in ATTR["head"]]
    return phrases

@torch.no_grad()
def text_embed(model, tok, texts: list[str], dev: torch.device) -> torch.Tensor:
    """
    将一组文本编码为张量，并保持在 dev（CUDA/MPS/CPU）上。
    返回形状：(N, D)，已做L2归一化。
    """
    out = []
    bs = 128
    for i in range(0, len(texts), bs):
        toks = tok(texts[i:i+bs]).to(dev)
        z = model.encode_text(toks).float()
        out.append(norm(z))  # 不转CPU，保持在dev
    return torch.cat(out, dim=0).to(dev)

@torch.no_grad()
def eval_metrics(T: torch.Tensor, I: torch.Tensor, Lt: list[str], Li: list[str]):
    """
    输入：已归一化的 T, I（在同一设备上）
    返回：Avg Top-1，相应的 Recall@K 字典，mAP
    """
    S = T @ I.T
    top1 = float(S.max(dim=1).values.mean().item())
    order = torch.argsort(S, dim=1, descending=True)

    def rec_at(k: int) -> float:
        hit = 0
        for i, y in enumerate(Lt):
            ranked = [Li[j] for j in order[i, :k].tolist()]
            if y in ranked:
                hit += 1
        return hit / len(Lt)

    def mean_ap() -> float:
        ap = 0.0
        for i, y in enumerate(Lt):
            ranked = [Li[j] for j in order[i].tolist()]
            total = ranked.count(y)
            if total == 0:
                continue
            h = 0
            ps = 0.0
            for r, p in enumerate(ranked, start=1):
                if p == y:
                    h += 1
                    ps += h / r
            ap += ps / total
        return ap / len(Lt)

    R = {1: rec_at(1), 5: rec_at(5), 10: rec_at(10)}
    return top1, R, mean_ap()

def main():
    ap = argparse.ArgumentParser("Auto-generate prompts from image centroids and evaluate")
    ap.add_argument("--img-centroids", required=True, help="图像类中心 .npy (P, D)")
    ap.add_argument("--img-centroids-labels", required=True, help="类中心标签 .txt (每行一个 pid)")
    ap.add_argument("--out-jsonl", default="data/prompts/market-50.auto_from_img.jsonl")
    ap.add_argument("--out-labels", default="data/prompts/market-50.auto_from_img.labels.txt")
    ap.add_argument("--out-text-emb", default="embeds/text/clip-b16_50.auto_from_img.npy")
    ap.add_argument("--model", default=os.getenv("CLIP_LOCAL_MODEL", "ViT-B-16"))
    ap.add_argument("--pretrained", default=os.getenv("CLIP_LOCAL_PRETRAIN", "laion2b_s34b_b88k"))
    args = ap.parse_args()

    dev = device()
    print(f"[auto] model={args.model}  pretrain={args.pretrained}  device={dev}")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=dev)
    model.eval()
    tok = open_clip.get_tokenizer(args.model)

    # 读入类中心（到 dev）与标签
    I = torch.tensor(np.load(args.img_centroids), dtype=torch.float32, device=dev)
    with open(args.img_centroids_labels, "r", encoding="utf-8") as f:
        Li = [ln.strip() for ln in f if ln.strip()]
    assert I.shape[0] == len(Li), "类中心数量与标签不一致"
    I = norm(I)

    # 预生成各槽位的候选短语与其向量（保持在 dev）
    phrases = build_phrases()
    embeds = {slot: text_embed(model, tok, txts, dev) for slot, txts in phrases.items()}

    # 固定域前缀
    prefix = "surveillance photo of a pedestrian, full-body, outdoors, "

    # 为每个 pid 选择属性短语（通过 v @ E^T 最大化）
    prompts = []
    with torch.no_grad():
        for idx in range(I.shape[0]):
            v = I[idx].unsqueeze(0)  # (1, D) on dev
            picked = {}
            for slot, E in embeds.items():
                s = (v @ E.T).squeeze(0)  # (N,)
                j = int(torch.argmax(s).item())
                picked[slot] = phrases[slot][j]
            # 组装句子（去重、保持顺序）
            desc_parts = [
                picked["upper_color"],
                picked["upper_type"],
                picked["lower_type"],
                picked["shoes"],
                picked["bag"],
                picked["head"],
            ]
            seen = set()
            cleaned = []
            for seg in desc_parts:
                if seg not in seen:
                    cleaned.append(seg)
                    seen.add(seg)
            text = prefix + ", ".join(cleaned)
            prompts.append({"id": Li[idx], "text": text})

    # 输出 JSONL 与标签
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for obj in prompts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(args.out_labels, "w", encoding="utf-8") as f:
        f.write("\n".join([p["id"] for p in prompts]) + "\n")
    print(f"[auto] saved prompts: {args.out_jsonl} (lines={len(prompts)})")
    print(f"[auto] saved labels : {args.out_labels}")

    # 文本嵌入（保留在 dev 上评测；同时另存为 .npy）
    texts = [p["text"] for p in prompts]
    T_dev = text_embed(model, tok, texts, dev)        # (P, D) on dev
    os.makedirs(os.path.dirname(args.out_text_emb), exist_ok=True)
    np.save(args.out_text_emb, T_dev.detach().cpu().numpy().astype(np.float32))
    print(f"[auto] saved text embeds: {args.out_text_emb}  shape={tuple(T_dev.shape)}")

    # 评测：AutoPrompt vs ImageCentroids
    top1, R, mAP = eval_metrics(T_dev, I, [p["id"] for p in prompts], Li)
    print("\n=== AutoPrompt vs ImageCentroids ===")
    print(f"Avg Top-1 Sim: {top1:.4f}")
    print(f"Recall@1: {R[1]:.4f}  Recall@5: {R[5]:.4f}  Recall@10: {R[10]:.4f}")
    print(f"mAP: {mAP:.4f}")

if __name__ == "__main__":
    main()
