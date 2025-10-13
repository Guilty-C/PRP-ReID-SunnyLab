# tools/auto_prompts_from_img_v3.py
from __future__ import annotations
import os
import json
import argparse
import itertools
from typing import List, Dict, Tuple

import numpy as np
import torch
import open_clip


# ---------------------------
# Utils
# ---------------------------
def dev() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def rl(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def l2n_np(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


@torch.no_grad()
def text_embed(
    model,
    tok,
    texts: List[str],
    device: torch.device,
    bs: int = 128,
) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), bs):
        toks = tok(texts[i : i + bs]).to(device)
        z = model.encode_text(toks).float()
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        out.append(z)
    return torch.cat(out, dim=0)


# ---------------------------
# v3 词库（只改 prompt，不改算法/模型）
# ---------------------------
BASE_COLORS = [
    "black",
    "white",
    "red",
    "blue",
    "navy blue",
    "light blue",
    "dark blue",
    "green",
    "olive",
    "yellow",
    "orange",
    "brown",
    "beige",
    "khaki",
    "gray",
    "light gray",
    "dark gray",
    "purple",
    "pink",
]
UPPER_TYPES = ["t-shirt", "shirt", "jacket", "coat", "hoodie", "sweater", "blazer"]
LOWER_TYPES = ["jeans", "trousers", "pants", "shorts", "skirt", "sweatpants"]
SHOES_TYPES = ["sneakers", "boots", "dress shoes", "sandals", "running shoes"]
BAG_NOUNS = ["backpack", "shoulder bag", "crossbody bag", "handbag"]
HEAD_NOUNS = ["hat", "cap", "hood"]

# 更贴近 CLIP 的前缀集（会自动选最匹配的前缀）
PREFIX_POOL = [
    "a photo of a person",
    "a CCTV image of a person",
    "a streetcam photo of a person",
    "a surveillance photo of a person",
    "a pedestrian photo",
]

# v3 模板：更口语、更稳定
TEMPLATES = [
    "{prefix}, full body, outdoors. {upper}. {lower}. {shoes}{bag}{head}",
    "{prefix}, walking outdoors, full body. {upper} and {lower}. {shoes}{bag}{head}",
    "{prefix}, full-length shot outdoors. wearing {upper}, with {lower}. {shoes}{bag}{head}",
]

# 默认过滤掉容易带来噪音/否定的短语（v3 不直接生成这些）
BLACKLIST_SUBSTR = ["no bag", "no headwear", "hood up"]


def build_slot_phrases() -> Dict[str, List[str]]:
    # 统一用 “color + noun” 的名词短语
    uppers = [f"{c} {t}" for c in BASE_COLORS for t in UPPER_TYPES]
    lowers = [f"{c} {t}" for c in BASE_COLORS for t in LOWER_TYPES]
    shoes = SHOES_TYPES[:]  # 简化成鞋类名词
    bags = BAG_NOUNS[:]
    heads = HEAD_NOUNS[:]
    return {"upper": uppers, "lower": lowers, "shoes": shoes, "bag": bags, "head": heads}


def gate_keep(slot: str, sim: float, anchor: float, thresh: float) -> bool:
    """
    次要槽位的“门控”：
    - 上衣/下装不门控；
    - 鞋/包/头饰只有当与图像中心的相似度 >= thresh * anchor 才写入，
      其中 anchor = (上衣相似度 + 下装相似度) / 2。
    """
    if slot in ("upper", "lower"):
        return True
    return sim >= thresh * anchor


def clean_text(s: str) -> str:
    s = s.replace("..", ".").replace(" ,", ",").replace("  ", " ").strip()
    return s


# ---------------------------
# Main
# ---------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("auto prompts v3 (prompt-only improvements)")
    ap.add_argument("--img-centroids", required=True)
    ap.add_argument("--img-centroids-labels", required=True)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--gallery-labels", default=None)

    ap.add_argument("--topk", type=int, default=3, help="每个槽位取前 top-k 候选")
    ap.add_argument("--variants", type=int, default=12, help="每个 pid 生成的候选上限")
    ap.add_argument("--gate", type=float, default=0.85, help="次要槽位的门控比例阈值（0~1）")
    ap.add_argument(
        "--prefix-pool-weight",
        type=float,
        default=0.2,
        help="前缀相似度在打分中的权重（0~0.5）",
    )

    ap.add_argument("--out-jsonl", default="data/prompts/market-50.auto_v3.jsonl")
    ap.add_argument("--out-labels", default="data/prompts/market-50.auto_v3.labels.txt")
    ap.add_argument(
        "--out-text-emb-all", default="embeds/text/clip-l14_50.auto_v3.all.npy"
    )
    ap.add_argument(
        "--out-text-emb-avg", default="embeds/text/clip-l14_50.auto_v3.avg_by_pid.npy"
    )

    ap.add_argument("--model", default=os.getenv("CLIP_LOCAL_MODEL", "ViT-L-14"))
    ap.add_argument(
        "--pretrained",
        default=os.getenv("CLIP_LOCAL_PRETRAIN", "datacomp_xl_s13b_b90k"),
    )
    args = ap.parse_args()

    device = dev()
    print(
        f"[v3] model={args.model}  pretrain={args.pretrained}  device={device.type}"
    )
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()
    tok = open_clip.get_tokenizer(args.model)

    # 读数据
    I = np.load(args.img_centroids).astype(np.float32)  # (P, D)
    Li = rl(args.img_centroids_labels)  # P
    P, D = I.shape

    # 词库向量
    phrases = build_slot_phrases()
    embeds: Dict[str, torch.Tensor] = {}
    for slot, lst in phrases.items():
        embeds[slot] = text_embed(model, tok, lst, device)

    # 前缀向量
    E_prefix = text_embed(model, tok, PREFIX_POOL, device)  # (Np, D)

    # 逐 pid 生成文本
    I_t = torch.tensor(l2n_np(I), dtype=torch.float32, device=device)  # (P, D)
    all_texts: List[str] = []
    all_labels: List[str] = []
    rows_json: List[Dict] = []

    for pi, pid in enumerate(Li):
        v = I_t[pi : pi + 1]  # (1, D)

        # 槽位 top-k
        picked: Dict[str, List[Tuple[str, float, int]]] = {}
        for slot, E in embeds.items():
            k = min(max(1, args.topk), E.shape[0])
            s = (v @ E.T).squeeze(0)  # (n_slot,)
            vals, idxs = torch.topk(s, k=k)
            picked[slot] = [
                (phrases[slot][int(i.item())], float(vals[j].item()), int(i.item()))
                for j, i in enumerate(idxs)
            ]

        # 前缀 top-2
        sp = (v @ E_prefix.T).squeeze(0)  # (Np,)
        pk = min(2, E_prefix.shape[0])
        pvals, pidx = torch.topk(sp, k=pk)
        top_prefixes = [
            (PREFIX_POOL[int(i.item())], float(pvals[j].item()))
            for j, i in enumerate(pidx)
        ]

        # 基准强度（上衣/下装的平均相似度）
        anchor = (picked["upper"][0][1] + picked["lower"][0][1]) / 2.0

        # 组装候选
        cands: List[Tuple[str, float]] = []
        for uc, lc, sh in itertools.product(
            picked["upper"], picked["lower"], picked["shoes"]
        ):
            bag_c = (
                picked["bag"][0]
                if gate_keep("bag", picked["bag"][0][1], anchor, args.gate)
                else None
            )
            head_c = (
                picked["head"][0]
                if gate_keep("head", picked["head"][0][1], anchor, args.gate)
                else None
            )

            # 文本片段
            upper = uc[0]  # "red jacket"
            lower = lc[0]  # "blue jeans"
            shoes = sh[0]  # "sneakers"
            bag = (", carrying a " + bag_c[0]) if bag_c else ""
            head = (", wearing a " + head_c[0]) if head_c else ""

            # 3 个模板 × 2 个前缀
            for prefix, pv in top_prefixes:
                for tpl in TEMPLATES:
                    text = tpl.format(
                        prefix=prefix, upper=upper, lower=lower, shoes=shoes, bag=bag, head=head
                    )
                    text = clean_text(text)

                    low = text.lower()
                    if any(b in low for b in BLACKLIST_SUBSTR):
                        continue

                    score = (
                        uc[1]
                        + lc[1]
                        + sh[1]
                        + (bag_c[1] if bag_c else 0.0)
                        + (head_c[1] if head_c else 0.0)
                        + float(args.prefix_pool_weight) * pv
                    )
                    cands.append((text, score))

        # 去重、排序、截断
        cands.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        uniq: List[Tuple[str, float]] = []
        for t, s in cands:
            if t not in seen:
                uniq.append((t, s))
                seen.add(t)
            if len(uniq) >= max(1, int(args.variants)):
                break

        # 收集
        for t, s in uniq:
            all_texts.append(t)
            all_labels.append(pid)
            rows_json.append({"id": pid, "text": t, "score": round(float(s), 4)})

    # 编码 & per-pid 平均
    Z = text_embed(model, tok, all_texts, device).cpu().numpy().astype(np.float32)  # (M, D)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    pid_to_rows: Dict[str, List[int]] = {p: [] for p in Li}
    for i, p in enumerate(all_labels):
        if p in pid_to_rows:
            pid_to_rows[p].append(i)

    Z_avg = np.stack(
        [
            (
                Z[pid_to_rows[p]].mean(axis=0)
                / (np.linalg.norm(Z[pid_to_rows[p]].mean(axis=0)) + 1e-8)
            )
            if pid_to_rows[p]
            else np.zeros((D,), dtype=np.float32)
            for p in Li
        ],
        axis=0,
    ).astype(np.float32)

    # 保存
    os.makedirs(os.path.dirname(args.out_jsonl) or "data/prompts", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows_json:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    label_dir = os.path.dirname(args.out_labels) or "data/prompts"
    os.makedirs(label_dir, exist_ok=True)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        f.write("\n".join(all_labels) + "\n")

    os.makedirs(os.path.dirname(args.out_text_emb_all) or "embeds/text", exist_ok=True)
    np.save(args.out_text_emb_all, Z)
    np.save(args.out_text_emb_avg, Z_avg)

    print(f"[v3] saved: {args.out_jsonl} (lines={len(rows_json)})")
    print(f"[v3] saved: {args.out_labels}")
    print(f"[v3] saved: {args.out_text_emb_all}  shape={Z.shape}")
    print(f"[v3] saved: {args.out_text_emb_avg}  shape={Z_avg.shape}")

    # 评测（可选）
    def eval_pair(T: np.ndarray, I: np.ndarray, Lt: List[str], Li_: List[str]):
        Tn = l2n_np(T)
        In = l2n_np(I)
        S = Tn @ In.T
        order = np.argsort(-S, axis=1)

        def r(k: int) -> float:
            hit = 0
            for i, y in enumerate(Lt):
                ranked = [Li_[j] for j in order[i, :k]]
                if y in ranked:
                    hit += 1
            return hit / len(Lt)

        # mAP
        mAP = 0.0
        for i, y in enumerate(Lt):
            ranked = [Li_[j] for j in order[i]]
            tot = ranked.count(y)
            if tot == 0:
                continue
            h = 0
            ps = 0.0
            for rnk, p in enumerate(ranked, 1):
                if p == y:
                    h += 1
                    ps += h / rnk
            mAP += ps / tot

        top1 = float(S.max(axis=1).mean())
        return r(1), r(5), r(10), mAP / len(Lt), top1

    r1, r5, r10, mapv, top1 = eval_pair(Z_avg, I, Li, Li)
    print(
        f"\n== AvgText(v3) vs ImageCentroids ==\n"
        f"R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  mAP={mapv:.4f}  AvgTop1={top1:.4f}"
    )

    if args.gallery and args.gallery_labels:
        G = np.load(args.gallery).astype(np.float32)
        Lg = rl(args.gallery_labels)
        r1, r5, r10, mapv, top1 = eval_pair(Z_avg, G, Li, Lg)
        print(
            f"\n== AvgText(v3) vs Gallery(Images) ==\n"
            f"R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  mAP={mapv:.4f}  AvgTop1={top1:.4f}"
        )


if __name__ == "__main__":
    main()
