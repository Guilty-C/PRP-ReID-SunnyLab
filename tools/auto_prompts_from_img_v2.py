# tools/auto_prompts_from_img_v2.py
from __future__ import annotations
import os, json, argparse, itertools
from typing import List, Dict, Tuple
import numpy as np
import torch
import open_clip

# ---------------------------
# Utils
# ---------------------------
def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def l2n(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def text_embed(model, tok, texts: List[str], dev: torch.device, bs: int = 128) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), bs):
        toks = tok(texts[i:i+bs]).to(dev)
        z = model.encode_text(toks).float()
        out.append(l2n(z))
    return torch.cat(out, dim=0).to(dev)

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

# ---------------------------
# Phrase space (v2, finer)
# ---------------------------
BASE_COLORS = [
    "black","white","red","blue","navy blue","light blue","dark blue",
    "green","olive","yellow","orange","brown","beige","khaki",
    "gray","light gray","dark gray","purple","pink"
]

UPPER_TYPES  = ["t-shirt","shirt","jacket","coat","hoodie","sweater","suit jacket"]
UPPER_PATTERNS = ["striped","plaid","solid color","with a logo"]
LOWER_COLORS = BASE_COLORS
LOWER_TYPES  = ["jeans","long pants","shorts","skirt","trousers","sweatpants"]
SHOES_TYPES  = ["sneakers","boots","dress shoes","sandals","running shoes"]
BAG_TYPES    = ["carrying a backpack","carrying a shoulder bag","carrying a crossbody bag","carrying a handbag","no bag"]
HEAD_ITEMS   = ["wearing a hat","wearing a cap","hood up","no headwear"]

def build_slot_phrases() -> Dict[str, List[str]]:
    phrases = {
        "upper_color":  [f"wearing a {c} top" for c in BASE_COLORS],
        "upper_type":   [f"wearing a {t}" for t in UPPER_TYPES],
        "upper_pattern":[f"{p} top" for p in UPPER_PATTERNS],
        "lower_color":  [f"wearing {c} bottoms" for c in LOWER_COLORS],
        "lower_type":   [f"wearing {t}" for t in LOWER_TYPES],
        "shoes":        [f"wearing {t}" for t in SHOES_TYPES],
        "bag":          BAG_TYPES[:],
        "head":         HEAD_ITEMS[:],
    }
    return phrases

TEMPLATES = [
    "{prefix}{uc}, {ut}, {up}, {lc} {lt}, {sh}, {bg}, {hd}",
    "{prefix}{ut} in {uc_plain}, {lt} in {lc_plain}, {sh}, {bg}, {hd}",
    "{prefix}{uc}{pat_clause}, {ut}, {lc} {lt}, {sh}, {bg}, {hd}",
]

def pattern_clause(up: str) -> str:
    # up like "striped top" / "solid color top" / "with a logo top"
    if up.startswith("solid color"): return ""
    seg = up[:-4].strip() if up.endswith(" top") else up
    return f" with a {seg}"

def clean_commas(s: str) -> str:
    s = s.replace(",,", ",").replace(", ,", ", ").replace(" ,", ",")
    while "  " in s: s = s.replace("  ", " ")
    return s.strip()

# ---------------------------
# Core logic
# ---------------------------
@torch.no_grad()
def choose_topk_per_slot(
    v: torch.Tensor, embeds: Dict[str, torch.Tensor], phrases: Dict[str, List[str]], topk: int
) -> Dict[str, List[Tuple[str, float]]]:
    picked: Dict[str, List[Tuple[str, float]]] = {}
    for slot, E in embeds.items():
        s = (v @ E.T).squeeze(0)  # (N,)
        vals, idxs = torch.topk(s, k=min(topk, E.shape[0]))
        picked[slot] = [(phrases[slot][int(i.item())], float(vv.item())) for vv, i in zip(vals, idxs)]
    return picked

def assemble_variants(picked: Dict[str, List[Tuple[str,float]]], max_variants: int) -> List[Tuple[str, float, Dict[str,str]]]:
    """
    组合多槽位 topk 短语，按分数和排序，取前 max_variants。
    返回：(文本, 分数, 选中短语dict)
    """
    slots_order = ["upper_color","upper_type","upper_pattern","lower_color","lower_type","shoes","bag","head"]
    cand_lists = [picked[s] for s in slots_order]
    combos = list(itertools.product(*cand_lists))  # K^S 组合

    ranked = []
    for combo in combos:
        choice = {s: ph for s, (ph, sc) in zip(slots_order, combo)}
        score  = sum(sc for (_, sc) in combo)

        prefix = "surveillance photo of a pedestrian, full-body, outdoors, "
        uc, ut, up, lc, lt, sh, bg, hd = (
            choice["upper_color"], choice["upper_type"], choice["upper_pattern"],
            choice["lower_color"], choice["lower_type"], choice["shoes"],
            choice["bag"], choice["head"]
        )

        # 预处理去掉 "wearing"/"wearing a"
        def strip_wearing(s: str) -> str:
            return s.replace("wearing a ", "").replace("wearing ", "")

        uc_plain = strip_wearing(uc)
        lc_plain = strip_wearing(lc)

        # 三个模板
        t1 = TEMPLATES[0].format(prefix=prefix, uc=uc, ut=ut, up=up, lc=lc, lt=lt, sh=sh, bg=bg, hd=hd)
        t2 = TEMPLATES[1].format(prefix=prefix, uc_plain=uc_plain, ut=ut, lc_plain=lc_plain, lt=lt, sh=sh, bg=bg, hd=hd)
        t3 = TEMPLATES[2].format(prefix=prefix, uc=uc, pat_clause=pattern_clause(up), ut=ut, lc=lc, lt=lt, sh=sh, bg=bg, hd=hd)

        texts = [t1, t2, t3]
        # 去重并清理
        seen=set(); uniq=[]
        for t in texts:
            t = clean_commas(t)
            if t not in seen:
                uniq.append(t); seen.add(t)

        for t in uniq:
            ranked.append((t, score, choice))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:max_variants]

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("AutoPrompt v2: multi-variant slotting + avg-by-pid")
    ap.add_argument("--img-centroids", required=True, help="(P,D) image centroids .npy")
    ap.add_argument("--img-centroids-labels", required=True, help="pids .txt for centroids")
    ap.add_argument("--gallery", default=None, help="(N,D) raw image embeds .npy (optional)")
    ap.add_argument("--gallery-labels", default=None, help="labels .txt for gallery (optional)")
    ap.add_argument("--topk", type=int, default=2, help="Top-K candidates per slot")
    ap.add_argument("--variants", type=int, default=8, help="Max variants per pid (after templates)")
    ap.add_argument("--out-jsonl", default="data/prompts/market-50.auto_v2.jsonl")
    ap.add_argument("--out-labels", default="data/prompts/market-50.auto_v2.labels.txt")
    ap.add_argument("--out-text-emb-avg", default="embeds/text/clip-b16_50.auto_v2.avg_by_pid.npy")
    ap.add_argument("--out-text-emb-all", default="embeds/text/clip-b16_50.auto_v2.all.npy")
    ap.add_argument("--model", default=os.getenv("CLIP_LOCAL_MODEL", "ViT-B-16"))
    ap.add_argument("--pretrained", default=os.getenv("CLIP_LOCAL_PRETRAIN", "laion2b_s34b_b88k"))
    args = ap.parse_args()

    dev = device()
    print(f"[v2] model={args.model}  pretrain={args.pretrained}  device={dev}")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=dev)
    model.eval()
    tok = open_clip.get_tokenizer(args.model)

    # 1) 类中心
    I = torch.tensor(np.load(args.img_centroids), dtype=torch.float32, device=dev)
    Li = read_lines(args.img_centroids_labels)
    assert I.shape[0] == len(Li), "Centroids and labels mismatch"
    I = l2n(I)

    # 2) 槽位短语 & 向量（在 dev）
    phrases = build_slot_phrases()
    embeds = {slot: text_embed(model, tok, lst, dev) for slot, lst in phrases.items()}

    # 3) 为每个 pid 组合多版本 prompts
    all_prompts: List[Dict] = []
    rows_pid_idx = []
    for idx in range(I.shape[0]):
        v = I[idx:idx+1, :]
        picked = choose_topk_per_slot(v, embeds, phrases, topk=args.topk)
        variants = assemble_variants(picked, max_variants=args.variants)
        pid = Li[idx]
        for (text, score, choice) in variants:
            all_prompts.append({"id": pid, "text": text, "score": round(score,4), "choice": choice})
            rows_pid_idx.append(idx)
    print(f"[v2] generated {len(all_prompts)} prompt variants for {len(Li)} pids "
          f"(~{len(all_prompts)//len(Li)} per pid)")

    # 4) 保存 JSONL + labels（all）
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for obj in all_prompts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(args.out_labels, "w", encoding="utf-8") as f:
        f.write("\n".join([obj["id"] for obj in all_prompts]) + "\n")
    print(f"[v2] saved: {args.out_jsonl}  lines={len(all_prompts)}")
    print(f"[v2] saved: {args.out_labels}")

    # 5) 编码所有文本变体，按 pid 均值聚合
    texts = [obj["text"] for obj in all_prompts]
    T_all = text_embed(model, tok, texts, dev)                 # (M,D)
    np.save(args.out_text_emb_all, T_all.detach().cpu().numpy().astype(np.float32))
    print(f"[v2] saved: {args.out_text_emb_all}  shape={tuple(T_all.shape)}")

    # 按 pid 聚合
    P = I.shape[0]; D = T_all.shape[1]
    T_avg = torch.zeros((P, D), dtype=torch.float32, device=dev)
    counts = torch.zeros((P,), dtype=torch.float32, device=dev)
    for row, pid_idx in enumerate(rows_pid_idx):
        T_avg[pid_idx] += T_all[row]
        counts[pid_idx] += 1.0
    T_avg = l2n(T_avg / counts.unsqueeze(-1))
    os.makedirs(os.path.dirname(args.out_text_emb_avg), exist_ok=True)
    np.save(args.out_text_emb_avg, T_avg.detach().cpu().numpy().astype(np.float32))
    print(f"[v2] saved: {args.out_text_emb_avg}  shape={tuple(T_avg.shape)}")

    # 6) 评测 1：AvgText(v2) vs Centroids
    def eval_pair(T: torch.Tensor, I: torch.Tensor, Lt: List[str], Li: List[str], title: str):
        T = l2n(T); I = l2n(I)
        S = T @ I.T
        top1 = float(S.max(dim=1).values.mean().item())
        order = torch.argsort(S, dim=1, descending=True)
        def r_at(k: int) -> float:
            hit = 0
            for i, y in enumerate(Lt):
                ranked = [Li[j] for j in order[i, :k].tolist()]
                if y in ranked: hit += 1
            return hit / len(Lt)
        # mAP
        mAP = 0.0
        for i, y in enumerate(Lt):
            ranked = [Li[j] for j in order[i].tolist()]
            tot = ranked.count(y)
            if tot == 0: continue
            h = 0; ps = 0.0
            for r, p in enumerate(ranked, start=1):
                if p == y: h += 1; ps += h / r
            mAP += ps / tot
        mAP /= len(Lt)
        print(f"\n== {title} ==")
        print(f"R@1={r_at(1):.4f}  R@5={r_at(5):.4f}  R@10={r_at(10):.4f}  mAP={mAP:.4f}  AvgTop1={top1:.4f}")

    eval_pair(T_avg, I, Lt=Li, Li=Li, title="AvgText(v2) vs ImageCentroids")

    # 7) 评测 2（可选）：AvgText(v2) vs 原始861张图
    if args.gallery and args.gallery_labels:
        G = torch.tensor(np.load(args.gallery), dtype=torch.float32, device=dev)
        Lg = read_lines(args.gallery_labels)
        assert G.shape[0] == len(Lg), "gallery mismatch"
        eval_pair(T_avg, G, Lt=Li, Li=Lg, title="AvgText(v2) vs Gallery(Images)")
    else:
        print("[v2] Skip gallery eval (provide --gallery & --gallery-labels to enable)")

if __name__ == "__main__":
    main()
