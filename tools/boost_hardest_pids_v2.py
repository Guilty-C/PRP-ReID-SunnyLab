# tools/boost_hardest_pids_v2.py
from __future__ import annotations
import os, json, argparse, itertools
from typing import List, Dict, Tuple
import numpy as np
import torch
import open_clip

# ---------------------------
# Helpers
# ---------------------------
def dev():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def l2n_np(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def l2n_t(T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return T / (T.norm(dim=-1, keepdim=True) + eps)

def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

@torch.no_grad()
def text_embed(model, tok, texts: List[str], device: torch.device, bs: int = 128) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), bs):
        toks = tok(texts[i:i+bs]).to(device)
        z = model.encode_text(toks).float()
        out.append(l2n_t(z))
    return torch.cat(out, dim=0).to(device)

# ---------------------------
# Phrase space (same as v2)
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
    return {
        "upper_color":  [f"wearing a {c} top" for c in BASE_COLORS],
        "upper_type":   [f"wearing a {t}" for t in UPPER_TYPES],
        "upper_pattern":[f"{p} top" for p in UPPER_PATTERNS],
        "lower_color":  [f"wearing {c} bottoms" for c in LOWER_COLORS],
        "lower_type":   [f"wearing {t}" for t in LOWER_TYPES],
        "shoes":        [f"wearing {t}" for t in SHOES_TYPES],
        "bag":          BAG_TYPES[:],
        "head":         HEAD_ITEMS[:],
    }

TEMPLATES = [
    "{prefix}{uc}, {ut}, {up}, {lc} {lt}, {sh}, {bg}, {hd}",
    "{prefix}{ut} in {uc_plain}, {lt} in {lc_plain}, {sh}, {bg}, {hd}",
    "{prefix}{uc}{pat_clause}, {ut}, {lc} {lt}, {sh}, {bg}, {hd}",
]

def pattern_clause(up: str) -> str:
    if up.startswith("solid color"): return ""
    seg = up[:-4].strip() if up.endswith(" top") else up
    return f" with a {seg}"

def clean_commas(s: str) -> str:
    s = s.replace(",,", ",").replace(", ,", ", ").replace(" ,", ",")
    while "  " in s: s = s.replace("  ", " ")
    return s.strip()

@torch.no_grad()
def choose_topk_per_slot(v: torch.Tensor, embeds: Dict[str, torch.Tensor], phrases: Dict[str, List[str]], topk: int):
    picked: Dict[str, List[Tuple[str, float]]] = {}
    for slot, E in embeds.items():
        s = (v @ E.T).squeeze(0)  # (N,)
        vals, idxs = torch.topk(s, k=min(topk, E.shape[0]))
        picked[slot] = [(phrases[slot][int(i.item())], float(vv.item())) for vv, i in zip(vals, idxs)]
    return picked

def assemble_variants(picked: Dict[str, List[Tuple[str,float]]], max_variants: int):
    slots_order = ["upper_color","upper_type","upper_pattern","lower_color","lower_type","shoes","bag","head"]
    cand_lists = [picked[s] for s in slots_order]
    combos = list(itertools.product(*cand_lists))

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
        def strip_wearing(s: str): return s.replace("wearing a ", "").replace("wearing ", "")
        uc_plain = strip_wearing(uc); lc_plain = strip_wearing(lc)
        t1 = TEMPLATES[0].format(prefix=prefix, uc=uc, ut=ut, up=up, lc=lc, lt=lt, sh=sh, bg=bg, hd=hd)
        t2 = TEMPLATES[1].format(prefix=prefix, uc_plain=uc_plain, ut=ut, lc_plain=lc_plain, lt=lt, sh=sh, bg=bg, hd=hd)
        t3 = TEMPLATES[2].format(prefix=prefix, uc=uc, pat_clause=pattern_clause(up), ut=ut, lc=lc, lt=lt, sh=sh, bg=bg, hd=hd)
        texts = [t1, t2, t3]
        seen=set(); uniq=[]
        for t in texts:
            t = clean_commas(t)
            if t not in seen:
                uniq.append(t); seen.add(t)
        for t in uniq:
            ranked.append((t, score, choice))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:max_variants]

# ---------------------------
# Hardest mining & boosting
# ---------------------------
def mine_hardest_pids(T_avg: np.ndarray, Lt: List[str], G: np.ndarray, Lg: List[str], topk_pids: int = 10):
    """按 Gallery 排名难度挖最难 pid。以“正确 pid 的最佳名次”优先，tie-break 用 top1 相似度。"""
    Tn = l2n_np(T_avg); Gn = l2n_np(G)
    S = Tn @ Gn.T  # (P, N)
    hardest = []
    for i, pid in enumerate(Lt):
        s = S[i]
        order = np.argsort(-s)  # desc
        # 找该 pid 在 gallery 的所有索引与最好名次
        idxs = [j for j, lab in enumerate(Lg) if lab == pid]
        if idxs:
            ranks = [int(np.where(order==j)[0][0]) + 1 for j in idxs]
            best_rank = min(ranks)
        else:
            best_rank = G.shape[0] + 1  # 不存在则最差
        top1_sim = float(s.max())
        hardest.append((pid, best_rank, top1_sim))
    # 排序规则：先按 best_rank 降序（名次越大越难），再按 top1_sim 升序（相似度越低越难）
    hardest.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    return [h[0] for h in hardest[:topk_pids]], hardest

def main():
    ap = argparse.ArgumentParser("Boost hardest pids with extra prompt variants (v2-compatible)")
    ap.add_argument("--img-centroids", required=True)
    ap.add_argument("--img-centroids-labels", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--gallery-labels", required=True)

    ap.add_argument("--base-all-embeds", required=True, help="existing v2 all text embeds (M,D)")
    ap.add_argument("--base-all-labels", required=True, help="labels for v2 all (M lines, each pid)")
    ap.add_argument("--base-avg-embeds", required=True, help="existing v2 avg_by_pid (P,D)")

    ap.add_argument("--hardest-k", type=int, default=10)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--variants", type=int, default=20)

    ap.add_argument("--out-jsonl", default="data/prompts/market-50.auto_v2.l14.boost.jsonl")
    ap.add_argument("--out-all", default="embeds/text/clip-l14_50.auto_v2.all.boost.npy")
    ap.add_argument("--out-all-labels", default="data/prompts/market-50.auto_v2.l14.labels.boost.txt")
    ap.add_argument("--out-avg", default="embeds/text/clip-l14_50.auto_v2.avg_by_pid.boost.npy")

    ap.add_argument("--model", default=os.getenv("CLIP_LOCAL_MODEL","ViT-L-14"))
    ap.add_argument("--pretrained", default=os.getenv("CLIP_LOCAL_PRETRAIN","datacomp_xl_s13b_b90k"))
    args = ap.parse_args()

    # 读入数据
    I = np.load(args.img_centroids)                   # (P,D)
    Li = read_lines(args.img_centroids_labels)        # P
    G = np.load(args.gallery)                         # (N,D)
    Lg = read_lines(args.gallery_labels)              # N
    assert I.shape[0]==len(Li) and G.shape[0]==len(Lg)

    T_all = np.load(args.base_all_embeds)             # (M,D)
    L_all = read_lines(args.base_all_labels)          # M
    T_avg = np.load(args.base_avg_embeds)             # (P,D)
    assert len(L_all)==T_all.shape[0]

    # 统一顺序：按 Li（centroid 顺序）排列 avg 文本
    # 这里假设 T_avg 已经对齐 Li（就是你从 v2 脚本输出的 avg_by_pid）
    Lt = Li[:]  # avg 的 pid 顺序

    # 1) 挖 hardest pids
    hardest_pids, full_stats = mine_hardest_pids(T_avg, Lt, G, Lg, topk_pids=args.hardest_k)
    print(f"[boost] hardest pids (k={args.hardest_k}): {hardest_pids}")

    # 2) 模型 & 槽位短语向量
    device = dev()
    print(f"[boost] model={args.model}  pretrain={args.pretrained}  device={device}")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(args.model)
    phrases = build_slot_phrases()
    embeds = {slot: text_embed(model, tok, lst, device) for slot, lst in phrases.items()}

    # 3) 为每个 hardest pid 生成更多变体
    pid_to_idx = {p:i for i,p in enumerate(Li)}   # centroid 顺序
    all_new_texts = []
    all_new_labels = []
    jsonl_rows = []

    I_t = torch.tensor(I, dtype=torch.float32, device=device)
    I_t = l2n_t(I_t)

    for pid in hardest_pids:
        idx = pid_to_idx[pid]
        v = I_t[idx:idx+1, :]
        picked = choose_topk_per_slot(v, embeds, phrases, topk=args.topk)
        variants = assemble_variants(picked, max_variants=args.variants)
        for (text, score, choice) in variants:
            all_new_texts.append(text)
            all_new_labels.append(pid)
            jsonl_rows.append({"id": pid, "text": text, "score": round(score,4), "choice": choice})

    # 4) 编码新增文本并与旧 all 合并
    T_new = text_embed(model, tok, all_new_texts, device)
    T_new_np = T_new.detach().cpu().numpy().astype(np.float32)
    T_all_boost = np.concatenate([T_all, T_new_np], axis=0)
    L_all_boost = L_all + all_new_labels

    # 保存 JSONL & 新的 all
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for obj in jsonl_rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(args.out_all), exist_ok=True)
    np.save(args.out_all, T_all_boost)
    with open(args.out_all_labels, "w", encoding="utf-8") as f:
        f.write("\n".join(L_all_boost) + "\n")
    print(f"[boost] saved: {args.out_jsonl} (lines={len(jsonl_rows)})")
    print(f"[boost] saved: {args.out_all}  shape={T_all_boost.shape}")
    print(f"[boost] saved: {args.out_all_labels}")

    # 5) 按 pid 重新均值（旧变体 + 新增变体）
    # 统计每个 pid 的所有索引
    idxs_by_pid = {p: [] for p in Li}
    for i, p in enumerate(L_all_boost):
        if p in idxs_by_pid:
            idxs_by_pid[p].append(i)
    P, D = I.shape[0], T_all_boost.shape[1]
    T_avg_boost = np.zeros((P, D), dtype=np.float32)
    for pi, pid in enumerate(Li):
        rows = idxs_by_pid[pid]
        if not rows:
            # 理论不会发生（原来每个 pid 就有 variants），兜底用旧 avg
            T_avg_boost[pi] = T_avg[pi]
        else:
            M = l2n_np(T_all_boost[rows])
            T_avg_boost[pi] = l2n_np(M.mean(axis=0, keepdims=True))[0]

    np.save(args.out_avg, T_avg_boost)
    print(f"[boost] saved: {args.out_avg}  shape={T_avg_boost.shape}")

    # 6) 简易评测（Centroids + Gallery）
    def eval_pair(T, I, Lt, Li):
        Tn = l2n_np(T); In = l2n_np(I)
        S = Tn @ In.T
        order = np.argsort(-S, axis=1)
        def r_at(k):
            hit=0
            for i,y in enumerate(Lt):
                ranked = [Li[j] for j in order[i,:k]]
                if y in ranked: hit+=1
            return hit/len(Lt)
        # mAP
        mAP=0.0
        for i,y in enumerate(Lt):
            ranked=[Li[j] for j in order[i]]
            tot=ranked.count(y)
            if tot==0: continue
            h=0; ps=0.0
            for r,p in enumerate(ranked,1):
                if p==y: h+=1; ps+=h/r
            mAP+=ps/tot
        return r_at(1), r_at(5), r_at(10), mAP/len(Lt)

    r1,r5,r10,mapv = eval_pair(T_avg_boost, I, Li, Li)
    print(f"[eval] Centroids | R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} mAP={mapv:.3f}")
    r1,r5,r10,mapv = eval_pair(T_avg_boost, G, Li, Lg)
    print(f"[eval] Gallery   | R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} mAP={mapv:.3f}")

if __name__ == "__main__":
    main()
