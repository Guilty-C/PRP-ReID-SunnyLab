# tools/auto_prompts_from_img_v3p.py
import os, json, argparse, random
from collections import defaultdict
import numpy as np
import torch
import open_clip

# ----------------- 词表与归一 -----------------
COLOR_CANON = {
    "navy":"blue","dark blue":"blue","indigo":"blue","sky blue":"blue",
    "khaki":"beige","tan":"beige","cream":"beige",
    "grey":"gray","light grey":"gray","dark grey":"gray",
    "maroon":"red","burgundy":"red",
    "teal":"green","olive":"green",
    "violet":"purple","lavender":"purple",
}
COLORS = ["black","white","gray","blue","green","red","yellow","orange","purple","pink","brown","beige"]
UPPER = ["jacket","hoodie","coat","sweater","shirt","t-shirt","polo","cardigan"]
LOWER = ["jeans","trousers","pants","shorts","skirt"]
SHOES = ["sneakers","shoes","boots","loafers","sandals"]
PREFIX_POOL = [
    "a street photo of a person",
    "a candid photo of a pedestrian",
    "a surveillance-style photo of a person",
    "a full-body photo of a person",
]

def canon_color(c):
    c = c.lower().strip()
    return COLOR_CANON.get(c, c)

# ----------------- 工具函数 -----------------
def l2n(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def torch_l2n(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def encode_texts(model, tokenizer, device, texts, bs=256):
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            tok = tokenizer(texts[i:i+bs]).to(device)
            feat = model.encode_text(tok)
            outs.append(torch_l2n(feat))
    return torch.cat(outs, 0)

def pick_hard_negs(v, label, C, L, G=None, LG=None, k_c=6, k_g=6):
    negs = []
    sc = C @ v
    for j in np.argsort(-sc):
        if L[j] != label:
            negs.append(C[j])
        if len(negs) >= k_c: break
    if G is not None and LG is not None:
        sg = G @ v
        cnt = 0
        for j in np.argsort(-sg):
            if LG[j] != label:
                negs.append(G[j]); cnt += 1
            if cnt >= k_g: break
    return np.stack(negs, 0) if len(negs) else None

def render(prefix, uc, ut, lc, lt, sh):
    return f"{prefix}, wearing a {uc} {ut} and {lc} {lt}, {sh}."

# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-centroids", required=True)
    ap.add_argument("--img-centroids-labels", required=True)
    ap.add_argument("--gallery")
    ap.add_argument("--gallery-labels")
    ap.add_argument("--variants", type=int, default=12, help="每个ID保留的最终高分prompt条数")
    ap.add_argument("--gate", type=float, default=0.10, help="最低正类相似（自适应会在兜底时放宽）")
    ap.add_argument("--prefix-pool-weight", type=float, default=0.25)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--out-text-emb-all", required=True)
    ap.add_argument("--out-text-emb-avg", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.environ.get("CLIP_LOCAL_MODEL", "ViT-L-14")
    pretrain   = os.environ.get("CLIP_LOCAL_PRETRAIN", "datacomp_xl_s13b_b90k")
    print(f"[v3p] model={model_name}  pretrain={pretrain}  device={device}")

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrain, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    C = l2n(np.load(args.img_centroids).astype(np.float32))
    L = [ln.strip() for ln in open(args.img_centroids_labels, encoding="utf-8") if ln.strip()]
    assert C.shape[0] == len(L)
    D = C.shape[1]

    G = LG = None
    if args.gallery and args.gallery_labels:
        G  = l2n(np.load(args.gallery).astype(np.float32))
        LG = [ln.strip() for ln in open(args.gallery_labels, encoding="utf-8") if ln.strip()]
        print(f"[v3p] gallery loaded: {G.shape}")

    C_t = torch.from_numpy(C).to(device)
    G_t = torch.from_numpy(G).to(device) if G is not None else None

    all_prompts, all_labels, all_text_embs = [], [], []

    for i, pid in enumerate(L):
        v = C[i]; v_t = C_t[i]
        negs = pick_hard_negs(v, pid, C, L, G, LG, 6, 6)
        negs_t = torch.from_numpy(negs).to(device) if negs is not None else None

        rng = random.Random(hash(pid) & 0xffffffff)
        # 提前挑 prefix
        n_pref = max(1, int(round(args.prefix_pool_weight * len(PREFIX_POOL))))
        prefixes = rng.sample(PREFIX_POOL, k=n_pref)

        # ---------- (1) 先独立为每个槽位挑“最像正类、最不像负类”的候选 ----------
        def best_slot_choice(slot, pool, uc, ut, lc, lt, sh, gate=args.gate):
            # 返回[(score, text, feat, token), ...]（已按 score 降序）
            cands = []
            texts = []
            for tok in pool:
                uc1,ut1,lc1,lt1,sh1 = uc,ut,lc,lt,sh
                if slot=="uc": uc1 = tok
                elif slot=="ut": ut1 = tok
                elif slot=="lc": lc1 = tok
                elif slot=="lt": lt1 = tok
                elif slot=="sh": sh1 = tok
                for prefix in prefixes:
                    texts.append(render(prefix, uc1, ut1, lc1, lt1, sh1))
            T = encode_texts(model, tokenizer, device, texts)
            # 正类相似
            s_pos = (T @ v_t)  # (M,)
            # 过滤太差的
            keep = (s_pos >= gate).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                keep = torch.arange(T.size(0), device=T.device)  # 放宽兜底
            T = T[keep]; s_pos = s_pos[keep]
            # 负类均值
            if negs_t is not None and negs_t.numel() > 0:
                s_neg = torch.matmul(negs_t, T.T).mean(0)  # (M,)
                score = s_pos - s_neg
            else:
                score = s_pos
            # 汇总
            kept_texts = [texts[j.item()] for j in keep]
            order = torch.argsort(score, descending=True).tolist()
            out = []
            for j in order:
                out.append((score[j].item(), kept_texts[j], T[j].cpu().numpy().astype(np.float32)))
            return out

        # 先用中性值初始化（保证可组合）
        init_uc, init_ut, init_lc, init_lt, init_sh = "black","jacket","black","pants","shoes"

        # 挑每个槽位的 Top-K（K=2）备选
        topK = 2
        uc_list = [canon_color(c) for c in COLORS]
        lc_list = [canon_color(c) for c in COLORS]

        top_uc = best_slot_choice("uc", uc_list, init_uc, init_ut, init_lc, init_lt, init_sh)[:topK]
        top_ut = best_slot_choice("ut", UPPER,    init_uc, init_ut, init_lc, init_lt, init_sh)[:topK]
        top_lc = best_slot_choice("lc", lc_list,  init_uc, init_ut, init_lc, init_lt, init_sh)[:topK]
        top_lt = best_slot_choice("lt", LOWER,    init_uc, init_ut, init_lc, init_lt, init_sh)[:topK]
        top_sh = best_slot_choice("sh", SHOES,    init_uc, init_ut, init_lc, init_lt, init_sh)[:topK]

        # ---------- (2) 组合这些 Top-K 形成候选，再统一按 margin 选前 variants ----------
        combo_texts = []
        for s_uc,_ in [(t.split("wearing a ")[1].split(" ")[0],t) for _,t,_ in top_uc]:
            s_uc = canon_color(s_uc)
            for token_ut,_ in [(t.split("wearing a ")[1].split(" ")[1].strip(",."),t) for _,t,_ in top_ut]:
                for s_lc,_ in [(t.split("and ")[1].split(" ")[0],t) for _,t,_ in top_lc]:
                    s_lc = canon_color(s_lc)
                    for token_lt,_ in [(t.split("and ")[1].split(" ")[1].strip(",."),t) for _,t,_ in top_lt]:
                        for token_sh,_ in [(t.split(",")[-1].strip().strip("."),t) for _,t,_ in top_sh]:
                            for prefix in prefixes:
                                combo_texts.append(render(prefix, s_uc, token_ut, s_lc, token_lt, token_sh))

        # 去重
        combo_texts = list(dict.fromkeys(combo_texts))
        if len(combo_texts) == 0:
            # 兜底一条
            combo_texts = [render(prefixes[0], init_uc, init_ut, init_lc, init_lt, init_sh)]

        T_all = encode_texts(model, tokenizer, device, combo_texts)
        s_pos = (T_all @ v_t)
        if negs_t is not None and negs_t.numel() > 0:
            s_neg = torch.matmul(negs_t, T_all.T).mean(0)
            score = s_pos - s_neg
        else:
            score = s_pos
        order = torch.argsort(score, descending=True).tolist()
        keepN = min(args.variants, len(order))
        sel = order[:keepN]

        # 收集输出
        for j in sel:
            all_prompts.append({"id": pid, "text": combo_texts[j]})
            all_labels.append(pid)
        all_text_embs.append(T_all[sel].cpu().numpy().astype(np.float32))

    E = np.concatenate(all_text_embs, 0)  # (50*variants, D)

    # PID 平均
    by_pid = defaultdict(list)
    for obj, e in zip(all_prompts, E):
        by_pid[obj["id"]].append(e)
    pids_sorted = sorted(by_pid.keys())
    A = np.stack([l2n(np.stack(by_pid[p], 0)).mean(0) for p in pids_sorted], 0)
    A = l2n(A).astype(np.float32)

    # 保存
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_text_emb_all), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for o in all_prompts: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    with open(args.out_labels, "w", encoding="utf-8") as f:
        f.write("\n".join([o["id"] for o in all_prompts]) + "\n")
    np.save(args.out_text_emb_all, E)
    np.save(args.out_text_emb_avg, A)
    print(f"[v3p] saved: {args.out_jsonl} (lines={len(all_prompts)})")
    print(f"[v3p] saved: {args.out_labels}")
    print(f"[v3p] saved: {args.out_text_emb_all}  shape={E.shape}")
    print(f"[v3p] saved: {args.out_text_emb_avg}  shape={A.shape}")

    # 简易评测
    def quick(T, I, Lt, Li, tag):
        Tn, In = l2n(T), l2n(I)
        S = Tn @ In.T
        order = np.argsort(-S, axis=1)
        r1=r5=r10=0
        for i,y in enumerate(Lt):
            ranked=[Li[j] for j in order[i]]
            r1+=(ranked[0]==y); r5+=(y in ranked[:5]); r10+=(y in ranked[:10])
        print(f"{tag} | R@1={r1/len(Lt):.4f}  R@5={r5/len(Lt):.4f}  R@10={r10/len(Lt):.4f}  AvgTop1={S.max(1).mean():.4f}")

    quick(A, C, pids_sorted, L, "== AvgText(v3p) vs ImageCentroids ==")
    if G is not None and LG is not None:
        quick(A, G, pids_sorted, LG, "== AvgText(v3p) vs Gallery(Images) ==")

if __name__ == "__main__":
    main()
