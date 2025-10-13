# tools/prompt_tune_margin_v2.py  (fixed)
import os, argparse, json, random
import numpy as np
import torch
import open_clip

# 与 v3p 同步的词表
COLOR_CANON = {
    "navy":"blue","dark blue":"blue","indigo":"blue","sky blue":"blue",
    "khaki":"beige","tan":"beige","cream":"beige",
    "grey":"gray","light grey":"gray","dark grey":"gray",
    "maroon":"red","burgundy":"red",
    "teal":"green","olive":"green",
    "violet":"purple","lavender":"purple",
}
COLORS = ["black","white","gray","blue","green","red","yellow","orange","purple","pink","brown","beige"]
UPPER  = ["jacket","hoodie","coat","sweater","shirt","t-shirt","polo","cardigan"]
LOWER  = ["jeans","trousers","pants","shorts","skirt"]
SHOES  = ["sneakers","shoes","boots","loafers","sandals"]
PREFIX_POOL = [
    "a street photo of a person",
    "a candid photo of a pedestrian",
    "a surveillance-style photo of a person",
    "a full-body photo of a person",
]

def canon_color(c): return COLOR_CANON.get(c.lower().strip(), c.lower().strip())
def l2n(x, eps=1e-8): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)
def torch_l2n(x, eps=1e-8): return x/(x.norm(dim=-1, keepdim=True)+eps)
def render(prefix, uc, ut, lc, lt, sh): return f"{prefix}, wearing a {uc} {ut} and {lc} {lt}, {sh}."

def encode_texts(model, tokenizer, device, texts, bs=256):
    outs=[]
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            tok = tokenizer(texts[i:i+bs]).to(device)
            feat = model.encode_text(tok)
            outs.append(torch_l2n(feat))
    return torch.cat(outs, 0)

def pick_hard_negs(v, label, C, L, G=None, LG=None, k_c=6, k_g=6):
    negs=[]
    sc = C @ v
    for j in np.argsort(-sc):
        if L[j]!=label: negs.append(C[j])
        if len(negs)>=k_c: break
    if G is not None and LG is not None:
        sg = G @ v; cnt=0
        for j in np.argsort(-sg):
            if LG[j]!=label: negs.append(G[j]); cnt+=1
            if cnt>=k_g: break
    return np.stack(negs,0) if len(negs) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-centroids", required=True)
    ap.add_argument("--img-centroids-labels", required=True)
    ap.add_argument("--gallery")
    ap.add_argument("--gallery-labels")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--gate", type=float, default=0.10)   # 仅用于初始化参考门槛
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--out-text-emb", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.environ.get("CLIP_LOCAL_MODEL","ViT-L-14")
    pretrain   = os.environ.get("CLIP_LOCAL_PRETRAIN","datacomp_xl_s13b_b90k")
    print(f"[tune_v2] model={model_name} pretrain={pretrain} device={device}")

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrain, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    # 数据加载
    C = l2n(np.load(args.img_centroids).astype(np.float32))
    L = [ln.strip() for ln in open(args.img_centroids_labels, encoding="utf-8") if ln.strip()]
    assert C.shape[0]==len(L)
    C_t = torch.from_numpy(C).to(device)

    G=LG=None
    if args.gallery and args.gallery_labels:
        G  = l2n(np.load(args.gallery).astype(np.float32))
        LG = [ln.strip() for ln in open(args.gallery_labels, encoding="utf-8") if ln.strip()]
    G_t = torch.from_numpy(G).to(device) if G is not None else None

    results=[]; out_embs=[]

    def margin(feats, v_t, negs_t):
        # feats: (N, D) tensor
        s_pos = torch.mv(feats, v_t)                    # (N,)
        if negs_t is not None and negs_t.numel()>0:
            s_neg = torch.matmul(negs_t, feats.T).mean(0)  # (N,)
            return s_pos - s_neg
        return s_pos

    for i, pid in enumerate(L):
        v   = C[i]
        v_t = C_t[i]
        negs = pick_hard_negs(v, pid, C, L, G, LG, 6, 6)
        negs_t = torch.from_numpy(negs).to(device) if negs is not None else None
        rng = random.Random(hash(pid) & 0xffffffff)
        prefix = rng.choice(PREFIX_POOL)

        # ---------- 初始化：若初始中性描述低于 gate，做一次“正类最高”的兜底 ----------
        uc, ut, lc, lt, sh = "black","jacket","black","pants","shoes"
        init_txt = render(prefix, uc, ut, lc, lt, sh)
        init_feat = encode_texts(model, tokenizer, device, [init_txt])[0]
        if torch.dot(init_feat, v_t) < args.gate:
            # 穷举小词表（~ 12*8*12*5*5 ≈ 28,800）一次，批量编码选正类最高
            texts=[]; slots=[]
            for c1 in COLORS:
                for u1 in UPPER:
                    for c2 in COLORS:
                        for l1 in LOWER:
                            for s1 in SHOES:
                                texts.append(render(prefix, canon_color(c1), u1, canon_color(c2), l1, s1))
                                slots.append((canon_color(c1), u1, canon_color(c2), l1, s1))
            T = encode_texts(model, tokenizer, device, texts)
            s_pos = torch.mv(T, v_t)
            j = int(torch.argmax(s_pos).item())
            uc,ut,lc,lt,sh = slots[j]
            best_txt  = texts[j]
            best_feat = T[j]
        else:
            best_txt  = init_txt
            best_feat = init_feat

        # ---------- 贪心多轮，每轮依次尝试五个槽位 ----------
        for _ in range(args.max_steps):
            improved = False
            # 定义一个 helper：给定当前槽位池，批量评估 margin，挑最优
            def try_slot(slot_name, pool):
                nonlocal uc,ut,lc,lt,sh, best_txt, best_feat, improved
                # 批量构造候选
                texts=[]; cfgs=[]
                for tok in pool:
                    uc1,ut1,lc1,lt1,sh1 = uc,ut,lc,lt,sh
                    if slot_name=="uc": uc1 = canon_color(tok)
                    elif slot_name=="ut": ut1 = tok
                    elif slot_name=="lc": lc1 = canon_color(tok)
                    elif slot_name=="lt": lt1 = tok
                    elif slot_name=="sh": sh1 = tok
                    texts.append(render(prefix, uc1, ut1, lc1, lt1, sh1))
                    cfgs.append((uc1,ut1,lc1,lt1,sh1))
                # 批量编码 + 计算 margin
                T = encode_texts(model, tokenizer, device, texts)
                sc = margin(T, v_t, negs_t)
                # 选最优，若提升则更新槽位 + best
                j = int(torch.argmax(sc).item())
                if sc[j].item() > torch.dot(best_feat, v_t).item() - (torch.matmul(negs_t, best_feat).mean().item() if (negs_t is not None and negs_t.numel()>0) else 0) + 1e-6:
                    uc,ut,lc,lt,sh = cfgs[j]
                    best_txt  = texts[j]
                    best_feat = T[j]
                    improved = True

            try_slot("uc", COLORS)
            try_slot("ut", UPPER)
            try_slot("lc", COLORS)
            try_slot("lt", LOWER)
            try_slot("sh", SHOES)

            if not improved:
                break

        results.append({"id": pid, "text": best_txt})
        out_embs.append(best_feat.cpu().numpy().astype(np.float32))

    E = l2n(np.stack(out_embs,0)).astype(np.float32)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl,"w",encoding="utf-8") as f:
        for o in results: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    with open(args.out_labels,"w",encoding="utf-8") as f:
        f.write("\n".join([o["id"] for o in results])+"\n")
    np.save(args.out_text_emb, E)
    print(f"[save] {args.out_jsonl}  (50 lines)")
    print(f"[save] {args.out_text_emb}  shape={E.shape}")

if __name__ == "__main__":
    main()
