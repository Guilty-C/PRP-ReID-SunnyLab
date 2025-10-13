# tools/prompt_tune_margin.py
from __future__ import annotations
import argparse, os, re, json, numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
import open_clip
from tqdm import tqdm

# -------------------- utils --------------------
def rl(p): 
    return [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]

def l2n(x, eps=1e-8): 
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

_pid4 = re.compile(r"(\d{4})")
def to_pid4(s: str) -> str:
    m = _pid4.search(s)
    return m.group(1) if m else s

# -------------------- CLIP --------------------
@torch.no_grad()
def build_model(device: str = "cuda"):
    model_name = os.getenv("CLIP_LOCAL_MODEL", "ViT-L-14")
    pretrain   = os.getenv("CLIP_LOCAL_PRETRAIN", "datacomp_xl_s13b_b90k")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrain, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device="cuda", batch_size=256):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        toks = tokenizer(batch).to(device)
        txt = model.encode_text(toks)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        feats.append(txt.float().cpu().numpy())
    return np.concatenate(feats, 0)

# -------------------- prompt pools --------------------
PREFIX_POOL = [
    "a person",
    "a street photo of a person",
    "a surveillance photo of a person",
    "a full-body photo of a person",
]
UPPER_TYPE = ["t-shirt","shirt","jacket","hoodie","coat","polo shirt","sweater","vest","blazer","long-sleeve shirt"]
LOWER_TYPE = ["jeans","trousers","pants","shorts","skirt"]
COLORS     = ["black","white","blue","red","green","yellow","gray","brown","purple","pink","orange","beige","navy"]
SHOES      = ["sneakers","shoes","running shoes","leather shoes","boots","canvas shoes"]
BAG_PHRASES= ["carrying a backpack","carrying a shoulder bag","carrying a handbag","with a tote bag","with no bag"]
HEAD_PHRASES=["wearing a hat","no hat"]

SLOTS = [
    ("prefix", PREFIX_POOL),
    ("upper_color", COLORS),
    ("upper_type", UPPER_TYPE),
    ("lower_color", COLORS),
    ("lower_type", LOWER_TYPE),
    ("shoes_color", COLORS),
    ("shoes_type", SHOES),
    ("bag", BAG_PHRASES),
    ("head", HEAD_PHRASES),
]

def render_prompt(slots: Dict[str, str]) -> str:
    """
    以 prefix 作为起点，其它短语仅在 prefix 未包含时追加，避免重复堆词。
    """
    prefix = (slots.get("prefix") or "a person").lower().strip()
    parts = [prefix]

    def add_phrase(s: str):
        if not s: 
            return
        s = s.lower().strip()
        if s and s not in prefix:  # 避免重复
            parts.append(s)

    ucol, utype = slots.get("upper_color"), slots.get("upper_type")
    lcol, ltype = slots.get("lower_color"), slots.get("lower_type")
    scol, stype = slots.get("shoes_color"), slots.get("shoes_type")
    if ucol and utype: add_phrase(f"wearing a {ucol} {utype}")
    if lcol and ltype: add_phrase(f"and {lcol} {ltype}")
    if scol and stype: add_phrase(f"with {scol} {stype}")
    if slots.get("bag"):  add_phrase(slots["bag"])
    if slots.get("head"): add_phrase(slots["head"])

    return ", ".join([p for p in parts if p])

def score_margin(t: np.ndarray, C: np.ndarray, idx: int, lam: float = 0.5, topM: int = 5) -> float:
    """
    t: (D,) 文本向量, 已L2
    C: (P,D) 各pid图像类中心, 已L2
    idx: 自己pid在C中的行号
    """
    s = C @ t  # (P,)
    mine = float(s[idx])
    others = np.sort(np.delete(s, idx))[-topM:] if topM > 1 else np.array([np.max(np.delete(s, idx))])
    imp = float(np.mean(others))
    return mine - lam * imp

def greedy_refine_for_pid(
    base_prompt: str,
    C: np.ndarray, pid_index: int,
    model, tokenizer, device="cuda",
    steps: int = 12, lam: float = 0.5, topM: int = 5,
    gate_improve: float = 1e-3,
) -> Tuple[str, float]:
    """
    以 base_prompt 为真正起点；按 slot 贪心替换/添加，只有带来正增益才接受。
    """
    slots: Dict[str, str] = {}
    if base_prompt and base_prompt.strip():
        slots["prefix"] = base_prompt.strip().lower()
    else:
        slots["prefix"] = "a person"

    cur = render_prompt(slots)
    cur_feat = encode_texts(model, tokenizer, [cur], device=device)
    cur_feat = l2n(cur_feat)[0]
    best = score_margin(cur_feat, C, pid_index, lam=lam, topM=topM)
    improved = True

    for _ in range(steps):
        if not improved:
            break
        improved = False
        best_delta, best_key, best_val, best_feat = 0.0, None, None, None

        for key, pool in SLOTS:
            cur_val = slots.get(key)
            for cand in pool:
                if cand == cur_val:
                    continue
                slots[key] = cand
                txt = render_prompt(slots)
                feat = encode_texts(model, tokenizer, [txt], device=device)
                feat = l2n(feat)[0]
                sc = score_margin(feat, C, pid_index, lam=lam, topM=topM)
                delta = sc - best
                if delta > best_delta:
                    best_delta, best_key, best_val, best_feat = delta, key, cand, feat
            # 回滚
            if cur_val is None:
                slots.pop(key, None)
            else:
                slots[key] = cur_val

        if best_delta > gate_improve and best_key is not None:
            slots[best_key] = best_val
            best += best_delta
            cur = render_prompt(slots)
            cur_feat = best_feat
            improved = True

    return cur, best

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser("Discriminative prompt tuning (margin-greedy) for CLIP ReID")
    ap.add_argument("--img-centroids", required=True)
    ap.add_argument("--img-centroids-labels", required=True)
    ap.add_argument("--base-jsonl", default=None, help="可选：作为起点的jsonl（例如 v3 输出），字段 text/id")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5)
    ap.add_argument("--topm", type=int, default=5)
    ap.add_argument("--gate", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    # 输出
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--out-text-emb", required=True)
    # 评测（可选）
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--gallery-labels", default=None)
    args = ap.parse_args()

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model, tokenizer = build_model(device)

    C = np.load(args.img_centroids).astype(np.float32)
    C = l2n(C)
    Lc = [to_pid4(x) for x in rl(args.img_centroids_labels)]
    pid2idx = {p:i for i,p in enumerate(Lc)}

    base = {}
    if args.base_jsonl and os.path.exists(args.base_jsonl):
        with open(args.base_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                pid = to_pid4(str(obj.get("id") or obj.get("pid") or obj.get("label") or ""))
                txt = str(obj.get("text") or obj.get("prompt") or "")
                if pid and txt and pid not in base:
                    base[pid] = txt

    results = []
    for pid in tqdm(Lc, desc="[tune]"):
        prompt0 = base.get(pid, "a person")
        txt, sc = greedy_refine_for_pid(
            base_prompt=prompt0,
            C=C, pid_index=pid2idx[pid],
            model=model, tokenizer=tokenizer, device=device,
            steps=args.max_steps, lam=args.lam, topM=args.topm, gate_improve=args.gate
        )
        results.append((pid, txt))

    # 保存
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for pid, txt in results:
            f.write(json.dumps({"id": pid, "text": txt}, ensure_ascii=False) + "\n")
    with open(args.out_labels, "w", encoding="utf-8") as f:
        f.write("\n".join([pid for pid,_ in results]) + "\n")

    texts = [txt for _,txt in results]
    Z = encode_texts(model, tokenizer, texts, device=device)
    Z = l2n(Z.astype(np.float32))
    np.save(args.out_text_emb, Z)

    # 快速评测：Centroids
    print(f"[save] {args.out_jsonl}  ({len(texts)} lines)")
    print(f"[save] {args.out_text_emb}  shape={Z.shape}")
    S = Z @ C.T
    order = np.argsort(-S, axis=1)
    ranks = []
    for i,p in enumerate(Lc):
        ranked = [Lc[j] for j in order[i]]
        r = ranked.index(p)+1 if p in ranked else -1
        ranks.append(r)
    r1 = sum(1 for r in ranks if 0 < r <= 1)/len(ranks)
    r5 = sum(1 for r in ranks if 0 < r <= 5)/len(ranks)
    r10= sum(1 for r in ranks if 0 < r <=10)/len(ranks)
    print(f"[quick] Centroids R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} AvgTop1={float(S.max(1).mean()):.4f}")

    # 可选：Gallery
    if args.gallery and args.gallery_labels and os.path.exists(args.gallery) and os.path.exists(args.gallery_labels):
        G = np.load(args.gallery).astype(np.float32)
        G = l2n(G)
        Lg = [to_pid4(x) for x in rl(args.gallery_labels)]
        Sg = Z @ G.T
        ordg = np.argsort(-Sg, axis=1)
        r1g = sum(1 for i,p in enumerate(Lc) if p in [Lg[j] for j in ordg[i,:1]])/len(Lc)
        r5g = sum(1 for i,p in enumerate(Lc) if p in [Lg[j] for j in ordg[i,:5]])/len(Lc)
        r10g= sum(1 for i,p in enumerate(Lc) if p in [Lg[j] for j in ordg[i,:10]])/len(Lc)
        print(f"[quick] Gallery   R@1={r1g:.3f} R@5={r5g:.3f} R@10={r10g:.3f} AvgTop1={float(Sg.max(1).mean()):.4f}")

if __name__ == "__main__":
    main()
