# tools/reweight_text_by_img.py
from __future__ import annotations
import argparse, os
import numpy as np

def rl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def l2n(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def softmax(x, tau=0.06):
    x = x / max(tau, 1e-8)
    x = x - x.max()  # for numerical stability
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def eval_pair(T, I, Lt, Li):
    T = l2n(T); I = l2n(I)
    S = T @ I.T
    order = np.argsort(-S, axis=1)
    def r_at(k):
        hit=0
        for i,y in enumerate(Lt):
            ranked=[Li[j] for j in order[i,:k]]
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
            if p==y:
                h+=1; ps+=h/r
        mAP+=ps/tot
    return r_at(1), r_at(5), r_at(10), mAP/len(Lt), float(S.max(axis=1).mean())

def main():
    ap = argparse.ArgumentParser("Reweight auto text variants by image centroids (softmax weighting)")
    ap.add_argument("--text-all", required=True, help="all text embeds (.npy), e.g. ...auto_v2.all.boost.npy")
    ap.add_argument("--text-all-labels", required=True, help="labels for all text embeds (.txt)")
    ap.add_argument("--img-centroids", required=True, help="(P,D) image centroids (.npy)")
    ap.add_argument("--img-centroids-labels", required=True, help="pids for centroids (.txt)")
    ap.add_argument("--gallery", required=True, help="(N,D) gallery image embeds (.npy)")
    ap.add_argument("--gallery-labels", required=True, help="labels for gallery (.txt)")
    ap.add_argument("--tau", type=float, default=0.06, help="softmax temperature for weighting")
    ap.add_argument("--topn", type=int, default=12, help="per-pid keep top-N variants by sim to centroid before weighting (<=0 to disable)")
    ap.add_argument("--out", default="embeds/text/clip-l14_50.auto_v2.avg_by_pid.rew.npy")
    args = ap.parse_args()

    T_all = np.load(args.text_all).astype(np.float32)             # (M,D)
    L_all = rl(args.text_all_labels)                              # len M
    I     = np.load(args.img_centroids).astype(np.float32)        # (P,D)
    Li    = rl(args.img_centroids_labels)                         # len P
    G     = np.load(args.gallery).astype(np.float32)              # (N,D)
    Lg    = rl(args.gallery_labels)                               # len N

    assert len(L_all)==T_all.shape[0], "all labels size mismatch npy"
    assert len(Li)==I.shape[0], "centroid labels mismatch npy"
    P, D = I.shape
    print(f"[reweight] variants: {T_all.shape[0]}  pids: {P}  dim: {D}  tau={args.tau}  topn={args.topn}")

    # 预归一化
    Tn = l2n(T_all)
    In = l2n(I)

    # 建立每个 pid 的变体索引列表
    rows_by_pid = {p: [] for p in Li}
    for idx, pid in enumerate(L_all):
        if pid in rows_by_pid:
            rows_by_pid[pid].append(idx)

    # 重加权均值
    T_rew = np.zeros((P, D), dtype=np.float32)
    for pi, pid in enumerate(Li):
        rows = rows_by_pid.get(pid, [])
        if not rows:
            # 不该发生；兜底用原始 centroid 向量形状维度填零向量
            T_rew[pi] = np.zeros((D,), dtype=np.float32)
            continue
        V = Tn[rows]                                  # (m, D)
        s = (V @ In[pi:pi+1].T).reshape(-1)           # (m,)
        # 先选 top-N
        if args.topn and args.topn > 0 and len(rows) > args.topn:
            top_idx = np.argsort(-s)[:args.topn]
            V = V[top_idx]; s = s[top_idx]
        w = softmax(s, tau=args.tau).astype(np.float32)  # (m,)
        T_rew[pi] = l2n((w[:,None] * V).sum(axis=0, keepdims=True))[0]

    # 保存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, T_rew)
    print(f"[reweight] saved: {args.out}  shape={T_rew.shape}")

    # 评测
    r1,r5,r10,mapv,top1 = eval_pair(T_rew, I, Li, Li)
    print(f"[eval] Centroids | R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} mAP={mapv:.3f} AvgTop1={top1:.3f}")
    r1,r5,r10,mapv,top1 = eval_pair(T_rew, G, Li, Lg)
    print(f"[eval] Gallery   | R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} mAP={mapv:.3f} AvgTop1={top1:.3f}")

if __name__ == "__main__":
    main()
