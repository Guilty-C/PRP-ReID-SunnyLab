# tools/eval_textset_image_topk.py
from __future__ import annotations
import argparse, os, re, numpy as np
from collections import defaultdict

def rl(p):
    return [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]

def l2n(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

_pid4_re = re.compile(r"(\d{4})")
def to_pid4(s: str) -> str:
    """从任意标签里抽取4位数字pid；若找不到，原样返回（但一般应都能匹配）。"""
    m = _pid4_re.search(s)
    return m.group(1) if m else s

def eval_max_ensemble(Z_all, L_all, I, L_img, paths=None, k=10, dump=None, restrict_pids=None):
    """
    Z_all: (Nv, D) 文本变体向量（多个pid，每个pid多条变体）
    L_all: Nv 个变体标签（可能含后缀）；本函数内部会归一化到4位pid并按pid分组
    I    : (Ni, D) 图库向量
    L_img: Ni 个图库pid（可能是原始形式）；本函数内部会归一化到4位pid
    paths: Ni 个图片路径（可选，仅用于dump可视化）
    k    : Top-K
    dump : tsv路径（可选），输出每个pid的topk检索样例
    restrict_pids: 可选集合，只在这些pid上做评测（且过滤图库到这些pid）
    """
    Z_all = l2n(Z_all.astype(np.float32))
    I = l2n(I.astype(np.float32))

    # 归一化标签到4位pid
    L_all_pid = [to_pid4(s) for s in L_all]
    L_img_pid = [to_pid4(s) for s in L_img]

    # 可选：只评测指定pid，并把图库过滤到这些pid，贴近“challenge子集”评测
    if restrict_pids is not None:
        restrict_pids = set(to_pid4(p) for p in restrict_pids)
        keep_idx = [i for i,p in enumerate(L_img_pid) if p in restrict_pids]
        I = I[keep_idx, :]
        L_img_pid = [L_img_pid[i] for i in keep_idx]
        if paths:
            paths = [paths[i] for i in keep_idx]

    # 分组：pid -> 该pid的所有变体行号
    groups = defaultdict(list)
    for i, p in enumerate(L_all_pid):
        groups[p].append(i)

    # 只保留在图库中实际出现过的pid，避免评测无效类
    present = set(L_img_pid)
    pids = sorted([p for p in groups.keys() if p in present])
    if len(pids) == 0:
        raise RuntimeError("没有任何文本pid与图库pid匹配，请检查标签归一化是否正确。")

    # 评测
    R1=R5=R10=0
    mAP = 0.0
    top1_vals = []

    if dump:
        os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
        fout = open(dump, "w", encoding="utf-8")
        fout.write("pid\tbest_img_pid\tbest_sim\trank_of_gt\ttopk_paths\n")

    for pid in pids:
        idx = groups[pid]
        Zp = Z_all[idx, :]            # (M, D)
        S  = Zp @ I.T                 # (M, Ni)
        smax = S.max(axis=0)          # (Ni,) 该pid对每张图的最大相似度
        order = np.argsort(-smax)     # (Ni,)
        ranked_pids = [L_img_pid[j] for j in order]

        # R@K
        hit1 = int(pid in ranked_pids[:1])
        hit5 = int(pid in ranked_pids[:5])
        hit10= int(pid in ranked_pids[:10])
        R1 += hit1; R5 += hit5; R10 += hit10

        # mAP
        tot = ranked_pids.count(pid)
        if tot > 0:
            h=0; ps=0.0
            for rnk, pp in enumerate(ranked_pids, 1):
                if pp == pid:
                    h += 1
                    ps += h / rnk
            mAP += ps / tot

        # Avg Top-1 sim
        top1_vals.append(float(smax[order[0]]))

        if dump:
            best_pid = ranked_pids[0]
            rank_of_gt = ranked_pids.index(pid)+1 if pid in ranked_pids else -1
            if paths:
                topk_paths = [paths[j] for j in order[:k]]
            else:
                topk_paths = [str(order[j]) for j in range(k)]
            fout.write(f"{pid}\t{best_pid}\t{smax[order[0]]:.4f}\t{rank_of_gt}\t" +
                       "|".join(topk_paths) + "\n")

    if dump: fout.close()

    P = float(len(pids))
    return {
        "R@1": R1/P, "R@5": R5/P, "R@10": R10/P,
        "mAP": mAP/P, "AvgTop1": float(np.mean(top1_vals)),
        "NumPIDs": int(P), "NumImages": int(I.shape[0])
    }

def main():
    ap = argparse.ArgumentParser("Eval with text-variant MAX ensemble (labels auto-normalized to pid4)")
    ap.add_argument("--text-all", required=True)
    ap.add_argument("--text-all-labels", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels-image", required=True)
    ap.add_argument("--paths-image", default=None)
    ap.add_argument("--restrict-pids", default=None, help="可选：文件，列出要评测的pid(4位)；图库也会过滤到这些pid")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dump", default=None, help="tsv path to dump per-pid topk")
    args = ap.parse_args()

    Z_all = np.load(args.text_all)
    L_all = rl(args.text_all_labels)
    I = np.load(args.image)
    L_img = rl(args.labels_image)
    paths = rl(args.paths_image) if args.paths_image and os.path.exists(args.paths_image) else None

    restrict = rl(args.restrict_pids) if args.restrict_pids and os.path.exists(args.restrict_pids) else None

    assert len(L_all) == Z_all.shape[0], "text-all 与标签数量不一致"
    if paths: assert len(paths) == I.shape[0], "paths 与图库数量不一致"

    m = eval_max_ensemble(Z_all, L_all, I, L_img, paths=paths, k=args.k, dump=args.dump, restrict_pids=restrict)
    print(f"[PIDs={m['NumPIDs']} | Images={m['NumImages']}] "
          f"R@1={m['R@1']:.4f}  R@5={m['R@5']:.4f}  R@10={m['R@10']:.4f}  mAP={m['mAP']:.4f}  AvgTop1={m['AvgTop1']:.4f}")

if __name__ == "__main__":
    main()
