# tools/eval_textset_image_rerank.py
import argparse, os, re, numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

def rl(p): 
    return [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]

def l2n(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def to_pid4_list(lines: List[str]) -> List[str]:
    out = []
    rx = re.compile(r"(\d{4})")
    for s in lines:
        m = rx.search(s)
        out.append(m.group(1) if m else s)
    return out

def pool_scores(S, mode="lse", tau=8.0, topm=3):
    """
    S: (M, N)  M条文本变体 vs N张图像的相似度（余弦）
    """
    if mode == "max":
        return S.max(axis=0)
    if mode == "mean":
        return S.mean(axis=0)
    if mode == "lse":
        # 数值稳定：s_pool = m + (1/tau)*log( mean(exp(tau*(S - m))) )
        m = S.max(axis=0, keepdims=True)             # (1, N)
        A = tau * (S - m)                            # (M, N)
        return (m.reshape(-1) + (np.log(np.exp(A).mean(axis=0)) / float(tau)))
    if mode == "topm":
        m = min(topm, S.shape[0])
        idx = np.argpartition(-S, m-1, axis=0)[:m, :]
        vals = np.take_along_axis(S, idx, axis=0)
        return vals.mean(axis=0)
    raise ValueError(f"Unknown pool mode: {mode}")

def make_pid_order(labels_all: List[str]) -> List[str]:
    """按首次出现顺序获取 pid 列表（稳定且与变体分组一致）"""
    seen, order = set(), []
    for p in labels_all:
        if p not in seen:
            seen.add(p); order.append(p)
    return order

def group_indices(labels_all: List[str]) -> Dict[str, List[int]]:
    g = defaultdict(list)
    for i,p in enumerate(labels_all):
        g[p].append(i)
    return g

def eval_retrieval(pid_order: List[str], pooled_scores: np.ndarray, Li: List[str], k: int = 10):
    """
    pooled_scores: (P, N) 每个 pid 对所有图像的池化后得分
    """
    P, N = pooled_scores.shape
    assert len(pid_order) == P
    order = np.argsort(-pooled_scores, axis=1)
    # metrics
    r1=r5=r10=0
    ap_acc=0.0
    best_sim = pooled_scores.max(axis=1).mean()
    for i,p in enumerate(pid_order):
        ranked_idx = order[i]
        topk_labels = [Li[j] for j in ranked_idx[:k]]
        r1 += (p == topk_labels[0])
        r5 += (p in topk_labels[:5])
        r10+= (p in topk_labels[:10])
        # mAP
        hits=0; prec_sum=0.0; total=0
        for rnk, j in enumerate(ranked_idx, 1):
            if Li[j]==p:
                hits += 1
                total += 1
                prec_sum += hits / rnk
        ap_acc += (prec_sum / total) if total>0 else 0.0
    P = len(pid_order)
    return {
        "R@1": r1/P, "R@5": r5/P, "R@10": r10/P,
        "mAP": ap_acc/P, "AvgTop1": float(best_sim)
    }, order

def main():
    ap = argparse.ArgumentParser("Evaluate text-set -> image reranking with pooling")
    ap.add_argument("--text-all", required=True)
    ap.add_argument("--text-all-labels", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels-image", required=True)
    ap.add_argument("--paths-image", default=None)

    ap.add_argument("--pool", choices=["max","mean","lse","topm"], default="lse")
    ap.add_argument("--tau", type=float, default=8.0, help="for lse")
    ap.add_argument("--topm", type=int, default=3, help="for topm")

    # 两阶段
    ap.add_argument("--stage1-text", default=None, help="(P,D) 每个pid一个向量；若缺省则用 text-all 的均值代替")
    ap.add_argument("--stage1-labels", default=None, help="可选；若不给，则默认顺序=按 text-all-labels 的首次出现顺序")
    ap.add_argument("--stage1-topn", type=int, default=800)

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    # 加载
    T_all = np.load(args.text_all).astype(np.float32)
    Lt_all = to_pid4_list(rl(args.text_all_labels))
    I = np.load(args.image).astype(np.float32)
    Li = to_pid4_list(rl(args.labels_image))
    paths = rl(args.paths_image) if (args.paths_image and os.path.exists(args.paths_image)) else None

    T_all = l2n(T_all)
    I = l2n(I)

    # 规范 pid 列表与分组
    pid_order = make_pid_order(Lt_all)
    groups = group_indices(Lt_all)

    # Stage1（可选）：每个 pid 先用一个向量筛 TopN 图像
    if args.stage1_text and os.path.exists(args.stage1_text):
        T1 = np.load(args.stage1_text).astype(np.float32)
        T1 = l2n(T1)
        # 若提供了 labels，则以提供的顺序为准；否则假设与 pid_order 对齐
        if args.stage1_labels and os.path.exists(args.stage1_labels):
            L1 = to_pid4_list(rl(args.stage1_labels))
            if len(L1) != len(pid_order):
                print(f"[warn] stage1-labels 与 pid_order 长度不符（{len(L1)} vs {len(pid_order)}），将按交集重排")
                # 对齐交集（保留 pid_order 顺序）
                keep_idx = [L1.index(p) for p in pid_order if p in L1]
                T1 = T1[keep_idx, :]
                pid_order = [pid_order[i] for i in range(len(pid_order)) if pid_order[i] in L1]
        else:
            if T1.shape[0] != len(pid_order):
                print(f"[warn] stage1-text 行数({T1.shape[0]}) != pid数({len(pid_order)}), 将退化为 text-all 均值")
                T1 = None
    else:
        T1 = None

    if T1 is None:
        # 用 text-all 的均值作为 Stage1 表示
        T1 = []
        for pid in pid_order:
            idx = groups[pid]
            T1.append(T_all[idx, :].mean(axis=0))
        T1 = np.stack(T1, 0)
        T1 = l2n(T1)

    # Stage1 选 TopN 索引
    S1 = T1 @ I.T                             # (P, N)
    topn = max(1, min(args.stage1_topn, I.shape[0]))
    idx_topn = np.argpartition(-S1, topn-1, axis=1)[:, :topn]   # (P, topn)

    # Stage2：对每个 pid 的候选集进行“文本变体池化”
    pooled = np.full((len(pid_order), I.shape[0]), fill_value=-1e9, dtype=np.float32)
    for i, pid in enumerate(pid_order):
        cand_cols = idx_topn[i]                          # (topn,)
        idx_group = groups[pid]                          # 该 pid 的所有文本变体 id
        S_sub = T_all[idx_group, :] @ I[cand_cols, :].T  # (Mi, topn)
        pooled_sub = pool_scores(S_sub, mode=args.pool, tau=args.tau, topm=args.topm)  # (topn,)
        pooled[i, cand_cols] = pooled_sub               # 其它列保持 -1e9（相当于没入选）

    # 计算指标
    metrics, order = eval_retrieval(pid_order, pooled, Li, k=args.k)
    print(f"[PIDs={len(pid_order)} | Images={I.shape[0]}] "
          f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}  "
          f"mAP={metrics['mAP']:.4f}  AvgTop1={metrics['AvgTop1']:.4f}")

    # 可选：dump 结果
    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as f:
            f.write("pid\trank\timg_idx\timg_label\tscore\tpath\n")
            for i, pid in enumerate(pid_order):
                ranked_idx = order[i][:args.k]
                for rnk, j in enumerate(ranked_idx, 1):
                    lab = Li[j]
                    sc  = pooled[i, j]
                    pth = paths[j] if paths and j < len(paths) else ""
                    f.write(f"{pid}\t{rnk}\t{j}\t{lab}\t{sc:.6f}\t{pth}\n")

if __name__ == "__main__":
    main()
