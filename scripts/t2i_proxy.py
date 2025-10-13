#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ===== 工具函数 =====
def l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0: return x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def read_jsonl_pids(jsonl: Path) -> List[int]:
    pids = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            # id 形如 "pid-66"
            pid = obj.get("id") or obj.get("pid") or obj.get("PID")
            if isinstance(pid, str) and pid.startswith("pid-"):
                pid_i = int(pid.split("-")[1])
            else:
                pid_i = int(pid)
            pids.append(pid_i)
    return pids

def read_mapping(csv_path: Path) -> Tuple[List[str], List[int], List[int]]:
    rels, pids, cams = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rels.append(r["relpath"])
            pids.append(int(r["pid"]))
            cam = r.get("cam") or r.get("camera") or "-1"
            try:
                cams.append(int(cam))
            except Exception:
                cams.append(-1)
    return rels, pids, cams

def group_mean_by_pid(X: np.ndarray, pids: List[int]) -> Dict[int, np.ndarray]:
    buf: Dict[int, List[np.ndarray]] = {}
    for v, pid in zip(X, pids):
        buf.setdefault(pid, []).append(v)
    out: Dict[int, np.ndarray] = {}
    for pid, vs in buf.items():
        out[pid] = l2n(np.stack(vs, 0).mean(0, keepdims=True))[0]
    return out

def ridge_fit(T: np.ndarray, I: np.ndarray, reg: float = 1e-2) -> np.ndarray:
    """
    线性岭回归：min ||T W - I||^2 + λ||W||^2
    解析解：W = (T^T T + λI)^(-1) T^T I
    T: (N, Dt), I: (N, Di) -> W: (Dt, Di)
    """
    Dt, Di = T.shape[1], I.shape[1]
    A = T.T @ T + reg * np.eye(Dt, dtype=np.float64)
    B = T.T @ I
    W = np.linalg.solve(A, B)  # (Dt, Di)
    return W.astype("float32")

def eval_pid_level(sim: np.ndarray, q_pids: List[int], g_pids: List[int]) -> Tuple[np.ndarray, float, int, float]:
    """
    文本→图像评测（不使用 cam junk 规则）：同 pid 为正样本。
    返回：CMC 曲线、mAP、有效 query 数、ATS
    """
    Q, G = sim.shape
    ranks = np.argsort(-sim, axis=1)  # 按相似度降序
    cmc_curve = np.zeros(G, dtype=np.float64)
    ap_list = []
    validQ = 0
    best_per_q = []

    for qi in range(Q):
        pid = q_pids[qi]
        order = ranks[qi]
        pos = [gi for gi in order if g_pids[gi] == pid]
        if not pos:
            continue
        validQ += 1
        # CMC：首命中位置
        first_hit = next((rk for rk, gi in enumerate(order) if g_pids[gi] == pid), None)
        if first_hit is not None:
            cmc_curve[first_hit:] += 1
        # AP
        hits, prec_sum = 0, 0.0
        for rk, gi in enumerate(order, start=1):
            if g_pids[gi] == pid:
                hits += 1
                prec_sum += hits / rk
                if hits == sum(1 for x in g_pids if x == pid):
                    break
        ap_list.append(prec_sum / max(1, sum(1 for x in g_pids if x == pid)))
        # ATS
        best_per_q.append(sim[qi, order[0]])

    if validQ == 0:
        return np.zeros(G), 0.0, 0, 0.0
    cmc_curve /= validQ
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    ats = float(np.mean(best_per_q)) if best_per_q else 0.0
    return cmc_curve, mAP, validQ, ats

# ===== 主流程 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-emb", required=True, help="embeds/text/*.npy")
    ap.add_argument("--prompts", required=True, help="data/prompts/market-mini.jsonl")
    ap.add_argument("--img-query", required=True, help="embeds/image/*/query.npy")
    ap.add_argument("--img-gallery", required=True, help="embeds/image/*/gallery.npy")
    ap.add_argument("--q-mapping", required=True, help="data/market-mini/mapping_query.csv")
    ap.add_argument("--g-mapping", required=True, help="data/market-mini/mapping_gallery.csv")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--reg", type=float, default=1e-2, help="ridge regularization")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载文本嵌入及其 PID
    Xt = np.load(args.text_emb)  # (Nt, Dt)
    t_pids = read_jsonl_pids(Path(args.prompts))  # len == Nt
    assert Xt.shape[0] == len(t_pids), "text-emb 与 prompts 行数不一致"
    Xt = l2n(Xt.astype("float32"))
    pid2t = group_mean_by_pid(Xt, t_pids)  # 每个 pid 一个文本向量

    # 2) 加载图像嵌入和映射
    Xq = np.load(args.img_query).astype("float32")
    Xg = np.load(args.img_gallery).astype("float32")
    Xq = l2n(Xq); Xg = l2n(Xg)
    q_rels, q_pids, q_cams = read_mapping(Path(args.q_mapping))
    g_rels, g_pids, g_cams = read_mapping(Path(args.g_mapping))
    assert Xq.shape[0] == len(q_pids)
    assert Xg.shape[0] == len(g_pids)

    # 3) 用 query 侧做 T→I 对齐（按 pid 聚合）
    pid2q = group_mean_by_pid(Xq, q_pids)      # 每个 pid 一个 query 图像向量
    common_pids = sorted(set(pid2t.keys()) & set(pid2q.keys()))
    T = np.stack([pid2t[pid] for pid in common_pids], 0)  # (N, Dt)
    I = np.stack([pid2q[pid] for pid in common_pids], 0)  # (N, Di)
    W = ridge_fit(T.astype("float64"), I.astype("float64"), reg=float(args.reg))  # (Dt, Di)

    # 4) 将所有 pid 的文本向量映射到图像空间
    all_pids_sorted = sorted(pid2t.keys())
    T_all = np.stack([pid2t[pid] for pid in all_pids_sorted], 0)            # (P, Dt)
    Z = l2n((T_all @ W).astype("float32"))                                   # (P, Di)

    # 5) 与 gallery 做相似度并评测
    sim = Z @ Xg.T  # (P, G)
    cmc, mAP, validQ, ats = eval_pid_level(sim, all_pids_sorted, g_pids)
    rank1 = float(cmc[0]) if cmc.size else 0.0

    # 6) 保存结果
    res = {
        "ProviderText": Path(args.text_emb).stem,
        "ProviderImage": Path(args.img_gallery).parent.name,
        "dims": {"text": int(Xt.shape[1]), "image": int(Xg.shape[1])},
        "ridge_reg": float(args.reg),
        "metrics": {"Rank-1": round(rank1, 4), "mAP": round(mAP, 4), "ATS": round(ats, 4), "valid_queries": int(validQ),
                    "Q_pid": int(sim.shape[0]), "G": int(sim.shape[1])},
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 可视化 CMC
    try:
        L = min(50, len(cmc))
        if L > 0:
            plt.figure()
            plt.plot(range(1, L + 1), cmc[:L])
            plt.xlabel("Rank"); plt.ylabel("CMC"); plt.title("T2I CMC")
            plt.grid(True, linestyle="--", linewidth=0.6)
            plt.savefig(out_dir / "cmc.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print("[warn] plot failed:", e)

    # 保存 topK 结果（便于排错/可视化）
    topk = min(50, sim.shape[1])
    idx = np.argpartition(-sim, kth=topk-1, axis=1)[:, :topk]
    row_sorted = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-row_sorted, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    np.save(out_dir / "top_idx.npy", top_idx)

    print(f"[done] T2I -> {out_dir}")
    print(f"Rank-1={rank1:.4f} mAP={mAP:.4f} ATS={ats:.4f} validQ={validQ}")

if __name__ == "__main__":
    main()
