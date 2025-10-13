#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PID_RE = re.compile(r"^(?P<pid>-?\d{1,4})_c(?P<cam>\d)")

def parse_pid_cam(name: str):
    m = PID_RE.match(name)
    if not m:
        return None, None
    return int(m.group("pid")), int(m.group("cam"))

def read_mapping(csv_path: Path) -> tuple[list[str], list[int]]:
    rels, pids = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rels.append(row["relpath"])
            pids.append(int(row["pid"]))
    return rels, pids

def cmc_map(sim: np.ndarray, q_meta: list[tuple[int,int]], g_meta: list[tuple[int,int]]):
    """Compute CMC and mAP with Market-like rules (drop same-pid same-cam as junk)."""
    Q, G = sim.shape
    ranks = np.argsort(-sim, axis=1)  # desc
    cmc_curve = np.zeros(G, dtype=np.float64)
    ap_list = []
    valid_queries = 0

    for qi in range(Q):
        q_pid, q_cam = q_meta[qi]
        order = ranks[qi]
        pos = []
        junk = set()
        for gi in order:
            g_pid, g_cam = g_meta[gi]
            if g_pid == q_pid and g_cam == q_cam:
                junk.add(gi)
            elif g_pid == q_pid:
                pos.append(gi)
        if len(pos) == 0:
            continue
        valid_queries += 1
        filt = [gi for gi in order if gi not in junk]
        # CMC
        first_hit = next((rank for rank, gi in enumerate(filt) if gi in pos), None)
        if first_hit is not None:
            cmc_curve[first_hit:] += 1
        # AP
        hits = 0
        prec_sum = 0.0
        for rank, gi in enumerate(filt, start=1):
            if gi in pos:
                hits += 1
                prec_sum += hits / rank
                if hits == len(pos):
                    break
        ap_list.append(prec_sum / len(pos))

    if valid_queries == 0:
        return np.zeros(G), 0.0, 0
    cmc_curve /= valid_queries
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    return cmc_curve, mAP, valid_queries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="path to query.npy")
    ap.add_argument("--gallery", required=True, help="path to gallery.npy")
    ap.add_argument("--q-mapping", required=True)
    ap.add_argument("--g-mapping", required=True)
    ap.add_argument("--subset-root", required=True, help="data/market-mini root for resolving relpaths")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    Xq = np.load(args.query)
    Xg = np.load(args.gallery)
    # cosine (already l2-normalized)
    sim = Xq @ Xg.T  # (Q,G)

    q_rels, _ = read_mapping(Path(args.q_mapping))
    g_rels, _ = read_mapping(Path(args.g_mapping))
    q_meta, g_meta = [], []
    for rel in q_rels:
        pid, cam = parse_pid_cam(os.path.basename(rel))
        q_meta.append((pid, cam))
    for rel in g_rels:
        pid, cam = parse_pid_cam(os.path.basename(rel))
        g_meta.append((pid, cam))

    cmc, mAP, valid_q = cmc_map(sim, q_meta, g_meta)
    rank1 = float(cmc[0]) if len(cmc) > 0 else 0.0
    ats = float(np.mean(np.max(sim, axis=1)))  # Average Top-1 Similarity

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)

    # Save top-k
    topk = min(args.topk, sim.shape[1])
    idx = np.argpartition(-sim, kth=topk-1, axis=1)[:, :topk]
    row_sorted = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-row_sorted, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    top_scores = np.take_along_axis(sim, top_idx, axis=1)
    np.save(outd / "top_idx.npy", top_idx)
    np.save(outd / "top_scores.npy", top_scores)

    # Summary JSON
    summary = {
        "metrics": {
            "Rank-1": round(rank1, 4),
            "mAP": round(mAP, 4),
            "ATS": round(ats, 4),
            "valid_queries": int(valid_q),
            "Q": int(sim.shape[0]),
            "G": int(sim.shape[1]),
        },
        "files": {
            "query_npy": str(Path(args.query).as_posix()),
            "gallery_npy": str(Path(args.gallery).as_posix()),
            "q_mapping": str(Path(args.q_mapping).as_posix()),
            "g_mapping": str(Path(args.g_mapping).as_posix()),
        }
    }
    with open(outd / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Plot CMC
    try:
        plt.figure()
        L = min(50, len(cmc))
        plt.plot(range(1, L+1), cmc[:L])
        plt.xlabel("Rank"); plt.ylabel("CMC"); plt.title("CMC Curve")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.savefig(outd / "cmc.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("[warn] plotting failed:", e)

    print(f"[done] Retrieve+Eval -> {outd}")
    print("Rank-1=%.4f mAP=%.4f ATS=%.4f validQ=%d" % (rank1, mAP, ats, valid_q))

if __name__ == "__main__":
    main()
