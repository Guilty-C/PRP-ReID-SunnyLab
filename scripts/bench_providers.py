#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ---- local imports
from providers import get_provider
# 直接复用 retrieve 里的规则，用一份独立实现避免导入路径问题
import re
PID_RE = re.compile(r"^(?P<pid>-?\d{1,4})_c(?P<cam>\d)")

def parse_pid_cam(name: str) -> Tuple[int | None, int | None]:
    m = PID_RE.match(name)
    if not m:
        return None, None
    return int(m.group("pid")), int(m.group("cam"))

def read_mapping(csv_path: Path) -> Tuple[list[str], list[int]]:
    rels, pids = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rels.append(row["relpath"])
            pids.append(int(row["pid"]))
    return rels, pids

def cmc_map(sim: np.ndarray, q_meta: list[Tuple[int,int]], g_meta: list[Tuple[int,int]]):
    """Market-1501 风格：同 pid 同 cam 视为 junk，不计入评测。"""
    Q, G = sim.shape
    ranks = np.argsort(-sim, axis=1)
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

def run_one_provider(provider_name: str, subset_dir: Path, out_root: Path) -> dict:
    """Embed -> Retrieve -> Eval -> Save results.json; return row for summary."""
    q_csv = subset_dir / "mapping_query.csv"
    g_csv = subset_dir / "mapping_gallery.csv"
    q_rels, _ = read_mapping(q_csv)
    g_rels, _ = read_mapping(g_csv)
    q_paths = [str(subset_dir / r) for r in q_rels]
    g_paths = [str(subset_dir / r) for r in g_rels]

    provider = get_provider(provider_name)

    t0 = time.perf_counter()
    Xq = provider.embed_images(q_paths)
    t1 = time.perf_counter()
    Xg = provider.embed_images(g_paths)
    t2 = time.perf_counter()

    # 余弦（已 L2 归一化）
    sim = Xq @ Xg.T

    # 元信息（pid, cam）
    q_meta, g_meta = [], []
    for rel in q_rels:
        pid, cam = parse_pid_cam(os.path.basename(rel))
        q_meta.append((pid, cam))
    for rel in g_rels:
        pid, cam = parse_pid_cam(os.path.basename(rel))
        g_meta.append((pid, cam))

    cmc, mAP, valid_q = cmc_map(sim, q_meta, g_meta)
    rank1 = float(cmc[0]) if len(cmc) else 0.0
    ats = float(np.mean(np.max(sim, axis=1)))

    # 保存该 provider 的结果
    out_dir = out_root / provider_name
    out_dir.mkdir(parents=True, exist_ok=True)
    topk = min(50, sim.shape[1])
    idx = np.argpartition(-sim, kth=topk-1, axis=1)[:, :topk]
    row_sorted = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-row_sorted, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    top_scores = np.take_along_axis(sim, top_idx, axis=1)
    np.save(out_dir / "top_idx.npy", top_idx)
    np.save(out_dir / "top_scores.npy", top_scores)

    result = {
        "metrics": {
            "Provider": provider_name,
            "Rank-1": round(rank1, 4),
            "mAP": round(mAP, 4),
            "ATS": round(ats, 4),
            "valid_queries": int(valid_q),
            "Q": int(sim.shape[0]),
            "G": int(sim.shape[1]),
        },
        "files": {
            "q_mapping": str(q_csv.as_posix()),
            "g_mapping": str(g_csv.as_posix()),
        },
        "timing_ms": {
            "embed_query": round((t1 - t0) * 1000, 2),
            "embed_gallery": round((t2 - t1) * 1000, 2),
            "ms_per_image_query": round((t1 - t0) * 1000 / max(1, len(q_paths)), 2),
            "ms_per_image_gallery": round((t2 - t1) * 1000 / max(1, len(g_paths)), 2),
        },
        "dim": int(Xq.shape[1]) if Xq.size else 0,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 画 CMC
    try:
        import matplotlib.pyplot as plt
        L = min(50, len(cmc))
        if L > 0:
            plt.figure()
            plt.plot(range(1, L + 1), cmc[:L])
            plt.xlabel("Rank"); plt.ylabel("CMC")
            plt.title(f"CMC – {provider_name}")
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.savefig(out_dir / "cmc.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"[warn] plotting failed for {provider_name}:", e)

    # 汇总行
    row = {
        "provider": provider_name,
        "rank1": round(rank1, 4),
        "mAP": round(mAP, 4),
        "ATS": round(ats, 4),
        "validQ": int(valid_q),
        "Q": int(sim.shape[0]),
        "G": int(sim.shape[1]),
        "dim": result["dim"],
        "ms/img(query)": result["timing_ms"]["ms_per_image_query"],
        "ms/img(gallery)": result["timing_ms"]["ms_per_image_gallery"],
    }
    print("[done]", provider_name, "->", row)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="data/market-mini")
    ap.add_argument("--providers", required=True, help="comma-separated list, e.g., clip-local,mock")
    ap.add_argument("--out", required=True, help="outputs/market-mini/exp003")
    args = ap.parse_args()

    subset = Path(args.subset)
    out_root = Path(args.out)
    prov_list = [p.strip() for p in args.providers.split(",") if p.strip()]

    rows = []
    for p in prov_list:
        try:
            rows.append(run_one_provider(p, subset, out_root / p))
        except KeyError as e:
            print(f"[skip] {p}: not registered ({e})")
        except Exception as e:
            print(f"[error] {p} failed:", e)

    # 写 summary.csv
    if rows:
        summary_csv = out_root / "summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = ["provider", "rank1", "mAP", "ATS", "validQ", "Q", "G", "dim", "ms/img(query)", "ms/img(gallery)"]
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[ok] summary -> {summary_csv}")

        # 画一个简单柱状图（Rank-1）
        try:
            providers = [r["provider"] for r in rows]
            r1 = [r["rank1"] for r in rows]
            plt.figure()
            plt.bar(providers, r1)
            plt.title("Rank-1 Comparison")
            plt.ylabel("Rank-1")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(out_root / "summary_rank1.png", dpi=150)
            plt.close()
            print(f"[ok] plot -> {out_root / 'summary_rank1.png'}")
        except Exception as e:
            print("[warn] summary plot failed:", e)
    else:
        print("[warn] no rows collected; nothing to summarize.")

if __name__ == "__main__":
    main()
