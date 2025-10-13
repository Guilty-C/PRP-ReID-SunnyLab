#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from providers import get_provider
from scripts.utils_io import write_metadata

def read_mapping(csv_path: Path) -> tuple[list[str], list[int]]:
    rels, pids = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rels.append(row["relpath"])
            pids.append(int(row["pid"]))
    return rels, pids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="Path to data/market-mini")
    ap.add_argument("--provider", required=True, help="e.g., mock")
    ap.add_argument("--out", required=True, help="Output dir for npy files")
    args = ap.parse_args()

    subset = Path(args.subset)
    q_csv = subset / "mapping_query.csv"
    g_csv = subset / "mapping_gallery.csv"
    if not q_csv.exists() or not g_csv.exists():
        raise SystemExit("mapping CSVs not found. Run mk_subset.py first.")

    q_rels, _ = read_mapping(q_csv)
    g_rels, _ = read_mapping(g_csv)

    q_paths = [str(subset / rel) for rel in q_rels]
    g_paths = [str(subset / rel) for rel in g_rels]

    provider = get_provider(args.provider)
    Xq = provider.embed_images(q_paths)
    Xg = provider.embed_images(g_paths)

    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    np.save(outd / "query.npy", Xq)
    np.save(outd / "gallery.npy", Xg)

    write_metadata(
        outd, provider=provider.name, modality="image",
        dim=int(Xq.shape[1]), l2_normalized=True
    )
    print(f"[done] image embeds -> {outd} query={Xq.shape} gallery={Xg.shape}")

if __name__ == "__main__":
    main()
