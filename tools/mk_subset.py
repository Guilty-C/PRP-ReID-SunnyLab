#!/usr/bin/env python3
"""
Reproducible Market-1501 mini subset maker (robust dirs).
- Accepts query dirs: ["query", "gt_query"]
- Accepts gallery dirs: ["bounding_box_test", "bounding_box_train"]
- Prefers test; falls back to train with a warning.
"""
import argparse, json, random, re, hashlib, shutil
from pathlib import Path
from typing import List, Dict

JUNK_PID = {-1, 0}
PID_RE = re.compile(r"^(?P<pid>-?\d{1,4})_c(?P<cam>\d)")

# ---- optional Pillow (lazy) ----
_PIL_OK = None
def _ensure_pillow():
    global _PIL_OK
    if _PIL_OK is not None:
        return _PIL_OK
    try:
        from PIL import Image  # noqa: F401
        _PIL_OK = True
    except Exception:
        _PIL_OK = False
    return _PIL_OK

def parse_pid_cam(name: str):
    m = PID_RE.match(name)
    if not m:
        return None, None
    return int(m.group("pid")), int(m.group("cam"))

def sha1_of_file(fp: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def is_low_res(fp: Path, min_w=64, min_h=128) -> bool:
    if min_w <= 0 or min_h <= 0:
        return False
    if not _ensure_pillow():
        return False
    try:
        from PIL import Image
        with Image.open(fp) as im:
            w, h = im.size
        return (w < min_w) or (h < min_h)
    except Exception:
        return True

def _choose_dir(root: Path, names: list[str]) -> Path | None:
    for n in names:
        p = root / n
        if p.exists() and p.is_dir():
            return p
    return None

def collect_market_paths(root: Path):
    # Try common names (case-insensitive on Windows anyway)
    q_dir = _choose_dir(root, ["query", "gt_query"])
    g_dir = _choose_dir(root, ["bounding_box_test", "bounding_box_train"])

    # If still missing, print what we see to help debugging
    if q_dir is None or g_dir is None:
        existing = [p.name for p in root.iterdir() if p.is_dir()]
        raise SystemExit(
            "Could not locate required Market-1501 folders under "
            f"{root}.\n- Looked for query in: ['query','gt_query'] -> found: {q_dir}\n"
            f"- Looked for gallery in: ['bounding_box_test','bounding_box_train'] -> found: {g_dir}\n"
            f"- Existing subdirs here: {existing}\n"
            "Tip: point --root to the folder that directly contains these subfolders."
        )

    # Prefer test if both exist
    if (root / "bounding_box_test").exists():
        g_dir = root / "bounding_box_test"
    elif (root / "bounding_box_train").exists():
        print("[warn] bounding_box_test not found; using bounding_box_train as gallery for this mini subset.")
        g_dir = root / "bounding_box_train"

    q_paths = sorted((q_dir).glob("*.jpg"))
    g_paths = sorted((g_dir).glob("*.jpg"))

    if not q_paths:
        raise SystemExit(f"No JPGs found in query dir: {q_dir}")
    if not g_paths:
        raise SystemExit(f"No JPGs found in gallery dir: {g_dir}")

    print(f"[info] query dir:  {q_dir} ({len(q_paths)} files)")
    print(f"[info] gallery dir:{g_dir} ({len(g_paths)} files)")
    return q_paths, g_paths

def index_by_pid(paths: List[Path]) -> Dict[int, List[Path]]:
    by = {}
    for p in paths:
        pid, cam = parse_pid_cam(p.name)
        if pid is None or pid in JUNK_PID:
            continue
        by.setdefault(pid, []).append(p)
    return by

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path containing query/ (or gt_query/) and bounding_box_test/ (or *_train/)")
    ap.add_argument("--ids", type=int, default=20)
    ap.add_argument("--imgs-per-id", type=int, default=6)
    ap.add_argument("--out", type=str, default="data/market-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-w", type=int, default=64)
    ap.add_argument("--min-h", type=int, default=128)
    args = ap.parse_args()

    if (args.min_w > 0 or args.min_h > 0) and not _ensure_pillow():
        print("[warn] Pillow unavailable; proceeding WITHOUT low-resolution filtering. "
              "Use --min-w 0 --min-h 0 to silence this.")

    random.seed(args.seed)
    root = Path(args.root)
    out = Path(args.out)
    (out / "query").mkdir(parents=True, exist_ok=True)
    (out / "gallery").mkdir(parents=True, exist_ok=True)

    q_paths, g_paths = collect_market_paths(root)
    by_q = index_by_pid(q_paths)
    by_g = index_by_pid(g_paths)

    candidate = sorted(set(by_q) & set(by_g))
    if not candidate:
        raise SystemExit("No overlapping PIDs between query and gallery.")

    if len(candidate) < args.ids:
        print(f"[warn] Requested {args.ids} IDs, but only {len(candidate)} available. Using all.")
    chosen = random.sample(candidate, k=min(args.ids, len(candidate)))

    import shutil, csv
    map_q_rows, map_g_rows = [], []
    kept_gallery = dup_skipped = lowres_skipped = 0

    for pid in chosen:
        # 1 query per PID
        q_choice = random.choice(by_q[pid])
        shutil.copy2(q_choice, out / "query" / q_choice.name)
        map_q_rows.append({"relpath": f"query/{q_choice.name}", "pid": pid})

        # up to M gallery per PID
        g_list = by_g[pid][:]
        random.shuffle(g_list)
        seen_hash = set()
        picked = 0
        for fp in g_list:
            if picked >= args.imgs_per_id: break
            if is_low_res(fp, args.min_w, args.min_h):
                lowres_skipped += 1; continue
            h = sha1_of_file(fp)
            if h in seen_hash:
                dup_skipped += 1; continue
            seen_hash.add(h)
            shutil.copy2(fp, out / "gallery" / fp.name)
            map_g_rows.append({"relpath": f"gallery/{fp.name}", "pid": pid})
            picked += 1; kept_gallery += 1

    with open(out / "mapping_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relpath", "pid"]); w.writeheader(); w.writerows(map_q_rows)
    with open(out / "mapping_gallery.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relpath", "pid"]); w.writeheader(); w.writerows(map_g_rows)

    stats = {
        "seed": args.seed,
        "num_pids": len(chosen),
        "gallery_per_pid_max": args.imgs_per_id,
        "kept_gallery": kept_gallery,
        "dup_skipped": dup_skipped,
        "lowres_skipped": lowres_skipped,
        "root": str(root),
    }
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("[done] subset at", out)

if __name__ == "__main__":
    main()
