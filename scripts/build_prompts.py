#!/usr/bin/env python3
"""
Build attribute-style prompt variations for the mini subset.

Input:
  --subset  : path to data/market-mini (must contain mapping_query.csv)
  --out     : output JSONL file (one record per line)
  --variants: number of prompt variants per PID per language (default 3)
  --seed    : RNG seed (default 42)
  --langs   : comma-separated languages to generate (default "en,zh")

Output JSONL schema (per line):
  {
    "id": "pid-XXXX",
    "lang": "en" | "zh",
    "variant": 0-based int,
    "prompt": "<text>"
  }
"""
import argparse
import csv
import json
import random
from pathlib import Path

TEMPLATES_EN = [
    "A full-body photo of a pedestrian for re-identification, neutral pose, clear appearance.",
    "Pedestrian appearance description for ReID: clothing, color, accessories, shoes, carrying items.",
    "ReID caption focusing on clothing texture, sleeve length, pants style, shoes, backpack presence."
]

TEMPLATES_ZH = [
    "用于行人重识别的全身照片描述，姿态自然，外观清晰。",
    "行人 ReID 外观描述：服装、颜色、配饰、鞋子、是否背包。",
    "强调衣物纹理、袖长、裤型、鞋子与背包存在的 ReID 文本描述。"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=str, required=True, help="data/market-mini directory (must contain mapping_query.csv)")
    ap.add_argument("--out", type=str, required=True, help="output JSONL path")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--langs", type=str, default="en,zh", help="comma-separated languages, e.g., en,zh or en")
    args = ap.parse_args()

    random.seed(args.seed)
    subset = Path(args.subset)
    map_q = subset / "mapping_query.csv"
    if not map_q.exists():
        raise SystemExit(f"[error] mapping_query.csv not found at {map_q}. Run mk_subset.py first.")

    # Collect unique PIDs from mapping_query.csv
    pids = set()
    with open(map_q, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "pid" not in reader.fieldnames:
            raise SystemExit("[error] mapping_query.csv missing 'pid' column.")
        for row in reader:
            try:
                pids.add(int(row["pid"]))
            except ValueError:
                continue
    pids = sorted(pids)
    if not pids:
        raise SystemExit("[error] No PIDs found in mapping_query.csv.")

    langs = [s.strip().lower() for s in args.langs.split(",") if s.strip()]
    unsupported = [l for l in langs if l not in {"en", "zh"}]
    if unsupported:
        raise SystemExit(f"[error] Unsupported languages: {unsupported}. Allowed: en, zh")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(outp, "w", encoding="utf-8") as f:
        for pid in pids:
            for v in range(args.variants):
                if "en" in langs:
                    tpl = TEMPLATES_EN[v % len(TEMPLATES_EN)]
                    f.write(json.dumps({
                        "id": f"pid-{pid}",
                        "lang": "en",
                        "variant": v,
                        "prompt": tpl
                    }, ensure_ascii=False) + "\n")
                    count += 1
                if "zh" in langs:
                    tpl = TEMPLATES_ZH[v % len(TEMPLATES_ZH)]
                    f.write(json.dumps({
                        "id": f"pid-{pid}",
                        "lang": "zh",
                        "variant": v,
                        "prompt": tpl
                    }, ensure_ascii=False) + "\n")
                    count += 1

    print(f"[done] prompts -> {outp} ({count} lines)")
    print(f"[info] PIDs: {len(pids)}, variants per lang: {args.variants}, langs: {','.join(langs)}")


if __name__ == "__main__":
    main()
