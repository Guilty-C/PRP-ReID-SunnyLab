import argparse
import json
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data/market1501")
    ap.add_argument("--split", choices=["query", "bounding_box_test"], required=True,
                   help="选择要索引的 split (query 或 bounding_box_test)")
    ap.add_argument("--out_index", default="./outputs/runs/index.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    samples = []
    p = os.path.join(args.data_root, args.split)
    if os.path.isdir(p):
        imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith((".jpg", ".png"))]
        samples = [{"image_id": os.path.basename(x), "path": x, "split": args.split} for x in imgs]

    with open(args.out_index, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[prepare_data] indexed {len(samples)} images -> {args.out_index}")

if __name__ == "__main__":
    main()
