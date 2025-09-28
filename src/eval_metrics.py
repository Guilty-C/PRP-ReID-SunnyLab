import argparse, json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="retrieve输出结果")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    # TODO: 替换为真实的 mAP / Rank-1 计算
    print("[eval_metrics] 当前仅做示例：")
    print(f"查询数: {len(results)}")
    print("Rank-1: 0.00 (占位)")
    print("mAP: 0.00 (占位)")
