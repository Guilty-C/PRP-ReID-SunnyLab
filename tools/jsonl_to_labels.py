# tools/jsonl_to_labels.py
from __future__ import annotations
import argparse, json, re, sys

def _as_pid4(v):
    """
    更鲁棒的解析：
    - int -> 4位
    - str -> 提取其中最后一段数字（如 'pid-9'、'person_0009'、'id=12'）
    - 统一转成 4 位（至少4位；若数字本身>9999则按其实际位数）
    """
    if isinstance(v, int):
        n = v
        width = 4 if n < 10_000 else len(str(n))
        return f"{n:0{width}d}"
    if isinstance(v, str):
        s = v.strip()
        nums = re.findall(r"\d+", s)
        if nums:
            n = int(nums[-1])              # 取最后一段数字，适配 'pid-0009-c1s1' 等
            width = 4 if n < 10_000 else len(str(n))
            return f"{n:0{width}d}"
        # 如果字符串本身是4位开头数字（极端fallback）
        if len(s) >= 4 and s[:4].isdigit():
            return s[:4]
    raise ValueError(f"无法解析为 pid4: {v!r}")

def main():
    ap = argparse.ArgumentParser("Extract labels (pid) from JSONL to .txt")
    ap.add_argument("--jsonl", required=True, help="输入 JSONL（每行一个对象）")
    ap.add_argument("--out", required=True, help="输出 labels.txt（每行一个 pid）")
    ap.add_argument("--fields", default="pid,person_id,id,label,name,text",
                    help="候选字段，按优先级逗号分隔（默认包含 pid,person_id,id,label,name,text）")
    args = ap.parse_args()

    fields = [x.strip() for x in args.fields.split(",") if x.strip()]
    labels = []
    used_field = None

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[jsonl_to_labels] 第{idx}行 JSON 解析失败：{e}", file=sys.stderr)
                raise

            val = None
            for k in fields:
                if k in obj:
                    val = obj[k]
                    used_field = used_field or k  # 记录第一次命中的字段名
                    break
            if val is None:
                raise SystemExit(f"[jsonl_to_labels] 第{idx}行找不到任一字段 {fields}；对象键有：{list(obj.keys())}")

            try:
                labels.append(_as_pid4(val))
            except Exception as e:
                raise SystemExit(f"[jsonl_to_labels] 第{idx}行解析失败：{e}；该行值={val!r}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")

    print(f"[jsonl_to_labels] saved: {args.out}  lines={len(labels)}")
    if used_field:
        print(f"[jsonl_to_labels] 首个命中的字段: {used_field}")

if __name__ == "__main__":
    main()
