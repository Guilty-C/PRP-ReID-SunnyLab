import os
import argparse
import csv
import json
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm   # ✅ 进度条

# 初始化 client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://usa.vimsai.com/v1"
)

def call_qwen(prompt_text):
    """调用 Qwen 模型生成描述"""
    try:
        resp = client.chat.completions.create(
            model="qwen-plus",   # ⚠️ 确认 VimsAI 支持的模型名
            messages=[
                {"role": "system", "content": "你是一个善于描述人物外观的助手，只描述人物可见外观"},
                {"role": "user", "content": prompt_text}
            ],
            timeout=20
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] 请求失败: {e}")
        return "描述失败"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, help="待处理的图片ID列表txt或JSON文件")
    parser.add_argument("--prompt_file", type=str, required=True, help="prompt文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出目录或文件")
    parser.add_argument("--verbose", action="store_true", help="是否显示逐条日志")  # ✅ 新增
    args = parser.parse_args()

    # 处理 out 路径
    args.out = os.path.normpath(args.out)
    if os.path.splitext(args.out)[1]:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_path = args.out
    else:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "captions_20.csv")

    out_path = os.path.abspath(out_path)
    print(f"[gen_caption] 输出文件绝对路径: {out_path}")

    # 读取 prompt
    try:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
    except Exception as e:
        print(f"[ERROR] 无法打开prompt文件: {e}")
        exit(1)

    # 解析 index 文件
    try:
        with open(args.index, "r", encoding="utf-8") as f:
            content = f.read().strip()
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    image_ids = []
                    for item in data:
                        if isinstance(item, dict):
                            if "image_id" in item:
                                image_ids.append(item["image_id"])
                            elif "path" in item:
                                base = os.path.basename(item["path"])
                                iid, _ = os.path.splitext(base)
                                image_ids.append(iid)
                        elif isinstance(item, str):
                            image_ids.append(item)
                    print(f"[gen_caption] 解析为JSON数组，共 {len(image_ids)} 个条目")
                else:
                    image_ids = [line.strip() for line in content.splitlines() if line.strip()]
            except json.JSONDecodeError:
                image_ids = [line.strip() for line in content.splitlines() if line.strip()]
    except Exception as e:
        print(f"[ERROR] 无法打开index文件: {e}")
        exit(1)

    # 只取前 20
    image_ids = image_ids[:20]
    print(f"[gen_caption] 开始处理前 {len(image_ids)} 张图片，输出到 {out_path}")

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] 无法创建输出目录: {e}")
            exit(1)

    try:
        with open(out_path, "w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["image_id", "prompt", "caption", "timestamp"])

            for i, img_id in enumerate(
                tqdm(image_ids, desc="[gen_caption] 生成中", unit="img"), 1
            ):
                caption = call_qwen(f"{prompt_text}\n图像ID: {img_id}")
                timestamp = datetime.now().isoformat()
                csv_writer.writerow([img_id, prompt_text, caption, timestamp])

                if args.verbose:  # ✅ 如果启用 verbose，打印详细日志
                    print(f"[gen_caption] 已完成 {i}/{len(image_ids)}: {img_id}")

        print("[gen_caption] 任务完成 ✅")
    except PermissionError as e:
        print(f"[ERROR] 权限错误：无法写入文件 '{out_path}'")
        print(f"详细错误: {e}")
        exit(1)
    except Exception as e:
        print(f"[ERROR] 无法写入文件: {e}")
        exit(1)
