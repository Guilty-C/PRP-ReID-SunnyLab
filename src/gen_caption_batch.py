# -*- coding: utf-8 -*-
# gen_caption_batch.py —— 批量版，兼容 path/image_id，带 tqdm 进度条

import os
import argparse
import csv
import json
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import GPT2Tokenizer

# 初始化分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 初始化 VimsAI 客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://usa.vimsai.com/v1"
)

def count_tokens(prompt_text: str) -> int:
    try:
        return len(tokenizer.encode(prompt_text))
    except Exception as e:
        print(f"[ERROR] token 统计失败: {e}")
        return 0

def call_model(prompt_text: str) -> str:
    """调用模型生成 caption"""
    try:
        resp = client.chat.completions.create(
            "qwen-plus",   # ⚠️ 确认 VimsAI 支持的模型名
            messages=[
                {"role": "system", "content": "你是一个善于描述人物外观的助手，只描述人物可见外观"},
                {"role": "user", "content": prompt_text}
            ],
            timeout=20
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {e}"

def extract_image_id(rec: dict) -> str:
    """统一提取 image_id"""
    if "image_id" in rec:
        return rec["image_id"]
    if "path" in rec:
        base = os.path.basename(rec["path"])
        iid, _ = os.path.splitext(base)
        return iid
    return "UNKNOWN"

def load_index(index_file: str):
    """支持 JSON / JSONL / TXT 三种索引文件"""
    with open(index_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # JSONL
    lines = [ln for ln in content.splitlines() if ln.strip()]
    try:
        return [json.loads(ln) for ln in lines]
    except Exception:
        # TXT
        return [{"image_id": ln.strip()} for ln in lines if ln.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file", type=str, help="索引文件(JSON/JSONL/TXT)")
    parser.add_argument("--prompt_file", type=str, required=True, help="prompt 文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出 CSV 文件")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="线程数")
    parser.add_argument("--verbose", action="store_true", help="是否逐条打印日志")
    args = parser.parse_args()

    # 读取 prompt
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    prompt_tokens = count_tokens(prompt_text)

    # 读取索引
    records = load_index(args.index_file)
    total = len(records)
    print(f"[gen_caption_batch] 共 {total} 条记录，将写入 {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp_out = args.out + ".tmp"

    with open(tmp_out, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_id", "path", "split", "prompt", "caption", "timestamp", "tokens"])

        # 分批处理
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for i in range(0, total, args.batch_size):
                batch = records[i:i + args.batch_size]
                futures[executor.submit(process_batch, batch, prompt_text, prompt_tokens)] = batch

            for fut in tqdm(as_completed(futures), total=len(futures), desc="[gen_caption_batch] 生成中"):
                captions = fut.result()
                for row in captions:
                    writer.writerow(row)
                    if args.verbose:
                        print(f"[gen_caption_batch] 已完成: {row[0]}")

    os.replace(tmp_out, args.out)
    print(f"[gen_caption_batch] 任务完成 ✅ 输出文件: {args.out}")

def process_batch(batch, prompt_text, prompt_tokens):
    rows = []
    for rec in batch:
        image_id = extract_image_id(rec)
        path = rec.get("path", "")
        split = rec.get("split", "")
        user_prompt = f"{prompt_text}\n图像ID: {image_id}"
        caption = call_model(user_prompt)
        timestamp = datetime.now().isoformat()
        rows.append([image_id, path, split, prompt_text, caption, timestamp, prompt_tokens])
    return rows

if __name__ == "__main__":
    main()
