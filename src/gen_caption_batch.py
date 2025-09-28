#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版 JSON 模式 gen_caption_batch.py
- 模型输出统一使用 "image_id"
- 解析逻辑支持顺序兜底
"""

import os
import sys
import csv
import json
import base64
import argparse
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

def load_index(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    records = []
    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
    elif ext == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append({"image_id": line, "path": line, "split": ""})
    else:
        raise ValueError(f"不支持的索引文件类型: {ext}")
    return records

def encode_image_to_dataurl(path):
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def extract_json_array(text):
    """正则提取 JSON 数组"""
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        return match.group(0)
    return None

def call_model(client, batch, prompt_text):
    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": "You are a precise assistant that outputs valid JSON only."}]
    }

    content = []
    image_ids = [rec["image_id"] for rec in batch]

    for rec in batch:
        try:
            data_url = encode_image_to_dataurl(rec["path"])
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        except Exception:
            content.append({"type": "image_url", "image_url": {"url": ""}})

    # ✅ 改进后的 prompt
    content.append({
        "type": "text",
        "text": (
            f"{prompt_text}\n"
            f"Input image_ids: {image_ids}\n"
            f"Please output a JSON array. Each object must contain exactly two fields:\n"
            f'  - "image_id": one from the input list\n'
            f'  - "caption": an English description of the person\n\n'
            f"Example:\n"
            f"[\n"
            f'  {{"image_id": "{image_ids[0]}", "caption": "A man in a blue shirt."}},\n'
            f'  {{"image_id": "{image_ids[1]}", "caption": "A woman in a red jacket."}}\n'
            f"]\n"
            f"⚠️ Only output the JSON array, no explanations."
        )
    })
    user_msg = {"role": "user", "content": content}

    # ======== API 调用 + 解析 ========
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="qwen-plus",
                messages=[system_msg, user_msg]
            )
            content_list = resp.choices[0].message.content
            text = ""
            if isinstance(content_list, list):
                text = "".join(seg.get("text", seg) if isinstance(seg, dict) else str(seg) for seg in content_list)
            else:
                text = str(content_list)

            # 提取 JSON 数组
            match = re.search(r"\[.*\]", text, re.S)
            if not match:
                raise ValueError("No JSON array found in response")
            json_str = match.group(0)
            data = json.loads(json_str)

            captions = []
            mapping = {str(d.get("image_id")): d.get("caption", "生成失败") for d in data if isinstance(d, dict)}
            for idx, rec in enumerate(batch):
                cap = mapping.get(str(rec["image_id"]))
                if not cap:
                    if idx < len(data) and isinstance(data[idx], dict):
                        cap = data[idx].get("caption", "生成失败")
                    else:
                        cap = "生成失败"
                captions.append(cap)
            return captions
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"[WARN] API 调用失败 (第 {attempt+1} 次): {e}, {wait}s 后重试...", file=sys.stderr)
            time.sleep(wait)

    # ======== 如果全部失败，兜底 ========
    return ["生成失败"] * len(batch)


def main():
    parser = argparse.ArgumentParser(description="批量生成图像描述 (Qwen API, JSON模式修正版)")
    parser.add_argument("index_file", help="索引文件 (JSON / JSONL / TXT)")
    parser.add_argument("--out", default="./outputs/captions", help="输出目录")
    parser.add_argument("--num_workers", type=int, default=8, help="线程数")
    parser.add_argument("--batch_size", type=int, default=1, help="每批图像数 (默认 1)")
    parser.add_argument("--prompt_file", help="自定义 prompt 文件")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
    else:
        prompt_text = "请描述图像中的人物，包括性别、上衣颜色、下装、鞋子，以及是否背背包。"

    records = load_index(args.index_file)
    if not records:
        print("[ERROR] 索引为空")
        return

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "captions.csv")
    failed_path = os.path.join(args.out, "failed.jsonl")

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_ids.add(row["image_id"])

    pending = [r for r in records if str(r.get("image_id")) not in done_ids]
    if not pending:
        print("[INFO] 已完成，无需生成")
        return

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    write_header = not os.path.exists(out_path)
    out_csv = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_csv)
    if write_header:
        writer.writerow(["image_id", "path", "split", "prompt", "caption", "timestamp"])

    batch_size = args.batch_size
    batches = [pending[i:i+batch_size] for i in range(0, len(pending), batch_size)]

    failed_records = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(call_model, client, batch, prompt_text): idx for idx, batch in enumerate(batches)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="处理批次"):
            idx = futures[fut]
            batch = batches[idx]
            captions = fut.result()
            for rec, cap in zip(batch, captions):
                ts = datetime.now().isoformat()
                writer.writerow([rec.get("image_id"), rec.get("path"), rec.get("split"),
                                 prompt_text, cap, ts])
                if cap == "生成失败":
                    failed_records.append(rec)
            out_csv.flush()

    out_csv.close()

    if failed_records:
        with open(failed_path, "w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[WARN] {len(failed_records)} 条生成失败，记录在 {failed_path}")
    else:
        print("[INFO] 全部生成成功 ✅")

    print(f"[INFO] 已完成，结果保存到 {out_path}")

if __name__ == "__main__":
    main()
