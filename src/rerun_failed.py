#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_failed.py (修正版)
- 读取 failed.jsonl
- 如果缺失 image_id，自动用 basename(path)
- 重新生成 caption 并更新 captions.csv
"""

import os
import sys
import csv
import json
import argparse
import time
import re
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
import base64

def encode_image_to_dataurl(path):
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def extract_json_array(text):
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        return match.group(0)
    return None

def call_model(client, rec, prompt_text):
    """单张图片调用模型"""
    system_msg = {"role": "system", "content": [{"type": "text", "text": "你是一个善于描述人物外观的助手"}]}

    image_id = rec.get("image_id") or os.path.basename(rec.get("path", "unknown.jpg"))

    try:
        data_url = encode_image_to_dataurl(rec["path"])
    except Exception:
        data_url = ""

    user_msg = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text",
             "text": (
                 f"{prompt_text}\n请严格只输出 JSON 数组，格式如下：\n"
                 f"[{{\"image_id\": \"{image_id}\", \"caption\": \"<描述>\"}}]\n"
                 f"不要输出任何额外说明。"
             )}
        ]
    }

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

            json_str = extract_json_array(text)
            if not json_str:
                raise ValueError("未找到 JSON 数组")

            data = json.loads(json_str)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                return data[0].get("caption", "生成失败")
            return "生成失败"
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"[WARN] {image_id} 调用失败 (第 {attempt+1} 次): {e}, {wait}s 后重试...", file=sys.stderr)
            time.sleep(wait)
    return "生成失败"

def main():
    parser = argparse.ArgumentParser(description="重新生成失败的 captions (修正版)")
    parser.add_argument("--failed", default="./outputs/captions/failed.jsonl", help="failed.jsonl 文件路径")
    parser.add_argument("--csv", default="./outputs/captions/captions.csv", help="captions.csv 文件路径")
    parser.add_argument("--prompt_file", help="自定义 prompt 文件")
    args = parser.parse_args()

    # prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
    else:
        prompt_text = "请描述图像中的人物，包括性别、上衣颜色、下装、鞋子，以及是否背背包。"

    # 读取失败记录
    failed_records = []
    with open(args.failed, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if "image_id" not in rec:
                    rec["image_id"] = os.path.basename(rec.get("path", "unknown.jpg"))
                failed_records.append(rec)

    if not failed_records:
        print("[INFO] 没有失败记录")
        return

    # 初始化 API
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 读取已有 CSV → 内存字典
    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 逐条处理
    for rec in tqdm(failed_records, desc="补跑失败项"):
        caption = call_model(client, rec, prompt_text)
        ts = datetime.now().isoformat()
        # 更新 CSV 中对应行
        for row in rows:
            if row["image_id"] == rec["image_id"]:
                row["caption"] = caption
                row["timestamp"] = ts
                break

    # 写回 CSV
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "path", "split", "prompt", "caption", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] 已完成补跑，更新写入 {args.csv}")

if __name__ == "__main__":
    main()
