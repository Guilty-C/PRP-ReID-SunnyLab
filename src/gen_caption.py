import os
import argparse
import csv
import json
from datetime import datetime
from openai import OpenAI

# 初始化 client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # 建议统一用 OPENAI_API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def call_qwen(prompt_text):
    """调用 Qwen 模型生成描述"""
    try:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个善于描述人物外观的助手，只描述人物可见外观"},
                {"role": "user", "content": prompt_text}
            ],
            timeout=20  # 设置超时时间，避免无限卡住
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] 请求失败: {e}")
        return "描述失败"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, help="待处理的图片ID列表txt或JSON文件")
    parser.add_argument("--prompt_file", type=str, required=True, help="prompt文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    # 检查out参数是目录还是文件，如果是文件则创建其父目录
    # 确保路径在Windows上正常工作
    args.out = os.path.normpath(args.out)
    
    if os.path.splitext(args.out)[1]:  # 有扩展名，认为是文件
        out_dir = os.path.dirname(args.out)
        if out_dir:  # 确保有父目录
            os.makedirs(out_dir, exist_ok=True)
        out_path = args.out
    else:  # 没有扩展名，认为是目录
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "captions_20.csv")
    
    # 标准化路径格式，确保Windows兼容性
    out_path = os.path.normpath(out_path)
    # 转换为绝对路径，提高Windows兼容性
    out_path = os.path.abspath(out_path)
    print(f"[gen_caption] 输出文件绝对路径: {out_path}")

    try:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
    except Exception as e:
        print(f"[ERROR] 无法打开prompt文件: {e}")
        exit(1)

    # 尝试解析索引文件，支持JSON和普通文本格式
    try:
        with open(args.index, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
            # 尝试作为JSON解析
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    # 如果是列表，提取其中的image_id或path字段
                    image_ids = []
                    for item in data:
                        if isinstance(item, dict):
                            if "image_id" in item:
                                image_ids.append(item["image_id"])
                            elif "path" in item:
                                # 从path中提取文件名作为image_id
                                base = os.path.basename(item["path"])
                                iid, _ = os.path.splitext(base)
                                image_ids.append(iid)
                        elif isinstance(item, str):
                            image_ids.append(item)
                    print(f"[gen_caption] 解析为JSON数组，共 {len(image_ids)} 个条目")
                else:
                    # 不是列表，尝试按普通文本处理
                    image_ids = [line.strip() for line in content.splitlines() if line.strip()]
            except json.JSONDecodeError:
                # JSON解析失败，按普通文本处理
                image_ids = [line.strip() for line in content.splitlines() if line.strip()]
    except Exception as e:
        print(f"[ERROR] 无法打开index文件: {e}")
        exit(1)

    # 只取前 20 张
    image_ids = image_ids[:20]
    print(f"[gen_caption] 开始处理前 20 张图片，输出到 {out_path}")

    # 检查输出目录是否存在，如果不存在则创建
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
            print(f"[gen_caption] 已创建输出目录: {out_dir}")
        except Exception as e:
            print(f"[ERROR] 无法创建输出目录: {e}")
            exit(1)

    # 检查是否可以写入文件
    try:
        with open(out_path, "w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["image_id", "prompt", "caption", "timestamp"])
            
            for i, img_id in enumerate(image_ids, 1):
                caption = call_qwen(f"{prompt_text}\n图像ID: {img_id}")
                timestamp = datetime.now().isoformat()
                
                csv_writer.writerow([img_id, prompt_text, caption, timestamp])

                # 每张都打印进度
                print(f"[gen_caption] 已完成 {i}/{len(image_ids)}: {img_id}")

        print("[gen_caption] 任务完成 ✅")
    except PermissionError as e:
        print(f"[ERROR] 权限错误：无法写入文件 '{out_path}'. 请检查文件权限或确认文件未被其他程序锁定。")
        print(f"详细错误: {e}")
        # 尝试使用备选路径
        alt_out_path = os.path.abspath(os.path.join("outputs", "captions_alt.csv"))
        print(f"[gen_caption] 尝试使用备选路径: {alt_out_path}")
        try:
            with open(alt_out_path, "w", encoding="utf-8", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["image_id", "prompt", "caption", "timestamp"])
                
                # 在备选路径中写入完整数据
                for i, img_id in enumerate(image_ids, 1):
                    caption = call_qwen(f"{prompt_text}\n图像ID: {img_id}")
                    timestamp = datetime.now().isoformat()
                    csv_writer.writerow([img_id, prompt_text, caption, timestamp])
                    
                print(f"[gen_caption] 已使用备选路径成功创建完整文件")
                # 成功使用备选路径后不退出，而是继续执行
                exit(0)
        except Exception as e2:
            print(f"[ERROR] 使用备选路径也失败: {e2}")
            exit(1)
    except Exception as e:
        print(f"[ERROR] 无法写入文件: {e}")
        exit(1)
