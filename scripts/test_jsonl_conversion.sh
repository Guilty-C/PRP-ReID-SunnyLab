#!/usr/bin/env bash
set -euo pipefail

# 确保输出目录存在
mkdir -p ./outputs/captions

# 如果已经有captions.jsonl文件，先删除
if [ -f ./outputs/captions/captions.jsonl ]; then
    echo "删除现有JSONL文件"
    rm ./outputs/captions/captions.jsonl
fi

# 执行CSV到JSONL的转换
python -c "import csv, json, os; csv_file = './outputs/captions/captions_20.csv'; jsonl_file = './outputs/captions/captions.jsonl'; with open(csv_file, 'r', encoding='utf-8') as f_in, open(jsonl_file, 'w', encoding='utf-8') as f_out: reader = csv.DictReader(f_in); for row in reader: try: image_data = json.loads(row['image_id']); if isinstance(image_data, list): for item in image_data: image_id = os.path.splitext(os.path.basename(item['path']))[0]; json_line = {'image_id': image_id, 'path': item['path'], 'caption': row['caption']}; print(f'生成JSON行: {json_line}'); f_out.write(json.dumps(json_line) + '\n'); except Exception as e: print(f'转换行时出错: {e}'); continue"

# 检查转换结果
echo -e "\n转换后的JSONL文件内容预览:"
head -n 2 ./outputs/captions/captions.jsonl

echo "\n转换后的JSONL文件行数: $(wc -l < ./outputs/captions/captions.jsonl)"

# 尝试验证JSONL文件格式
if python -c "import json; with open('./outputs/captions/captions.jsonl', 'r') as f: for line in f: try: json.loads(line); print(f'验证成功: {line[:50]}...'); break except Exception as e: print(f'JSON格式错误: {e}'); exit(1)";
then
    echo "\n✅ JSONL文件格式验证成功！"
else
    echo "\n❌ JSONL文件格式验证失败！"
fi