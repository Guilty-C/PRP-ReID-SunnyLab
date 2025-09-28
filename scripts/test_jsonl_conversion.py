import csv
import json
import os

# 确保输出目录存在
os.makedirs('./outputs/captions', exist_ok=True)

# 定义文件路径
csv_file = './outputs/captions/captions_20.csv'
jsonl_file = './outputs/captions/captions.jsonl'

# 检查CSV文件是否存在
if not os.path.exists(csv_file):
    print(f"错误：CSV文件 {csv_file} 不存在")
    exit(1)

# 删除现有JSONL文件（如果存在）
if os.path.exists(jsonl_file):
    print("删除现有JSONL文件")
    os.remove(jsonl_file)

# 执行CSV到JSONL的转换
print(f"开始从 {csv_file} 转换到 {jsonl_file}")
with open(csv_file, 'r', encoding='utf-8') as f_in, open(jsonl_file, 'w', encoding='utf-8') as f_out:
    reader = csv.DictReader(f_in)
    line_count = 0
    
    for row in reader:
        try:
            # 解析image_id字段中的JSON数据
            image_data = json.loads(row['image_id'])
            
            if isinstance(image_data, list):
                for item in image_data:
                    # 从path中提取image_id
                    image_id = os.path.splitext(os.path.basename(item['path']))[0]
                    # 创建JSON行
                    json_line = {
                        'image_id': image_id,
                        'path': item['path'],
                        'caption': row['caption']
                    }
                    # 写入JSONL文件
                    f_out.write(json.dumps(json_line) + '\n')
                    line_count += 1
                    # 只打印前3行作为示例
                    if line_count <= 3:
                        print(f'生成JSON行 {line_count}: {json.dumps(json_line)[:100]}...')
            else:
                print(f"警告：image_data不是列表类型: {type(image_data)}")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        except Exception as e:
            print(f"处理行时出错: {e}")
            import traceback
            traceback.print_exc()

# 检查转换结果
print(f"\n转换后的JSONL文件已生成: {jsonl_file}")
print(f"转换后的JSONL文件行数: {line_count}")

# 尝试验证JSONL文件格式
print("\n验证JSONL文件格式...")
valid_count = 0
invalid_count = 0
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            valid_count += 1
            # 检查必要字段
            required_fields = ['image_id', 'path', 'caption']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"第{i}行缺少字段: {missing_fields}")
                invalid_count += 1
            # 打印前2行作为验证示例
            if i <= 2:
                print(f"验证成功 (行{i}): {line.strip()[:100]}...")
        except json.JSONDecodeError as e:
            print(f"第{i}行JSON格式错误: {e}")
            print(f"错误内容: {line.strip()[:100]}...")
            invalid_count += 1

print(f"\n验证统计: 有效行={valid_count}, 无效行={invalid_count}")

if invalid_count == 0:
    print("✅ JSONL文件格式验证成功！")
else:
    print("❌ JSONL文件格式验证失败！")