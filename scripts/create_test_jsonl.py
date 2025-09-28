import json, os

# 定义输出文件路径
jsonl_file = './outputs/captions/captions.jsonl'

# 删除现有文件（如果存在）
if os.path.exists(jsonl_file):
    os.remove(jsonl_file)
    print(f'已删除现有文件: {jsonl_file}')

# 确保输出目录存在
os.makedirs('./outputs/captions', exist_ok=True)

# 创建模拟数据
mock_data = [
    {'image_id': '0001_c1s1_001051_00', 'path': './data/market1501/query/0001_c1s1_001051_00.jpg', 'caption': 'A person wearing a dark jacket and light pants.'},
    {'image_id': '0001_c2s1_000301_00', 'path': './data/market1501/query/0001_c2s1_000301_00.jpg', 'caption': 'A person with a backpack and blue shirt.'},
    {'image_id': '0001_c3s1_000551_00', 'path': './data/market1501/query/0001_c3s1_000551_00.jpg', 'caption': 'A person in a red coat and black pants.'},
    {'image_id': '-1_c1s1_000401_03', 'path': './data/market1501/bounding_box_test/-1_c1s1_000401_03.jpg', 'caption': 'A person wearing a white shirt and dark pants.'},
    {'image_id': '-1_c1s1_000451_04', 'path': './data/market1501/bounding_box_test/-1_c1s1_000451_04.jpg', 'caption': 'A person with a hat and striped shirt.'}
]

# 写入JSONL文件
with open(jsonl_file, 'w', encoding='utf-8') as f:
    for item in mock_data:
        f.write(json.dumps(item) + '\n')

print(f'已创建模拟JSONL文件: {jsonl_file}，包含 {len(mock_data)} 条记录')