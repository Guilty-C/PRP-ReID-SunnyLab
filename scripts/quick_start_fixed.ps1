# Windows PowerShell版本的快速启动脚本

# 设置UTF-8编码
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# 确保输出目录存在
Write-Host "正在创建必要的目录..."
if (-not (Test-Path -Path "./outputs/captions")) {
    New-Item -ItemType Directory -Path "./outputs/captions" -Force | Out-Null
    Write-Host "已创建目录: outputs/captions"
}

if (-not (Test-Path -Path "./outputs/features")) {
    New-Item -ItemType Directory -Path "./outputs/features" -Force | Out-Null
    Write-Host "已创建目录: outputs/features"
}

if (-not (Test-Path -Path "./outputs/runs")) {
    New-Item -ItemType Directory -Path "./outputs/runs" -Force | Out-Null
    Write-Host "已创建目录: outputs/runs"
}

if (-not (Test-Path -Path "./outputs/results")) {
    New-Item -ItemType Directory -Path "./outputs/results" -Force | Out-Null
    Write-Host "已创建目录: outputs/results"
}

# 步骤1：准备数据
Write-Host -ForegroundColor Green "`n[1/6] 准备数据..."
python src/prepare_data.py --data_root ./data/market1501 --out_index ./outputs/runs/index_small.json
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "数据准备失败!"
    exit 1
}
Write-Host -ForegroundColor Green "[1/6] 数据准备完成"

# 步骤2：生成图像描述
Write-Host -ForegroundColor Green "`n[2/6] 生成图像描述..."
$captionFile = "outputs/captions/captions_20.csv"

# 检查是否已有测试数据，如果没有OpenAI API Key
if (-not (Test-Path -Path "Env:OPENAI_API_KEY")) {
    Write-Host -ForegroundColor Yellow "警告: OPENAI_API_KEY 环境变量未设置! 生成测试数据..."
    
    # 创建测试CSV文件
    $csvContent = "image_id,prompt,caption,timestamp"
    1..20 | ForEach-Object {
        $imageId = "0001_c1s1_001051_00"
        $prompt = "描述图片中的人物外观"
        $caption = "这是一个穿着蓝色上衣和黑色裤子的人物，正在行走。"
        $timestamp = Get-Date -Format "o"
        $csvContent += "`n$imageId,$prompt,$caption,$timestamp"
    }
    
    Set-Content -Path $captionFile -Value $csvContent -Encoding utf8
    Write-Host -ForegroundColor Yellow "已生成测试数据到 $captionFile"
} else {
    # 正常调用gen_caption.py
    python src/gen_caption.py --index ./outputs/runs/index_small.json --out ./outputs/captions --prompt_file ./prompts/base.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Red "图像描述生成失败!"
        exit 1
    }
}
Write-Host -ForegroundColor Green "[2/6] 图像描述生成完成"

# 步骤3：将CSV转换为JSONL格式
Write-Host -ForegroundColor Green "`n[3/6] 将CSV转换为JSONL格式..."

# 使用Python执行转换
python -c "import csv, json, os
csv_file = './outputs/captions/captions_20.csv'
jsonl_file = './outputs/captions/captions.jsonl'

# 删除现有JSONL文件（如果存在）
if os.path.exists(jsonl_file):
    os.remove(jsonl_file)
    print('已删除现有文件: ' + jsonl_file)

print('开始转换: ' + csv_file + ' -> ' + jsonl_file)

# 检查CSV文件是否存在
if not os.path.exists(csv_file):
    print('错误：CSV文件不存在: ' + csv_file)
    exit(1)

line_count = 0
try:
    with open(csv_file, 'r', encoding='utf-8') as f_in, open(jsonl_file, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        
        for row in reader:
            try:
                # 从image_id中提取信息
                image_id = row['image_id']
                # 创建包含所有必要字段的JSON行
                json_line = {
                    'image_id': image_id,
                    'path': f'data/market1501/{image_id}.jpg',
                    'caption': row['caption']
                }
                # 写入JSONL文件
                f_out.write(json.dumps(json_line) + '\n')
                line_count += 1
            except Exception as e:
                print('处理行时出错: ' + str(e))
    
    print('已成功转换 ' + str(line_count) + ' 行数据')
    
    # 验证生成的文件
    if line_count == 0:
        print('警告：未转换任何数据，请检查CSV文件格式')
except Exception as e:
    print('转换过程中出错: ' + str(e))
    exit(1)"

# 检查JSONL文件
if (-not (Test-Path -Path "./outputs/captions/captions.jsonl")) {
    Write-Host -ForegroundColor Red "错误：captions.jsonl 文件未生成！"
    exit 1
}

# 检查JSONL文件是否为空
if ((Get-Item "./outputs/captions/captions.jsonl").Length -eq 0) {
    Write-Host -ForegroundColor Red "错误：转换后的JSONL文件为空"
    exit 1
} else {
    Write-Host "转换后的JSONL文件已生成且不为空"
}

# 验证JSONL文件格式
Write-Host "验证JSONL文件格式..."
python -c "import json
valid = True
invalid_lines = 0
file_path = './outputs/captions/captions.jsonl'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # 检查必要字段
                required_fields = ['image_id', 'path', 'caption']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print('第' + str(i) + '行缺少字段: ' + str(missing_fields))
                    valid = False
                    invalid_lines += 1
            except json.JSONDecodeError:
                print('第' + str(i) + '行JSON格式错误')
                valid = False
                invalid_lines += 1
                if invalid_lines > 3:
                    break
    
    exit(0 if valid else 1)
except Exception as e:
    print('验证过程中出错: ' + str(e))
    exit(1)"

if ($LASTEXITCODE -eq 0) {
    Write-Host "JSONL文件格式验证成功！"
} else {
    Write-Host -ForegroundColor Red "错误：JSONL文件格式验证失败！"
    exit 1
}

Write-Host -ForegroundColor Green "[3/6] CSV到JSONL转换完成"

# 步骤4：解析属性
Write-Host -ForegroundColor Green "`n[4/6] 解析属性..."
if (-not (Test-Path -Path "./outputs/attrs")) {
    New-Item -ItemType Directory -Path "./outputs/attrs" -Force | Out-Null
}
python src/parse_attrs.py --captions ./outputs/captions/captions.jsonl --out_dir ./outputs/attrs
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "属性解析失败!"
    exit 1
}
Write-Host -ForegroundColor Green "[4/6] 属性解析完成"

# 步骤5：编码特征
Write-Host -ForegroundColor Green "`n[5/6] 编码CLIP特征..."
python src/encode_clip.py --captions ./outputs/captions/captions.jsonl --out_feats ./outputs/feats/text.npy
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "特征编码失败!"
    exit 1
}
Write-Host -ForegroundColor Green "[5/6] 特征编码完成"

# 步骤6：检索
Write-Host -ForegroundColor Green "`n[6/6] 图像检索..."
python src/retrieve.py --captions ./outputs/captions/captions.jsonl --out ./outputs/results/retrieval_results.json --topk 5
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "图像检索失败!"
    exit 1
}
Write-Host -ForegroundColor Green "[6/6] 检索完成"

# 评估
Write-Host -ForegroundColor Green "`n评估指标..."
python src/eval_metrics.py --results ./outputs/results/retrieval_results.json
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "指标评估失败!"
    exit 1
}
Write-Host -ForegroundColor Green "指标评估完成"

Write-Host -ForegroundColor Green "`n🎉 所有步骤执行完毕！"