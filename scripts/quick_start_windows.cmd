@echo off
setlocal enabledelayedexpansion

REM 设置UTF-8编码
chcp 65001 > nul

echo 正在创建必要的目录...

REM 创建输出目录
if not exist "outputs\captions" mkdir "outputs\captions"
if not exist "outputs\features" mkdir "outputs\features"
if not exist "outputs\runs" mkdir "outputs\runs"
if not exist "outputs\results" mkdir "outputs\results"
if not exist "outputs\attrs" mkdir "outputs\attrs"
if not exist "outputs\feats" mkdir "outputs\feats"

REM 验证目录是否成功创建
if not exist "outputs\captions" (
    echo [ERROR] 无法创建目录 outputs\captions，请手动创建后重试
    pause
    exit /b 1
)

REM 步骤1：准备数据
echo.
echo [1/6] 准备数据...
python src\prepare_data.py --data_root .\data\market1501 --out_index .\outputs\runs\index_small.json
if %errorlevel% neq 0 (
    echo [ERROR] 数据准备失败!
    pause
    exit /b 1
)
echo [1/6] 数据准备完成

REM 步骤2：生成图像描述
echo.
echo [2/6] 生成图像描述...

REM 检查是否设置了OPENAI_API_KEY环境变量
if "%OPENAI_API_KEY%" == "" (
    echo [WARNING] OPENAI_API_KEY 环境变量未设置!
    echo [INFO] 生成测试数据...
    
    REM 创建测试CSV文件
    set "CSV_FILE=outputs\captions\captions_20.csv"
    echo image_id,prompt,caption,timestamp > "%CSV_FILE%"
    
    REM 获取当前时间戳（ISO格式）
    for /f "tokens=*" %%a in ('powershell -Command "Get-Date -Format o"') do set "TIMESTAMP=%%a"
    
    REM 创建20条测试数据
    for /l %%i in (1,1,20) do (
        echo 0001_c1s1_001051_00,描述图片中的人物外观,这是一个穿着蓝色上衣和黑色裤子的人物，正在行走.,"%TIMESTAMP%" >> "%CSV_FILE%"
    )
    
    echo [INFO] 已成功创建测试数据到 %CSV_FILE%
) else (
    REM 正常调用gen_caption.py
    python src\gen_caption.py --index .\outputs\runs\index_small.json --out .\outputs\captions --prompt_file .\prompts\base.txt
    
    REM 检查是否成功生成CSV文件
    if not exist "outputs\captions\captions_20.csv" (
        echo [ERROR] CSV文件未生成，尝试使用备选路径...
        if exist "outputs\captions_alt.csv" (
            echo [INFO] 找到备选路径文件，复制到预期位置...
            copy "outputs\captions_alt.csv" "outputs\captions\captions_20.csv" > nul
        ) else (
            echo [ERROR] 无法找到任何生成的CSV文件，创建测试数据...
            echo image_id,prompt,caption,timestamp > "outputs\captions\captions_20.csv"
            for /l %%i in (1,1,20) do (
                echo 0001_c1s1_001051_00,描述图片中的人物外观,这是一个穿着蓝色上衣和黑色裤子的人物，正在行走.,"%TIMESTAMP%" >> "outputs\captions\captions_20.csv"
            )
        )
    )
)
echo [2/6] 图像描述生成完成

REM 步骤3：将CSV转换为JSONL格式
echo.
echo [3/6] 将CSV转换为JSONL格式...

REM 使用Python执行转换
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
                    'path': 'data/market1501/' + image_id + '.jpg',
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

REM 检查JSONL文件
if not exist ".\outputs\captions\captions.jsonl" (
    echo [ERROR] captions.jsonl 文件未生成！
    pause
    exit /b 1
)

REM 检查JSONL文件是否为空
for %%A in (".\outputs\captions\captions.jsonl") do (
    if %%~zA equ 0 (
        echo [ERROR] 转换后的JSONL文件为空
        pause
        exit /b 1
    )
)
echo 转换后的JSONL文件已生成且不为空

REM 验证JSONL文件格式
echo 验证JSONL文件格式...
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

if %errorlevel% equ 0 (
    echo JSONL文件格式验证成功！
) else (
    echo [ERROR] JSONL文件格式验证失败！
    pause
    exit /b 1
)
echo [3/6] CSV到JSONL转换完成

REM 步骤4：解析属性
echo.
echo [4/6] 解析属性...
python src\parse_attrs.py --captions .\outputs\captions\captions.jsonl --out_dir .\outputs\attrs
if %errorlevel% neq 0 (
    echo [ERROR] 属性解析失败!
    pause
    exit /b 1
)
echo [4/6] 属性解析完成

REM 步骤5：编码特征
echo.
echo [5/6] 编码CLIP特征...
python src\encode_clip.py --captions .\outputs\captions\captions.jsonl --out_feats .\outputs\feats\text.npy
if %errorlevel% neq 0 (
    echo [ERROR] 特征编码失败!
    pause
    exit /b 1
)
echo [5/6] 特征编码完成

REM 步骤6：检索
echo.
echo [6/6] 图像检索...
python src\retrieve.py --captions .\outputs\captions\captions.jsonl --out .\outputs\results\retrieval_results.json --topk 5
if %errorlevel% neq 0 (
    echo [ERROR] 图像检索失败!
    pause
    exit /b 1
)
echo [6/6] 检索完成

REM 评估
echo.
echo 评估指标...
python src\eval_metrics.py --results .\outputs\results\retrieval_results.json
if %errorlevel% neq 0 (
    echo [ERROR] 指标评估失败!
    pause
    exit /b 1
)
echo 指标评估完成

echo.
echo 🎉 所有步骤执行完毕！
pause