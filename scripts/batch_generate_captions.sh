#!/bin/bash
set -euo pipefail

# 设置UTF-8编码
echo "[INFO] 设置UTF-8编码..."
export LC_ALL=C.UTF-8

echo "[INFO] 批量生成图像描述脚本"
echo "[INFO] 此脚本将批量为图像生成详细描述"

echo ""

# 检查操作系统类型
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "[INFO] 在Windows环境下运行（Git Bash或Cygwin）"
else
    echo "[INFO] 在类Unix环境下运行"
fi

# 检查Python环境
echo -e "\n[INFO] 检查Python环境..."
python --version > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "[ERROR] 未找到Python环境，请确保Python已正确安装并添加到系统PATH中"
    read -p "按Enter键退出..." 
    exit 1
fi

# 创建必要的输出目录
echo -e "\n[INFO] 创建必要的输出目录..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows环境下使用PowerShell创建目录
    powershell -Command "New-Item -ItemType Directory -Path './outputs/captions' -Force; New-Item -ItemType Directory -Path './outputs/runs' -Force"
else
    # 非Windows环境使用标准命令
    mkdir -p ./outputs/captions ./outputs/runs
fi

echo "[INFO] 已创建必要的输出目录"

# 步骤1：准备数据
echo -e "\n[1/4] 准备数据..."
PYTHON_SCRIPT="$(pwd)/src/prepare_data.py"
DATA_ROOT="$(pwd)/data/market1501"
INDEX_FILE="$(pwd)/outputs/runs/index_batch.json"

python "$PYTHON_SCRIPT" --data_root "$DATA_ROOT" --out_index "$INDEX_FILE"
if [[ $? -ne 0 ]]; then
    echo "[ERROR] 数据准备失败，请检查错误信息"
    read -p "按Enter键退出..." 
    exit 1
fi
echo "[1/4] 数据准备完成"

# 步骤2：检查API密钥
echo -e "\n[2/4] 检查API环境..."
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[WARNING] OPENAI_API_KEY 环境变量未设置!"
    echo "[INFO] 请先设置API密钥，使用以下命令之一:"
    echo "[INFO] 1. 临时设置: export OPENAI_API_KEY=your_api_key_here"
    echo "[INFO] 2. 使用API设置助手: scripts/setup_api_and_run.cmd (Windows) 或 scripts/setup_and_run.sh (Linux/Mac)"
    read -p "按Enter键退出..." 
    exit 1
else
    echo "[INFO] OPENAI_API_KEY 环境变量已设置"
fi
echo "[2/4] API环境检查完成"

# 步骤3：询问批处理参数
echo -e "\n[3/4] 设置批处理参数..."
read -p "请输入每批处理的图片数量 [默认: 20]: " BATCH_SIZE
if [[ -z "$BATCH_SIZE" ]]; then
    BATCH_SIZE=20
fi

echo "[INFO] 每批处理 $BATCH_SIZE 张图片"
OUTPUT_DIR="$(pwd)/outputs/captions/batch"

# 创建输出目录
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    powershell -Command "New-Item -ItemType Directory -Path '$OUTPUT_DIR' -Force"
else
    mkdir -p "$OUTPUT_DIR"
fi

echo "[INFO] 输出目录: $OUTPUT_DIR"
echo "[3/4] 批处理参数设置完成"

# 步骤4：执行批量生成
echo -e "\n[4/4] 开始批量生成图像描述..."
echo "[INFO] 调用 gen_caption_batch.py 进行批量处理"

# 获取绝对路径
GEN_CAPTION_SCRIPT="$(pwd)/src/gen_caption_batch.py"
PROMPT_FILE="$(pwd)/prompts/base.txt"

# 执行批量生成，但不因为其错误而中断脚本
python "$GEN_CAPTION_SCRIPT" --index "$INDEX_FILE" --out "$OUTPUT_DIR" --prompt_file "$PROMPT_FILE" --batch_size "$BATCH_SIZE" || echo "[WARNING] gen_caption_batch.py执行遇到问题，请检查上面的错误信息"

# 检查执行结果
echo -e "\n[INFO] 批量生成图像描述过程完成"
if [[ -f "$OUTPUT_DIR/captions_all.csv" ]]; then
    echo "[INFO] 批量生成图像描述成功完成!"
    echo "[INFO] 生成的文件:"
    echo "[INFO] - 总的CSV文件: $OUTPUT_DIR/captions_all.csv"
    echo "[INFO] - 批次CSV文件: $OUTPUT_DIR/captions_part*.csv"
else
    echo "[ERROR] 批量生成图像描述过程中发生错误，未找到生成的文件"
    echo "[INFO] 请检查上面的错误信息"
fi

echo -e "\n[INFO] 批量注释功能执行结束"
read -p "按Enter键退出..." 
exit $?