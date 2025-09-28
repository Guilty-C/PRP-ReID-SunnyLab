#!/bin/bash

# 设置错误处理
set -euo pipefail

# 创建并激活虚拟环境
echo "[1/4] 创建并激活Python虚拟环境..."
VENV_DIR="./venv"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "  虚拟环境已创建: $VENV_DIR"
else
    echo "  虚拟环境已存在，跳过创建"
fisource "$VENV_DIR/bin/activate"
echo "  虚拟环境已激活"

# 安装项目依赖
echo "\n[2/4] 安装项目依赖..."
pip install --upgrade pip
pip install -r requirements.txt
echo "  依赖安装完成"

# 设置OpenAI API密钥
echo "\n[3/4] 设置OpenAI API密钥..."
export OPENAI_API_KEY="sk-184bd95b1eb94dc189f12db32ab6cc37"
echo "  API密钥已设置"

# 创建必要的输出目录
echo "\n[4/4] 创建必要的输出目录..."
mkdir -p ./outputs/captions ./outputs/feats ./outputs/attrs ./outputs/runs
echo "  输出目录已创建"

# 提供运行命令选项
echo "\n\n环境设置完成！您可以运行以下命令开始实验："
echo "\n选项 1: 运行完整实验流程"
echo "bash scripts/quick_start.sh"
echo "\n选项 2: 仅运行评估指标"
echo "bash scripts/eval_small.sh"
echo "\n选项 3: 运行提示词效果评估（需先准备描述文件）"
echo "python src/eval_prompt_effectiveness.py --config configs/prompt_evaluation.yaml --descriptions_file ./outputs/captions/captions.jsonl --images_dir ./data/market1501 --output_file ./outputs/eval_results.json"

# 保持虚拟环境激活
echo "\n提示: 您当前已在虚拟环境中，运行完实验后可以使用 'deactivate' 命令退出虚拟环境。"