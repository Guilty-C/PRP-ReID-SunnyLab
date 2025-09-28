# 创建并激活虚拟环境
Write-Host "[1/4] 创建并激活Python虚拟环境..." -ForegroundColor Green
$VENV_DIR = ".\venv"
if (-Not (Test-Path -Path $VENV_DIR -PathType Container)) {
    python -m venv $VENV_DIR
    Write-Host "  虚拟环境已创建: $VENV_DIR" -ForegroundColor Yellow
} else {
    Write-Host "  虚拟环境已存在，跳过创建" -ForegroundColor Yellow
}

# 在Windows上激活虚拟环境的方式
$ActivateScript = Join-Path -Path $VENV_DIR -ChildPath "Scripts\Activate.ps1"
if (Test-Path -Path $ActivateScript) {
    . $ActivateScript
    Write-Host "  虚拟环境已激活" -ForegroundColor Yellow
} else {
    Write-Host "  警告：无法找到激活脚本，可能需要手动激活虚拟环境" -ForegroundColor Red
}

# 安装项目依赖
Write-Host "\n[2/4] 安装项目依赖..." -ForegroundColor Green
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "  依赖安装完成" -ForegroundColor Yellow

# 设置OpenAI API密钥
Write-Host "\n[3/4] 设置OpenAI API密钥..." -ForegroundColor Green
$env:OPENAI_API_KEY = "sk-184bd95b1eb94dc189f12db32ab6cc37"
Write-Host "  API密钥已设置" -ForegroundColor Yellow

# 创建必要的输出目录
Write-Host "\n[4/4] 创建必要的输出目录..." -ForegroundColor Green
New-Item -ItemType Directory -Path ".\outputs\captions" -Force | Out-Null
New-Item -ItemType Directory -Path ".\outputs\feats" -Force | Out-Null
New-Item -ItemType Directory -Path ".\outputs\attrs" -Force | Out-Null
New-Item -ItemType Directory -Path ".\outputs\runs" -Force | Out-Null
Write-Host "  输出目录已创建" -ForegroundColor Yellow

# 提供运行命令选项
Write-Host "\n\n环境设置完成！您可以运行以下命令开始实验：" -ForegroundColor Green
Write-Host "\n选项 1: 运行完整实验流程" -ForegroundColor Cyan
Write-Host "bash scripts/quick_start.sh  # 在Git Bash中运行"
Write-Host "或手动在PowerShell中运行每一步："
Write-Host "python src/prepare_data.py --data_root ./data/market1501 --out_index ./outputs/runs/index_small.json"
Write-Host "python src/gen_caption.py --index ./outputs/runs/index_small.json --prompt_file ./prompts/base.txt --out ./outputs/captions/captions.jsonl"
Write-Host "python src/parse_attrs.py --captions ./outputs/captions/captions.jsonl --out_dir ./outputs/attrs"
Write-Host "python src/encode_clip.py --captions ./outputs/captions/captions.jsonl --out_feats ./outputs/feats/text.npy"
Write-Host "python src/retrieve.py --feats ./outputs/feats/text.npy --topk 5"
Write-Host "python src/eval_metrics.py"

Write-Host "\n选项 2: 仅运行评估指标" -ForegroundColor Cyan
Write-Host "python src/eval_metrics.py"

Write-Host "\n选项 3: 运行提示词效果评估（需先准备描述文件）" -ForegroundColor Cyan
Write-Host "python src/eval_prompt_effectiveness.py --config configs/prompt_evaluation.yaml --descriptions_file ./outputs/captions/captions.jsonl --images_dir ./data/market1501 --output_file ./outputs/eval_results.json"

# 保持虚拟环境激活的提示
Write-Host "\n提示: 您当前已在虚拟环境中，运行完实验后可以使用 'deactivate' 命令退出虚拟环境。" -ForegroundColor Yellow