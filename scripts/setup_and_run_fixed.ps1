# Create and activate virtual environment
Write-Host "[1/4] Creating and activating Python virtual environment..." -ForegroundColor Green
$VENV_DIR = ".\venv"
if (-Not (Test-Path -Path $VENV_DIR -PathType Container)) {
    python -m venv $VENV_DIR
    Write-Host "  Virtual environment created: $VENV_DIR" -ForegroundColor Yellow
} else {
    Write-Host "  Virtual environment already exists, skipping creation" -ForegroundColor Yellow
}

# Activate virtual environment on Windows
$ActivateScript = Join-Path -Path $VENV_DIR -ChildPath "Scripts\Activate.ps1"
if (Test-Path -Path $ActivateScript) {
    . $ActivateScript
    Write-Host "  Virtual environment activated" -ForegroundColor Yellow
} else {
    Write-Host "  Warning: Cannot find activation script, you may need to activate manually" -ForegroundColor Red
}

# Install project dependencies
Write-Host "\n[2/4] Installing project dependencies..." -ForegroundColor Green
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "  Dependencies installed successfully" -ForegroundColor Yellow

# Set OpenAI API key
Write-Host "\n[3/4] Setting OpenAI API key..." -ForegroundColor Green
$env:OPENAI_API_KEY = "sk-184bd95b1eb94dc189f12db32ab6cc37"
Write-Host "  API key set successfully" -ForegroundColor Yellow

# Create necessary output directories
Write-Host "\n[4/4] Creating necessary output directories..." -ForegroundColor Green
New-Item -ItemType Directory -Path ".\outputs\captions" -Force | Out-Null
New-Item -ItemType Directory -Path ".\outputs\feats" -Force | Out-Null
New-Item -ItemType Directory -Path ".\outputs\attrs" -Force | Out-Null
New-Item -ItemType Directory -Path ".\outputs\runs" -Force | Out-Null
Write-Host "  Output directories created" -ForegroundColor Yellow

# Provide running command options
Write-Host "\n\nEnvironment setup completed! You can run the following commands to start experiments:" -ForegroundColor Green
Write-Host "\nOption 1: Run complete experiment flow" -ForegroundColor Cyan
Write-Host "bash scripts/quick_start.sh  # Run in Git Bash"
Write-Host "Or run each step manually in PowerShell:" -ForegroundColor White
Write-Host "python src/prepare_data.py --data_root ./data/market1501 --out_index ./outputs/runs/index_small.json"
Write-Host "python src/gen_caption.py --index ./outputs/runs/index_small.json --prompt_file ./prompts/base.txt --out ./outputs/captions/captions.jsonl"
Write-Host "python src/parse_attrs.py --captions ./outputs/captions/captions.jsonl --out_dir ./outputs/attrs"
Write-Host "python src/encode_clip.py --captions ./outputs/captions/captions.jsonl --out_feats ./outputs/feats/text.npy"
Write-Host "python src/retrieve.py --feats ./outputs/feats/text.npy --topk 5"
Write-Host "python src/eval_metrics.py"

Write-Host "\nOption 2: Run evaluation metrics only" -ForegroundColor Cyan
Write-Host "python src/eval_metrics.py"

Write-Host "\nOption 3: Run prompt effectiveness evaluation (need descriptions file first)" -ForegroundColor Cyan
Write-Host "python src/eval_prompt_effectiveness.py --config configs/prompt_evaluation.yaml --descriptions_file ./outputs/captions/captions.jsonl --images_dir ./data/market1501 --output_file ./outputs/eval_results.json"

# Keep virtual environment activated
Write-Host "\nNote: You are currently in virtual environment. After running experiments, you can use 'deactivate' command to exit." -ForegroundColor Yellow