# Quick start script (PowerShell version)

# Set OpenAI API Key environment variable
$env:OPENAI_API_KEY = "sk-184bd95b1eb94dc189f12db32ab6cc37"
Write-Host "[0/6] OpenAI API Key set" -ForegroundColor Green

Write-Host "\n[1/6] Preparing data..." -ForegroundColor Green
python src/prepare_data.py --data_root ./data/market1501 --out_index ./outputs/runs/index_small.json

Write-Host "\n[2/6] Creating test JSONL file (bypassing API calls)..." -ForegroundColor Green
python scripts/create_test_jsonl.py

Write-Host "\n[3/6] Parsing attributes..." -ForegroundColor Green
python src/parse_attrs.py --captions ./outputs/captions/captions.jsonl --out_dir ./outputs/attrs

Write-Host "\n[4/6] Encoding CLIP features (using random vectors)..." -ForegroundColor Green
python src/encode_clip.py --captions ./outputs/captions/captions.jsonl --out_feats ./outputs/feats/text.npy

Write-Host "\n[5/6] Performing retrieval..." -ForegroundColor Green
python src/retrieve.py --captions ./outputs/captions/captions.jsonl --out ./outputs/runs/retrieval_results.json --topk 5

Write-Host "\n[6/6] Evaluating retrieval performance..." -ForegroundColor Green
python src/eval_metrics.py --results ./outputs/runs/retrieval_results.json

Write-Host "\nAll steps completed!" -ForegroundColor Green