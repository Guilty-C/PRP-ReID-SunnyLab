# 快速启动脚本（PowerShell版本）

# 设置OpenAI API Key环境变量
$env:OPENAI_API_KEY = "sk-184bd95b1eb94dc189f12db32ab6cc37"
Write-Host "[0/6] OpenAI API Key已设置" -ForegroundColor Green

Write-Host "\n[1/6] 准备数据..." -ForegroundColor Green
python src/prepare_data.py --data_root ./data/market1501 --out_index ./outputs/runs/index_small.json

Write-Host "\n[2/6] 生成图片描述..." -ForegroundColor Green
python src/gen_caption.py --index ./outputs/runs/index_small.json --prompt_file ./prompts/base.txt --out ./outputs/captions/captions.jsonl

Write-Host "\n[3/6] 解析属性..." -ForegroundColor Green
python src/parse_attrs.py --captions ./outputs/captions/captions.jsonl --out_dir ./outputs/attrs

Write-Host "\n[4/6] 编码CLIP特征（当前使用随机向量）..." -ForegroundColor Green
python src/encode_clip.py --captions ./outputs/captions/captions.jsonl --out_feats ./outputs/feats/text.npy

Write-Host "\n[5/6] 执行检索..." -ForegroundColor Green
python src/retrieve.py --captions ./outputs/captions/captions.jsonl --out ./outputs/runs/retrieval_results.json --topk 5

Write-Host "\n[6/6] 评估检索性能..." -ForegroundColor Green
python src/eval_metrics.py --results ./outputs/runs/retrieval_results.json

Write-Host "\n所有步骤执行完毕！" -ForegroundColor Green