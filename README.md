# PRP-ReID: CLIP-powered Person Re-ID with Prompt Tuning and PWS Scoring

![python](https://img.shields.io/badge/Python-3.10+-blue) ![pytorch](https://img.shields.io/badge/PyTorch-≥2.0-red) ![cuda](https://img.shields.io/badge/CUDA-11.8%7C12.x-green)

## Table of Contents
- [1. Quickstart (TL;DR)](#1-quickstart-tldr)
- [2. Dataset Layouts](#2-dataset-layouts)
- [3. Training (Image ReID)](#3-training-image-reid)
- [4. Evaluation (Image ReID)](#4-evaluation-image-reid)
- [5. Text→Person Retrieval](#5-textperson-retrieval)
- [6. Offline Caption/Attribute Generation](#6-offline-captionattribute-generation)
- [7. Build/Search Index (FAISS)](#7-buildsearch-index-faiss)
- [8. Video ReID](#8-video-reid)
- [9. Prompt Worth Score (PWS) Grading](#9-prompt-worth-score-pws-grading)
- [10. Online Routing (Optional)](#10-online-routing-optional)
- [11. Configuration Guide](#11-configuration-guide)
- [12. Reproduce Known Results](#12-reproduce-known-results)
- [13. Troubleshooting](#13-troubleshooting)
- [14. License & Citation](#14-license--citation)

## Documentation
- [Project Process](PROCESS.md)

## 1. Quickstart (TL;DR)

```bash
# 1) Create env
conda create -n reid python=3.10 -y
conda activate reid

# 2) Install deps
pip install -r requirements.txt
# Optional developer install:
# pip install -e .
# If FAISS GPU is needed:
# pip install faiss-gpu-cu12  # adjust to CUDA version

# 3) Download datasets (Market-1501 shown)
# Place as:
#  datasets/
#    Market-1501/
#      bounding_box_train/
#      bounding_box_test/
#      query/
```

Tested on Python 3.10.13, PyTorch 2.1.0+cu121, CUDA 12.1 with an RTX 4060. Set `DATA_DIR` (e.g., `export DATA_DIR=$(pwd)/datasets`) or pass `--data_root` flags so scripts can locate datasets.

## 2. Dataset Layouts

All scripts assume Market-1501-style folders and can be adapted via `--data_root` arguments. Recommended structure:

```
datasets/
  Market-1501/
    bounding_box_train/
    bounding_box_test/
    gt_bbox/              # optional
    query/
  DukeMTMC-reID/
    bounding_box_train/
    bounding_box_test/
    query/
  MSMT17/
    train/
    test/
    query/
  CUHK-PEDES/
    images/
    captions.jsonl        # optional paired captions
```

If you keep datasets elsewhere, either symlink them into `datasets/` or update the `--data_root` arguments/config entries accordingly.

## 3. Training (Image ReID)

The repository bundles a pipeline driver rather than a dedicated trainer. It orchestrates data prep, captioning, retrieval, and scoring in one go.

```bash
python src/run_pipeline.py \
  --data_root datasets/Market-1501 \
  --prompt_file prompts/base.txt \
  --num all \
  --out_dir outputs/market_pipeline
```

Outputs land in `outputs/market_pipeline/exp###/` (auto-incremented). Use `--num 500` to run quick smoke tests. Resume by pointing `--out_dir` to the same folder; new runs append `expNNN` subdirectories.

## 4. Evaluation (Image ReID)

`src/eval_clip_results.py` computes Rank-1/mAP from the JSON emitted by `src/retrieve.py` or `run_pipeline.py`.

```bash
python src/eval_clip_results.py \
  --results outputs/market_pipeline/exp001/results.json
```

Metrics (mAP, Rank-1, mean similarity) stream to stdout. `src/eval_metrics.py` is still a placeholder—prefer `eval_clip_results.py` for quantitative checks. Logs remain beside the `results.json` file.

## 5. Text→Person Retrieval

Use CLIP to retrieve gallery images from natural-language prompts.

```bash
python src/retrieve.py \
  --captions outputs/market_pipeline/exp001/captions.csv \
  --gallery datasets/Market-1501/bounding_box_test \
  --out outputs/market_pipeline/exp001/retrieval_top20.json \
  --topk 20
```

The script loads `openai/clip-vit-base-patch32` by default, encodes captions, and retrieves the top-K gallery matches by cosine similarity.

**Persistent gallery cache.** `retrieve.py` automatically caches gallery embeddings under `outputs/cache/gallery/` (configurable via `--gallery_cache`). The first run scans/encodes the entire gallery and writes three files:

- `embeddings.npy`: float32/float16 matrix of gallery features (set with `--dtype`).
- `manifest.csv`: tracks each image path, size/mtime (ETag), optional SHA1 (`--hash sha1`), and index.
- `meta.json`: CLIP model metadata and cache schema (`cache_version=1`).

Subsequent runs reuse the cache and only encode new or modified files. Deleted files are dropped from the manifest automatically. Use `--rebuild_cache` to force a full refresh. Large datasets can be memory-mapped by reusing the saved `.npy` file and switching to `--dtype fp16` if you need to shrink disk/memory usage.

**Optional FAISS index.** Pass `--save_faiss` (and optionally `--faiss_path`) to persist a `faiss.index` alongside the cache. If the gallery is unchanged, `retrieve.py` loads the serialized FAISS index instead of rebuilding. Without FAISS, cosine search runs fully in-memory.

Additional knobs:

```bash
python src/retrieve.py \
  --captions outputs/market_pipeline/exp001/captions.csv \
  --gallery datasets/Market-1501/bounding_box_test \
  --out outputs/market_pipeline/exp001/retrieval_top20.json \
  --gallery_cache outputs/cache/gallery \
  --batch_size 64 \
  --num_workers 0 \
  --hash sha1 \
  --save_faiss
```

Windows users should keep `--num_workers 0` (default). Linux defaults to 2 workers for faster disk throughput.

## 6. Offline Caption/Attribute Generation

Batch prompts and cached captions so that online inference uses zero extra tokens.

```bash
python src/gen_caption_batch.py \
  outputs/market_pipeline/exp001/index.json \
  --prompt_file prompts/base.txt \
  --out outputs/captions/market_batch.csv \
  --batch_size 16 \
  --num_workers 4
```

For one-off runs you can use `src/gen_caption.py` with identical arguments minus batching. Combine with `src/parse_attrs.py` if you need structured attributes before retrieval.

## 7. Build/Search Index (FAISS)

The repo currently exports CLIP features to `.npy` files; swap in FAISS if you need ANN search.

```bash
# Build a JSON index of image metadata
python src/prepare_data.py \
  --data_root datasets/Market-1501 \
  --split query \
  --out_index outputs/indexes/market_query.json

# Encode captions to vectors (for FAISS ingestion)
python src/encode_clip.py \
  --captions outputs/captions/market_batch.csv \
  --out_feats outputs/features/market_text.npy

# (Optional) After exporting features, use your FAISS script to build the index.
# Example placeholder:
# python your_faiss_builder.py --feats outputs/features/market_text.npy --out indexes/market.faiss

# CLIP-based search without FAISS (in-memory cosine similarity)
python src/retrieve.py \
  --captions outputs/captions/market_batch.csv \
  --gallery datasets/Market-1501/bounding_box_test \
  --out outputs/results/retrieval.json \
  --topk 50
```

Feel free to integrate FAISS by loading the saved features and persisting them into a GPU index (`faiss-gpu` works with CUDA 12.x).

## 8. Video ReID

Dedicated video ReID scripts are not shipped yet. Export frame sequences per track and reuse the image pipeline until video-specific encoders are added.

## 9. Prompt Worth Score (PWS) Grading

Evaluate prompt strategies with the built-in PWS tooling.

```bash
python -m tools.pws_eval \
  --logs out/with_prompt.jsonl \
  --baseline-logs out/baseline.jsonl \
  --prompt-type B --sla-ms 300 \
  --weights configs/pws_weights.yaml \
  --costs configs/pws_costs.yaml \
  --complexity 2 \
  --risk-drift 0.1 --risk-bias 0.0 --risk-privacy 0.0 --risk-repro 0.0 \
  --out-dir out/pws_B
```

Artifacts:
- `report.md`: ≤400-word markdown summary with the deployment recommendation.
- `summary.json`: PWS mean/CI, deltas, latency penalty, and per-query costs.
- `decision_matrix.csv`: Deployment matrix for dashboards.

## 10. Online Routing (Optional)

```python
from reid.pws.core import route

if route(top1_score, tau=0.72, pws_mean=0.8, pws_low95=0.3):
    # run prompt-enhanced path
    pass
else:
    # use baseline retrieval
    pass
```

Routing only switches to the prompt arm when the candidate’s `top1_score` is below the threshold and both `PWS_mean` and `PWS_lower95` are positive.

## 11. Configuration Guide

Key YAML files live in `configs/`:

- `env.yaml`: Dataset roots and default output folders (edit to match your filesystem or export `DATA_DIR`).
- `pipeline.yaml`: Feature switches for data prep, captioning, encoding, retrieval, and evaluation.
- `prompt_evaluation.yaml`: Settings for `src/eval_prompt_effectiveness.py` (CLIP/text scoring weights).
- `pws_weights.yaml`: Weight coefficients used by `reid.pws.core.compute_pws`.
- `pws_costs.yaml`: Token/GPU pricing and offline budget inputs for PWS.

Override individual fields with CLI flags (e.g., `--data_root`) or by editing the YAML files before running the pipeline.

## 12. Reproduce Known Results

The shipped CLIP baseline retrieves Market-1501 images directly; Rank-1/mAP depend on prompt quality. A typical run with `openai/clip-vit-base-patch32` over the Market-1501 query split yields ~45–50% Rank-1 and ~35–40% mAP after basic prompt tuning. Metrics will stay near zero if you rely on the placeholder scripts (`src/eval_metrics.py`).

For DukeMTMC-reID and MSMT17, reuse the same commands with adjusted `--data_root` and prompt files. Expect lower scores until you fine-tune prompts or switch to stronger encoders (e.g., ViT-B/16).

## 13. Troubleshooting

- **CUDA/FAISS installs:** Match the wheel (`faiss-gpu-cu12`) to your CUDA toolkit. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Missing datasets:** Ensure folder names exactly match those in the dataset layout or update `--data_root`/`env.yaml`.
- **CLIP downloads stuck:** Pre-download the model via `transformers-cli download openai/clip-vit-base-patch32` or set `HF_HOME` to a writable cache.
- **p95 latency too high:** Lower batch sizes, enable mixed precision, or pre-cache gallery embeddings to avoid recomputing per query.
- **Text search weak:** Confirm captions exist (`outputs/captions/*.csv`), regenerate them with richer prompts, and re-run retrieval.
- **PWS says “no deploy”:** Inspect `decision_matrix.csv`, lower `--complexity`, reduce SLA breaches, or cut per-query token/GPU costs.

## Experiments

- [EXP-2025-10-11 — Prompt Improvements for Text→Person Retrieval](experiments/EXP-2025-10-11-prompt-improvements.md)

## 14. License & Citation

A standalone license file is not included; treat the code as “all rights reserved” unless you obtain explicit permission from the maintainers.

If you reference this project in academic work, cite it as:

```bibtex
@misc{prp_reid,
  title        = {PRP-ReID: Prompt-Augmented Person Re-Identification},
  author       = {SunnyLab},
  year         = {2024},
  howpublished = {GitHub repository},
  url          = {https://github.com/SunnyLab/PRP-ReID}
}
```
