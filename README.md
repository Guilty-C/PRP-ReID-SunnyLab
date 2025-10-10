# PRP-ReID (Prompt-Augmented Person Re-Identification)

![status](https://img.shields.io/badge/status-alpha-orange) ![python](https://img.shields.io/badge/python-3.10-blue) ![pytorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.12%20%7C%202.x-red) ![os](https://img.shields.io/badge/OS-Windows%20%7C%20Linux-lightgrey)

Prompt-augmented person re-identification (ReID) playground that stitches together LLM-generated captions with CLIP-style encoders for cross-modal text→image retrieval on Market-1501. The project is **alpha quality**—pipelines run end-to-end with stubs where models are missing, so expect to fill in TODOs before any serious benchmarking. (No qualitative screenshots are shipped yet; contribute yours!)

---

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Enable Real Encoders (Recommended)](#enable-real-encoders-recommended)
- [Windows vs. Linux](#windows-vs-linux)
- [Architecture Overview](#architecture-overview)
- [Benchmarks](#benchmarks)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Features

- LLM-generated captions → CLIP embeddings → text→image retrieval workflows scripted end-to-end.
- Market-1501 dataset support with JSON index builders and optional k-reciprocal re-ranking hook.
- Batching utilities in `scripts/` for caption generation; mirrored shell (`.sh`) and cmd (`.cmd`) entry points for Linux and Windows.
- ⚠️ **Known limitation:** `src/encode_clip.py` currently emits placeholder/random vectors and `src/eval_metrics.py` prints dummy metrics. Follow [Enable Real Encoders (Recommended)](#enable-real-encoders-recommended) to swap in OpenCLIP + FAISS.

## Quick Start

> Runs the scripted alpha pipeline on Market-1501. Replace dataset paths and API keys as needed. Commands assume repo root.

```bash
mkdir -p outputs/captions outputs/feats outputs/results outputs/runs
python src/prepare_data.py --data_root data/market1501 --split query --out_index outputs/runs/index_small.json
python src/gen_caption.py --index outputs/runs/index_small.json --prompt_file prompts/base.txt --out outputs/captions/captions_20.csv
python -c "import csv,json;rows=list(csv.DictReader(open('outputs/captions/captions_20.csv',encoding='utf-8')));open('outputs/captions/captions.jsonl','w',encoding='utf-8').writelines(json.dumps({'image_id':r['image_id'],'caption':r['caption'],'path':r.get('path','')})+'\\n' for r in rows)"
python src/encode_clip.py --captions outputs/captions/captions.jsonl --out_feats outputs/feats/text.npy  # placeholder features
python src/retrieve.py --captions outputs/captions/captions.jsonl --gallery data/market1501/bounding_box_test --out outputs/results/retrieval_results.json --topk 5
python src/eval_metrics.py --results outputs/results/retrieval_results.json  # prints stub metrics
python src/rerank_optional.py  # optional; prints TODO
bash scripts/quick_start.sh  # Linux orchestrated smoke test
scripts\quick_start_windows.cmd  # Windows orchestrated smoke test
```

For large caption batches run:

```bash
bash scripts/batch_generate_captions.sh
# Windows:
scripts\batch_generate_captions.cmd
```

## Installation

```bash
conda create -n prp-reid python=3.10 -y
conda activate prp-reid
# Choose the right CUDA build:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r requirements.txt
# Optional for scalable retrieval:
# pip install faiss-gpu-cu12
# Optional if enabling OpenCLIP backend:
# pip install open_clip_torch
```

## Datasets

### Market-1501

```
data/market1501/
├── bounding_box_test/
├── bounding_box_train/
├── gt_bbox/               # optional evaluation split
└── query/
```

Build a compact JSON index for any split:

```bash
python src/prepare_data.py --data_root data/market1501 --split query --out_index data/market1501/index_query.json
```

> `prepare_data.py` currently writes `{ "image_id": ..., "path": ..., "split": ... }` entries; adjust downstream scripts if you change field names.

### DukeMTMC-reID *(TODO)*

```
data/dukemtmc/
├── bounding_box_test/
├── bounding_box_train/
└── query/
```

- TODO: Add DukeMTMC index builder (reuse `prepare_data.py` with dataset-specific splits) and evaluation scripts.

### MSMT17 *(TODO)*

```
data/msmt17/
├── test/
├── train/
└── query/
```

- TODO: Confirm official folder names, create metadata conversion, and validate caption prompts for large scale.

> **Dataset licensing:** Download and use datasets according to their original licenses. This repo does not redistribute images.

## Usage

### Prepare index

```bash
python src/prepare_data.py --data_root data/market1501 --split query --out_index data/market1501/index_query.json
```

- Generates a JSON list consumed by captioning/encoding scripts.
- For split-specific indexes pass `--split query` or `--split bounding_box_test`.

### Generate captions

Single batch (first 20 entries) using `src/gen_caption.py`:

```bash
python src/gen_caption.py \
  --index data/market1501/index_small.json \
  --prompt_file prompts/base.txt \
  --out outputs/captions/captions_20.csv --verbose
```

Batch mode via helper scripts:

```bash
bash scripts/batch_generate_captions.sh
# Windows:
scripts\batch_generate_captions.cmd
```

> Prompts live under `prompts/`. `docs/BATCH_GENERATION_GUIDE.md` and `docs/batch_caption_guide.md` contain additional token budgeting tips.

### Encode & Retrieve *(current alpha state)*

- `src/encode_clip.py` saves **placeholder/random** text features to `outputs/feats/text.npy` (see `requirements.txt` note). Replace with OpenCLIP per the [Enable Real Encoders](#enable-real-encoders-recommended) section.
- `src/retrieve.py` loads captions, encodes gallery images on-the-fly with CLIP, and writes top-k lists to JSON. Without real text features the similarity scores are not meaningful.
- `src/eval_metrics.py` prints stubbed Rank-1/mAP values; integrate real metrics before publishing results.

### Evaluate metrics

```bash
python src/eval_metrics.py \
  --results outputs/results/retrieval_results.json \
  --gt_root data/market1501 \
  --metrics mAP,Rank-1
```

> CLI arguments are parsed but metric computation is TODO—expect placeholder output.

### Re-ranking (optional)

```bash
python src/retrieve.py ... --save_dist outputs/results/dist.npy

python src/rerank_optional.py \
  --dist outputs/results/dist.npy \
  --k1 20 --k2 6 --lambda 0.3 \
  --out outputs/results/retrieval_rerank.json
```

> `rerank_optional.py` is currently a stub logging `TODO`. Hook in k-reciprocal re-ranking when distance matrices become available.

## Enable Real Encoders (Recommended)

> **Proposed workflow**—implements real CLIP embeddings using [OpenCLIP](https://github.com/mlfoundations/open_clip) and optional FAISS acceleration. Contributions welcome.

1. Install the extras:
   ```bash
   pip install open_clip_torch
   # Optional for large galleries
   pip install faiss-gpu-cu12
   ```
2. Update `src/encode_clip.py` to:
   - Load `open_clip.create_model_and_transforms` with models like `ViT-B-32` and convert captions into text features (float32, L2-normalized).
   - Persist text/image features separately, e.g. `outputs/features/text.npy` & `outputs/features/img.npy`.
3. Call the **proposed** CLI:
   ```bash
   # Proposed: real features with OpenCLIP
   # pip install open_clip_torch
   python src/encode_clip.py \
     --images_index data/market1501/index.json \
     --captions_csv outputs/captions/captions_20.csv \
     --out_dir outputs/features \
     --model openclip:ViT-B-32 --device cuda --batch 128 --amp

   python src/retrieve.py \
     --text_feats outputs/features/text.npy \
     --image_feats outputs/features/img.npy \
     --topk 50 --l2norm \
     --out_json outputs/results/retrieval.json

   python src/eval_metrics.py \
     --results outputs/results/retrieval.json \
     --gt_root data/market1501 \
     --metrics mAP,Rank-1
   ```
4. Integrate FAISS (optional): build an IVF/HNSW index for gallery embeddings and expose `--faiss` flag inside `retrieve.py`.

## Windows vs. Linux

| Task | Linux / WSL | Windows |
| --- | --- | --- |
| Quick smoke test | `bash scripts/quick_start.sh` | `scripts\quick_start_windows.cmd` |
| Batch captioning | `bash scripts/batch_generate_captions.sh` | `scripts\batch_generate_captions.cmd` |
| Virtual env | `conda activate prp-reid` | Same (Anaconda Prompt) |
| Path separators | `/workspace/...` | `C:\path\to\repo\...` (escape backslashes in JSON) |

> Scripts auto-create `outputs/` folders and generate mock captions when `OPENAI_API_KEY` is missing—useful for offline smoke tests.

## Architecture Overview

- **Captioner** (`gen_caption.py`, `gen_caption_batch.py`): hits Qwen API through VimsAI to turn prompts into natural language.
- **Indexer** (`prepare_data.py`): scans dataset splits and emits JSON metadata.
- **Encoders** (`encode_clip.py`, TODO for OpenCLIP): convert text/image pairs into joint feature space.
- **Retriever** (`retrieve.py`): performs cosine similarity search, ready for FAISS offloading.
- **Metrics** (`eval_metrics.py`, `rerank_optional.py`): placeholders for mAP / CMC and k-reciprocal re-ranking.

```mermaid
flowchart LR
  A[Images + CamID + PID] --> B[prepare_data / index]
  B --> C[LLM captions (prompts/*)]
  C --> D[Text Encoder (CLIP)]
  A --> E[Image Encoder (CLIP)]
  D --> F[Text Feats]
  E --> G[Image Feats]
  F --> H[Similarity + (optional) Re-ranking]
  G --> H
  H --> I[mAP / CMC Rank-1]
```

## Benchmarks

| Dataset | Text encoder | Image encoder | mAP | Rank-1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Market-1501 | TBD (OpenCLIP ViT-B/32) | TBD | Coming soon | Coming soon | Seed=0, batch=128, AMP, rerank k1=20/k2=6/λ=0.3 |

**Repro Plan:** fix `torch.manual_seed(0)`, normalize embeddings, evaluate with official Market-1501 split, optionally enable k-reciprocal re-ranking after distance matrix export.

## Troubleshooting

- **CUDA/PyTorch mismatch** – Ensure the `pip install --index-url ...` command matches your local CUDA (e.g., `cu118`, `cpu`). Use `torch.cuda.is_available()` to verify GPU visibility.
- **Windows path quoting** – Escape backslashes or use raw strings when editing JSON indexes (`C:\\data\\market1501\\...`).
- **Caption API rate limits** – Batch prompts (`gen_caption_batch.py`) and cache CSV/JSONL outputs. Consider throttling with `--sleep` flag (add one if needed).
- **Large file handles** – When captioning tens of thousands of images, raise `ulimit -n 4096` on Linux to avoid `Too many open files` errors.
- **FAISS wheels missing** – CPU-only environments can use `faiss-cpu`. For CUDA 12.1 run `pip install faiss-gpu-cu12` from conda-forge or official wheels.

## Roadmap

1. Replace placeholder encoders with OpenCLIP (text + image) and persist real features.
2. Re-introduce FAISS/HNSW retrieval for scalable galleries.
3. Extend dataset loaders and evaluation for DukeMTMC-reID and MSMT17.
4. Add supervised ReID baselines (e.g., BNNeck + ID/Triplet loss) for comparison.
5. Implement proper mAP/CMC computation and k-reciprocal re-ranking.
6. Set up automated tests (pytest) and CI for lint + smoke runs.
7. Publish an explicit open-source license.

## Contributing

- Fork, create feature branches, and open pull requests referencing related issues.
- Follow Python formatting with `black` and lint with `ruff` (add to your toolchain).
- Document new CLI flags in `README.md` and update Windows/Linux scripts for parity.
- Use descriptive commit messages; include dataset provenance in PR descriptions when relevant.

## License

No license granted yet—please do not redistribute.

## Citation

```
@misc{prp_reid_2024,
  title  = {PRP-ReID: Prompt-Augmented Person Re-Identification},
  author = {SunnyLab Contributors},
  note   = {Work in progress. Please cite original Market-1501 and CLIP/OpenCLIP papers.},
  year   = {2024}
}
```

Related works to cite alongside this project:
- Market-1501 dataset: Zheng et al., ICCV 2015.
- OpenAI CLIP: Radford et al., 2021.
- OpenCLIP: Cherti et al., 2023.
- k-reciprocal re-ranking for ReID: Zhong et al., CVPR 2017.
