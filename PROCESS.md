# PRP-ReID Project Process

## 1. Title & Overview
PRP-ReID combines prompt-engineered large-language-model captioning with CLIP-based retrieval to tackle person re-identification across Market-1501-style datasets. The shipped pipeline chains data indexing, caption generation, retrieval, and metric computation without fine-tuning. Current status: the end-to-end pipeline (`src/run_pipeline.py`) works with Market-1501-style layouts and CLIP encoders, gallery caching is production-ready, and PWS tooling produces prompt-worth summaries. Known gaps **[TBD]**: training/fine-tuning code is absent, dataset download links/checksums are not documented, and `src/encode_clip.py` is missing imports (`argparse`, `os`, `os.path as op`) which must be restored before use.

## 2. Repo Map (High-level)
- `src/` – Core scripts: pipeline driver, data prep, captioning, retrieval, evaluation, cache management, and utilities.【F:src/run_pipeline.py†L1-L116】【F:src/retrieve.py†L1-L264】
- `reid/` – Inference and PWS routing helpers for online decision logic.【F:reid/inference.py†L1-L58】
- `configs/` – YAML defaults for environment paths, pipeline toggles, and PWS weighting.【F:configs/env.yaml†L1-L53】【F:configs/pipeline.yaml†L1-L23】
- `prompts/` – Prompt templates consumed by caption generators (not shown; update as needed). **[TBD]** Document per-file prompt intents.
- `scripts/` – OS-specific automation (bash/PowerShell/Batch) for quick starts, testing, and demos.【F:scripts/quick_start.sh†L1-L160】
- `tools/` – Prompt Worth Score evaluator and reporting CLI.【F:tools/pws_eval.py†L1-L88】
- `tests/` – Pytest suites covering gallery cache behavior and PWS math.【F:tests/test_gallery_cache.py†L1-L103】
- `docs/` – Additional guides (prompt tuning, batch captioning, assumptions, risks). **[TBD]** Sync with this process doc as features evolve.

## 3. Environment Setup
- **OS**: Linux (Ubuntu 20.04/22.04) or Windows WSL2. Scripts assume POSIX paths; Windows batch/PowerShell equivalents exist under `scripts/`.
- **Python**: 3.10 (tested with 3.10.13). PyTorch ≥2.0 with CUDA 11.8/12.x recommended for GPU acceleration.【F:README.md†L21-L44】

### Conda workflow
```bash
conda create -n prp-reid python=3.10 -y
conda activate prp-reid
pip install -r requirements.txt
# Optional: pip install faiss-gpu-cu12  # match your CUDA if FAISS search is needed
```
`requirements.txt` installs PyTorch, torchvision, OpenAI SDK, pandas, transformers, etc.【F:requirements.txt†L1-L9】 Install `faiss-gpu` only if you plan to persist FAISS indices in `src/retrieve.py`.

### Pure pip (virtualenv)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU / CUDA
`src/retrieve.py` auto-selects CUDA if available (`torch.cuda.is_available()`), otherwise CPU.【F:src/retrieve.py†L120-L129】 Ensure your PyTorch build matches the installed CUDA runtime (11.8 or 12.x). For FAISS, install the matching `faiss-gpu` wheel.

### API keys
Captioning scripts expect `OPENAI_API_KEY` targeting the VimsAI-compatible endpoint (`https://usa.vimsai.com/v1`).【F:src/gen_caption_batch.py†L17-L43】 Export the key before generating captions:
```bash
export OPENAI_API_KEY="sk-..."
```

### Verify the installation
Run the smallest smoke test to ensure dependencies import and filesystem permissions work:
```bash
python src/prepare_data.py --data_root datasets/Market-1501 --split query --out_index outputs/checks/index.json
```
Expected log: `[prepare_data] indexed N images -> outputs/checks/index.json`.【F:src/prepare_data.py†L5-L23】 Delete the generated `outputs/checks/` folder after verification.

## 4. Data Preparation
- **Supported datasets**: Market-1501, DukeMTMC-reID, MSMT17 (plus optional CUHK-PEDES captions). Layout must follow the Market-1501 split naming for the pipeline defaults.【F:README.md†L46-L70】

### Expected directory structure
```
datasets/
  Market-1501/
    bounding_box_train/
    bounding_box_test/
    query/
  DukeMTMC-reID/
    bounding_box_train/
    bounding_box_test/
    query/
  MSMT17/
    train/
    test/
    query/
```

Place this under `datasets/` or set `--data_root` to the dataset root. The pipeline assumes query images live under `<data_root>/query` and gallery images under `<data_root>/bounding_box_test` (Market/Duke) or `<data_root>/test` (adjust manually for MSMT). `src/prepare_data.py` indexes `query` or `bounding_box_test` splits without recursion; keep images flat within those folders.【F:src/prepare_data.py†L5-L19】

### Downloading
Official links/checksums are not tracked—marking **[TBD]**. Use the dataset providers’ official websites, and verify integrity manually (e.g., via SHA256).

### Optional environment variables
You may export `DATA_DIR=/abs/path/to/datasets` and pass `--data_root "$DATA_DIR/Market-1501"` to scripts.【F:README.md†L21-L44】

### Building query indices
```bash
python src/prepare_data.py \
  --data_root datasets/Market-1501 \
  --split query \
  --out_index outputs/indexes/market_query.json
```
Repeat with `--split bounding_box_test` if you need gallery metadata for offline captioning or cache precomputation.

## 5. Training
This repository does **not** include supervised training/fine-tuning. Instead, use the pipeline driver to orchestrate data prep, captioning, retrieval, and evaluation end-to-end.【F:README.md†L72-L85】【F:src/run_pipeline.py†L38-L107】

### End-to-end pipeline command
```bash
python src/run_pipeline.py \
  --data_root datasets/Market-1501 \
  --prompt_file prompts/base.txt \
  --num 500 \
  --out_dir outputs/market_pipeline
```
- `--num`: limit processed queries for smoke tests (`all` by default).【F:src/run_pipeline.py†L47-L55】
- Outputs land in `outputs/market_pipeline/expNNN/` with `index.json`, `captions.csv`, `results.json`, and `pipeline.log` per experiment.【F:src/run_pipeline.py†L34-L107】
- The driver truncates the generated index when `--num` is set and appends command logs plus final metric summary.

### Prompt selection
Choose prompt templates from `prompts/` (e.g., `prompts/base.txt`). Update or add new prompts and reference them via `--prompt_file`. Documenting each template’s intent remains **[TBD]**.

### Caption model throughput
`src/gen_caption_batch.py` parallelizes captioning with a thread pool; tweak `--batch_size` and `--num_workers` based on your API rate limits.【F:src/gen_caption_batch.py†L75-L118】 Ensure `OPENAI_API_KEY` points to a model that accepts your prompts (`qwen-plus` by default).

## 6. Evaluation
Use the CLIP-focused evaluator to compute Rank-1 and mAP from retrieval outputs.
```bash
python src/eval_clip_results.py \
  --results outputs/market_pipeline/exp001/results.json
```
The script parses Market-1501-style filenames to ignore same-camera matches and prints queries processed, average top-1 similarity, Rank-1 (%), and mAP (%).【F:src/eval_clip_results.py†L16-L91】

`src/eval_metrics.py` is a placeholder and should not be used until real metrics replace the stubs (**[TBD]**).【F:src/eval_metrics.py†L1-L15】

## 7. Inference / Retrieval Pipeline
`src/retrieve.py` encodes gallery images (with incremental caching) and queries to compute cosine-similarity matches.
```bash
python src/retrieve.py \
  --captions outputs/market_pipeline/exp001/captions.csv \
  --gallery datasets/Market-1501/bounding_box_test \
  --out outputs/market_pipeline/exp001/retrieval_top20.json \
  --topk 20 \
  --gallery_cache outputs/cache/gallery \
  --batch_size 32 \
  --num_workers 2
```
Key behaviors:
- Accepts captions in CSV or JSONL (`image_id`, `caption`).【F:src/retrieve.py†L18-L37】
- Builds/updates a cache (`embeddings.npy`, `manifest.csv`, `meta.json`) keyed by file ETags or SHA1; reuse with `--gallery_cache`, enforce rebuild via `--rebuild_cache`, and choose hash mode via `--hash {mtime,sha1}`.【F:src/retrieve.py†L104-L205】
- Optional FAISS persistence when `faiss-gpu` is installed (`--save_faiss`, `--faiss_path`).【F:src/retrieve.py†L205-L231】
- Results are saved as `{query_image_id: [(gallery_path, score), ...]}` JSON; logs report query counts and output paths.【F:src/retrieve.py†L233-L260】

For pure caption encoding (without retrieval), fix `src/encode_clip.py` imports (**[TBD]**) and run it to export text features once the script is corrected.

## 8. Experiment Tracking & Reproducibility
- **Seeds**: `configs/env.yaml` declares a default experiment seed (42) but scripts currently do not re-seed PyTorch/NumPy; add seeding where required (**[TBD]**).【F:configs/env.yaml†L12-L34】
- **Run folders**: `src/run_pipeline.py` auto-increments `expNNN` directories under `--out_dir` and logs executed commands plus summary metrics.【F:src/run_pipeline.py†L14-L107】 Adopt a naming convention like `outputs/<dataset>/<prompt>_exp###` to track prompt variants.
- **Config overrides**: edit YAML files in `configs/` or supply CLI arguments (e.g., `--data_root`, `--prompt_file`). Store YAML snapshots with results for reproducibility.
- **PWS evaluations**: Use `tools/pws_eval.py` with JSONL logs from inference to capture deployment decisions; outputs include Markdown reports, JSON summaries, and CSV decision matrices.【F:tools/pws_eval.py†L1-L88】 Keep logs versioned alongside experiment outputs.

## 9. Troubleshooting
- **Dataset not found**: `FileNotFoundError: No images found in gallery` indicates wrong `--gallery` path; ensure layout matches Section 4 and directories contain `.jpg/.png` files.【F:src/retrieve.py†L165-L200】
- **Caption API failures**: `[ERROR]` entries from `gen_caption_batch.py` usually signal missing `OPENAI_API_KEY` or model issues. Validate connectivity and rate limits.【F:src/gen_caption_batch.py†L30-L128】
- **FAISS import errors**: The retrieval script falls back to pure NumPy if `faiss` is missing; install `faiss-gpu-cuXX` to enable indexing or omit `--save_faiss` to silence warnings.【F:src/retrieve.py†L205-L260】
- **Windows multiprocessing**: Use `--num_workers 0` on Windows to avoid DataLoader spawn errors (default behavior).【F:src/retrieve.py†L110-L112】
- **Placeholder modules**: `src/eval_metrics.py` prints zeros and `src/encode_clip.py` lacks imports; update them before relying on their outputs (**[TBD]**).
- **CLI help**: Append `--help` to any script (e.g., `python src/retrieve.py --help`) to inspect supported flags.

## 10. Development Workflow
- **Branching**: Fork or branch off `main`, keep feature branches scoped.
- **Commits/PRs**: Follow conventional commits (e.g., `feat:`, `fix:`, `docs:`). Include experiment context in commit messages when results depend on data revisions.
- **Testing**: Run pytest before submitting PRs:
  ```bash
  pytest tests/test_gallery_cache.py
  pytest tests/test_pws.py
  ```
  `tests/test_gallery_cache.py` validates incremental cache updates; `tests/test_pws.py` covers PWS bootstrapping logic.【F:tests/test_gallery_cache.py†L1-L103】
- **Style**: Respect existing script patterns (argparse + print logging). Avoid wrapping imports in try/except (per repo guidelines).
- **Continuous docs**: Update `PROCESS.md` and README when CLI arguments or configs change.

## 11. Roadmap / Open TODOs
- **[TBD]** Publish dataset download mirrors, SHA256 checksums, and automated verification.
- **[TBD]** Restore missing imports in `src/encode_clip.py` and document its usage once fixed.
- **[TBD]** Implement supervised fine-tuning or adapter training paths (currently inference-only).
- **[TBD]** Extend seeding/config plumbing so seeds propagate into PyTorch/NumPy for deterministic runs.
- **[TBD]** Document prompt template intent and expected token budgets under `prompts/`.
- **[TBD]** Replace `src/eval_metrics.py` placeholder with the same mAP/Rank-1 logic used in `src/eval_clip_results.py`.

## 12. Citation / Acknowledgments
This project builds on CLIP (`openai/clip-vit-base-patch32`) for vision-language embeddings and classic ReID datasets (Market-1501, DukeMTMC-reID, MSMT17). Please cite the respective dataset and CLIP papers when publishing results. Formal citation entries remain **[TBD]** pending project publication notes.
