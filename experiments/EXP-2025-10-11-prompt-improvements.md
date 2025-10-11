# EXP-2025-10-11 — Prompt Improvements for Text→Person Retrieval (Market-1501)

**Date:** 2025-10-11 (Asia/Tokyo)  
**Repo:** `reid-prompt`  
**Goal:** Baseline text→image retrieval with current caption template, then document 3 prompt-improvement methods and next steps.

## Setup

- **Dataset:** Market-1501 (single-query).  
  - Query: `./data/market1501/query` (3368 images)  
  - Gallery: `./data/market1501/bounding_box_test` (~19,732 images)  
- **Environment:** Windows 11, Python 3.11, Git Bash.  
- **Scripts:** `src/prepare_data.py`, `src/gen_caption_batch.py`, `src/retrieve.py`, `src/eval_clip_results.py`  
- **Notes:** `ThreadPoolExecutor` requires `num_workers ≥ 1`; Windows console may crash on emojis; avoid them.

## Commands (reproducible)

```bash
# 0) Index queries
/d/Python311/python.exe src/prepare_data.py \
  --data_root "./data/market1501" \
  --split query \
  --out_index "outputs/market_pipeline/exp002/index.json"

# 1) Generate captions (single worker to avoid Windows issues)
#   Template: prompts/base.txt  (single-sentence attribute focus)
/d/Python311/python.exe src/gen_caption_batch.py \
  "outputs/market_pipeline/exp002/index.json" \
  --prompt_file "prompts/base.txt" \
  --out "outputs/market_pipeline/exp002/captions.csv" \
  --batch_size 8 --num_workers 1

# 2) Retrieve (current script has no --device flag; runs on CPU)
/d/Python311/python.exe src/retrieve.py \
  --captions "outputs/market_pipeline/exp002/captions.csv" \
  --gallery  "./data/market1501/bounding_box_test" \
  --out      "outputs/market_pipeline/exp002/results.json" \
  --topk 5   # NOTE: small topk; underestimates mAP

# 3) Evaluate
/d/Python311/python.exe src/eval_clip_results.py \
  --results "outputs/market_pipeline/exp002/results.json"
```

## Observed Results (with `topk=5`)

* **Queries:** 3368
* **Average Top-1 Similarity:** 0.3183
* **Rank-1 Accuracy:** **0.15%**
* **mAP:** **0.34%**
* **Gallery encoding time:** ~22m15s on CPU for ~19,732 images (progress ~14.8 it/s).

> Important: With `topk=5`, correct matches beyond rank 5 are not logged, which depresses mAP/R1. For fair evaluation, set `--topk` ≥ gallery size (e.g., `--topk 20000`) and re-evaluate.

## Issues & Fixes

* **Emoji crash (GBK):** Replace any “✅” prints with “[OK]”.
* **`num_workers=0` invalid:** Use `--num_workers 1` for `gen_caption_batch.py`.
* **No GPU flag in retrieve:** `--device` is unsupported; runs CPU-only unless code is extended.

## Three Improvement Methods (and prompt templates)

### 1) Structured, Single-Line Attribute Captioning

**Why:** Machine-checkable, CSV-safe, reduces drift.

```
Role: Vision-language captioner.
Task: Describe ONLY visible attributes of the person in a single-line JSON. No speculation.
Output exactly one line, no extra text.

Schema:
{"gender":"","top":{"color":"","type":""},"bottom":{"color":"","type":""},"shoes":{"color":"","type":""},"accessories":[]}

Rules:
- Lowercase; leave "" if unknown/occluded.
- No race/age/identity.
- Colors: black, white, gray, blue, red, green, yellow, brown, pink, purple, orange.
- Types: t-shirt, shirt, hoodie, jacket, coat, dress, jeans, trousers, shorts, skirt, sneakers, shoes, boots, sandals, backpack, hat, glasses, mask.

Return ONLY the JSON for IMAGE={{IMAGE_PATH}}:
```

### 2) Query Canonicalizer + Compact Rewrites

**Why:** Stabilizes user queries and provides 2–3 dense prompts for CLIP encoding.

```
Role: Query canonicalizer.
Input: a free-text user query about a person. Output one-line JSON with canonical attributes AND 3 short paraphrases.
No extra text.

Schema:
{"attributes":{"gender":"","top":{"color":"","type":""},"bottom":{"color":"","type":""},"shoes":{"color":"","type":""},"accessories":[]},
 "prompts":["","",""]}

Rules:
- Map colors/types to the same sets as captioning.
- Keep "" if unspecified.
- Each prompt ≤ 12 words, attribute-first, no names/brands.

User query: {{QUERY_TEXT}}
Return ONLY the JSON:
```

### 3) Hard-Case Rerank Judge (Selective Second Stage)

**Why:** Spend tokens only when baseline is uncertain; reorders top-K by attribute match.

```
Role: Attribute-based reranker.
Task: Given a canonical query JSON and top-K candidates {id, caption, sim}, return a JSON ranking.
No explanations, no extra text.

Input:
query = {{QUERY_JSON_SINGLE_LINE}}
candidates = [
  {"id":"{{ID1}}","caption":"{{CAP1}}","sim":{{SIM1}}},
  {"id":"{{ID2}}","caption":"{{CAP2}}","sim":{{SIM2}}},
  ...
]

Rules:
- Prefer exact attribute matches (color→type→accessories→gender).
- Break ties by higher sim.
- Do NOT add attributes not present in inputs.

Output one line JSON:
{"ranking":["id_best","id_2","id_3",...],"notes":""}
```

## Planned Rerun (fair evaluation)

Re-run retrieval with a large `topk`:

```bash
/d/Python311/python.exe src/retrieve.py \
  --captions "outputs/market_pipeline/exp002/captions.csv" \
  --gallery  "./data/market1501/bounding_box_test" \
  --out      "outputs/market_pipeline/exp002/results_topk20000.json" \
  --topk 20000
/d/Python311/python.exe src/eval_clip_results.py \
  --results "outputs/market_pipeline/exp002/results_topk20000.json"
```

## Artifacts

* `outputs/market_pipeline/exp002/index.json`
* `outputs/market_pipeline/exp002/captions.csv`
* `outputs/market_pipeline/exp002/results.json` (topk=5)
* `experiments/EXP-2025-10-11-prompt-improvements.md` (this file)

## Checklist

* [x] Captions generated (`num_workers=1`).
* [x] Retrieval completed (CPU).
* [x] Evaluation saved.
* [ ] Rerun with `topk ≥ 20000`.
* [ ] Optional: add visual rerank on top-K.
* [ ] Optional: add device flag to `retrieve.py` and/or embedding cache.

---
