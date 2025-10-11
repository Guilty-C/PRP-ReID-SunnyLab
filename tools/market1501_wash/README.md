# Market-1501 Washing Utility

This tool prepares a cleaned Market-1501 training subset that is suitable for
classification-style pre-training. It removes low-quality images, prunes
near-duplicates, and selects a camera-balanced "core" subset for each identity.

## Installation

```bash
cd tools/market1501_wash
python -m venv .venv && . .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
python wash_market1501.py \
  --src /path/to/Market-1501-v15.09.15 \
  --dst ../../data/market1501_clean \
  --core_per_id 6 --blur_th 80 --dup_hamming 6 --min_imgs_per_id 4
```

## Outputs

After the script completes, the destination directory will contain:

```
data/market1501_clean/
  train_full_clean/
    0001/*.jpg
    0002/*.jpg
  train_core/
    0001/*.jpg
    0002/*.jpg
  market1501_index.csv
  wash_stats.txt
```

## Notes

* Only images inside `bounding_box_train/` are read.
* The process prepares data for consistent backbone training; it does not
  compute evaluation metrics such as mAP or Rank-1.
* Thresholds (`--blur_th`, `--dup_hamming`, etc.) are configurable to match the
  quality of your local dataset copy.
