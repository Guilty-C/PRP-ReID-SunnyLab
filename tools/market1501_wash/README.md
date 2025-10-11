# Market-1501 Washer (cross-platform)

Utility script for cleaning the Market-1501 training split. It indexes the
dataset, filters out blurry or tiny images, removes near-duplicates, and builds
both a "full clean" training set and a balanced core subset per identity.

## Setup

```bash
cd tools/market1501_wash
python -m venv .venv
# Activate:
# macOS/Linux (bash/zsh)
. .venv/bin/activate
# Windows (Git Bash)
. .venv/Scripts/activate
# Windows (CMD)
.venv\Scripts\activate.bat
# Windows (PowerShell)
./.venv/Scripts/Activate.ps1

pip install -r requirements.txt
```

## Run

```bash
# Example (Windows paths with spaces -> quote them)
python wash_market1501.py \
  --src "D:\\datasets\\Market-1501-v15.09.15" \
  --dst "D:\\PRP SunnyLab\\reid-prompt\\data\\market1501_clean" \
  --core_per_id 6 --blur_th 80 --dup_hamming 6 --min_imgs_per_id 4
```

Tips:

* `--src` should point to the folder that contains `bounding_box_train`. The
  washer also recognises common wrapper layouts such as
  `Market-1501/bounding_box_train` and `Market-1501-v15.09.15/bounding_box_train`.
* Paths are normalised, so both Windows (`C:\\...`) and POSIX (`/home/...`)
  layouts are supported. Always wrap paths that include spaces in quotes.
* Use `--dry-run` to validate the dataset and print counts without copying
  files.

## Outputs

After a successful run, the destination directory contains:

```
market1501_clean/
  train_full_clean/
    0001/*.jpg
    0002/*.jpg
  train_core/
    0001/*.jpg
    0002/*.jpg
  market1501_index.csv
  wash_stats.txt
```

## Common issues

* **"'bounding_box_train' not found"** – Ensure `--src` is the dataset root and
  refer back to the layout tips above.
* **Virtual environment activation on Windows** – Use the script that matches
  your shell (Git Bash/CMD/PowerShell) as shown in the setup instructions.

