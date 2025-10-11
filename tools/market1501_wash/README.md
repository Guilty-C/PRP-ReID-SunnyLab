# Market-1501 Washer (Cross-Platform)

This tool cleans the Market-1501 training split by filtering tiny/blurred images and removing near-duplicates.  
**Tested on Windows (CMD/PowerShell/Git Bash), macOS, and Linux.**

## Prerequisites
- Python ≥ 3.9
- A local copy of **Market-1501**. Your `--src` must be the folder that **directly contains** `bounding_box_train/`.
  - Typical layouts accepted:
    ```
    <SRC>/bounding_box_train/
    <SRC>/Market-1501/bounding_box_train/
    <SRC>/Market-1501-v15.09.15/bounding_box_train/
    ```

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
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run (Examples)

### Windows (PowerShell/CMD)

```powershell
python wash_market1501.py `
  --src "D:\PRP SunnyLab\reid-prompt\data\market1501" `
  --dst "D:\PRP SunnyLab\reid-prompt\data\market1501_clean" `
  --core_per_id 6 --blur_th 80 --dup_hamming 6 --min_imgs_per_id 4
```

### Windows (Git Bash)

```bash
python wash_market1501.py \
  --src "D:\\PRP SunnyLab\\reid-prompt\\data\\market1501" \
  --dst "D:\\PRP SunnyLab\\reid-prompt\\data\\market1501_clean" \
  --core_per_id 6 --blur_th 80 --dup_hamming 6 --min_imgs_per_id 4
```

### macOS/Linux

```bash
python wash_market1501.py \
  --src ~/datasets/Market-1501-v15.09.15 \
  --dst ./_market1501_clean \
  --core_per_id 6 --blur_th 80 --dup_hamming 6 --min_imgs_per_id 4
```

### Dry-Run (validate + index only)

```bash
python wash_market1501.py --src "<SRC>" --dst "<DST>" --dry_run
```

## Export manifest

After washing, create a manifest and deterministic train/val split from the cleaned folder:

### Windows (PowerShell)

```powershell
python export_manifest.py `
  --relative --posix_paths `
  "D:\PRP SunnyLab\reid-prompt\data\market1501_clean"
```

### Windows (Git Bash)

```bash
python export_manifest.py \
  --relative --posix_paths \
  "D:\\PRP SunnyLab\\reid-prompt\\data\\market1501_clean"
```

## Expected Layout

The tool will look for:

```
<SRC>/bounding_box_train/
```

It will create:

```
<DST>/
  wash_stats.txt
  ... (cleaned images by ID)
```

## Troubleshooting

* **Error: 'Could not locate `bounding_box_train`'**
  Ensure `--src` is the folder that directly contains `bounding_box_train`.
  Accepted wrappers include `Market-1501/` and `Market-1501-v15.09.15/`.

* **WARNING skip reason=invalid_filename**
  The tool now parses standard Market-1501 names like `1500_c6s3_086542_02.jpg`.
  If you see this warning, check for unrelated files or rename them to the Market-1501 pattern.

* **Paths with spaces (Windows)**
  Always quote your paths, e.g. `"D:\PRP SunnyLab\reid-prompt\data\market1501"`.

## Notes

* Extensions `.jpg/.jpeg/.png` are supported (case-insensitive).
* PID `-1` files are treated as junk and omitted from the cleaned splits but are still indexed for reporting.
