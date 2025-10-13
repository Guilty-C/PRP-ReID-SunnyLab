from __future__ import annotations
import json, time
from pathlib import Path

def write_metadata(out_dir: str | Path, **kw):
    p = Path(out_dir) / "metadata.json"
    kw = {**kw, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(kw, f, indent=2, ensure_ascii=False)
    return p
