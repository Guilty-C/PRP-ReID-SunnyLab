import json, argparse, os, os.path as op

def simple_parse(caption:str):
    # 占位：真实项目可接入结构化解析/正则/小模型
    return {"upper":{"type":"top","color":"unknown"},
            "lower":{"type":"pants","color":"unknown"},
            "shoes":{"type":"shoes","color":"unknown"},
            "bag":{"has":False,"type":"","color":""},
            "accessories":[], "special_cues":[]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="./outputs/captions/captions.jsonl")
    ap.add_argument("--out_dir", default="./outputs/attrs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    attrs = {}
    with open(args.captions, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            attrs[obj["path"]] = simple_parse(obj["caption"])
    out = op.join(args.out_dir, "attrs.json")
    with open(out, "w", encoding="utf-8") as w:
        json.dump(attrs, w, ensure_ascii=False, indent=2)
    print(f"[parse_attrs] parsed {len(attrs)} -> {out}")

if __name__ == "__main__":
    main()
