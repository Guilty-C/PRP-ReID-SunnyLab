import argparse, json, os, os.path as op
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="./outputs/captions/captions.jsonl")
    ap.add_argument("--out_feats", default="./outputs/feats/text.npy")
    args = ap.parse_args()

    # 占位：真实项目请用 CLIP 编码
    caps = []
    with open(args.captions, "r", encoding="utf-8") as f:
        for line in f:
            caps.append(json.loads(line)["caption"])
    # 用随机向量代替
    dim = 512
    feats = np.random.randn(len(caps), dim).astype("float32")
    os.makedirs(op.dirname(args.out_feats), exist_ok=True)
    np.save(args.out_feats, feats)
    print(f"[encode_clip] encoded {len(caps)} -> {args.out_feats} (dim={dim})")

if __name__ == "__main__":
    main()
