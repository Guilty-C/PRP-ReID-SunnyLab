# tools/analyze_confusions.py
import argparse, numpy as np
from collections import Counter, defaultdict

def rl(p): return [ln.strip() for ln in open(p,encoding='utf-8') if ln.strip()]
def l2n(X,eps=1e-8): return X/(np.linalg.norm(X,axis=1,keepdims=True)+eps)

def main():
    ap = argparse.ArgumentParser("analyze top-1 confusions for single-vector per pid")
    ap.add_argument("--text", required=True)            # 50xD (单向量/avg_by_pid)
    ap.add_argument("--labels-text", required=True)     # 50 行 pid（与 --text 对齐）
    ap.add_argument("--image", required=True)           # NxD
    ap.add_argument("--labels-image", required=True)    # N 行 pid
    args = ap.parse_args()

    T = l2n(np.load(args.text).astype(np.float32))
    Lt= rl(args.labels_text)
    I = l2n(np.load(args.image).astype(np.float32))
    Li= rl(args.labels_image)

    S = T @ I.T
    best = S.argmax(axis=1)
    pred = [Li[j] for j in best]
    conf = Counter()
    per_pid = defaultdict(list)

    for y, p in zip(Lt, pred):
        if p != y:
            conf[(y,p)] += 1
        per_pid[y].append(p)

    print("Top confused pairs (truth -> predicted):")
    for (y,p),c in conf.most_common(15):
        print(f"{y} -> {p}  ({c})")

if __name__ == "__main__":
    main()
