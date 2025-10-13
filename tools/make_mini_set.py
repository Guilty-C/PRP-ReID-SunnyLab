import argparse, shutil
from pathlib import Path
import random, re

PAT = re.compile(r'^(-?\d{1,4})_c(\d+)s(\d+)_\d+_\d+\.jpg$', re.I)

def pid_from_name(name: str):
    m = PAT.match(name)
    return None if not m else int(m.group(1))

def collect_by_pid(root: Path):
    d = {}
    for sub in ["bounding_box_test", "query"]:
        for p in (root/sub).glob("*.jpg"):
            pid = pid_from_name(p.name)
            if pid is None or pid < 0:  # 跳过 junk (-1)
                continue
            d.setdefault(pid, {"test": [], "query": []})
            d[pid]["test" if sub=="bounding_box_test" else "query"].append(p)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原 Market-1501 根目录（含 bounding_box_test / query）")
    ap.add_argument("--dst", required=True, help="输出 mini 根目录")
    ap.add_argument("--pids", nargs="*", type=int, help="手动指明的 PID 列表，如 1 2 3 4")
    ap.add_argument("--n", type=int, default=10, help="若未指定 pids，则随机抽取 n 个 PID")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst/"bounding_box_test").mkdir(exist_ok=True)
    (dst/"query").mkdir(exist_ok=True)

    by_pid = collect_by_pid(src)
    all_pids = sorted([p for p in by_pid.keys() if p > 0])

    if args.pids:
        sel = [p for p in args.pids if p in by_pid]
    else:
        random.seed(args.seed)
        sel = sorted(random.sample(all_pids, min(args.n, len(all_pids))))

    print(f"[mini] 用 PID: {sel} (共 {len(sel)} 个)")
    cnt_test = cnt_query = 0
    for pid in sel:
        for p in by_pid[pid]["test"]:
            shutil.copy2(p, dst/"bounding_box_test"/p.name)
            cnt_test += 1
        for p in by_pid[pid]["query"]:
            shutil.copy2(p, dst/"query"/p.name)
            cnt_query += 1

    print(f"[mini] 拷贝完成: test={cnt_test}, query={cnt_query} → {dst}")

if __name__ == "__main__":
    main()
