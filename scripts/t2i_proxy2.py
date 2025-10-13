
#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0: return x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def read_jsonl_pids(jsonl: Path) -> List[int]:
    pids=[]
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o=json.loads(line)
            pid=o.get("id") or o.get("pid") or o.get("PID")
            if isinstance(pid,str) and pid.startswith("pid-"): pid=int(pid.split("-")[1])
            else: pid=int(pid)
            pids.append(pid)
    return pids

def read_mapping(csv_path: Path) -> Tuple[List[str], List[int], List[int]]:
    rels,pids,cams=[],[],[]
    with open(csv_path,"r",encoding="utf-8") as f:
        rd=csv.DictReader(f)
        for r in rd:
            rels.append(r["relpath"]); pids.append(int(r["pid"]))
            cam=r.get("cam") or r.get("camera") or "-1"
            try: cams.append(int(cam))
            except: cams.append(-1)
    return rels,pids,cams

def group_mean_by_pid(X: np.ndarray, pids: List[int]) -> Dict[int,np.ndarray]:
    buf: Dict[int,List[np.ndarray]]={}
    for v,p in zip(X,pids): buf.setdefault(p,[]).append(v)
    out={}
    for p,vs in buf.items(): out[p]=l2n(np.stack(vs,0).mean(0,keepdims=True))[0]
    return out

def ridge_fit(T: np.ndarray, I: np.ndarray, reg: float=1e-2) -> np.ndarray:
    Dt,Di=T.shape[1],I.shape[1]
    A=T.T@T + reg*np.eye(Dt, dtype=np.float64)
    B=T.T@I
    W=np.linalg.solve(A,B)
    return W.astype("float32")

def oproc_fit(T: np.ndarray, I: np.ndarray) -> np.ndarray:
    # Solve: min ||T W - I|| s.t. W^T W = I  -> SVD(T^T I) = U Σ V^T, W = U V^T
    M = T.T @ I
    U,_,Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    return W.astype("float32")

def eval_pid(sim: np.ndarray, q_pids: List[int], g_pids: List[int]) -> Tuple[np.ndarray,float,int,float]:
    Q,G=sim.shape
    ranks=np.argsort(-sim,axis=1)
    cmc=np.zeros(G,dtype=np.float64); aps=[]; valid=0; best=[]
    for qi in range(Q):
        pid=q_pids[qi]; order=ranks[qi]
        pos=[gi for gi in order if g_pids[gi]==pid]
        if not pos: continue
        valid+=1
        first = next((rk for rk,gi in enumerate(order) if g_pids[gi]==pid), None)
        if first is not None: cmc[first:]+=1
        hits=0; prec=0.0; P=sum(1 for x in g_pids if x==pid)
        for rk,gi in enumerate(order, start=1):
            if g_pids[gi]==pid:
                hits+=1; prec+=hits/rk
                if hits==P: break
        aps.append(prec/max(1,P))
        best.append(sim[qi, order[0]])
    if valid==0: return np.zeros(G),0.0,0,0.0
    cmc/=valid
    return cmc, float(np.mean(aps)), valid, float(np.mean(best))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--text-emb", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--img-query", required=True)
    ap.add_argument("--img-gallery", required=True)
    ap.add_argument("--q-mapping", required=True)
    ap.add_argument("--g-mapping", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", choices=["ridge","oprocrustes"], default="oprocrustes")
    ap.add_argument("--train", choices=["pid-mean","expand"], default="expand")
    ap.add_argument("--reg", type=float, default=1e-2)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--center-t", action="store_true")
    ap.add_argument("--center-i", action="store_true")
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Xt=np.load(args.text_emb).astype("float32"); Xt=l2n(Xt)
    t_pids=read_jsonl_pids(Path(args.prompts)); assert Xt.shape[0]==len(t_pids)
    pid2t=group_mean_by_pid(Xt,t_pids)

    Xq=np.load(args.img_query).astype("float32"); Xg=np.load(args.img_gallery).astype("float32")
    # 居中需在 l2 归一化之前
    if args.center:
        try:
            Xg = Xg - imean
        except NameError:
            pass
    Xq=l2n(Xq); Xg=l2n(Xg)
    # 如果训练端做了居中，评测端的 gallery 也减去同一个 imean
    try:
        args
        if args.center:
            Xg = l2n(Xg - imean)
    except Exception:
        pass
    _,q_pids,_=read_mapping(Path(args.q_mapping))
    _,g_pids,_=read_mapping(Path(args.g_mapping))

    # --- Build training pairs ---
    if args.train=="pid-mean":
        pid2q=group_mean_by_pid(Xq,q_pids)
        common=sorted(set(pid2t)&set(pid2q))
        T=np.stack([pid2t[p] for p in common],0); I=np.stack([pid2q[p] for p in common],0)
        q_eval_pids=common; Z_src=np.stack([pid2t[p] for p in q_eval_pids],0)
    else: # expand: 每张 query 图像作为一个样本
        T_list=[]; I_list=[]; q_eval_pids=[]
        for v,p in zip(Xq,q_pids):
            if p in pid2t:
                T_list.append(pid2t[p]); I_list.append(v); q_eval_pids.append(p)
        T=np.stack(T_list,0); I=np.stack(I_list,0)
        Z_src=np.stack([pid2t[p] for p in q_eval_pids],0)

    # --- Fit mapping ---
    # 可选：分别对 T（文本）/ I（图像）做均值居中
    do_ct = (args.center_t or args.center)
    do_ci = (args.center_i or args.center)
    if do_ct:
        tmean = T.mean(0, keepdims=True)
        T = T - tmean
    else:
        tmean = 0
    if do_ci:
        imean = I.mean(0, keepdims=True)
        I = I - imean
    else:
        imean = 0
    if args.map=="ridge":
        W=ridge_fit(T.astype("float64"), I.astype("float64"), reg=float(args.reg))
    else:
        W=oproc_fit(T.astype("float64"), I.astype("float64"))

    Z=l2n((Z_src - (tmean if isinstance(tmean,int) else tmean)) @ W)
    sim=Z @ Xg.T
    cmc,mAP,validQ,ATS=eval_pid(sim, q_eval_pids, g_pids)
    r1=float(cmc[0]) if cmc.size else 0.0

    res={"map":args.map,"train":args.train,"reg":float(args.reg),
         "metrics":{"Rank-1":round(r1,4),"mAP":round(mAP,4),"ATS":round(ATS,4),"validQ":int(validQ)},
         "dims":{"text":int(Xt.shape[1]),"image":int(Xg.shape[1])}}
    with open(out/"results.json","w",encoding="utf-8") as f: json.dump(res,f,indent=2,ensure_ascii=False)
    L=min(50,len(cmc))
    if L>0:
        plt.figure(); plt.plot(range(1,L+1), cmc[:L]); plt.xlabel("Rank"); plt.ylabel("CMC"); plt.title(f"T2I CMC – {args.map}/{args.train}")
        plt.grid(True,linestyle="--",linewidth=0.6); plt.savefig(out/"cmc.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"[done] {args.map}/{args.train} -> {out}")
    print(f"Rank-1={r1:.4f} mAP={mAP:.4f} ATS={ATS:.4f} validQ={validQ}")
if __name__=="__main__": main()
