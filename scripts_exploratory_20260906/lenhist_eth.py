"""【探索性,非协议】帧长直方图特征族 —— 以太网版（UNSW / CIC）。

自采上这一族把网关三类从 0.69 抬到 0.87、5 类 outer macro +0.155（`lenhist_test.py`，
泄漏检查 `lenhist_verify.py` 进行中）。CIC 与 UNSW 都是网关侧以太网抓包，**包长同样可得，
而两边的特征池同样只有 `len_*` 的矩（均值/标准差/分位数），没有任何离散长度结构**。
按"找出三个数据集全部可区分的点"的口径，这一族必须在这两个池上也试。

**行集对齐**：直接复用 `extract_features_generic` 自己的 `read_packets` /
`assign_device_streams` / `load_mac_map`，窗口栅格用它的全天共享 origin
（`day_origin = streams.time_epoch.min()`，`window_id = floor((t−origin)/10)`），
因此产出可按 (device, day, window_id) 与既有特征表直接连接，绝不重实现。

**目标长度只从指定的源天导出**（类间/类内方差比 top-K），目标天从不参与选择。

用法：
    python lenhist_eth.py --pcap-dir <dir> --mac-map <csv> --days d1,d2,... \
        --src-days d1 --out <csv>
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

sys.path.insert(0, "/home/lmy/iot-device-classification/code/scripts/core")
import extract_features_generic as EG          # noqa: E402

WIN=10.0; TOPK=32; MIN_PKT=2

def per_window_counts(pcap, day, mac_map):
    packets = EG.read_packets(pcap, set(mac_map), verbose=False)
    if packets.empty: return {}
    streams = EG.assign_device_streams(packets, mac_map)
    if streams.empty: return {}
    origin = float(streams["time_epoch"].min())
    rel = streams["time_epoch"].to_numpy() - origin
    wid = np.floor(rel/WIN).astype(np.int64)
    dev = streams["device"].to_numpy(); ln = streams["length"].to_numpy().astype(int)
    out={}
    order=np.lexsort((wid,dev))
    dev,wid,ln = dev[order],wid[order],ln[order]
    # 分段扫描，避免 groupby 的对象开销
    b=0
    for i in range(1,len(dev)+1):
        if i==len(dev) or dev[i]!=dev[b] or wid[i]!=wid[b]:
            if i-b>=MIN_PKT:
                out[(dev[b],day,int(wid[b]))]=Counter(ln[b:i])
            b=i
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pcap-dir",type=Path,required=True)
    ap.add_argument("--mac-map",type=Path,required=True)
    ap.add_argument("--days",type=str,required=True)
    ap.add_argument("--src-days",type=str,required=True,help="只从这些天导出目标长度")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--pattern",type=str,default="{day}.pcap")
    a=ap.parse_args()
    t0=time.time()
    mac_map=EG.load_mac_map(a.mac_map)
    days=[d.strip() for d in a.days.split(",")]
    src=set(d.strip() for d in a.src_days.split(","))
    print(f"设备 {len(mac_map)} 台   天 {days}   源天 {sorted(src)}",flush=True)

    per={}
    for day in days:
        p=a.pcap_dir/a.pattern.format(day=day)
        if not p.exists():
            cands=list(a.pcap_dir.rglob(f"*{day}*.pcap"))
            if not cands: print(f"  [缺] {day}",flush=True); continue
            p=cands[0]
        c=per_window_counts(p,day,mac_map)
        per.update(c)
        print(f"  {day}  窗 {len(c)}   {time.time()-t0:.0f}s",flush=True)

    srck=[k for k in per if k[1] in src]
    if not srck: sys.exit("源天没有窗口")
    allc=Counter()
    for k in srck: allc.update(per[k].keys())
    cand=[L for L,n in allc.items() if n>=50]
    frac={k:{L:per[k].get(L,0)/sum(per[k].values()) for L in cand} for k in srck}
    labs=sorted({k[0] for k in srck})
    score={}
    for L in cand:
        by={d:np.array([frac[k][L] for k in srck if k[0]==d]) for d in labs}
        mus=np.array([v.mean() for v in by.values()]); wit=np.mean([v.var() for v in by.values()])
        score[L]=mus.var()/(wit+1e-12)
    tgt=[L for L,_ in sorted(score.items(),key=lambda x:-x[1])[:TOPK]]
    print(f"\n源天候选长度 {len(cand)} 种 → 目标 top{TOPK}: {sorted(tgt)}",flush=True)

    rows=[]
    for (dev,day,wid),c in per.items():
        tot=sum(c.values()); r={"device":dev,"day":day,"window_id":wid}
        for L in tgt:
            r[f"lenhist_cnt_{L}"]=c.get(L,0); r[f"lenhist_frac_{L}"]=c.get(L,0)/tot
        p=np.array([v/tot for v in c.values()]); top=c.most_common(1)[0]
        r["lenhist_nuniq"]=len(c); r["lenhist_entropy"]=float(-(p*np.log(p+1e-12)).sum())
        r["lenhist_top1_len"]=int(top[0]); r["lenhist_top1_frac"]=top[1]/tot
        r["lenhist_cover_topk"]=sum(c.get(L,0) for L in tgt)/tot
        rows.append(r)
    R=pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    R.to_csv(a.out,index=False)
    print(f"写出 {a.out}   {len(R)} 行 × {len(R.columns)} 列",flush=True)
    print(f"逐天窗口数：\n{R.groupby('day').size().to_string()}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
