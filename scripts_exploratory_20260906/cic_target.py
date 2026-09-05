"""【探索性,非协议】CIC 靶清单：把 630 对切成"不该分"与"该修得动"。

用户 2026-09-05 的判据：
    指纹相同 → 本来就不该分成两类（真要分得换模态，与本方法无关）
    指纹不同 → 数据【可证地】支持区分，分类器却做不到  ← 这才是要解决的
后者的好处是**可分性有独立证明**：证据来自一组从未进过分类器的稳定观测量，
所以修不动就是方法的问题，不能推给数据。

输入：
  cic_fingerprint.csv   协议栈指纹（TTL / tcp.window / 心跳 / 做不做DNS / 端点集 / DNS名集），
                        两天各算一次，只用两天一致的项
  same_device.csv       同设备检测器（域内≥0.90 且 跨捕获≤0.60 → flag）

输出：
  1. 2×2：指纹相同/不同 × 检测器是否开火     —— 特异性硬检验：
        指纹【明确不同】的对上开火 = 检测器误报（那是两台真设备）
  2. 靶清单：指纹不同 且 cross_acc 低         —— 该修得动却没修动的
  3. 靶清单的规模：占全部对多少、占"低 cross_acc"的对多少
"""
from __future__ import annotations
import json, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd

FP  = Path("/home/lmy/cic_probe/cic_fingerprint.csv")
SD  = Path("/home/lmy/cic_probe/same_device.csv")
OUT = Path("/home/lmy/cic_probe/cic_target.csv")
HB_TOL = 1.0        # 心跳秒差容忍
CROSS_BAD = 0.80    # "分不开"的门槛（检测器的 flag 用的是 0.60，这里放宽看全景）

def jac(a,b):
    A,B=set(json.loads(a) if isinstance(a,str) else []), set(json.loads(b) if isinstance(b,str) else [])
    return len(A&B)/max(len(A|B),1) if (A or B) else 1.0

def main():
    fp=pd.read_csv(FP)
    # 只保留两天都出现的设备，并检查逐项跨天一致
    ok=fp.groupby("device").filter(lambda g: g.day.nunique()==2)
    devs=sorted(ok.device.unique())
    print(f"指纹表：{len(devs)} 台设备有两天数据",flush=True)
    stable={}
    for dev,g in ok.groupby("device"):
        g=g.sort_values("day"); a,b=g.iloc[0],g.iloc[1]
        stable[dev]={
            "ttl": a.ttl_mode if a.ttl_mode==b.ttl_mode else np.nan,
            "win": a.win_mode if a.win_mode==b.win_mode else np.nan,
            "hb":  (a.heartbeat+b.heartbeat)/2 if abs(a.heartbeat-b.heartbeat)<=HB_TOL else np.nan,
            "dns": a.does_dns if a.does_dns==b.does_dns else np.nan,
            "nep": a.n_endpoints if a.n_endpoints==b.n_endpoints else np.nan,
            "eps": a.endpoints, "nms": a.dns_names}
    for k in ["ttl","win","hb","dns","nep"]:
        n=sum(1 for d in stable if not pd.isna(stable[d][k]))
        print(f"  {k:4s} 两天一致的设备数 {n}/{len(stable)}",flush=True)

    def cmp_pair(x,y):
        A,B=stable[x],stable[y]; diff=[]
        for k in ["ttl","win","dns","nep"]:
            if pd.isna(A[k]) or pd.isna(B[k]): continue
            if A[k]!=B[k]: diff.append(k)
        if not pd.isna(A["hb"]) and not pd.isna(B["hb"]) and abs(A["hb"]-B["hb"])>HB_TOL:
            diff.append("hb")
        if jac(A["eps"],B["eps"])<0.34: diff.append("eps")
        if jac(A["nms"],B["nms"])<0.34: diff.append("dnsname")
        return diff

    if not SD.exists():
        print(f"\n[等] {SD} 还没生成，先只报指纹分组",flush=True)
        rows=[{"a":x,"b":y,"fp_diff":len(cmp_pair(x,y)),"which":",".join(cmp_pair(x,y))}
              for x,y in itertools.combinations(devs,2)]
        R=pd.DataFrame(rows)
        print(R.fp_diff.value_counts().sort_index().to_string(),flush=True)
        R.to_csv(OUT,index=False); return

    sd=pd.read_csv(SD)
    ac=[c for c in sd.columns if c in ("a","dev_a","device_a","x")][0]
    bc=[c for c in sd.columns if c in ("b","dev_b","device_b","y")][0]
    rows=[]
    for _,r in sd.iterrows():
        x,y=r[ac],r[bc]
        if x not in stable or y not in stable:
            rows.append({**r.to_dict(),"fp_diff":np.nan,"which":""}); continue
        d=cmp_pair(x,y)
        rows.append({**r.to_dict(),"fp_diff":len(d),"which":",".join(d)})
    R=pd.DataFrame(rows); R.to_csv(OUT,index=False)
    R2=R.dropna(subset=["fp_diff"])
    print(f"\n可比对数 {len(R2)}/{len(R)}",flush=True)

    print("\n=== 2×2：指纹是否不同 × 检测器是否开火 ===",flush=True)
    R2["fp_same"]=R2.fp_diff==0
    print(pd.crosstab(R2.fp_same, R2.flag).to_string(),flush=True)
    fpfire=R2[(~R2.fp_same)&(R2.flag)]
    print(f"\n  指纹【明确不同】却被判'不该分'的对：{len(fpfire)} 个  ← 检测器误报",flush=True)
    for _,r in fpfire.head(10).iterrows():
        print(f"    {r[ac]:26s} {r[bc]:26s} 差异项={r['which']}  "
              f"域内={r.in_dom:.3f} 跨={r.cross_acc:.3f}",flush=True)

    print("\n=== 靶清单：指纹不同（可证两台设备）且跨捕获分不开 ===",flush=True)
    T=R2[(~R2.fp_same)&(R2.cross_acc<CROSS_BAD)].sort_values("cross_acc")
    print(f"  共 {len(T)} 对（cross_acc < {CROSS_BAD}）",flush=True)
    for _,r in T.head(25).iterrows():
        print(f"    {r[ac]:26s} {r[bc]:26s} 跨={r.cross_acc:.3f} 域内={r.in_dom:.3f}  "
              f"指纹差异={r['which']}",flush=True)

    print("\n=== 规模对照 ===",flush=True)
    bad=R2[R2.cross_acc<CROSS_BAD]
    print(f"  全部 cross_acc<{CROSS_BAD} 的对：{len(bad)}",flush=True)
    print(f"    其中指纹相同（不该分，不是我们的活）：{(bad.fp_diff==0).sum()}",flush=True)
    print(f"    其中指纹不同（该修得动）：           {(bad.fp_diff>0).sum()}  ← 真正的靶",flush=True)
    print(f"\n  同名型号真值对照：同型号对 {int(R2.same_coarse.sum())} 个，"
          f"其中指纹相同 {int(R2[(R2.same_coarse)&(R2.fp_diff==0)].shape[0])} 个",flush=True)

if __name__=="__main__":
    main()
