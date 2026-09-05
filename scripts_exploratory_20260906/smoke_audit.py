"""审计改动后的烟测 —— 只验已改的三处，已烧天，不碰保留集。

本次验证的改动（`c3_confirm.py`）：
  缺陷 1  施加顺序按区大小从大到小 + 写完后统一记账
  缺陷 8  RF 去掉 class_weight（原与 fit_w 的 sample_weight 相乘 ⇒ 少数类权重被平方）
  缺陷 9  加权拟合失败不再静默退回，打印一次警告

**不含缺陷 7**（专用侧平滑）—— 一次只验一批改动，出问题才知道是谁的。

只跑臂 B：臂 A 用 kNN，完全不碰 MK/fit_w，本次改动影响不到它。
单种子（42）。施加天 = 已烧的 16-10-11 + 16-10-12。

判据：
  跑通、无异常、无"加权拟合失败"警告            → 改动无连带
  过闸对数与配置和 C3 确证时【可能不同】         → 缺陷 8 会改变 RF 候选的行为，属预期
  端到端不应大幅下降                            → 若掉超过 0.01，说明缺陷 8 改反了
"""
import sys, time
sys.path.insert(0,"/home/lmy/cic_probe")
import c3_confirm as C
import numpy as np, pandas as pd

C.RESERVE=["16-10-11","16-10-12"]
C.RES_L=C.LH_MAIN
C.SEEDS=[42]
C.OUT=C.Path("/home/lmy/iot-device-classification/results/smoke_audit_20260906")
C.OUT.mkdir(exist_ok=True)

def gate_reserve():
    out={}
    for day in C.RESERVE:
        d=pd.read_csv(C.FULL%day,low_memory=False,encoding="utf-8-sig",
                      usecols=["device","label","day","window_id"])
        out[day]=set(C.IID.day_gate(d,day))
    return out
C.gate_reserve=gate_reserve
_orig=pd.read_csv
def _read(path,*a,**k):
    if path is None:
        return pd.concat([_orig(C.FULL%d,low_memory=False,encoding="utf-8-sig")
                          for d in C.RESERVE],ignore_index=True)
    return _orig(path,*a,**k)
pd.read_csv=_read
C.RES_F=None

t0=time.time()
res_gates=C.gate_reserve()
sets=[C.gate_day(C.TRAIN),C.gate_day(C.TUNE)]+[res_gates[d] for d in C.RESERVE]
devs=set.intersection(*sets)
print(f"烟测（审计改动 1/8/9）  {len(devs)} 类  施加天 {C.RESERVE}（已烧）",flush=True)
print(f"RF 是否仍带 class_weight: "
      f"{'是（缺陷 8 未生效！）' if 'class_weight' in str(C.MK('rf',42).get_params()) and C.MK('rf',42).get_params().get('class_weight') else '否（缺陷 8 已生效）'}",flush=True)
b=C.arm_b(devs,42)
print(f"\n{'='*80}\n=== 烟测判读 ===",flush=True)
print(f"  过闸 {b['n_gated']} 对   端到端 base={b['base']:.4f} "
      f"平滑={b['smooth']:.4f} 修后={b['fixed']:.4f}  Δ={b['delta']:+.4f}",flush=True)
print(f"  对照 C3 确证（保留 16 天，改动前）：base 0.8620 平滑 0.9183 修后 0.9263 Δ=+0.0080",flush=True)
print(f"  对照 上次已烧天试跑（改动前）：      base 0.8521 平滑 0.9166 修后 0.9191 Δ=+0.0024",flush=True)
print(f"\n  判据：跑通无警告 → 改动无连带；Δ 若掉超过 0.01 → 缺陷 8 改反了。",flush=True)
print(f"  总耗时 {time.time()-t0:.0f}s",flush=True)
