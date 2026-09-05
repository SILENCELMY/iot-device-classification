"""C3 的轻度验证 —— 只用【已烧天】，保留集一行不读。

正式跑是一次性的，先在已烧天上把 C3b′ 的机制过一遍：
    导出配置  16-09-30（选参天，与正式跑相同）
    施加      16-10-11 + 16-10-12（已烧天，充当"没见过的天"）
    问        闸门在选参天放行的对，到别的天上【整区 Δ 的符号】还成不成立

若这里就翻符号 → 正式跑几乎必挂，应先改方法而不是烧保留集。
顺带验：区搜索是否给 Belkin 选出 R3∩R1；整区记账下放行/拒绝是否与端到端同号。

臂 A 同样只用已烧天，确认曲线形状（16-09-23 训练 → 16-10-11/12 测试）。
"""
import sys
sys.path.insert(0,"/home/lmy/cic_probe")
import c3_confirm as C

# 把"保留集"的角色交给已烧天，走主特征表/主 lenhist
C.RESERVE=["16-10-11","16-10-12"]
C.RES_F=None                      # 下面重写 loader
C.RES_L=C.LH_MAIN
C.OUT=C.Path("/home/lmy/iot-device-classification/results/c3_pilot_burned_20260905")
C.OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd

def gate_reserve():
    out={}
    for day in C.RESERVE:
        d=pd.read_csv(C.FULL%day,low_memory=False,encoding="utf-8-sig",
                      usecols=["device","label","day","window_id"])
        out[day]=set(C.IID.day_gate(d,day))
    return out
C.gate_reserve=gate_reserve

_orig_read=pd.read_csv
def _read(path,*a,**k):
    """臂 B 读'保留集'时，改成拼两个已烧天的特征表。"""
    if path is None:
        return pd.concat([_orig_read(C.FULL%d,low_memory=False,encoding="utf-8-sig")
                          for d in C.RESERVE],ignore_index=True)
    return _orig_read(path,*a,**k)
pd.read_csv=_read

print("轻度验证：施加天 =",C.RESERVE,"（全部已烧）  保留集不读",flush=True)
C.main()
