"""C3 脚本的跑通验证。**不读保留集的科学结果。**

分两步（协议 §6 步骤 3：选参天允许调试，保留天不允许）：

  步骤 1  保留集只做 **schema 检查** —— 列名 / 连接键覆盖率 / day 取值 / 行数。
          纯元数据，不建模、不比标签与预测，不构成"看过测试集"。
  步骤 2  把 `16-09-30`（选参天，已烧）顶替保留集的位置，跑完整两臂，
          只看"跑不跑得通"，不看数字含义。
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0,"/home/lmy/cic_probe")
import c3_confirm as C

print("="*90); print("步骤 1  保留集 schema 检查（元数据）"); print("="*90)
f=pd.read_csv(C.RES_F,low_memory=False,encoding="utf-8-sig",
              usecols=["device","label","day","window_id"])
l=pd.read_csv(C.RES_L,usecols=["device","day","window_id"])
print(f"features  {len(f):8d} 行   day 取值 {sorted(f.day.unique())}")
print(f"lenhist   {len(l):8d} 行   day 取值 {sorted(l.day.unique())}")
print(f"features 的 device 数 {f.device.nunique()}   lenhist 的 device 数 {l.device.nunique()}")
kf=set(map(tuple,f[["device","day","window_id"]].values))
kl=set(map(tuple,l[["device","day","window_id"]].values))
print(f"连接键：features {len(kf)}  lenhist {len(kl)}  交集 {len(kf&kl)}")
print(f"features 能连上的比例 {len(kf&kl)/len(kf):.6f}   （<1 说明有窗拿不到 lenhist）")
assert len(kf&kl)/len(kf) > 0.99, "连接率过低，先查栅格对齐"
print("PASS  schema 与连接键对齐")

print(); print("="*90); print("步骤 2  用 16-09-30 顶替保留集，跑通两臂"); print("="*90)
C.RESERVE=["16-09-30"]
C.RES_F=C.FULL%"16-09-30"
C.RES_L=C.LH_MAIN
C.AGGS=[1]   # 臂 A 只跑 1 个 k，本次只验臂 B 的闸门
C.SEEDS=[42]
C.TOPN=3
C.OUT=C.Path("/tmp/c3_smoke_out"); C.OUT.mkdir(exist_ok=True)
C.main()
print("\nPASS  两臂均跑通")
