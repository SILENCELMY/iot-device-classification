"""【探索性,非协议】CIC 时钟偏斜可行性探针 —— 只回答三个问题，不建特征、不跑模型。

动机：CIC 型号级剩余 0.073 缺口集中在 4 个 ESP 型号上，逐类对搜遍 144 种配置在 inner 上
也只到 0.71–0.80 —— **配置层没有解，缺的是表征**（见 cic-has-no-headroom / cic-label-space-was-wrong）。
那几个 ESP 同芯片同固件，行为特征原理上分不开；能分开同型号硬件的是**硬件级、跨捕获稳定**
的量。TCP 时间戳时钟偏斜（晶振物理属性）是其中最经典的一个，且 CIC 的 pcap 里确实带
`tcp.options.timestamp.tsval`（抽样约 70% 的包）。

**方法身份**：不是"加特征提分"，而是把偏斜作为**候选特征族**交给逐类对验收程序，
让闸门自己决定在哪些对上用（见 method-is-candidate-acceptance-not-deletion）。
若它只在 ESP 那几对开火、其余 300+ 对不动，既补了 CIC 的洞，也是方法本身的一次演示。

**本脚本只做可行性判定**：
  Q1 能否估出       每 (设备, 天) 有多少可用样本、拟合残差多大
  Q2 跨天稳不稳     同一台在 3 天之间的 ticks/s 相差多少 ppm
  Q3 设备间分不分   不同台之间相差多少 ppm，与 Q2 的天内波动比
判据：Q2 的跨天漂移 ≪ Q3 的设备间间隔 → 可用；两者同量级 → 不可用，别做了。

注意 tsval 会因设备重启而重置：按负跳变切段，取最长段拟合。
"""
from __future__ import annotations
import subprocess, sys, time, io
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/lmy/iot-device-classification/dataset/cic2022")
EXTR = ROOT/"extracted"
DAYS = [("1102_Idle",   EXTR/"2-Idle/2021_11_02_Idle.pcap"),
        ("1103_Active", EXTR/"5-Active/2021_11_03_Active.pcap"),
        ("1108_Active", EXTR/"5-Active/2021_11_08_Active.pcap")]
OUT  = Path("/home/lmy/cic_probe/cic_skew.csv")
NOMINAL = np.array([1.,10.,100.,128.,250.,1000.])
MIN_N = 200          # 一段至少这么多点才拟合

def esp_macs():
    m = pd.read_csv(ROOT/"device_mac_map.csv")
    sel = m[m.device_id.str.contains("Gosund|Teckin|Yutron|GlobeLamp", case=False, na=False)]
    return dict(zip(sel.mac.str.lower(), sel.device_id))

def run_tshark(pcap, macs):
    filt = " or ".join(f"eth.src=={m}" for m in macs)
    cmd = ["tshark","-r",str(pcap),"-Y",f"tcp.options.timestamp.tsval and ({filt})",
           "-T","fields","-e","eth.src","-e","frame.time_epoch",
           "-e","tcp.options.timestamp.tsval","-E","separator=\t"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  [tshark 失败] {p.stderr[:200]}", flush=True); return pd.DataFrame()
    if not p.stdout.strip(): return pd.DataFrame()
    d = pd.read_csv(io.StringIO(p.stdout), sep="\t", header=None,
                    names=["mac","t","tsval"], dtype={"mac":str})
    d["tsval"] = pd.to_numeric(d.tsval.astype(str).str.split(",").str[0], errors="coerce")
    d["t"]     = pd.to_numeric(d.t, errors="coerce")
    return d.dropna()

def fit_one(t, v):
    """按负跳变切段取最长段，最小二乘拟合 tsval~t，返回 (ticks/s, 残差std, n, 时长)。"""
    o = np.argsort(t); t, v = t[o], v[o]
    br = np.where(np.diff(v) < -1e6)[0] + 1        # 重启 / 回绕
    segs = np.split(np.arange(len(t)), br)
    seg = max(segs, key=len)
    if len(seg) < MIN_N: return None
    tt, vv = t[seg], v[seg]
    dur = tt[-1]-tt[0]
    if dur < 60: return None
    A = np.vstack([tt-tt[0], np.ones(len(tt))]).T
    sl, ic = np.linalg.lstsq(A, vv-vv[0], rcond=None)[0]
    res = (vv-vv[0]) - (A@np.array([sl,ic]))
    return float(sl), float(np.std(res)), int(len(seg)), float(dur)

def main():
    t0=time.time(); macs = esp_macs()
    print(f"ESP 族设备 {len(macs)} 台", flush=True)
    rows=[]
    for day, pcap in DAYS:
        if not pcap.exists(): print(f"[缺] {pcap}", flush=True); continue
        print(f"\n=== {day}  {pcap.name} ({pcap.stat().st_size/1e6:.0f} MB) ===", flush=True)
        d = run_tshark(pcap, list(macs))
        if d.empty: print("  无带时间戳的包", flush=True); continue
        print(f"  取到 {len(d)} 个带 tsval 的包，涉及 {d.mac.nunique()} 台", flush=True)
        for mac, g in d.groupby("mac"):
            name = macs.get(mac.lower(), mac)
            r = fit_one(g.t.to_numpy(), g.tsval.to_numpy())
            if r is None:
                print(f"    {name:26s} 样本不足 (n={len(g)})", flush=True); continue
            sl, resid, n, dur = r
            hz = NOMINAL[np.argmin(np.abs(NOMINAL-sl))]
            ppm = (sl/hz - 1.0)*1e6
            rows.append({"day":day,"device":name,"mac":mac,"ticks_per_s":sl,
                         "hz_nominal":hz,"skew_ppm":ppm,"resid_std":resid,"n":n,"dur_s":dur})
            print(f"    {name:26s} {sl:12.6f} tick/s  标称 {hz:6.0f}  偏斜 {ppm:+9.1f} ppm  "
                  f"n={n:6d}  时长 {dur/3600:.2f} h  残差std={resid:.1f}", flush=True)

    if not rows: print("\n没有可用样本，判定：不可行"); return
    R = pd.DataFrame(rows); R.to_csv(OUT, index=False)

    print("\n=== Q2 跨天稳定性（同一台在不同天的 ppm）===", flush=True)
    P = R.pivot_table(index="device", columns="day", values="skew_ppm")
    print(P.round(1).to_string(), flush=True)
    P = P.dropna(thresh=2)
    drift = (P.max(axis=1)-P.min(axis=1)) if len(P) else pd.Series(dtype=float)
    if len(drift):
        print(f"\n  跨天极差 ppm：中位 {drift.median():.1f}  最大 {drift.max():.1f}  "
              f"（{len(drift)} 台有 ≥2 天）", flush=True)

    print("\n=== Q3 设备间可分性（每天内两两 ppm 间隔）===", flush=True)
    gaps=[]
    for day, g in R.groupby("day"):
        v = np.sort(g.skew_ppm.values)
        if len(v)<2: continue
        dif = np.diff(v)
        gaps.append(dif.min())
        print(f"  {day}: {len(v)} 台  ppm 范围 [{v.min():+.1f}, {v.max():+.1f}]  "
              f"最小相邻间隔 {dif.min():.1f}  中位间隔 {np.median(dif):.1f}", flush=True)

    print("\n=== 判定 ===", flush=True)
    if len(drift) and gaps:
        d_med, g_min = drift.median(), float(np.median(gaps))
        print(f"  跨天漂移中位 {d_med:.1f} ppm   设备间最小间隔中位 {g_min:.1f} ppm   "
              f"比值 {d_med/max(g_min,1e-9):.2f}", flush=True)
        if d_med < g_min/3:
            print("  >> 可行：跨天漂移远小于设备间间隔，偏斜是稳定的设备指纹。", flush=True)
        elif d_med < g_min:
            print("  >> 边缘：可分但裕度小，需在型号级（同型多台合并）上再看。", flush=True)
        else:
            print("  >> 不可行：跨天漂移与设备间间隔同量级，别做了。", flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)

if __name__=="__main__":
    main()
