"""【探索性,非协议】CIC 深层指纹探针：TLS 握手 + TCP 选项 —— 在宣称"不可区分"之前把枚举拓宽。

来由：CIC 的残余（前三对占 68.5% 错误）在补上包长分布后仍不动（+0.0061），
四项栈指纹（tcp.window / 心跳 / 做不做DNS / 端点数）组内完全相同，
且 ESP 的 lwIP 不协商 TCP 时间戳（实测 0 个包带 tsval），时钟偏斜不可用。

**但"在已枚举空间内不可区分"的强度完全取决于枚举有多宽**，而今天已经因为
"用我们建过的表征论证表征极限"摔过一次（见 dont-claim-representation-limit）。
所以在把 CIC 写成"根源不可区分"之前，先补两族最可能区分同固件设备的标准指纹：

  TLS ClientHello   cipher suite 列表与顺序、扩展类型与顺序、supported_groups、
                    ALPN、SNI、TLS 版本 —— 固件里 TLS 库的编译期配置，明文可见
  TCP 选项           SYN 包里选项的【顺序】、窗口缩放因子、MSS —— 协议栈编译期配置

判据：
  组内设备的这些指纹若出现稳定差异（且两天一致）→ 枚举尚未穷尽，"不可区分"不能写
  若组内完全相同                                 → 与已有四项证据合并，结论可封

**只做描述，不建特征。** 每项都看两天，只有跨天一致的项才算身份。
"""
from __future__ import annotations
import subprocess, io, time, json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/lmy/iot-device-classification/dataset/cic2022")
EXTR = ROOT/"extracted"
DAYS = [("1102_Idle", EXTR/"2-Idle/2021_11_02_Idle.pcap"),
        ("1108_Idle", EXTR/"2-Idle/2021_11_08_Idle.pcap")]
OUT  = Path("/home/lmy/cic_probe/cic_deep_fp.csv")

def esp_macs():
    m = pd.read_csv(ROOT/"device_mac_map.csv")
    s = m[m.device_id.str.contains("Gosund|Teckin|Yutron|GlobeLamp", case=False, na=False)]
    return dict(zip(s.mac.str.lower(), s.device_id))

def tsh(pcap, disp, fields):
    cmd=["tshark","-r",str(pcap),"-Y",disp,"-T","fields"]
    for f in fields: cmd+=["-e",f]
    cmd+=["-E","separator=\t","-E","occurrence=a"]     # a = 全部出现，保留顺序
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode!=0 or not p.stdout.strip():
        return pd.DataFrame(columns=fields)
    return pd.read_csv(io.StringIO(p.stdout),sep="\t",header=None,names=fields,dtype=str)

def main():
    t0=time.time(); macs=esp_macs()
    filt=" or ".join(f"eth.src=={m}" for m in macs)
    print(f"ESP 族 {len(macs)} 台\n",flush=True)
    rows=[]
    for day,p in DAYS:
        if not p.exists(): print(f"[缺] {p}",flush=True); continue
        print(f"=== {day} ===",flush=True)

        # --- TLS ClientHello ---
        tls=tsh(p, f"tls.handshake.type==1 and ({filt})",
                ["eth.src","tls.handshake.version","tls.handshake.ciphersuite",
                 "tls.handshake.extension.type","tls.handshake.extensions_server_name",
                 "tls.handshake.extensions_supported_group"])
        print(f"  ClientHello {len(tls)} 个",flush=True)
        for mac,g in tls.groupby("eth.src"):
            r=g.iloc[0]
            cs=str(r.get("tls.handshake.ciphersuite",""))
            ext=str(r.get("tls.handshake.extension.type",""))
            sni=sorted(set(x for x in g["tls.handshake.extensions_server_name"].dropna() if x))
            rows.append({"day":day,"device":macs.get(mac.lower(),mac),"kind":"tls",
                "n":len(g),"ver":r.get("tls.handshake.version",""),
                "ciphers":cs,"exts":ext,
                "groups":str(r.get("tls.handshake.extensions_supported_group","")),
                "sni":json.dumps(sni)})
            print(f"    {macs.get(mac.lower(),mac):26s} n={len(g):3d}  "
                  f"cipher数={len(cs.split(',')) if cs else 0}  "
                  f"ext={ext[:60]}  SNI={sni[:2]}",flush=True)

        # --- TCP 选项（SYN 包）---
        syn=tsh(p, f"tcp.flags.syn==1 and tcp.flags.ack==0 and ({filt})",
                ["eth.src","tcp.option_kind","tcp.options.wscale.shift","tcp.options.mss_val"])
        print(f"  SYN {len(syn)} 个",flush=True)
        for mac,g in syn.groupby("eth.src"):
            r=g.iloc[0]
            kinds=str(r.get("tcp.option_kind",""))
            rows.append({"day":day,"device":macs.get(mac.lower(),mac),"kind":"tcpopt",
                "n":len(g),"opt_order":kinds,
                "wscale":str(r.get("tcp.options.wscale.shift","")),
                "mss":str(r.get("tcp.options.mss_val",""))})
            print(f"    {macs.get(mac.lower(),mac):26s} n={len(g):3d}  "
                  f"选项顺序={kinds}  wscale={r.get('tcp.options.wscale.shift','')}  "
                  f"mss={r.get('tcp.options.mss_val','')}",flush=True)
        print(flush=True)

    if not rows:
        print("没有取到任何 TLS/TCP 选项样本"); return
    R=pd.DataFrame(rows); R.to_csv(OUT,index=False)

    print("=== 判读：组内设备之间这些指纹有没有差异 ===",flush=True)
    GA=["GlobeLampESPB1680C","GosundESP032979Plug","GosundESP039AAFSocket",
        "GosundESP0C3994Plug","GosundESP10098FSocket","GosundESP10ACD8Plug",
        "GosundESP147FF9Plug","GosundESP1ACEE1Socket"]
    GB=["TeckinPlug1","TeckinPlug2","YutronPlug1","YutronPlug2"]
    for kind, keys in [("tls",["ver","ciphers","exts","groups","sni"]),
                       ("tcpopt",["opt_order","wscale","mss"])]:
        K=R[R.kind==kind]
        if K.empty: print(f"\n  [{kind}] 无样本"); continue
        for gname,members in [("A组(win4380)",GA),("B组(win2920)",GB)]:
            sub=K[K.device.isin(members)]
            if sub.empty: continue
            print(f"\n  [{kind}] {gname}  {sub.device.nunique()} 台有样本",flush=True)
            for k in keys:
                if k not in sub.columns: continue
                vals=sub.groupby("device")[k].first()
                u=vals.nunique(dropna=False)
                print(f"     {k:12s} 不同取值 {u} 种"
                      + ("   ← 组内有差异，枚举未穷尽" if u>1 else "   组内一致"),flush=True)
                if u>1:
                    for dev,v in vals.items():
                        print(f"        {dev:26s} {str(v)[:70]}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
