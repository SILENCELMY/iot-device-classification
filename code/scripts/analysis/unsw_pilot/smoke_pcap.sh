#!/bin/bash
# UNSW pilot — 五问之 4 的证据：通用 Ethernet 特征能否从 pcap 稳定提取
# 输出全部落 results/unsw_pilot/smoke_<day>.txt
# 严格 §16.2：只碰 pcap，不碰官方 CSV。
set -u
DAY="${1:-16-09-30}"
ROOT="$HOME/iot-device-classification"
PCAP="$ROOT/dataset/unsw/pcap/$DAY.pcap"
OUT="$ROOT/results/unsw_pilot/smoke_$DAY.txt"
PY="$HOME/anaconda3/bin/python3"

mkdir -p "$(dirname "$OUT")"
{
echo "==============================================================="
echo " UNSW pilot smoke — pcap 字段与时间分辨率检验（五问之 4）"
echo " day        : $DAY"
echo " pcap       : $PCAP"
echo " run at     : $(date -Is)"
echo " tshark     : $(tshark --version 2>/dev/null | head -1)"
echo "==============================================================="
echo
echo "--- [1] capinfos ---"
capinfos -c -u -a -e -y -T -M "$PCAP" 2>&1
echo
echo "--- [2] 链路层类型（应为 Ethernet） ---"
capinfos -t "$PCAP" 2>&1 | head -5
echo
echo "--- [3] 提取器所用的 4 个 tshark 字段，前 10 包 ---"
echo "frame.time_epoch<TAB>frame.len<TAB>eth.src<TAB>eth.dst"
tshark -r "$PCAP" -T fields -e frame.time_epoch -e frame.len -e eth.src -e eth.dst \
       -E header=n -E separator=/t -E occurrence=a -c 10 2>/dev/null
echo
echo "--- [4] 字段缺失率（前 500k 包） ---"
tshark -r "$PCAP" -T fields -e frame.time_epoch -e frame.len -e eth.src -e eth.dst \
       -E header=n -E separator=/t -E occurrence=a -c 500000 2>/dev/null | \
"$PY" -c "
import sys
n=0; miss=[0,0,0,0]
for line in sys.stdin:
    p=line.rstrip('\n').split('\t')
    p += ['']*(4-len(p)); n+=1
    for i in range(4):
        if not p[i].strip(): miss[i]+=1
names=['frame.time_epoch','frame.len','eth.src','eth.dst']
print(f'packets checked: {n:,}')
for name,m in zip(names,miss):
    print(f'  {name:20s} missing {m:,}  ({m/max(n,1)*100:.4f}%)')
"
echo
echo "--- [5] 时间戳分辨率（§16.2 的关键判据；CSV 为秒级整数，pcap 必须是亚秒） ---"
tshark -r "$PCAP" -T fields -e frame.time_epoch -E header=n -c 500000 2>/dev/null | \
"$PY" -c "
import sys, numpy as np
a=np.array([float(x) for x in sys.stdin if x.strip()])
print(f'packets                     : {len(a):,}')
print(f'unique timestamps           : {len(np.unique(a)):,}')
print(f'unique ratio                : {len(np.unique(a))/len(a):.6f}')
print(f'all timestamps integer-sec? : {bool(np.all(a==np.floor(a)))}   <- 必须 False')
d=np.diff(a); d=d[d>0]
print(f'positive interarrival gaps  : {len(d):,}')
print(f'  min   = {d.min():.9f} s')
print(f'  p1    = {np.percentile(d,1):.9f} s')
print(f'  p50   = {np.percentile(d,50):.9f} s')
print(f'  p99   = {np.percentile(d,99):.6f} s')
print(f'  max   = {d.max():.6f} s')
print(f'  frac  < 1 s   : {(d<1.0).mean():.4f}')
print(f'  frac  < 0.1 s : {(d<0.1).mean():.4f}   <- burst 阈值 (§ burst_packet_ratio)')
print(f'  frac  < 0.01 s: {(d<0.01).mean():.4f}')
"
echo
echo "--- [6] 全天 MAC 清点（eth.src / eth.dst 各自的唯一 MAC 计数，全文件） ---"
tshark -r "$PCAP" -T fields -e eth.src -e eth.dst -E header=n -E separator=/t 2>/dev/null | \
"$PY" -c "
import sys, collections
src=collections.Counter(); dst=collections.Counter(); n=0
for line in sys.stdin:
    p=line.rstrip('\n').split('\t'); p+=['']*(2-len(p)); n+=1
    if p[0].strip(): src[p[0].strip().lower().split(',')[0]]+=1
    if p[1].strip(): dst[p[1].strip().lower().split(',')[0]]+=1
print(f'total packets  : {n:,}')
print(f'unique eth.src : {len(src)}')
print(f'unique eth.dst : {len(dst)}')
allm=set(src)|set(dst)
print(f'unique MACs (src or dst): {len(allm)}')
print()
print('top 40 MACs by (src+dst) packet count:')
tot=collections.Counter()
for m in allm: tot[m]=src.get(m,0)+dst.get(m,0)
for m,c in tot.most_common(40):
    print(f'  {m}  total={c:>9,}  src={src.get(m,0):>9,}  dst={dst.get(m,0):>9,}')
"
echo
echo "=== smoke done: $(date -Is) ==="
} > "$OUT" 2>&1
echo "wrote $OUT"
