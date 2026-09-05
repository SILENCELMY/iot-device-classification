#!/bin/bash
# 服务器停机后的只读体检。不启动任何计算，不写任何文件（除 stdout）。
echo "════════ 1. 本次开机 / 上次关机 ════════"
echo "现在        : $(date)"
echo "本次开机    : $(uptime -s)"
echo "已运行      : $(uptime -p)"
echo "--- 重启记录 ---"; last -x reboot 2>/dev/null | head -5
echo "--- 关机记录（上次停机没有，若这次也没有=非正常关机）---"
last -x shutdown 2>/dev/null | head -5 || echo "（无）"

echo
echo "════════ 2. 上一次启动的日志末尾 ════════"
echo "--- 最后 40 行 ---"
journalctl -b -1 -e --no-pager 2>/dev/null | tail -40 || echo "（读不到，需 sudo）"
echo "--- 上次启动的 error 级 ---"
journalctl -b -1 -p err --no-pager 2>/dev/null | tail -25 || echo "（读不到）"
echo "--- 上次启动的内核环最后 30 行 ---"
journalctl -b -1 -k --no-pager 2>/dev/null | tail -30 || echo "（读不到）"

echo
echo "════════ 3. OOM 假设（我今晚并行跑了多个吃内存的任务）════════"
echo "总内存 / 当前占用："; free -h
echo "--- 上次启动里的 OOM 痕迹 ---"
n=$(journalctl -b -1 --no-pager 2>/dev/null | grep -icE "out of memory|Killed process|oom-kill" || echo 0)
echo "匹配条数: $n"
journalctl -b -1 --no-pager 2>/dev/null | grep -iE "out of memory|Killed process|oom-kill" | tail -10
echo "--- 本次启动至今 ---"
dmesg -T 2>/dev/null | grep -iE "out of memory|Killed process|oom-kill" | tail -5 || echo "（dmesg 需 sudo）"

echo
echo "════════ 4. 磁盘 / IO / 温度 ════════"
df -h | grep -vE "tmpfs|udev"
echo "--- 上次启动的 IO / 硬件错误 ---"
journalctl -b -1 --no-pager 2>/dev/null | grep -iE "I/O error|ata[0-9]|nvme.*error|hung task|watchdog|thermal|mce" | tail -15 || echo "（无或读不到）"

echo
echo "════════ 5. 服务状态 ════════"
systemctl --failed --no-pager 2>/dev/null | head -15
echo "--- 蒲公英 ---"
systemctl status pgyvpn --no-pager 2>/dev/null | head -12 || echo "（无 pgyvpn 服务单元）"
ip -4 addr 2>/dev/null | grep -E "inet 172\.16|pgy" || echo "（无 172.16 地址）"

echo
echo "════════ 6. 今晚的产物是否完好 ════════"
echo "--- 检测器日志 ---"
ls -la /home/lmy/cic_probe/detector.log 2>/dev/null || echo "（不存在）"
tail -3 /home/lmy/cic_probe/detector.log 2>/dev/null
echo "--- 仓库状态 ---"
cd /home/lmy/iot-device-classification 2>/dev/null && {
  git log --oneline -3
  echo "工作区: $(git status --short | wc -l) 个变更"
}

echo
echo "════════ 需要 sudo 的部分（我跑不了，要你贴输出）════════"
cat <<'TIP'
  sudo tail -200 /var/log/syslog
  sudo tail -200 /var/log/kern.log
  sudo dmesg -T | grep -iE "oom|killed|hung task|watchdog|thermal|ata|nvme|I/O error"
  sudo smartctl -a /dev/nvme0n1     # 或 lsblk 看盘符
TIP
