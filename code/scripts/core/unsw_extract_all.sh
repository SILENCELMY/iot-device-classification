#!/usr/bin/env bash
# UNSW 20-day feature extraction (mainline, for protocol 16.4 test 1).
# Writes to results/unsw_features_full/ ; never touches results/unsw_pilot/
# or any independent-line path.
set -u
cd /home/lmy/iot-device-classification || exit 1

OUT=results/unsw_features_full
LOG=$OUT/logs
mkdir -p "$LOG"
PY=/home/lmy/anaconda3/bin/python

# Largest pcap first for better parallel packing.
DAYS="16-10-12 16-10-04 16-09-28 16-10-11 16-10-07 16-10-05 16-09-23 16-09-24 \
16-09-30 16-10-06 16-09-29 16-10-01 16-10-03 16-10-08 16-09-26 16-10-09 \
16-10-10 16-10-02 16-09-25 16-09-27"

echo "[start] $(date -Is)  host=$(hostname)  nproc=$(nproc)" | tee "$LOG/_driver.log"
echo "[plan] 20 days, 8-way parallel, out=$OUT" | tee -a "$LOG/_driver.log"

run_day() {
  d="$1"
  /home/lmy/anaconda3/bin/python code/scripts/core/extract_features_generic.py \
    --pcap-dir dataset/unsw/pcap \
    --mac-map dataset/unsw/device_mac_map.csv \
    --output "results/unsw_features_full/features_day_${d}.csv" \
    --days "$d" > "results/unsw_features_full/logs/extract_${d}.log" 2>&1
  rc=$?
  echo "[day] $d rc=$rc $(date -Is)" >> results/unsw_features_full/logs/_driver.log
  return $rc
}
export -f run_day

echo "$DAYS" | tr ' ' '\n' | grep -v '^$' | xargs -I{} -P 8 bash -c 'run_day {}'

echo "[extract-done] $(date -Is)" >> "$LOG/_driver.log"

# ---- integrity summary ----
{
  echo "day,rows,bytes,md5,rc_log_tail"
  for d in $DAYS; do
    f="$OUT/features_day_${d}.csv"
    if [ -f "$f" ]; then
      r=$(( $(wc -l < "$f") - 1 ))
      b=$(stat -c%s "$f")
      m=$(md5sum "$f" | cut -d' ' -f1)
      echo "${d},${r},${b},${m},ok"
    else
      echo "${d},NA,NA,NA,MISSING"
    fi
  done
} > "$OUT/extraction_summary.csv"

# ---- determinism cross-check against the four pilot days ----
{
  echo "# UNSW extraction cross-check vs results/unsw_pilot (read-only)"
  echo "# generated $(date -Is)"
  for d in 16-09-23 16-09-30 16-10-11 16-10-12; do
    new="$OUT/features_day_${d}.csv"
    old="results/unsw_pilot/features_day_${d}.csv"
    if [ -f "$new" ] && [ -f "$old" ]; then
      mn=$(md5sum "$new" | cut -d' ' -f1)
      mo=$(md5sum "$old" | cut -d' ' -f1)
      if [ "$mn" = "$mo" ]; then s=IDENTICAL; else s=DIFFERS; fi
      echo "$d $s new=$mn pilot=$mo"
    else
      echo "$d SKIP (missing one side)"
    fi
  done
} > "$OUT/PILOT_CROSSCHECK.md"

echo "[all-done] $(date -Is)" >> "$LOG/_driver.log"
