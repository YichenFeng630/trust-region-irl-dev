#!/usr/bin/env bash
# Run a command while sampling nvidia-smi, then summarise GPU use.
#   gpu_monitor.sh OUT_PREFIX command [args...]
# Writes OUT_PREFIX_gpu.csv (whole GPU, every 0.5 s), OUT_PREFIX_apps.csv (per process, every 1 s),
# OUT_PREFIX_cmd.log (the command's output) and prints a summary.
set -u
OUT="$1"; shift
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -lms 500 > "${OUT}_gpu.csv" &
MON1=$!
( while true; do nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | sed "s/^/$(date +%s.%N),/"; sleep 1; done ) > "${OUT}_apps.csv" &
MON2=$!
sleep 2                                   # idle baseline before the command starts
T0=$(date +%s)
"$@" > "${OUT}_cmd.log" 2>&1
RC=$?
T1=$(date +%s)
sleep 1
kill $MON1 $MON2 2>/dev/null; wait $MON1 $MON2 2>/dev/null
echo "command exit $RC, $((T1 - T0)) s" | tee "${OUT}_summary.txt"
"${PYTHON:-python3}" -I - "$OUT" "$T0" "$T1" <<'EOF' | tee -a "${OUT}_summary.txt"
import sys, csv, datetime
out, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = []
for r in csv.reader(open(f"{out}_gpu.csv")):
    ts = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
    rows.append((ts, *[float(x) for x in r[1:]]))
base = [r for r in rows if r[0] < t0]
run = [r for r in rows if t0 <= r[0] <= t1]
col = lambda rs, i: [r[i] for r in rs]
mean = lambda x: sum(x) / max(len(x), 1)
print(f"GPU memory total {rows[0][4]:.0f} MiB")
if base:
    print(f"idle baseline: memory used {mean(col(base, 3)):.0f} MiB, utilisation {mean(col(base, 1)):.0f} %")
print(f"during the run ({len(run)} samples): memory used peak {max(col(run, 3)):.0f} MiB, mean {mean(col(run, 3)):.0f} MiB; "
      f"GPU utilisation mean {mean(col(run, 1)):.0f} %, peak {max(col(run, 1)):.0f} %; "
      f"power mean {mean(col(run, 5)):.0f} W, peak {max(col(run, 5)):.0f} W; temperature peak {max(col(run, 6)):.0f} C")
procs = {}
for r in csv.reader(open(f"{out}_apps.csv")):
    if len(r) >= 4 and r[3].strip().isdigit():
        key = (r[1].strip(), r[2].strip())
        procs[key] = max(procs.get(key, 0), int(r[3]))
for (pid, name), mem in procs.items():
    print(f"process {pid} {name.split('/')[-1]}: peak {mem} MiB")
EOF
exit $RC
