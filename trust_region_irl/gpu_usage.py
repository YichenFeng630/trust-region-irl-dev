"""Record GPU use with nvidia-smi while a block of code runs.

    with GpuUsage("out/gpu_rollout.csv") as gpu:
        run()
    print(gpu.summary)

nvidia-smi samples the GPU this process uses (the first entry of CUDA_VISIBLE_DEVICES, else GPU 0) every
`interval_ms` into a CSV: utilisation, memory, power, temperature. The summary covers the samples inside the block:
wall time, mean / peak utilisation, peak memory of the whole GPU, mean / peak power, energy (power integrated over
time), the GPU model and driver, and the XLA settings. Memory of this process:
  process_memory_mib            what nvidia-smi reports for this process at the end of the block: JAX's memory pool,
                                which only grows, so it is the high-water mark of the process so far
  jax_peak_bytes_in_use_mib     JAX's own peak of memory actually in use since the process started, closer to what
                                is really needed
Set XLA_PYTHON_CLIENT_PREALLOCATE=false, otherwise the pool is 75 % of the GPU from the start.
Without nvidia-smi the block runs and only the wall time is recorded.
"""
import csv
import datetime
import os
import shutil
import subprocess
import time

FIELDS = ("timestamp", "utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw", "temperature.gpu")


def _gpu_id():
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    return ids[0].strip() if ids and ids[0].strip() else "0"


def _query(fields, gpu):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", gpu],
                         capture_output=True, text=True, check=True).stdout
    return [x.strip() for x in out.strip().split(",")]


def _jax_peak_bytes():
    try:
        import jax
        dev = jax.devices()[0]
        if dev.platform != "gpu":
            return None
        stats = dev.memory_stats() or {}
        return stats.get("peak_bytes_in_use")
    except Exception:
        return None


def _process_memory_mib(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "-i", gpu],
                             capture_output=True, text=True, check=True).stdout
        for line in out.strip().splitlines():
            pid, mem = (x.strip() for x in line.split(","))
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:
        return None
    return None


class GpuUsage:
    def __init__(self, csv_path, interval_ms=200):
        self.csv_path, self.interval_ms = str(csv_path), interval_ms
        self.available = shutil.which("nvidia-smi") is not None
        self.gpu = _gpu_id()
        self.summary = {"xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true")}

    def __enter__(self):
        if self.available:
            name, driver, total = _query("name,driver_version,memory.total", self.gpu)
            self.summary.update(gpu_name=name, gpu_driver=driver, gpu_memory_total_mib=float(total),
                                gpu_memory_used_before_mib=float(_query("memory.used", self.gpu)[0]))
            self._file = open(self.csv_path, "w")
            self._file.write(",".join(FIELDS) + "\n"); self._file.flush()
            self._proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits",
                                           "-i", self.gpu, "-lms", str(self.interval_ms)], stdout=self._file, stderr=subprocess.DEVNULL)
            time.sleep(self.interval_ms / 1000)   # let the first sample land before the block starts
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.t1 = time.time()
        self.summary["wall_s"] = self.t1 - self.t0
        if not self.available:
            return False
        time.sleep(self.interval_ms / 1000)
        self._proc.terminate(); self._proc.wait(); self._file.close()
        rows = []
        with open(self.csv_path) as f:
            for r in csv.DictReader(f):
                try:
                    ts = datetime.datetime.strptime(r["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                    rows.append((ts, float(r["utilization.gpu"]), float(r["memory.used"]), float(r["power.draw"]), float(r["temperature.gpu"])))
                except (ValueError, KeyError):
                    continue
        inside = [r for r in rows if self.t0 <= r[0] <= self.t1]
        if inside:
            mean = lambda i: sum(r[i] for r in inside) / len(inside)
            energy_j = sum((b[0] - a[0]) * (a[3] + b[3]) / 2 for a, b in zip(inside, inside[1:]))
            self.summary.update(samples=len(inside), utilization_mean_pct=mean(1), utilization_peak_pct=max(r[1] for r in inside),
                                gpu_memory_used_peak_mib=max(r[2] for r in inside), power_mean_w=mean(3), power_peak_w=max(r[3] for r in inside), energy_wh=energy_j / 3600,
                                temperature_peak_c=max(r[4] for r in inside))
        own = _process_memory_mib(self.gpu)
        if own is not None:
            self.summary["process_memory_mib"] = own
        peak = _jax_peak_bytes()
        if peak is not None:
            self.summary["jax_peak_bytes_in_use_mib"] = peak / 2**20
        return False
