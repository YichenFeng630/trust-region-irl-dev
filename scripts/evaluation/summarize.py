"""Summarise evaluation results over seeds: one row per method, the value per seed, mean and standard deviation.

Reads the metrics.json files written by scripts/pizza/plot_policy_paths.py and scripts/balance/eval_policies.py.
Their entries are named "<run>/<latest|best>"; the method and the seed are read from the run name, "<method>_seeds<N>"
(also "<method>_s<N>"), e.g. DP_seeds0, AIRL_seeds1. Writes OUT.md (a table) and OUT.csv (one row per run).

    python scripts/evaluation/summarize.py OUT metrics.json [metrics.json ...] [--checkpoint latest] [--keys k1 k2 ...]
"""
import argparse
import csv
import json
import re
from collections import defaultdict

import numpy as np

PIZZA_KEYS = ["coverage_with_cut", "success_with_cut", "coverage_final", "success_final", "clean_finish", "would_be_cut",
              "stop_action_norm_frac_below_005", "compute.ms_per_control_step", "compute.jax_peak_bytes_in_use_mib",
              "compute.utilization_mean_pct", "compute.energy_wh"]
BALANCE_KEYS = ["all.goal_dist_cm_not_lost", "all.goal_dist_2nd_half_cm_not_lost", "all.ball_lost", "all.goal_dist_cm",
                "compute.ms_per_control_step", "compute.jax_peak_bytes_in_use_mib", "compute.utilization_mean_pct", "compute.energy_wh"]
RUN = re.compile(r"^(?P<method>.+?)_s(?:eeds?)?(?P<seed>\d+)$")

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("metrics", nargs="+")
parser.add_argument("--checkpoint", default="latest", help="latest | best: which checkpoint of each run to report")
parser.add_argument("--keys", nargs="*", default=None, help="metric keys, nested with '.' (default: by environment)")
args = parser.parse_args()


def get(d, key):
    for part in key.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


rows = []
for path in args.metrics:
    for name, m in json.load(open(path)).items():
        if "/" not in name or not name.endswith("/" + args.checkpoint):
            continue                                   # skips "dynamics", "zero action", "expert" and the other checkpoint
        run = name.split("/")[0]
        hit = RUN.match(run)
        method, seed = (hit["method"], int(hit["seed"])) if hit else (run, None)
        rows.append(dict(method=method, seed=seed, run=run, file=path, metrics=m))
if not rows:
    raise SystemExit(f"no '<run>/{args.checkpoint}' entries in {args.metrics}")
keys = args.keys or (BALANCE_KEYS if "all" in rows[0]["metrics"] else PIZZA_KEYS)

with open(f"{args.out}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "seed", "run", "file"] + keys)
    for r in sorted(rows, key=lambda r: (r["method"], r["seed"] if r["seed"] is not None else -1)):
        w.writerow([r["method"], r["seed"], r["run"], r["file"]] + [get(r["metrics"], k) for k in keys])

by_method = defaultdict(list)
for r in rows:
    by_method[r["method"]].append(r)
fmt = lambda x: "" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
lines = [f"checkpoint: {args.checkpoint}; values: mean ± std over seeds (per-seed values in {args.out}.csv)", "",
         "| method | seeds | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
for method, rs in sorted(by_method.items()):
    cells = []
    for k in keys:
        v = [get(r["metrics"], k) for r in rs]
        v = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
        cells.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else fmt(float(v[0])) if v else "")
    seeds = ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"] if r["seed"] is not None else -1))
    lines.append(f"| {method} | {seeds} | " + " | ".join(cells) + " |")
open(f"{args.out}.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
