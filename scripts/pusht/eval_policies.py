"""Evaluate PushT policies on fixed random start poses with the success rule of the PushT evaluation pipeline
(Mohammed0Saeed/trust-region-irl main f49513b, experiments/evaluation/env_policy.py).

Start poses: the environment's random sampler (random_TPos: the T within --spawn-radius of the goal, any yaw, clear of the
end effector), one per key of jax.random.split(PRNGKey(--seed), --n-eval). With --seed 0 --n-eval 10 these are the 10 poses
of that pipeline; every checkpoint sees the same poses. A rollout ends at its first success (position error and
orientation error both below --threshold, the environment's own termination rule), when the block leaves the table or the
state turns NaN ("diverged"), or at the horizon ("timeout").

Per checkpoint: success rate, final position and orientation error, intersection over union of the block T and the goal
T (recorded only, it does not decide success), steps to success. Writes to OUT_DIR: metrics.json, initial_poses.csv,
policy_results_<label>.csv (one row per rollout, the columns of the pipeline's policy_results.csv), rollout_<label>.npz,
paths.png (block and end-effector paths of --n-show rollouts per checkpoint) and success.png (share of rollouts that have
succeeded, over time).

Any saved model works (TRIRL, AIRL, PPO: deterministic mean action; Diffusion Policy: closed-loop action chunks with
sampling noise drawn from --seed, so the result is reproducible). Each checkpoint is rolled out twice on the same poses:
the first run includes the JIT compilation, the second is timed while nvidia-smi records GPU use (gpu_<label>.csv;
summary under "compute" in metrics.json; set XLA_PYTHON_CLIENT_PREALLOCATE=false).

    python scripts/pusht/eval_policies.py OUT_DIR CKPT [CKPT ...] [--n-eval 512] [--seed 0] [--spawn-radius 0.2] [--impl jax]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.environments.pusht_mjx.iou import iou, planar_yaw, t_rectangles
from trust_region_irl.policy_checkpoint import load_actor
from trust_region_irl.gpu_usage import GpuUsage

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("ckpts", nargs="+", help="model archives (e.g. runs/.../models/latest.model or latest.model.zip)")
parser.add_argument("--n-eval", type=int, default=512, help="number of random start poses")
parser.add_argument("--n-show", type=int, default=8, help="rollouts drawn per checkpoint")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--spawn-radius", type=float, default=0.20, help="m, maximum distance of the T origin from the goal")
parser.add_argument("--threshold", type=float, default=0.05, help="success: position error (m) and orientation error (rad) both below this")
parser.add_argument("--impl", default="jax", help="MJX backend: jax | warp")
args = parser.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
N = args.n_eval
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C_GOAL, C_BLOCK, C_EE, C_FAIL = "#9a9994", "#2a78d6", "#eb6834", "#e34948"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

env = PushT(render=False, random_TPos=True, spawn_radius=args.spawn_radius, success_threshold=args.threshold,
            impl=args.impl, nr_envs=N)
H = env.horizon
OBS_DIM, ACT_DIM = env.single_observation_space.shape[0], env.single_action_space.shape[0]


def make_rollout(actor):
    @jax.jit
    def rollout(params, keys, actor_key):
        s0 = env.reset(keys, True)
        a0 = actor.init(s0.next_observation, actor_key)
        def step(carry, t):
            s, a = carry
            action, a = actor.act(params, a, s.next_observation, t, s.terminated | s.truncated)
            s2 = env.step(s, action)
            # actual_next_observation and the error info still describe the state reached on the step an episode ends
            rec = dict(block=s2.actual_next_observation[:, 0:7], ee=s2.actual_next_observation[:, 7:10],
                       pos_err=s2.info["env_info/pos_err"], orient_err=s2.info["env_info/orient_err"],
                       terminated=s2.terminated, truncated=s2.truncated)
            return (s2, a), rec
        return s0.next_observation, s0.data.qpos[:, 0:7], jax.lax.scan(step, (s0, a0), jnp.arange(H))[1]
    return rollout


def label(ckpt):
    p = Path(ckpt)
    return f"{p.parent.parent.name}/{p.name.split('.')[0]}"   # <run>/<best|latest>


def episode_outcomes(rec):
    """First end of every rollout: step index, status, final errors and IoU. rec[k]: (H, N, ...)."""
    ended = rec["terminated"] | rec["truncated"]
    end = ended.argmax(0)                                   # every rollout ends at the latest on the horizon step
    i = np.arange(N)
    success = rec["terminated"][end, i]
    status = np.where(success, "success", np.where(end + 1 >= H, "timeout", "diverged"))
    pos_err, orient_err = rec["pos_err"][end, i], rec["orient_err"][end, i]
    block = rec["block"][end, i]
    ious = np.array([iou(b[0:2], b[3:7]) for b in block])
    return dict(end=end, steps=end + 1, status=status, success=success, pos_err=pos_err, orient_err=orient_err, iou=ious)


keys = jax.random.split(jax.random.PRNGKey(args.seed), N)
actor_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)   # Diffusion Policy's sampling noise, same for every checkpoint
results, runs, poses = {}, [], None
for ckpt in args.ckpts:
    actor, params, algorithm = load_actor(ckpt, OBS_DIM, ACT_DIM)
    rollout = make_rollout(actor)
    t0 = time.time()
    first = jax.block_until_ready(rollout(params, keys, actor_key))                       # includes the JIT compilation
    compile_and_run_s = time.time() - t0
    with GpuUsage(OUT / f"gpu_{label(ckpt).replace('/', '_')}.csv") as gpu:            # timed run, same poses and noise
        obs0, qpos0, rec = jax.block_until_ready(rollout(params, keys, actor_key))
    pairs = list(zip(jax.tree.leaves(first), jax.tree.leaves((obs0, qpos0, rec))))
    reproducible = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in pairs)
    max_diff_between_runs = max(float(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)).max()) for x, y in pairs)
    compute = dict(algorithm=algorithm, n_episodes=N, steps=H, first_run_incl_compile_s=compile_and_run_s,
                   rollout_s=gpu.summary["wall_s"], env_steps_per_s=N * H / gpu.summary["wall_s"],
                   ms_per_control_step=1000 * gpu.summary["wall_s"] / H, reproducible=reproducible,
                   max_abs_diff_between_runs=max_diff_between_runs,
                   backend=jax.default_backend(), **{k: v for k, v in gpu.summary.items() if k != "wall_s"})
    rec = {k: np.asarray(v) for k, v in rec.items()}       # rec[k]: (H, N, ...)
    qpos0 = np.asarray(qpos0)
    start = np.stack([qpos0[:, 0], qpos0[:, 1], np.mod(2.0 * np.arctan2(qpos0[:, 6], qpos0[:, 3]), 2.0 * np.pi)], axis=1)
    if poses is None:
        poses = start
        with open(OUT / "initial_poses.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rollout", "block_x_m", "block_y_m", "block_angle_rad"])
            for k, (x, y, a) in enumerate(poses):
                w.writerow([k + 1, f"{x:.5f}", f"{y:.5f}", f"{a:.5f}"])
    assert np.allclose(start, poses, atol=1e-6), "start poses differ between checkpoints"

    o = episode_outcomes(rec)
    succ = o["success"]
    m = dict(success=float(succ.mean()), diverged=float((o["status"] == "diverged").mean()),
             timeout=float((o["status"] == "timeout").mean()),
             steps_to_success_mean=float(o["steps"][succ].mean()) if succ.any() else None,
             final_pos_err_mean=float(o["pos_err"].mean()), final_orient_err_mean=float(o["orient_err"].mean()),
             final_total_err_mean=float((o["pos_err"] + o["orient_err"]).mean()), final_iou_mean=float(o["iou"].mean()),
             n=N, seed=args.seed, spawn_radius=args.spawn_radius, threshold=args.threshold, horizon=H)
    m["compute"] = compute
    name = label(ckpt); runs.append(name); results[name] = m
    print(name, json.dumps({k: v for k, v in m.items() if k != "compute"}), flush=True)
    with open(OUT / f"policy_results_{name.replace('/', '_')}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pose", "block_x_m", "block_y_m", "block_angle_rad", "pos_err", "orient_err", "iou", "success", "status", "steps"])
        for k in range(N):
            w.writerow([k + 1, *poses[k], o["pos_err"][k], o["orient_err"][k], o["iou"][k], int(succ[k]), o["status"][k], o["steps"][k]])
    np.savez_compressed(OUT / f"rollout_{name.replace('/', '_')}.npz", start_obs=np.asarray(obs0), block=rec["block"], ee=rec["ee"],
                        end=o["end"], success=succ, iou=o["iou"])
json.dump(results, open(OUT / "metrics.json", "w"), indent=1)


def draw_t(ax, x, y, yaw, **kw):
    for corners in t_rectangles(x, y, yaw):
        ax.add_patch(plt.Polygon(corners * 100, closed=True, fill=False, **kw))


fig, axes = plt.subplots(len(runs), args.n_show, figsize=(2.3 * args.n_show, 2.5 * len(runs) + 0.6), facecolor=SURF, squeeze=False)
for i, name in enumerate(runs):
    d = np.load(OUT / f"rollout_{name.replace('/', '_')}.npz")
    for j in range(min(args.n_show, N)):
        ax = axes[i, j]; ax.set_facecolor(SURF)
        e = int(d["end"][j]); block, ee = d["block"][:e + 1, j], d["ee"][:e + 1, j]
        s0 = d["start_obs"][j]
        draw_t(ax, 0.0, 0.0, 0.0, lw=1.2, color=C_GOAL)
        draw_t(ax, s0[0], s0[1], planar_yaw(s0[3:7]), lw=0.8, ls=":", color=INK2)
        ok = bool(d["success"][j])
        draw_t(ax, block[-1, 0], block[-1, 1], planar_yaw(block[-1, 3:7]), lw=1.0, color=C_BLOCK if ok else C_FAIL)
        ax.plot(np.r_[s0[7], ee[:, 0]] * 100, np.r_[s0[8], ee[:, 1]] * 100, color=C_EE, lw=0.7)
        ax.plot(np.r_[s0[0], block[:, 0]] * 100, np.r_[s0[1], block[:, 1]] * 100, color=C_BLOCK, lw=1.2)
        ax.set_xlim(-32, 32); ax.set_ylim(-32, 32); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d8d7d2")
        ax.set_title(f"{'success' if ok else 'fail'} @ {e + 1}  IoU {d['iou'][j]:.2f}", fontsize=7, color=INK2)
        if j == 0:
            ax.set_ylabel(name, fontsize=8, color=INK)
handles = [plt.Line2D([], [], color=C_GOAL, lw=2, label="goal T"), plt.Line2D([], [], color=INK2, ls=":", label="start T"),
           plt.Line2D([], [], color=C_BLOCK, lw=2, label="block path / final T (success)"),
           plt.Line2D([], [], color=C_FAIL, lw=2, label="final T (failure)"), plt.Line2D([], [], color=C_EE, lw=1, label="end-effector path")]
fig.suptitle(f"PushT rollouts until the first success (goal-centred cm, threshold {args.threshold})", fontsize=11, color=INK, y=0.995)
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "paths.png", dpi=130, facecolor=SURF); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURF); ax.set_facecolor(SURF)
t = np.arange(1, H + 1) / env.CONTROL_FREQ
for k, name in enumerate(runs):
    d = np.load(OUT / f"rollout_{name.replace('/', '_')}.npz")
    done_by = (d["success"][None, :] & (d["end"][None, :] <= np.arange(H)[:, None])).mean(1)
    ax.plot(t, done_by, color=SERIES[k % len(SERIES)], lw=2, label=f"{name} ({results[name]['success']:.3f})")
ax.set_title(f"share of the {N} rollouts that have reached the goal", fontsize=10, color=INK)
ax.set_xlabel("time (s, 20 Hz control)"); ax.set_ylabel("success"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.25)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "success.png", dpi=130, facecolor=SURF); plt.close(fig)
print("wrote", OUT)
