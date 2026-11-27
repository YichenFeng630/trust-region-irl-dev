"""Fit the SuperMimic world model offline on recorded PushT transitions and measure its prediction error (plan step 1 (a),
supermimic_trirl/PLAN_zh.md).

Training: the online fit's settings (k-step autoregressive windows, batch 2048, Adam 2e-4 with cosine decay to 0.03x,
global-norm clip 1.0) on the training environments of every data file (collect_pusht.py output); the last --holdout
share of each file's environments is held out.

Evaluation on the held-out environments: windows of 32 steps (one start every 8 steps), the model rolled out on the
logged actions from the logged start state. Per horizon j in 1..32: the T's position error (m) and angle error (rad),
the tool position error (m) and the mean arm joint error (rad), split by whether the T moved within the first j steps
(more than 2 mm or 0.02 rad). Two simple predictions for comparison: "unchanged" (the start state) and "constant
velocity" (the T's last-step change continued). Registered rule (plan step 1): on windows where the T moved, the
model's mean 8-step T position error and angle error are each at most half of the better simple prediction's.

    python scripts/supermimic/wm_offline.py OUT_DIR DATA.npz [DATA.npz ...] [--steps 20000]

Writes OUT_DIR/model.msgpack (params and normaliser, for grad_direction.py), config.json, metrics.json (errors per
horizon, the rule check, the training loss), errors.png.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit import pusht_model as pm
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.world_model import WorldModel
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.replay_buffer import ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("data", nargs="+")
parser.add_argument("--steps", type=int, default=20000, help="gradient steps")
parser.add_argument("--hidden", default="512,512,512")
parser.add_argument("--k", type=int, default=8, help="window length of the fit")
parser.add_argument("--batch", type=int, default=2048)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--holdout", type=float, default=0.1, help="share of each file's environments held out")
parser.add_argument("--eval-horizon", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
MOVE_POS, MOVE_ANG = 0.002, 0.02
L = args.eval_horizon


def load(path):
    d = np.load(path)
    return {k: d[k] for k in ("state", "next_state", "qpos", "qvel", "action", "done", "terminated", "diverged")}


def split(d, holdout):
    n = d["state"].shape[1]
    n_hold = max(1, int(np.ceil(holdout * n)))
    return ({k: v[:, :n - n_hold] for k, v in d.items()}, {k: v[:, n - n_hold:] for k, v in d.items()})


files = [load(p) for p in args.data]
T = min(f["state"].shape[0] for f in files)
files = [{k: v[:T] for k, v in f.items()} for f in files]
parts = [split(f, args.holdout) for f in files]
train = {k: np.concatenate([p[0][k] for p in parts], axis=1) for k in files[0]}
held = {k: np.concatenate([p[1][k] for p in parts], axis=1) for k in files[0]}
held_counts = [p[1]["state"].shape[1] for p in parts]
print(f"T {T}, train envs {train['state'].shape[1]}, held-out envs {held['state'].shape[1]} ({held_counts})", flush=True)

wm = WorldModel(pm.STATE_DIM, 2, args.hidden, quat=pm.QUAT, velocity_dims=pm.VELOCITY_DIMS)
f32 = lambda x: jnp.asarray(x, jnp.float32)
bad, done = f32(train["diverged"]), f32(train["done"])
buffer = ReplayBuffer.create(T, train["state"].shape[1], pm.STATE_DIM, 2).add(
    f32(train["state"]), f32(train["action"]), f32(train["next_state"]), done, bad)
norm = wm.update_normalizer(wm.init_normalizer(), f32(train["state"]).reshape(-1, pm.STATE_DIM),
                            f32(train["next_state"]).reshape(-1, pm.STATE_DIM), (1 - bad).reshape(-1), (1 - done).reshape(-1))
params = wm.init(jax.random.PRNGKey(args.seed))
schedule = optax.cosine_decay_schedule(args.lr, int(0.78 * args.steps), alpha=0.03)
tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
opt_state = tx.init(params)


@jax.jit
def train_step(params, opt_state, key, buffer, norm):
    s0, a, tgt, valid, vel_valid = buffer.sample_windows(key, args.batch, args.k)
    (loss, _), g = jax.value_and_grad(wm.loss, has_aux=True)(params, norm, s0, a, tgt, valid, vel_valid)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss


def quat_angle(q1, q2):
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0))


def eval_windows(d):
    """Held-out windows of L steps from start t0 = 1, 9, 17, ...: start state, previous state, actions, targets, masks."""
    starts = np.arange(1, d["state"].shape[0] - L + 1, 8)
    n = d["state"].shape[1]
    t0 = np.repeat(starts, n); e = np.tile(np.arange(n), len(starts))
    idx = t0[:, None] + np.arange(L)[None, :]
    s0, prev = d["state"][t0, e], d["state"][t0 - 1, e]
    prev_ok = d["done"][t0 - 1, e] < 0.5                       # the previous step is in the same episode
    actions, targets = d["action"][idx, e[:, None]], d["next_state"][idx, e[:, None]]
    dn, dv = d["done"][idx, e[:, None]].astype(float), d["diverged"][idx, e[:, None]].astype(float)
    ended_before = np.concatenate([np.zeros((len(t0), 1)), np.cumsum(dn, axis=1)[:, :-1]], axis=1) > 0
    valid = (~ended_before) & (dv < 0.5)
    return s0, prev, prev_ok, actions, targets, valid


rollout = jax.jit(jax.vmap(wm.rollout, in_axes=(None, None, 0, 0)))


def predict(params, s0, actions, chunk=8192):
    return np.concatenate([np.asarray(rollout(params, norm, f32(s0[i:i + chunk]), f32(actions[i:i + chunk])))
                           for i in range(0, len(s0), chunk)])


def errors(pred, targets):
    return {"T_pos": np.linalg.norm(pred[..., 0:3] - targets[..., 0:3], axis=-1),
            "T_angle": quat_angle(pred[..., 3:7], targets[..., 3:7]),
            "tool_pos": np.linalg.norm(pred[..., 7:10] - targets[..., 7:10], axis=-1),
            "arm_q": np.mean(np.abs(pred[..., 10:17] - targets[..., 10:17]), axis=-1)}


def simple_predictions(s0, prev, prev_ok):
    """(L, ...) predictions of the T pose: unchanged, and the last step's change continued (zero if no previous step)."""
    j = np.arange(1, L + 1)[None, :, None]
    p0, pp = s0[:, None, 0:3], np.where(prev_ok[:, None], prev[:, 0:3], s0[:, 0:3])[:, None]
    unchanged = {"pos": np.repeat(p0, L, axis=1), "quat": np.repeat(s0[:, None, 3:7], L, axis=1)}
    to_xyzw = lambda q: q[..., [1, 2, 3, 0]]
    r0 = R.from_quat(to_xyzw(s0[:, 3:7]))
    rp = R.from_quat(to_xyzw(np.where(prev_ok[:, None], prev[:, 3:7], s0[:, 3:7])))
    step_rotvec = (r0 * rp.inv()).as_rotvec()
    quats = np.stack([(R.from_rotvec(step_rotvec * k) * r0).as_quat()[:, [3, 0, 1, 2]] for k in range(1, L + 1)], axis=1)
    const_vel = {"pos": p0 + j * (p0 - pp), "quat": quats}
    return {"unchanged": unchanged, "constant_velocity": const_vel}


def evaluate(params, d):
    s0, prev, prev_ok, actions, targets, valid = eval_windows(d)
    pred = predict(params, s0, actions)
    err = errors(pred, targets)
    simple = simple_predictions(s0, prev, prev_ok)
    simple_err = {name: {"T_pos": np.linalg.norm(p["pos"] - targets[..., 0:3], axis=-1),
                         "T_angle": quat_angle(p["quat"], targets[..., 3:7])} for name, p in simple.items()}
    # did the T move within the first j steps (valid steps only)?
    disp = np.linalg.norm(targets[..., 0:3] - s0[:, None, 0:3], axis=-1)
    rot = quat_angle(targets[..., 3:7], s0[:, None, 3:7])
    moved_step = valid & ((disp > MOVE_POS) | (rot > MOVE_ANG))
    moved_by = np.cumsum(moved_step, axis=1) > 0
    out = {}
    for j in range(1, L + 1):
        v = valid[:, j - 1]
        row = {}
        for cat, sel in (("moved", v & moved_by[:, j - 1]), ("still", v & ~moved_by[:, j - 1])):
            stats = {"n": int(sel.sum())}
            for name, e in [("model", err)] + list(simple_err.items()):
                for q in e:
                    vals = e[q][sel, j - 1]
                    stats[f"{name}/{q}_mean"] = float(vals.mean()) if len(vals) else float("nan")
                    stats[f"{name}/{q}_median"] = float(np.median(vals)) if len(vals) else float("nan")
            row[cat] = stats
        out[j] = row
    return out


log, key, t_start = [], jax.random.PRNGKey(args.seed + 1), time.time()
for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    params, opt_state, loss = train_step(params, opt_state, sub, buffer, norm)
    if step % 100 == 0 or step == 1:
        log.append((step, float(loss)))
    if step % 2000 == 0 or step == args.steps:
        e8 = evaluate(params, held)[8]["moved"]
        print(f"step {step}: loss {float(loss):.4f}, held-out moved windows 8-step T pos {e8['model/T_pos_mean'] * 1000:.2f} mm, "
              f"angle {e8['model/T_angle_mean']:.4f} rad ({time.time() - t_start:.0f} s)", flush=True)

res = evaluate(params, held)
m8 = res[8]["moved"]
rule = {}
for q in ("T_pos", "T_angle"):
    best_simple = min(m8[f"unchanged/{q}_mean"], m8[f"constant_velocity/{q}_mean"])
    rule[q] = {"model": m8[f"model/{q}_mean"], "better_simple": best_simple, "pass": bool(m8[f"model/{q}_mean"] <= 0.5 * best_simple)}
rule["pass"] = rule["T_pos"]["pass"] and rule["T_angle"]["pass"]

(OUT / "model.msgpack").write_bytes(serialization.to_bytes({"params": params, "norm": norm}))
json.dump({"hidden": args.hidden, "k": args.k, "data": args.data, "T": T, "holdout": args.holdout, "held_out_envs": held_counts,
           "steps": args.steps, "batch": args.batch, "lr": args.lr, "seed": args.seed}, open(OUT / "config.json", "w"), indent=1)
json.dump({"rule_8_step_moved": rule, "errors": {str(j): r for j, r in res.items()}, "train_loss": log,
           "train_seconds": time.time() - t_start}, open(OUT / "metrics.json", "w"), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
js = np.arange(1, L + 1)
for ax, q, unit, scale in ((axes[0], "T_pos", "mm", 1000), (axes[1], "T_angle", "rad", 1)):
    for name, style in (("model", "-"), ("unchanged", ":"), ("constant_velocity", "--")):
        ax.plot(js, [res[j]["moved"][f"{name}/{q}_mean"] * scale for j in js], style, label=name.replace("_", " "))
    ax.set_xlabel("prediction horizon (steps of 0.05 s)"); ax.set_ylabel(f"{q.replace('_', ' ')} error ({unit})")
    ax.set_title("held-out windows where the T moved")
axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "errors.png", dpi=130)
print(json.dumps({"rule_8_step_moved": rule, "n_moved_8": m8["n"], "n_still_8": res[8]["still"]["n"]}, indent=1), flush=True)
