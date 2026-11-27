"""Does the world model's gradient point the right way in the real simulation? (plan step 1 (b); with --reward trirl
also step 2.5 (a), supermimic_trirl/PLAN_zh.md)

From held-out recorded states (the environments wm_offline.py held out), take the logged next H actions a (H, 2). The
model's gradient g = d/da sum_{j=1..H} gamma^(j-1) R(s_j) of the return predicted from the logged start state. In the
real simulation, restored to that start state, run a + delta g/rms(g) and a - delta g/rms(g) (rms over the H x 2
entries, so delta is the typical change per action entry) and compare the real returns: the share of starts where the
"+" sequence is better. A random direction instead of g gives the 50 % reference (ties where the T is never touched).

Starts are split by the logged real trajectory: "pushing" (the T moves at the start: speed > 5 mm/s or 0.05 rad/s),
"first contact" (the T is at rest at the start and moves within the H steps, > 2 mm or 0.02 rad), "no contact" (it does
not move). Registered rule (plan step 1): an H passes if, at delta 0.05, the "+" sequence is better in at least 65 % of
both the first-contact and the pushing starts.

R: --reward env (PushT's -(position error + orientation error)) or --reward trirl --reward-ckpt MODEL (the corrected
reward alpha D_corr(s) of a TRIRL-SPR checkpoint, the reward the PPO retraining uses).

    python scripts/supermimic/grad_direction.py OUT_DIR --model WM_DIR [--H 8,16,32] [--delta 0.02,0.05,0.1] [--n 512]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.stats import spearmanr

from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit import pusht_model as pm
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.world_model import WorldModel
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.pusht_sim import states_from_physics, replay_actions

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("--model", required=True, help="output directory of wm_offline.py")
parser.add_argument("--data", nargs="*", default=None, help="data files (default: the model's)")
parser.add_argument("--H", default="8,16,32")
parser.add_argument("--delta", default="0.02,0.05,0.1")
parser.add_argument("--n", type=int, default=512, help="starts per category and H")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--reward", default="env", choices=["env", "trirl"])
parser.add_argument("--reward-ckpt", default=None)
parser.add_argument("--chunk", type=int, default=2048, help="parallel real rollouts")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
Hs = [int(h) for h in args.H.split(",")]
deltas = [float(d) for d in args.delta.split(",")]
H_MAX = max(Hs)
PUSH_V, PUSH_W, MOVE_POS, MOVE_ANG, RULE_DELTA, RULE_SHARE = 0.005, 0.05, 0.002, 0.02, 0.05, 0.65

# world model
cfg = json.load(open(Path(args.model) / "config.json"))
wm = WorldModel(pm.STATE_DIM, 2, cfg["hidden"], quat=pm.QUAT, velocity_dims=pm.VELOCITY_DIMS)
template = {"params": wm.init(jax.random.PRNGKey(0)), "norm": wm.init_normalizer()}
restored = serialization.from_bytes(template, (Path(args.model) / "model.msgpack").read_bytes())
wm_params, norm = restored["params"], restored["norm"]

# reward
if args.reward == "env":
    reward = pm.env_reward
    reward_info = {"reward": "env"}
else:
    from trust_region_irl.policy_checkpoint import _read_checkpoint
    from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.byol import BYOL
    ckpt, ckpt_cfg = _read_checkpoint(args.reward_ckpt)
    reward_type = ckpt_cfg.get("reward_type", "byol-task-based-observation")
    indices = {"byol-task-based-observation": (0, 1, 2, 3, 4, 5, 6), "byol-embodiment-free": tuple(range(10)),
               "byol-observation-based": ()}[reward_type]
    byol = BYOL(input_indices=indices)
    corr = jax.tree.map(jnp.asarray, ckpt["corrected_reward"]["params"])
    corr = corr if "params" in corr else {"params": corr}
    alpha = float(ckpt_cfg.get("entropy_coef", 0.001))
    reward = lambda s: alpha * byol.apply(corr, pm.observation(s), jnp.zeros(2), method=byol._reward)
    reward_info = {"reward": "trirl", "ckpt": args.reward_ckpt, "reward_type": reward_type, "alpha": alpha}

# held-out environments of the model's data files (the same split as wm_offline.py)
data_files = args.data or cfg["data"]
held = []
for path in data_files:
    d = np.load(path)
    n = d["state"].shape[1]; n_hold = max(1, int(np.ceil(cfg["holdout"] * n)))
    held.append({k: d[k][:cfg["T"], n - n_hold:] for k in ("state", "next_state", "qpos", "qvel", "action", "done", "diverged")})
held = {k: np.concatenate([h[k] for h in held], axis=1) for k in held[0]}
T, N = held["state"].shape[:2]
print(f"held-out: T {T}, envs {N}", flush=True)


def quat_angle(q1, q2):
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0))


# candidate starts: the next H_MAX steps lie in one episode and nothing diverges
t0 = np.repeat(np.arange(T - H_MAX + 1), N); env_ix = np.tile(np.arange(N), T - H_MAX + 1)
idx = t0[:, None] + np.arange(H_MAX)[None, :]
ends = (held["done"][idx, env_ix[:, None]] | held["diverged"][idx, env_ix[:, None]]).any(axis=1)
t0, env_ix, idx = t0[~ends], env_ix[~ends], idx[~ends]
s0 = held["state"][t0, env_ix]
traj = held["next_state"][idx, env_ix[:, None]]                       # logged s_1..s_HMAX
pushing = (np.linalg.norm(s0[:, 24:27], axis=-1) > PUSH_V) | (np.linalg.norm(s0[:, 27:30], axis=-1) > PUSH_W)
moved_step = (np.linalg.norm(traj[..., 0:3] - s0[:, None, 0:3], axis=-1) > MOVE_POS) | (quat_angle(traj[..., 3:7], s0[:, None, 3:7]) > MOVE_ANG)
print(f"candidate starts {len(t0)}; pushing {pushing.mean():.3f}", flush=True)

env = PushT(render=False, random_TPos=True, spawn_radius=0.2, success_threshold=-1.0, nr_envs=args.chunk)
rng = np.random.default_rng(args.seed)


def model_return(a_seq, s, H):
    states = wm.rollout(wm_params, norm, s, a_seq[:H])
    r = jax.vmap(reward)(states)
    return jnp.sum(args.gamma ** jnp.arange(H) * r)


@jax.jit
def _replay_and_score(st, actions):
    full, done = replay_actions(env, st, actions)
    H = actions.shape[1]
    ret = jnp.sum(args.gamma ** jnp.arange(H) * jax.vmap(jax.vmap(reward))(full), axis=1)
    return ret, full, jnp.any(done, axis=1)


def real_returns(qpos, qvel, actions, H):
    """Real returns of action sequences (n, H, 2) from restored states, in chunks; NaN where the episode was cut.
    Also returns the real states (n, H, 30)."""
    rets, states = [], []
    for i in range(0, len(qpos), args.chunk):
        st = states_from_physics(env, qpos[i:i + args.chunk], qvel[i:i + args.chunk], jax.random.PRNGKey(0))
        ret, full, cut = _replay_and_score(st, jnp.asarray(actions[i:i + args.chunk], jnp.float32))
        ret = np.array(ret)
        ret[np.asarray(cut)] = np.nan
        rets.append(ret); states.append(np.asarray(full))
    return np.concatenate(rets), np.concatenate(states)


results, table = {"reward": reward_info, "model": args.model, "data": data_files}, []
for H in Hs:
    moved = moved_step[:, :H].any(axis=1)
    cats = {"pushing": np.where(pushing)[0], "first_contact": np.where(~pushing & moved)[0], "no_contact": np.where(~pushing & ~moved)[0]}
    grad_fn = jax.jit(jax.vmap(jax.value_and_grad(lambda a, s: model_return(a, s, H))))
    pred_fn = jax.jit(jax.vmap(lambda a, s: model_return(a, s, H)))
    for cat, pool in cats.items():
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=min(args.n, len(pool)), replace=False)
        a = held["action"][idx[pick, :H], env_ix[pick][:, None]]
        qpos, qvel = held["qpos"][t0[pick], env_ix[pick]], held["qvel"][t0[pick], env_ix[pick]]
        _, g = grad_fn(jnp.asarray(a, jnp.float32), jnp.asarray(s0[pick], jnp.float32))
        g = np.asarray(g)
        rms = np.sqrt(np.mean(g ** 2, axis=(1, 2)))
        u = rng.normal(size=a.shape)
        u_dir = u / np.sqrt(np.mean(u ** 2, axis=(1, 2)))[:, None, None]
        g_dir = g / np.maximum(rms, 1e-30)[:, None, None]
        # restore check: the logged actions from the restored state against the logged next states
        _, replay = real_returns(qpos, qvel, a, H)
        restore_err = np.linalg.norm(replay[..., 0:3] - traj[pick, :H, 0:3], axis=-1).max(axis=1)
        for delta in deltas:
            seqs = np.concatenate([a + delta * g_dir, a - delta * g_dir, a + delta * u_dir, a - delta * u_dir])
            rets, _ = real_returns(np.tile(qpos, (4, 1)), np.tile(qvel, (4, 1)), seqs, H)
            gp, gm, rp, rm = np.split(rets, 4)
            ok = np.isfinite(gp) & np.isfinite(gm) & (rms > 1e-12)
            okr = np.isfinite(rp) & np.isfinite(rm)
            pred_p = np.asarray(pred_fn(jnp.asarray(a + delta * g_dir, jnp.float32), jnp.asarray(s0[pick], jnp.float32)))
            pred_m = np.asarray(pred_fn(jnp.asarray(a - delta * g_dir, jnp.float32), jnp.asarray(s0[pick], jnp.float32)))
            real_gain, pred_gain = (gp - gm)[ok], (pred_p - pred_m)[ok]
            row = {"H": H, "delta": delta, "category": cat, "n": int(len(pick)), "n_used": int(ok.sum()),
                   "zero_gradient": int((rms <= 1e-12).sum()),
                   "plus_better": float(np.mean(real_gain > 0)) if ok.any() else float("nan"),
                   "tie": float(np.mean(real_gain == 0)) if ok.any() else float("nan"),
                   "minus_better": float(np.mean(real_gain < 0)) if ok.any() else float("nan"),
                   "random_plus_better": float(np.mean((rp - rm)[okr] > 0)) if okr.any() else float("nan"),
                   "random_tie": float(np.mean((rp - rm)[okr] == 0)) if okr.any() else float("nan"),
                   "real_gain_mean": float(real_gain.mean()) if ok.any() else float("nan"),
                   "predicted_gain_mean": float(pred_gain.mean()) if ok.any() else float("nan"),
                   "spearman_predicted_vs_real_gain": float(spearmanr(pred_gain, real_gain)[0]) if ok.sum() > 2 else float("nan"),
                   "restore_T_pos_err_median_m": float(np.median(restore_err)), "restore_T_pos_err_p95_m": float(np.percentile(restore_err, 95))}
            table.append(row)
            print(json.dumps(row), flush=True)

rule = {}
for H in Hs:
    share = {r["category"]: r["plus_better"] for r in table if r["H"] == H and r["delta"] == RULE_DELTA}
    fc, pu = share.get("first_contact", float("nan")), share.get("pushing", float("nan"))
    rule[H] = {"first_contact": fc, "pushing": pu, "pass": bool(fc >= RULE_SHARE and pu >= RULE_SHARE)}
results["rule"] = rule
results["table"] = table
json.dump(results, open(OUT / "metrics.json", "w"), indent=1)
print(json.dumps(rule, indent=1), flush=True)
