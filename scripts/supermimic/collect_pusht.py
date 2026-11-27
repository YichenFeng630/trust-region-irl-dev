"""Record PushT transitions for the offline world-model checks (plan step 1, supermimic_trirl/PLAN_zh.md).

Rolls out a Gaussian policy in pusht_mjx with random start poses (Mohammed's setting: within 0.20 m, success threshold
0.05) and saves time-major arrays (T, N, ...) to OUT.npz: state, next_state (the world model's 30-dim state:
observation + the T's velocity), qpos, qvel (physics before each step, to restore it later), action, done, terminated,
diverged; plus meta (policy, std, seed).

    python scripts/supermimic/collect_pusht.py OUT.npz --policy init [--std 0.4]
    python scripts/supermimic/collect_pusht.py OUT.npz --policy runs/.../models/best.model.zip [--std 0.2]
"""
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp

from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.pusht_sim import record_rollout
from trust_region_irl.policy_checkpoint import load_policy_params

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--policy", default="init", help="init (a freshly initialised TRIRL policy) or a model archive")
parser.add_argument("--std", type=float, default=None, help="action noise std (default: 0.4 for init, the policy's own otherwise)")
parser.add_argument("--nr-envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--spawn-radius", type=float, default=0.20)
args = parser.parse_args()

env = PushT(render=False, random_TPos=True, spawn_radius=args.spawn_radius, success_threshold=args.threshold,
            nr_envs=args.nr_envs)
policy = Policy(env.single_action_space.shape, 0.4, jnp.arange(env.single_observation_space.shape[0]))
if args.policy == "init":
    params = policy.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, env.single_observation_space.shape[0])))
    std = 0.4 if args.std is None else args.std
else:
    params = jax.tree.map(jnp.asarray, load_policy_params(args.policy))
    std = args.std
own_std = float(np.exp(np.asarray(params["params"]["policy_logstd"])).mean())
print(f"policy {args.policy}: own std {own_std:.3f}, rollout std {std if std is not None else own_std:.3f}", flush=True)

data = record_rollout(env, policy, params, None if std is None else jnp.float32(std), args.nr_envs, args.steps,
                      jax.random.PRNGKey(args.seed + 1))
meta = dict(policy=args.policy, std=std if std is not None else own_std, own_std=own_std, seed=args.seed,
            nr_envs=args.nr_envs, steps=args.steps, threshold=args.threshold, spawn_radius=args.spawn_radius)
np.savez(args.out, **data, meta=json.dumps(meta))

T_moving = np.linalg.norm(data["state"][..., 24:27], axis=-1) > 0.005
print(json.dumps({**meta, "episodes_ended": int(data["done"].sum()), "successes": int(data["terminated"].sum()),
                  "diverged": int(data["diverged"].sum()), "share_steps_T_moving": float(T_moving.mean())}), flush=True)
