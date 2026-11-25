"""Inference cost of the policy alone, without the simulator: the evaluation scripts measure policy + MJX together.

For each checkpoint, the policy produces actions for N observations (the first observations of N episodes of the
environment) for STEPS control steps, through the same actor as the evaluation scripts (Diffusion Policy: a new
action chunk every n_action_steps steps, so its cost is averaged over the chunk). The first run includes the JIT
compilation; the second is timed while nvidia-smi records GPU use. Writes OUT_DIR/bench_policy.json and gpu_*.csv.

    python scripts/gpu_cost/bench_policy.py OUT_DIR {pizza|balance|pusht} CKPT [CKPT ...] [--n-envs 512] [--steps 800]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from trust_region_irl.policy_checkpoint import load_actor
from trust_region_irl.gpu_usage import GpuUsage

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("env", choices=["pizza", "balance", "pusht"])
parser.add_argument("ckpts", nargs="+")
parser.add_argument("--n-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=None, help="control steps (default: the environment's horizon)")
parser.add_argument("--impl", default="jax")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--min-seconds", type=float, default=5.0, help="repeat the timed run until it lasts this long, so nvidia-smi gets enough samples")
args = parser.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)

if args.env == "pizza":
    from trust_region_irl.environments.pizza_dr_mjx.environment import PizzaSpreadDR
    env = PizzaSpreadDR(render=False, nr_envs=args.n_envs, impl=args.impl)
elif args.env == "balance":
    from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3
    env = BalanceFR3(render=False, nr_envs=args.n_envs, impl=args.impl)
else:
    from trust_region_irl.environments.pusht_mjx.environment import PushT
    env = PushT(render=False, random_TPos=True, spawn_radius=0.20, nr_envs=args.n_envs, impl=args.impl)
steps = args.steps or env.horizon
obs = jax.block_until_ready(env.reset(jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs), True).next_observation)
no_reset = jnp.zeros(args.n_envs, bool)


def label(ckpt):
    p = Path(ckpt)
    return f"{p.parent.parent.name}/{p.name.split('.')[0]}"


results = {}
for ckpt in args.ckpts:
    actor, params, algorithm = load_actor(ckpt, obs.shape[-1], env.single_action_space.shape[0])

    @jax.jit
    def run(params, key):
        def step(a, t):
            action, a = actor.act(params, a, obs, t, no_reset)
            return a, action[0]
        return jax.lax.scan(step, actor.init(obs, key), jnp.arange(steps))[1]

    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    t0 = time.time(); jax.block_until_ready(run(params, key)); first = time.time() - t0
    t0 = time.time(); jax.block_until_ready(run(params, key)); once = time.time() - t0
    repeats = max(1, int(np.ceil(args.min_seconds / max(once, 1e-6))))
    with GpuUsage(OUT / f"gpu_{label(ckpt).replace('/', '_')}.csv") as gpu:
        for _ in range(repeats):
            jax.block_until_ready(run(params, key))
    s = gpu.summary
    per_run = s["wall_s"] / repeats
    results[label(ckpt)] = dict(algorithm=algorithm, env=args.env, n_envs=args.n_envs, steps=steps, repeats=repeats, backend=jax.default_backend(),
                                first_run_incl_compile_s=first, run_s=per_run, ms_per_control_step=1000 * per_run / steps,
                                actions_per_s=args.n_envs * steps / per_run, **{k: v for k, v in s.items() if k != "wall_s"})
    print(label(ckpt), json.dumps(results[label(ckpt)]), flush=True)
json.dump(results, open(OUT / "bench_policy.json", "w"), indent=1)
