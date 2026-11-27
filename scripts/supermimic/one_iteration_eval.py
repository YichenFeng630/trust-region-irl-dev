"""Plan step 2.5 (b), supermimic_trirl/PLAN_zh.md: how far one TRIRL iteration moved the policy, and how much it raised
the return under the frozen reward.

START is the model archive the iteration started from (init_checkpoint of the runs): its policy, and its reward
r_k = alpha [(1 - eps) D_corr + eps beta D_k], the reward both updates optimised (freeze_reward=True). NEW are the
archives written after the iteration (PPO and BPTT runs, latest.model.zip).

- KL(pi_new || pi_start), averaged over the states of a fresh rollout of pi_start (nr-envs x kl-steps, seed --seed,
  not the training seed).
- Return: one full episode from each of nr-envs start poses, sampled actions, the same reset keys and the same noise
  for every policy (common random numbers). Per step gamma^t r_k(o_t); at success also
  gamma^(t+1) (r_k(o_{t+1}) + alpha H_terminal) / (1 - gamma), trirl_ppo_spr's terminal tail; cut at the time limit.
  Reported: mean, the paired difference to START and its standard error, success rate (sampled and mean action).

    python scripts/supermimic/one_iteration_eval.py OUT.json START.zip NEW.zip [NEW.zip ...] --threshold 0.05
"""
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp

from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.discriminator import get_discriminator
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.imagination import gaussian_kl
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.pusht_sim import record_rollout
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit import pusht_model as pm
from trust_region_irl.policy_checkpoint import _read_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("start")
parser.add_argument("new", nargs="+")
parser.add_argument("--threshold", type=float, default=0.05, help="success threshold of the iteration (curriculum stage)")
parser.add_argument("--spawn-radius", type=float, default=0.20)
parser.add_argument("--nr-envs", type=int, default=4096)
parser.add_argument("--kl-steps", type=int, default=128)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--gamma", type=float, default=0.99)
args = parser.parse_args()

env = PushT(render=False, random_TPos=True, spawn_radius=args.spawn_radius, success_threshold=args.threshold,
            nr_envs=args.nr_envs)
obs_dim, act_dim = env.single_observation_space.shape[0], env.single_action_space.shape[0]
policy = Policy((act_dim,), 0.4, jnp.arange(obs_dim))
N, T, gamma = args.nr_envs, env.horizon, args.gamma


def params_of(tree):
    p = jax.tree.map(jnp.asarray, tree["params"])
    return p if "params" in p else {"params": p}


# frozen reward of START, as trirl_supermimic's corr_reward
start_ckpt, cfg = _read_checkpoint(args.start)
alpha, eps = float(cfg.get("entropy_coef", 0.001)), float(cfg.get("epsilon", 0.2))
beta = 1.0 / alpha
reward_type = cfg.get("reward_type", "byol-task-based-observation")
byol = get_discriminator(None, env, reward_type=reward_type)   # the training code's discriminator (config is unused)
corr, disc = params_of(start_ckpt["corrected_reward"]), params_of(start_ckpt["discriminator"])
h_terminal = float(np.sum(np.log(env.single_action_space.high - env.single_action_space.low)))


def reward(obs):
    d = jax.vmap(lambda p, o: byol.apply(p, o, jnp.zeros(act_dim), method=byol._reward).reshape(()), in_axes=(None, 0))
    return alpha * ((1.0 - eps) * d(corr, obs) + eps * beta * d(disc, obs))


@jax.jit
def episodes(params, reset_key, noise_key, sampled):
    state = env.reset(jax.random.split(reset_key, N), False)
    noise = jax.random.normal(noise_key, (T, N, act_dim)) * sampled

    def body(carry, noise_t):
        state, alive, G, discount, success = carry
        obs = state.next_observation
        mean, logstd = policy.apply(params, obs)
        r = reward(obs)
        state = env.step(state, mean + jnp.exp(logstd) * noise_t)
        term = state.terminated.astype(jnp.float32)
        done = (state.terminated | state.truncated).astype(jnp.float32)
        tail = gamma / (1.0 - gamma) * (reward(state.actual_next_observation) + alpha * h_terminal)
        G = G + alive * discount * (r + term * tail)
        success = success + alive * term
        return (state, alive * (1.0 - done), G, discount * gamma, success), None

    init = (state, jnp.ones(N), jnp.zeros(N), jnp.ones(()), jnp.zeros(N))
    (_, alive, G, _, success), _ = jax.lax.scan(body, init, noise)
    return G, success


@jax.jit
def mean_kl(params_new, params_old, obs):
    m1, s1 = policy.apply(params_new, obs)
    m0, s0 = policy.apply(params_old, obs)
    kl = gaussian_kl(m1, jnp.broadcast_to(s1, m1.shape), m0, jnp.broadcast_to(s0, m0.shape))
    return jnp.mean(kl), jnp.median(kl)


key = jax.random.PRNGKey(args.seed)
kl_key, reset_key, noise_key = jax.random.split(key, 3)
start_params = params_of(start_ckpt["policy"])
kl_obs = pm.observation(jnp.asarray(record_rollout(env, policy, start_params, None, N, args.kl_steps, kl_key)["state"])).reshape(-1, obs_dim)

G0, succ0 = episodes(start_params, reset_key, noise_key, 1.0)
_, succ0_mean = episodes(start_params, reset_key, noise_key, 0.0)
G0 = np.asarray(G0)
std_of = lambda p: float(np.exp(np.asarray(p["params"]["policy_logstd"])).mean())
results = {"start": {"path": args.start, "return_mean": float(G0.mean()), "return_se": float(G0.std() / np.sqrt(N)),
                     "success_sampled": float(np.mean(succ0)), "success_mean_action": float(np.mean(succ0_mean)),
                     "std": std_of(start_params)},
           "config": {**vars(args), "alpha": alpha, "epsilon": eps, "reward_type": reward_type, "horizon": T},
           "new": []}
print(json.dumps(results["start"]), flush=True)
for path in args.new:
    params = params_of(_read_checkpoint(path)[0]["policy"])
    kl_m, kl_med = mean_kl(params, start_params, kl_obs)
    G, succ = episodes(params, reset_key, noise_key, 1.0)
    _, succ_mean = episodes(params, reset_key, noise_key, 0.0)
    diff = np.asarray(G) - G0
    row = {"path": path, "kl_mean": float(kl_m), "kl_median": float(kl_med), "return_mean": float(np.mean(G)),
           "return_diff_mean": float(diff.mean()), "return_diff_se": float(diff.std() / np.sqrt(N)),
           "success_sampled": float(np.mean(succ)), "success_mean_action": float(np.mean(succ_mean)), "std": std_of(params)}
    results["new"].append(row)
    print(json.dumps(row), flush=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print(f"wrote {args.out}")
