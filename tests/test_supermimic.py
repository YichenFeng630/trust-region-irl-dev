"""Tests for TRIRL + SuperMimic (trust_region_irl/algorithms/trirl_supermimic), plan step 1 (supermimic_trirl/PLAN_zh.md).

    XLA_FLAGS= python -m pytest tests/test_supermimic.py -k "not ppo_mode"            # GPU (MJX needs XLA_FLAGS empty)
    JAX_PLATFORMS=cpu python -m pytest tests/test_supermimic.py -k ppo_mode            # exact comparison with trirl_ppo_spr

The comparison with trirl_ppo_spr needs bit-reproducible runs, so it runs on the CPU; tests/conftest.py's GPU flag
--xla_gpu_deterministic_ops=true breaks MJX's physics on the GPU.
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest
from ml_collections import config_dict

from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.imagination import (
    gaussian_kl, gaussian_entropy, lambda_returns, make_actor_loss)
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.world_model import RunningStats, WorldModel
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.replay_buffer import ReplayBuffer
from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit import pusht_model as pm

GPU_DETERMINISTIC = jax.default_backend() == "gpu" and "xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", "")


# ---------------------------------------------------------------- returns, KL ---- #
def gae_absorbing_returns(rewards, rewards_next_state, values, next_values, terminations, gamma, lam, entropy_coef, h_terminal):
    """trirl_ppo_spr's calculate_gae_advantages_absorbing (returns = advantages + values), next values passed in."""
    terminal_tail = (gamma / (1.0 - gamma)) * (rewards_next_state + entropy_coef * h_terminal)
    delta = rewards + gamma * next_values * (1.0 - terminations) + terminations * terminal_tail - values
    adv = np.zeros_like(delta)
    adv[-1] = delta[-1]
    for t in range(delta.shape[0] - 2, -1, -1):
        adv[t] = delta[t] + gamma * lam * (1 - terminations[t]) * adv[t + 1]
    return adv + values


def test_lambda_returns_match_trirl_gae():
    """Eq. 3 with the terminal tail T = (r(s') + alpha H) / (1 - gamma) gives trirl_ppo_spr's GAE returns.
    The two agree when V(s_{t+1}) is the value of the next row, i.e. no time-limit cut inside the window, as in an
    imagined rollout (GAE's returns are a lambda-return only then)."""
    rng = np.random.default_rng(0)
    T, N, gamma, lam, alpha, h = 12, 5, 0.99, 0.95, 0.001, 1.386
    r, r_next = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    v = rng.normal(size=(T + 1, N))
    values, next_values = v[:-1], v[1:]
    term = (rng.uniform(size=(T, N)) < 0.2).astype(np.float64)
    expected = gae_absorbing_returns(r, r_next, values, next_values, term, gamma, lam, alpha, h)
    tails = (r_next + alpha * h) / (1.0 - gamma)
    got = lambda_returns(jnp.asarray(r), jnp.asarray(term), jnp.asarray(tails), jnp.asarray(next_values), gamma, lam)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


def test_gaussian_kl():
    """Equals trirl_ppo_spr's trust-region loss (KL(pi_theta || pi_old)) and the closed form; zero for equal policies."""
    rng = np.random.default_rng(1)
    mean, old_mean = rng.normal(size=(7, 2)), rng.normal(size=(7, 2))
    logstd, old_logstd = rng.normal(size=(2,)) * 0.3, rng.normal(size=(2,)) * 0.3
    std, old_std = np.exp(logstd), np.exp(old_logstd)
    # trirl_ppo_spr's lines, with its shapes: action_std (B, a), old_action_std (1, a)
    action_std, old_action_std = np.broadcast_to(std, mean.shape), old_std[None]
    code = 0.5 * np.sum(((old_mean - mean) / old_action_std) ** 2, axis=1) + \
        0.5 * np.sum(2.0 * (np.log(old_action_std) - np.log(action_std)) + (action_std / old_action_std) ** 2 - 1.0, axis=1)
    closed = np.sum(np.log(old_std / std) + (std ** 2 + (mean - old_mean) ** 2) / (2 * old_std ** 2) - 0.5, axis=1)
    got = np.asarray(gaussian_kl(jnp.asarray(mean), jnp.asarray(logstd), jnp.asarray(old_mean), jnp.asarray(old_logstd)))
    np.testing.assert_allclose(got, code, rtol=1e-5)
    np.testing.assert_allclose(got, closed, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(gaussian_kl(jnp.asarray(mean), jnp.asarray(logstd), jnp.asarray(mean), jnp.asarray(logstd))), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(gaussian_entropy(jnp.asarray(logstd))), np.sum(logstd + 0.5 * np.log(2 * np.pi * np.e)), rtol=1e-6)


# ------------------------------------------------------------- imagination ---- #
class LinPolicy(nn.Module):
    """mean = k s (1-D), state-independent log std, as the TRIRL policy's parameter layout."""
    k0: float = -0.5
    std: float = 0.3

    @nn.compact
    def __call__(self, x):
        k = self.param("k", nn.initializers.constant(self.k0), (1,))
        logstd = self.param("policy_logstd", nn.initializers.constant(np.log(self.std)), (1, 1))
        return x[..., :1] * k, logstd


class LinCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        c = self.param("c", nn.initializers.constant(0.7), (1,))
        return x[..., :1] * c


def toy_setup(gamma=0.99, H=6, std=0.3, straight_through=False, normalization="advantage_std", tail=5.0):
    policy, critic = LinPolicy(std=std), LinCritic()
    model_step = lambda wp, norm, s, a: s + 0.1 * a
    reward_fn = lambda rp, s, a, s_next: -s[0] ** 2
    terminal_fn = lambda rp, s_next: tail + s_next[0]
    success_fn = lambda s, thr: (jnp.abs(s[0]) < thr).astype(jnp.float32)
    soft_fn = lambda s, thr, temp: jax.nn.sigmoid((thr - jnp.abs(s[0])) / temp)
    actor_loss, imagined_return = make_actor_loss(policy, critic, model_step, lambda s: s, reward_fn, terminal_fn, success_fn,
                                                  soft_fn, H, gamma, 0.95, 0.0, straight_through=straight_through,
                                                  return_normalization=normalization)
    pp = policy.init(jax.random.PRNGKey(0), jnp.zeros((1,)))
    cp = critic.init(jax.random.PRNGKey(0), jnp.zeros((1,)))
    return actor_loss, imagined_return, pp, cp


def hand_return(s0, eps, k, std, gamma, thr, tail, c_v, H):
    s, c, G = s0, 1.0, 0.0
    for t in range(H):
        a = k * s + std * eps[t]
        s_next = s + 0.1 * a
        r = -s ** 2
        d = float(abs(s_next) < thr)
        G += gamma ** t * c * (r + gamma * d * (tail + s_next))
        c *= 1.0 - d
        s = s_next
    return G, G + gamma ** H * c * c_v * s


def test_imagined_return_by_hand():
    """Return of Eq. 2 with hard termination, terminal tail and the critic beyond H, against a plain loop."""
    H, gamma, thr = 6, 0.99, 0.2
    actor_loss, imagined_return, pp, cp = toy_setup(H=H)
    s0 = jnp.array([[0.5], [-0.3], [0.205], [0.1]])   # 0.1 ends at once, 0.5 never
    eps = jax.random.normal(jax.random.PRNGKey(3), (4, H, 1))
    G_H, ended = imagined_return(pp, None, None, None, s0, eps, thr)
    _, aux = actor_loss(pp, pp, cp, None, None, None, s0, eps, 0.0, thr)
    hand = [hand_return(float(s0[i, 0]), np.asarray(eps[i, :, 0]), -0.5, 0.3, gamma, thr, 5.0, 0.7, H) for i in range(4)]
    np.testing.assert_allclose(np.asarray(G_H), [h[0] for h in hand], rtol=1e-5)
    np.testing.assert_allclose(float(aux["metrics"]["imagination/return"]), np.mean([h[1] for h in hand]), rtol=1e-5)
    assert 0 < float(np.mean(np.asarray(ended))) < 1, "the example should have both ended and unfinished rollouts"


def test_bptt_gradient_linear_system():
    """s' = A s + B a, a = k s (no noise, no termination), r = -s^2: the gradient of the imagined return with respect to
    k equals the closed form -sum_t gamma^t 2 t B (A + B k)^(2t-1) s0^2, and the actor loss gradient is -1/W of it."""
    H, gamma, A, B, k = 7, 0.97, 1.0, 0.1, -0.5
    actor_loss, imagined_return, pp, cp = toy_setup(gamma=gamma, H=H, normalization="none")
    s0 = jnp.array([[0.8], [-0.4]])
    eps = jnp.zeros((2, H, 1))
    thr = -1.0   # never ends
    g = jax.grad(lambda p: jnp.sum(imagined_return(p, None, None, None, s0, eps, thr)[0]))(pp)["params"]["k"][0]
    m = A + B * k
    closed = -sum(gamma ** t * 2 * t * B * m ** (2 * t - 1) for t in range(1, H)) * float(jnp.sum(s0 ** 2))
    np.testing.assert_allclose(float(g), closed, rtol=1e-4)
    cp0 = jax.tree.map(jnp.zeros_like, cp)   # V = 0: no critic term
    g_loss = jax.grad(lambda p: actor_loss(p, p, cp0, None, None, None, s0, eps, 0.0, thr)[0])(pp)["params"]["k"][0]
    W = sum(gamma ** t for t in range(H))
    np.testing.assert_allclose(float(g_loss), -closed / 2 / W, rtol=1e-4)   # mean over the 2 starts


def test_kl_term_in_actor_loss():
    """eta KL(pi_theta || pi_k): zero at pi_theta = pi_k, and its gradient pulls a moved policy back."""
    actor_loss, _, pp, cp = toy_setup(H=4)
    s0, eps = jnp.array([[0.5], [-0.3]]), jnp.zeros((2, 4, 1))
    moved = {"params": {**pp["params"], "k": pp["params"]["k"] + 0.3}}
    no_kl = actor_loss(moved, pp, cp, None, None, None, s0, eps, 0.0, -1.0)[0]
    with_kl = actor_loss(moved, pp, cp, None, None, None, s0, eps, 10.0, -1.0)[0]
    same = actor_loss(pp, pp, cp, None, None, None, s0, eps, 0.0, -1.0)[0] - actor_loss(pp, pp, cp, None, None, None, s0, eps, 10.0, -1.0)[0]
    assert float(with_kl) > float(no_kl)
    np.testing.assert_allclose(float(same), 0.0, atol=1e-6)
    g = jax.grad(lambda p: actor_loss(p, pp, cp, None, None, None, s0, eps, 10.0, -1.0)[0] -
                 actor_loss(p, pp, cp, None, None, None, s0, eps, 0.0, -1.0)[0])(moved)["params"]["k"][0]
    assert float(g) > 0, "the KL term should push k back towards pi_k's"


def test_straight_through_termination():
    """Same return as the hard termination in the forward pass, a different gradient."""
    H, thr = 6, 0.2
    s0 = jnp.array([[0.5], [-0.3], [0.205], [0.1]])
    eps = jax.random.normal(jax.random.PRNGKey(3), (4, H, 1))
    hard = toy_setup(H=H)
    st = toy_setup(H=H, straight_through=True)
    f = lambda setup, p: setup[0](p, p, setup[3], None, None, None, s0, eps, 0.0, thr)[0]
    np.testing.assert_allclose(float(f(hard, hard[2])), float(f(st, st[2])), rtol=1e-6)
    g_hard = jax.grad(lambda p: f(hard, p))(hard[2])["params"]["k"][0]
    g_st = jax.grad(lambda p: f(st, p))(st[2])["params"]["k"][0]
    assert abs(float(g_hard) - float(g_st)) > 1e-6


# ------------------------------------------------------- world model, buffer ---- #
def test_running_stats():
    rng = np.random.default_rng(2)
    stats, rows = RunningStats.create(3), []
    for _ in range(4):
        x, w = rng.normal(size=(50, 3)) * [1, 2, 3] + [0, 1, -1], (rng.uniform(size=50) < 0.7).astype(np.float32)
        stats = stats.update(jnp.asarray(x, jnp.float32), jnp.asarray(w))
        rows.append(x[w > 0])
    rows = np.concatenate(rows)
    np.testing.assert_allclose(np.asarray(stats.mean), rows.mean(0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(stats.std(0.0)), rows.std(0), rtol=1e-4)
    assert int(stats.count) == len(rows)


def test_replay_windows_follow_time_and_episodes():
    """Windows are consecutive time steps of one environment among the rows still stored; a step is valid until an
    episode ends before it, and its velocity entries only if the episode does not end at it."""
    C, N, Tc, k, D = 12, 3, 4, 3, pm.STATE_DIM
    rng = np.random.default_rng(3)
    done = (rng.uniform(size=(16, N)) < 0.25).astype(np.float32)
    ids = np.arange(16)[:, None] * 10 + np.arange(N)[None, :]           # unique id of (global time, env)
    buf = ReplayBuffer.create(C, N, D, 2)
    for c in range(4):                                                  # 16 rows into 12: the first 4 are overwritten
        sl = slice(c * Tc, (c + 1) * Tc)
        st = np.zeros((Tc, N, D), np.float32); st[..., 0] = ids[sl]
        buf = buf.add(jnp.asarray(st), jnp.asarray(np.stack([ids[sl], ids[sl]], -1), jnp.float32),
                      jnp.asarray(st + 0.5), jnp.asarray(done[sl]), jnp.zeros((Tc, N)))
    s0, actions, targets, valid, vel_valid = jax.tree.map(np.asarray, buf.sample_windows(jax.random.PRNGKey(0), 500, k))
    t0, env = (s0[:, 0] // 10).astype(int), (s0[:, 0] % 10).astype(int)
    assert t0.min() >= 4 and (t0 + k - 1).max() <= 15                   # only stored rows
    for i in range(500):
        tt = t0[i] + np.arange(k)
        np.testing.assert_allclose(actions[i, :, 0], ids[tt, env[i]])
        np.testing.assert_allclose(targets[i, :, 0], ids[tt, env[i]] + 0.5)
        d = done[tt, env[i]]
        exp_valid = np.concatenate([[True], np.cumsum(d)[:-1] == 0])
        np.testing.assert_array_equal(valid[i] > 0.5, exp_valid)
        np.testing.assert_array_equal(vel_valid[i] > 0.5, exp_valid & (d < 0.5))
    states = np.asarray(buf.sample_states(jax.random.PRNGKey(1), 500))
    assert (states[:, 0] // 10).min() >= 4


def test_world_model_start_and_velocity_mask():
    """An untrained model predicts the mean change (zero output layer) with a unit quaternion; at an episode end the
    velocity targets do not enter the loss."""
    wm = WorldModel(pm.STATE_DIM, 2, "16,16", quat=pm.QUAT, velocity_dims=pm.VELOCITY_DIMS)
    params = wm.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(4)
    s = rng.normal(size=(64, pm.STATE_DIM)).astype(np.float32); s[:, 3:7] /= np.linalg.norm(s[:, 3:7], axis=1, keepdims=True)
    s_next = s + 0.01 * rng.normal(size=s.shape).astype(np.float32)
    norm = wm.update_normalizer(wm.init_normalizer(), jnp.asarray(s), jnp.asarray(s_next), jnp.ones(64), jnp.ones(64))
    pred = np.asarray(wm.step(params, norm, jnp.asarray(s[0]), jnp.zeros(2)))
    expected = s[0] + np.asarray(norm.delta.mean)
    expected[3:7] /= np.linalg.norm(expected[3:7])
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-6)
    targets = jnp.asarray(np.repeat(s_next[:8, None], 2, axis=1))
    a = jnp.zeros((8, 2, 2))
    valid, vel_valid = jnp.ones((8, 2)), jnp.zeros((8, 2))
    l1 = wm.loss(params, norm, jnp.asarray(s[:8]), a, targets, valid, vel_valid)[0]
    l2 = wm.loss(params, norm, jnp.asarray(s[:8]), a, targets.at[..., 24:].add(5.0), valid, vel_valid)[0]
    np.testing.assert_allclose(float(l1), float(l2), rtol=1e-6)


# ---------------------------------------------------------------- PushT heads ---- #
def test_goal_errors_match_the_environment_formula():
    """The arctan form of the orientation error equals the environment's arccos form, and has a finite gradient at zero."""
    rng = np.random.default_rng(5)
    q = rng.normal(size=(200, 4)); q[:20] = [1, 0, 0, 0]; q[20:40] = [1, 1e-4, -1e-4, 2e-4]; q[40:60, 0] *= 1e3
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s = np.zeros((200, pm.STATE_DIM), np.float32); s[:, 0:3] = rng.normal(size=(200, 3)) * 0.1; s[:, 3:7] = q
    got = jax.vmap(pm.goal_errors)(jnp.asarray(s)); env = jax.vmap(pm.goal_errors_env)(jnp.asarray(s))
    np.testing.assert_allclose(np.asarray(got[0]), np.asarray(env[0]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(got[1]), np.asarray(env[1]), atol=1e-3)   # arccos loses precision near 0
    zero = jnp.zeros(pm.STATE_DIM).at[3].set(1.0)
    g = jax.grad(lambda x: pm.env_reward(x))(zero)
    assert np.all(np.isfinite(np.asarray(g)))


def _pusht_env(n, threshold=0.3):
    from trust_region_irl.environments.pusht_mjx.environment import PushT
    return PushT(render=False, random_TPos=True, spawn_radius=0.2, success_threshold=threshold, nr_envs=n)


def test_pusht_heads_match_the_environment():
    """On real steps: errors, success and reward of the world-model state equal the environment's; at an episode end
    the state's velocity entries already belong to the next episode (zero at its start), hence the mask."""
    n = 32
    env = _pusht_env(n)
    vel = pm.t_velocity_slice(env)
    assert vel == (0, 6)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), n), False)
    key = jax.random.PRNGKey(1)
    diverged = False
    for t in range(20):
        key, sub = jax.random.split(key)
        state = env.step(state, jax.random.uniform(sub, (n, 2), minval=-1, maxval=1))
        diverged |= bool(np.any(np.asarray(state.info["env_info/diverged"]) > 0.5))
        full = pm.full_state(state.actual_next_observation, state.data.qvel, vel)
        pos, orn = jax.vmap(pm.goal_errors)(full)
        np.testing.assert_allclose(np.asarray(pos), np.asarray(state.info["env_info/pos_err"]), atol=1e-4)
        np.testing.assert_allclose(np.asarray(orn), np.asarray(state.info["env_info/orient_err"]), atol=2e-3)
        np.testing.assert_allclose(np.asarray(jax.vmap(pm.env_reward)(full)), np.asarray(state.reward), atol=2e-3)
        succ = np.asarray(jax.vmap(pm.success, in_axes=(0, None))(full, 0.3))
        near = np.abs(np.asarray(state.info["env_info/orient_err"]) - 0.3) < 2e-3   # the two formulas may differ at the edge
        np.testing.assert_array_equal(succ[~near], np.asarray(state.info["env_info/is_success"])[~near])
        done = np.asarray(state.terminated | state.truncated)
        if done.any():
            np.testing.assert_allclose(np.asarray(full)[done][:, 24:30], 0.0)
    if diverged and GPU_DETERMINISTIC:
        pytest.skip("MJX diverges on GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true; run with XLA_FLAGS=")


def test_restored_physics_replays_the_recorded_steps():
    """pusht_sim: from a logged (qpos, qvel) the logged actions reproduce the logged next states."""
    from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy
    from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.pusht_sim import record_rollout, states_from_physics, replay_actions
    n, T, t0, H = 8, 12, 3, 6
    env = _pusht_env(n, threshold=0.05)
    policy = Policy((2,), 0.4, jnp.arange(24))
    params = policy.init(jax.random.PRNGKey(0), jnp.zeros((1, 24)))
    d = record_rollout(env, policy, params, jnp.float32(0.6), n, T, jax.random.PRNGKey(2))
    if d["diverged"].any() and GPU_DETERMINISTIC:
        pytest.skip("MJX diverges on GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true; run with XLA_FLAGS=")
    keep = ~d["done"][t0:t0 + H].any(axis=0)
    assert keep.any()
    st = states_from_physics(env, d["qpos"][t0], d["qvel"][t0], jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.asarray(pm.full_state(st.next_observation, st.data.qvel, (0, 6))), d["state"][t0], atol=1e-5)
    full, done = replay_actions(env, st, jnp.asarray(np.swapaxes(d["action"][t0:t0 + H], 0, 1)))
    err = np.abs(np.asarray(full)[keep] - np.swapaxes(d["next_state"][t0:t0 + H], 0, 1)[keep])
    assert err[..., 0:3].max() < 1e-4, err[..., 0:3].max()


# ------------------------------------------------------------ training loop ---- #
def fake_expert_data(path, n=2000, seed=0, obs_dim=24):
    rng = np.random.default_rng(seed)
    ab = np.zeros(n, np.float32); ab[399::400] = 1.0
    np.savez(path, states=rng.normal(size=(n, obs_dim)).astype(np.float32), actions=rng.uniform(-1, 1, (n, 2)).astype(np.float32),
             next_states=rng.normal(size=(n, obs_dim)).astype(np.float32), absorbing=ab, rewards=np.zeros(n, np.float32))
    return path


def make_config(algorithm_module, data_path, nr_envs=8, nr_steps=16, iterations=2, **algorithm):
    from trust_region_irl.environments.pusht_mjx.default_config import get_config as pusht_env_config
    config = config_dict.ConfigDict()
    config.algorithm = algorithm_module.get_config(algorithm_module.__name__)
    config.environment = pusht_env_config("pusht_mjx")
    # save_model on, so the save callback runs (the tests replace model.save to capture what would be saved)
    config.runner = config_dict.ConfigDict(dict(save_model=True, track_console=False, track_tb=False, track_wandb=False))
    config.environment.nr_envs, config.environment.seed, config.environment.impl = nr_envs, 0, "jax"
    config.environment.random_TPos, config.environment.spawn_radius = True, 0.2
    config.environment.curriculum_success = True
    config.environment.success_threshold_values, config.environment.success_threshold_fractions = "0.5,0.05", "0.5,0.5"
    batch = nr_envs * nr_steps
    base = dict(data_path=str(data_path), nr_steps=nr_steps, total_timesteps=batch * iterations,
                evaluation_and_save_frequency=batch, evaluation_active=True, minibatch_size=batch // 2, nr_epochs=2,
                nr_epochs_disc=1, pretrain_env_steps=batch, prediction_steps=2, std_dev=0.4,
                reward_type="byol-task-based-observation")
    for k, v in {**base, **algorithm}.items():
        config.algorithm[k] = v
    return config


TINY_BPTT = dict(replay_capacity=32, wm_window=4, wm_batch_size=16, wm_minibatches_per_update=2, imagination_horizon=4,
                 imagination_batch_size=16, critic_minibatches=4, nr_model_updates=2, wm_hidden_dims="32,32")


def flat(tree):
    return np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree.leaves(tree)])


def run_and_capture(model):
    logged, saved = [], []
    model.log = lambda name, value, step: logged.append((int(step), name, float(np.asarray(value))))
    model.save = lambda *a, **k: saved.append((a, k))
    init_policy = flat(model.policy_state.params)
    model.train()
    return logged, saved, init_policy


@pytest.mark.parametrize("reward_source", ["trirl", "env", "zero"])
def test_bptt_training_runs(tmp_path, reward_source):
    """Two TRIRL iterations with SuperMimic's update on PushT: every logged value is finite, the world model and the
    policy change, the curriculum threshold reaches the imagination, and each save carries the world model."""
    from trust_region_irl.algorithms.trirl_supermimic import flax_full_jit as sm
    from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.trirl_supermimic import TRIRL_SuperMimic
    from trust_region_irl.environments.pusht_mjx.create_env import create_train_and_eval_env
    config = make_config(sm.default_config, fake_expert_data(tmp_path / "d.npz"), reward_source=reward_source, **TINY_BPTT)
    train_env, eval_env = create_train_and_eval_env(config)
    model = TRIRL_SuperMimic(config, train_env, eval_env, str(tmp_path / "run"), None)
    init_wm = flat(model.wm_state.params)
    logged, saved, init_policy = run_and_capture(model)
    names = {n for _, n, _ in logged}
    for key in ("world_model/loss", "imagination/return", "loss/critic_loss", "policy/real_kl", "imagination/model_return_H",
                "world_model/new_data_T_pos_err_4", "eval/is_success"):
        assert key in names, key
    bad = [(s, n, v) for s, n, v in logged if not np.isfinite(v)]
    assert not bad, bad[:5]
    assert len(saved) >= 2 and all("world_model" in k for _, k in saved)   # latest (and best) at both iterations
    wm_state = saved[-1][1]["world_model"][0]
    assert np.abs(flat(wm_state.params) - init_wm).max() > 1e-6, "the world model did not train"
    assert np.abs(flat(saved[-1][0][0].params) - init_policy).max() > 1e-7, "the policy did not change"
    kl = [v for _, n, v in logged if n == "policy/real_kl"]
    assert all(v >= 0 for v in kl) and max(kl) > 0
    if reward_source != "trirl":
        assert "loss/discriminator_loss" not in names


def test_bptt_save_and_load(tmp_path):
    """A saved SuperMimic model loads back with the same policy and world model, and the evaluation scripts' load_actor reads it."""
    from trust_region_irl.algorithms.trirl_supermimic import flax_full_jit as sm
    from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.trirl_supermimic import TRIRL_SuperMimic
    from trust_region_irl.environments.pusht_mjx.create_env import create_train_and_eval_env
    from trust_region_irl.policy_checkpoint import load_actor
    config = make_config(sm.default_config, fake_expert_data(tmp_path / "d.npz"), iterations=1, **TINY_BPTT)
    train_env, eval_env = create_train_and_eval_env(config)
    model = TRIRL_SuperMimic(config, train_env, eval_env, str(tmp_path / "run"), None)
    model.log = lambda *a: None
    model.train()
    path = tmp_path / "run" / "models" / "latest.model.zip"
    assert path.exists()
    config.runner.load_model = str(path)
    loaded = TRIRL_SuperMimic.load(config, train_env, eval_env, str(tmp_path / "run2"), None, [])
    actor, params, name = load_actor(str(path), 24, 2)
    obs = jnp.asarray(np.random.default_rng(0).normal(size=(4, 24)), jnp.float32)
    a1 = actor.act(jax.tree.map(jnp.asarray, params), {}, obs, 0, True)[0]
    a2 = loaded.policy.apply(loaded.policy_state.params, obs)[0]
    np.testing.assert_allclose(np.asarray(a1), np.asarray(a2), rtol=1e-6)
    assert flat(loaded.wm_state.params).size > 0 and float(loaded.wm_norm.state.count) > 0


def test_progress_saves_and_starting_from_a_checkpoint(tmp_path):
    """Plan step 2.5: save_at_fractions keeps a copy of latest.model at that point of the run; init_checkpoint starts a
    run from such an archive (policy, discriminator, corrected reward, and the critic with ppo); with freeze_reward the
    discriminator and the corrected reward stay as loaded while the policy moves."""
    from trust_region_irl.algorithms.trirl_supermimic import flax_full_jit as sm
    from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.trirl_supermimic import TRIRL_SuperMimic
    from trust_region_irl.environments.pusht_mjx.create_env import create_train_and_eval_env
    from trust_region_irl.policy_checkpoint import _read_checkpoint
    data = fake_expert_data(tmp_path / "d.npz")
    config = make_config(sm.default_config, data, iterations=4, policy_update="ppo", save_at_fractions="0.5")
    train_env, eval_env = create_train_and_eval_env(config)
    model = TRIRL_SuperMimic(config, train_env, eval_env, str(tmp_path / "run"), None)
    model.log = lambda *a: None
    model.train()
    models = tmp_path / "run" / "models"
    half = models / "progress_0.5.model.zip"
    assert half.exists() and not (models / "progress_0.25.model.zip").exists()
    ck_half, _ = _read_checkpoint(str(half))
    ck_last, _ = _read_checkpoint(str(models / "latest.model.zip"))
    assert np.abs(flat(ck_half["policy"]["params"]) - flat(ck_last["policy"]["params"])).max() > 0, "the copy is not from mid-run"

    for update in ("ppo", "bptt"):
        extra = TINY_BPTT if update == "bptt" else {}
        config = make_config(sm.default_config, data, iterations=2, policy_update=update, pretrain_env_steps=0,
                             init_checkpoint=str(half), freeze_reward=True, **extra)
        train_env, eval_env = create_train_and_eval_env(config)
        model = TRIRL_SuperMimic(config, train_env, eval_env, str(tmp_path / f"from_{update}"), None)
        parts = ("policy", "discriminator", "corrected_reward") + (("critic",) if update == "ppo" else ())
        for part in parts:
            np.testing.assert_array_equal(flat(getattr(model, f"{part}_state").params), flat(ck_half[part]["params"]), err_msg=part)
        logged, saved, init_policy = run_and_capture(model)
        policy_state, _, disc_state, corr_state = saved[-1][0][:4]
        np.testing.assert_array_equal(flat(disc_state.params), flat(ck_half["discriminator"]["params"]))
        np.testing.assert_array_equal(flat(corr_state.params), flat(ck_half["corrected_reward"]["params"]))
        assert np.abs(flat(policy_state.params) - init_policy).max() > 1e-7, f"{update}: the policy did not change"
        assert "loss/discriminator_loss" not in {n for _, n, _ in logged}
        assert all(np.isfinite(v) for _, _, v in logged)


def test_bptt_warmup_and_offline_world_model(tmp_path):
    """init_world_model loads wm_offline.py's file (parameters and normaliser); bptt_warmup_updates fits the world model
    and the critic before the first policy step, in the first iteration only."""
    import json
    from flax import serialization
    from trust_region_irl.algorithms.trirl_supermimic import flax_full_jit as sm
    from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.trirl_supermimic import TRIRL_SuperMimic
    from trust_region_irl.environments.pusht_mjx.create_env import create_train_and_eval_env
    wm = WorldModel(pm.STATE_DIM, 2, "32,32", quat=pm.QUAT, velocity_dims=pm.VELOCITY_DIMS)
    params = wm.init(jax.random.PRNGKey(7))
    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.normal(size=(64, pm.STATE_DIM)), jnp.float32)
    norm = wm.update_normalizer(wm.init_normalizer(), s, s + 0.01 * jnp.asarray(rng.normal(size=s.shape), jnp.float32),
                                jnp.ones(64), jnp.ones(64))
    d = tmp_path / "wm"
    d.mkdir()
    (d / "model.msgpack").write_bytes(serialization.to_bytes({"params": params, "norm": norm}))
    (d / "config.json").write_text(json.dumps({"hidden": "32,32"}))

    config = make_config(sm.default_config, fake_expert_data(tmp_path / "d.npz"), iterations=2, init_world_model=str(d),
                         bptt_warmup_updates=3, **TINY_BPTT)
    train_env, eval_env = create_train_and_eval_env(config)
    model = TRIRL_SuperMimic(config, train_env, eval_env, str(tmp_path / "run"), None)
    np.testing.assert_array_equal(flat(model.wm_state.params), flat(params))
    np.testing.assert_allclose(float(model.wm_norm.state.count), float(norm.state.count))
    logged, saved, _ = run_and_capture(model)
    assert all(np.isfinite(v) for _, _, v in logged)
    wm_state = saved[-1][1]["world_model"][0]
    # world-model gradient steps: warm-up 3 updates x 2, then 2 iterations x 2 updates x 2; policy steps: 2 x 2
    assert int(wm_state.step) == 3 * 2 + 2 * 2 * 2, int(wm_state.step)
    assert int(saved[-1][0][0].step) == 2 * 2, int(saved[-1][0][0].step)


@pytest.mark.slow
def test_ppo_mode_matches_trirl_ppo_spr(tmp_path):
    """policy_update=ppo is trirl_ppo_spr unchanged: same logged values (bit-exact on the CPU) and the same final policy."""
    if jax.default_backend() != "cpu":
        pytest.skip("bit-exact comparison: run with JAX_PLATFORMS=cpu")
    from trust_region_irl.algorithms.trirl_supermimic import flax_full_jit as sm
    from trust_region_irl.algorithms.trirl_ppo_spr import flax_full_jit as spr
    from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit.trirl_supermimic import TRIRL_SuperMimic
    from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.trirl_ppo import TRIRL_PPO
    from trust_region_irl.environments.pusht_mjx.create_env import create_train_and_eval_env
    data = fake_expert_data(tmp_path / "d.npz")
    runs = {}
    for name, module, cls, extra in (("spr", spr.default_config, TRIRL_PPO, {}),
                                     ("supermimic", sm.default_config, TRIRL_SuperMimic, {"policy_update": "ppo"})):
        config = make_config(module, data, **extra)
        train_env, eval_env = create_train_and_eval_env(config)
        model = cls(config, train_env, eval_env, str(tmp_path / name), None)
        logged, saved, _ = run_and_capture(model)
        runs[name] = ({(s, n): v for s, n, v in logged if not n.startswith("time/")}, flat(saved[-1][0][0].params))
    (log_a, pol_a), (log_b, pol_b) = runs["spr"], runs["supermimic"]
    common = set(log_a) & set(log_b)
    assert len(common) > 50 and set(log_a) <= set(log_b)
    diff = {k: (log_a[k], log_b[k]) for k in common if not (log_a[k] == log_b[k] or (np.isnan(log_a[k]) and np.isnan(log_b[k])))}
    assert not diff, list(diff.items())[:5]
    np.testing.assert_array_equal(pol_a, pol_b)
