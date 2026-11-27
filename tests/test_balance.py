"""Tests for the ball balancing options the baseline comparison uses (trust_region_irl/environments/balance_fr3_mjx, from
Santhosh's 55caec6): the observation modes, the positions_history rule in the environment and in the expert dataset,
lost_terminates, and AIRL's embodiment-free reward input on the 43-D observation.

    python -m pytest tests/test_balance.py -q
    XLA_FLAGS= python -m pytest tests/test_balance.py -q      # on GPU: conftest.py's deterministic flag breaks MJX physics
"""
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from ml_collections import config_dict

from trust_region_irl.algorithms.airl_ppo.flax_full_jit.default_config import get_config as airl_config
from trust_region_irl.algorithms.airl_ppo.flax_full_jit.discriminator import get_discriminator
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.dataset import create_indices, episode_ends_from
from trust_region_irl.environments.balance_fr3_mjx.create_env import create_train_and_eval_env
from trust_region_irl.environments.balance_fr3_mjx.default_config import get_config as env_config
from trust_region_irl.environments.balance_fr3_mjx.environment import BalanceFR3

DATA = Path(__file__).resolve().parents[1] / "trirl_dataset/rl_expert/balance_fr3_varied_300_hist.npz"
HIST = slice(16, 29)   # positions_history: the previous step's ball position, plate position and quaternion, goal position
POS = slice(0, 13)     # ... and the same quantities now (the first BalanceFR3.POS_DIM entries)


def skip_if_mjx_broken():
    """XLA_FLAGS=--xla_gpu_deterministic_ops=true (conftest.py's default) breaks MJX physics on GPU (pizza 2026-11-23)."""
    if jax.default_backend() == "gpu" and "xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""):
        pytest.skip("MJX physics is wrong on GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true; run with XLA_FLAGS=")


@pytest.mark.parametrize("obs_mode, dim, n_encoder", [("full", 39, 25), ("positions_history", 43, 29)])
def test_observation_layout(obs_mode, dim, n_encoder):
    env = BalanceFR3(nr_envs=1, obs_mode=obs_mode)
    assert env.single_observation_space.shape == (dim,)
    parts = sorted(env.obs_index.values(), key=lambda s: s.start)
    assert parts[0].start == 0 and parts[-1].stop == dim
    assert all(a.stop == b.start for a, b in zip(parts, parts[1:])), "obs_index parts must tile the observation"
    assert tuple(env.encoder_input_indices) == tuple(range(n_encoder)) == tuple(range(env.obs_index["arm_qpos"].start))
    if obs_mode == "positions_history":
        assert (env.obs_index["prev_ball_pos"].start, env.obs_index["prev_goal_pos"].stop) == (HIST.start, HIST.stop)
        assert env.POS_DIM == POS.stop


def test_defaults_unchanged_and_config_reaches_the_env():
    config = config_dict.ConfigDict()
    config.environment = env_config("balance_fr3_mjx")
    assert config.environment.obs_mode == "full" and config.environment.lost_terminates is False   # as before the port
    config.environment.obs_mode, config.environment.lost_terminates, config.environment.nr_envs = "positions_history", True, 2
    train_env, eval_env = create_train_and_eval_env(config)
    for env in (train_env, eval_env):
        assert env.obs_mode == "positions_history" and env.lost_terminates and env.single_observation_space.shape == (43,)


def test_positions_history_in_the_environment():
    """At reset the history repeats the current positions; after each step it holds the previous step's positions."""
    skip_if_mjx_broken()
    n = 4
    env = BalanceFR3(nr_envs=n, obs_mode="positions_history")
    s = env.reset(jax.random.split(jax.random.PRNGKey(0), n), True)
    prev = np.asarray(s.next_observation)
    np.testing.assert_array_equal(prev[:, HIST], prev[:, POS])
    actions = 0.1 * jax.random.uniform(jax.random.PRNGKey(1), (3, n, 2), minval=-1.0, maxval=1.0)
    for t in range(3):
        s = env.step(s, actions[t])
        assert not np.any(np.asarray(s.terminated | s.truncated)), "no episode should end in 3 small steps"
        obs = np.asarray(s.next_observation)
        np.testing.assert_array_equal(obs[:, HIST], prev[:, POS])
        assert np.abs(obs[:, POS] - prev[:, POS]).max() > 0, "the positions should move"
        prev = obs


@pytest.mark.parametrize("lost_terminates", [False, True])
def test_lost_terminates(lost_terminates):
    """A ball moved 0.5 m below the plate is lost at the next step; with lost_terminates the episode ends (terminated)."""
    skip_if_mjx_broken()
    n = 3
    env = BalanceFR3(nr_envs=n, obs_mode="positions_history", lost_terminates=lost_terminates)
    s = env.reset(jax.random.split(jax.random.PRNGKey(0), n), True)
    z = env.ball_qadr + 2
    s = s.replace(data=s.data.replace(qpos=s.data.qpos.at[:, z].add(-0.5)))
    s = env.step(s, jnp.zeros((n, 2)))
    assert np.all(np.asarray(s.info["env_info/ball_lost"]) == 1.0)
    assert not np.any(np.asarray(s.truncated))
    if lost_terminates:
        assert np.all(np.asarray(s.terminated))
        assert np.all(np.asarray(s.info["rollout/ball_lost"]) == 1.0)
        assert np.all(np.asarray(s.info["rollout/episode_length"]) == 1)
    else:
        assert not np.any(np.asarray(s.terminated))


def test_airl_embodiment_free_input_on_positions_history():
    """AIRL's g and h read observation 0-28 (everything but the arm joints), the input of TRIRL-SPR's
    byol-embodiment-free reward; changing the arm joints (29-42) leaves the discriminator output unchanged."""
    config = config_dict.ConfigDict()
    config.algorithm = airl_config("airl_ppo.flax_full_jit")
    config.algorithm.reward_input = "embodiment-free"
    config.environment = env_config("balance_fr3_mjx")
    config.environment.obs_mode, config.environment.lost_terminates, config.environment.nr_envs = "positions_history", True, 2
    env, _ = create_train_and_eval_env(config)
    disc = get_discriminator(config, env)
    assert disc.input_indices == tuple(range(29))
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    x, xn = jax.random.normal(k[0], (8, 43)), jax.random.normal(k[1], (8, 43))
    a, ab, logp = jnp.zeros((8, 2)), jnp.zeros(8), jnp.zeros(8)
    params = disc.init(k[2], x[:1], a[:1], xn[:1], ab[:1], logp[:1])
    out = jax.vmap(lambda s, sn: disc.apply(params, s, a[0], sn, ab[0], logp[0]))
    base = out(x, xn)
    joints, ball = jnp.zeros(43).at[29:].set(1.0), jnp.zeros(43).at[0].set(0.1)
    assert float(jnp.abs(out(x + joints, xn + joints) - base).max()) == 0.0
    assert float(jnp.abs(out(x + ball, xn + ball) - base).max()) > 1e-6


@pytest.mark.skipif(not DATA.exists(), reason=f"expert data not found: {DATA}")
def test_expert_dataset_follows_the_environment_rule():
    """The 300-demo dataset (Santhosh, positions_history): the history holds the previous step's positions as the env
    builds them, consecutive steps chain, 200 steps per demo with the last marked absorbing, and the Diffusion Policy
    windows (horizon 16, padding 1 / 7) are 300 x 193."""
    d = np.load(DATA)
    E, T = 300, 200
    assert d["states"].shape == (E * T, 43) and d["actions"].shape == (E * T, 2)
    assert np.all(d["episode_lengths"] == T)
    ends = episode_ends_from(d)
    np.testing.assert_array_equal(ends, np.arange(1, E + 1) * T)
    S, NS = d["states"].reshape(E, T, 43), d["next_states"].reshape(E, T, 43)
    np.testing.assert_array_equal(S[:, 1:, HIST], S[:, :-1, POS])
    np.testing.assert_array_equal(S[:, 0, HIST], S[:, 0, POS])
    np.testing.assert_array_equal(NS[:, :, HIST], S[:, :, POS])
    np.testing.assert_array_equal(NS[:, :-1], S[:, 1:])
    ab = d["absorbing"].reshape(E, T)
    assert np.all(ab[:, -1] == 1.0) and np.all(ab[:, :-1] == 0.0)
    assert len(create_indices(ends, 16, pad_before=1, pad_after=7)) == E * (T - 16 + 1 + 1 + 7)
    assert np.abs(d["actions"]).max() <= 1.02   # tilt-rate commands / MAX_TILT_VEL, as the env's action
