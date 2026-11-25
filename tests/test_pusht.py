"""Tests for the PushT additions of the baseline comparison (trust_region_irl/environments/pusht_mjx): the spawn radius
of random start poses, the episode success rollout/is_success, and the IoU that scripts/pusht/eval_policies.py records.

    python -m pytest tests/test_pusht.py -q
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from ml_collections import config_dict

from trust_region_irl.environments.pusht_mjx.create_env import create_train_and_eval_env
from trust_region_irl.environments.pusht_mjx.default_config import get_config as env_config
from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.environments.pusht_mjx.iou import T_BOX_CENTRES, T_BOX_HALF_SIZES, iou


@pytest.mark.parametrize("radius", [0.15, 0.20])
def test_spawn_radius(radius):
    """Random start poses lie within spawn_radius of the goal and fill the disk (read from qpos, set directly by reset)."""
    env = PushT(render=False, random_TPos=True, spawn_radius=radius, nr_envs=256)
    s = env.reset(jax.random.split(jax.random.PRNGKey(0), 256), True)
    r = np.linalg.norm(np.asarray(s.data.qpos[:, 0:2]) - np.asarray(env.goal_pos[0:2]), axis=1)
    assert r.max() <= radius + 1e-6
    assert r.max() > 0.9 * radius   # 256 draws, uniform over the disk area: all inside 0.9 R with probability 0.81^256


def test_spawn_radius_from_the_config():
    config = config_dict.ConfigDict()
    config.environment = env_config("pusht_mjx")
    assert config.environment.spawn_radius == 0.15   # default: the environment as before
    config.environment.spawn_radius, config.environment.nr_envs = 0.2, 4
    train_env, eval_env = create_train_and_eval_env(config)
    assert train_env.spawn_radius == 0.2 and eval_env.spawn_radius == 0.2


@pytest.mark.parametrize("threshold, success", [(10.0, 1.0), (0.0, 0.0)])
def test_rollout_is_success(threshold, success):
    """rollout/is_success is 1 for an episode that ends at the goal, 0 for one that ends at the horizon. Threshold 10:
    every episode succeeds on its first step (the errors are capped at 5 m and pi rad); threshold 0: none ever does."""
    horizon, n = 3, 4
    env = PushT(render=False, horizon=horizon, success_threshold=threshold, nr_envs=n)
    s = env.reset(jax.random.split(jax.random.PRNGKey(0), n), True)
    assert np.all(np.asarray(s.info["rollout/is_success"]) == 0.0)
    diverged = np.zeros(n, bool)
    for t in range(horizon):
        s = env.step(s, jnp.zeros((n, 2)))
        diverged |= np.asarray(s.info["env_info/diverged"]) > 0.5
        if success:
            assert np.all(np.asarray(s.terminated)) and np.all(np.asarray(s.info["rollout/is_success"]) == 1.0)
            assert np.all(np.asarray(s.info["rollout/episode_length"]) == 1)
    if not success:
        # the horizon case needs three physics steps; XLA_FLAGS=--xla_gpu_deterministic_ops=true (set by conftest.py)
        # breaks MJX on GPU, so the block "diverges" and the episodes are cut early. Run with XLA_FLAGS= to test it on GPU.
        if diverged.any() and jax.default_backend() == "gpu" and "xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""):
            pytest.skip("MJX diverges on GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true; run with XLA_FLAGS= for this case")
        assert not diverged.any(), "zero actions from the start pose should not make the block diverge"
        assert np.all(np.asarray(s.truncated)) and not np.any(np.asarray(s.terminated))
        assert np.all(np.asarray(s.info["rollout/is_success"]) == 0.0)
        assert np.all(np.asarray(s.info["rollout/episode_length"]) == horizon)


def yaw_quat(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def test_iou_limits():
    assert iou([0.0, 0.0], yaw_quat(0.0)) == pytest.approx(1.0)
    assert iou([1.0, 0.0], yaw_quat(0.0)) == 0.0
    assert iou([np.nan, 0.0], yaw_quat(0.0)) == 0.0
    assert iou([0.0, 0.0], yaw_quat(2 * np.pi)) == pytest.approx(1.0)


@pytest.mark.parametrize("pose", [(0.02, -0.01, 0.3), (0.05, 0.03, 2.0), (-0.03, 0.0, np.pi), (0.0, 0.0, -0.7)])
def test_iou_matches_point_sampling(pose):
    """The exact polygon IoU against counting random points inside both T's."""
    x, y, yaw = pose
    pts = np.random.default_rng(0).uniform(-0.3, 0.3, (400_000, 2))

    def inside(px, py, pyaw):
        c, s = np.cos(pyaw), np.sin(pyaw)
        d = pts - [px, py]
        local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
        return np.any([np.all(np.abs(local - cen) <= half, axis=1) for cen, half in zip(T_BOX_CENTRES, T_BOX_HALF_SIZES)], axis=0)

    goal, block = inside(0.0, 0.0, 0.0), inside(x, y, yaw)
    sampled = (goal & block).sum() / (goal | block).sum()
    assert iou([x, y], yaw_quat(yaw)) == pytest.approx(sampled, abs=0.01)
