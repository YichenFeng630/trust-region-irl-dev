"""Training-path tests for the Diffusion Policy port (trust_region_irl.algorithms.diffusion_policy.flax_full_jit).

    python -m pytest tests/test_dp.py             # about 2 min on a laptop GPU
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

from trust_region_irl.algorithms.diffusion_policy.flax_full_jit import DiffusionPolicy  # noqa: F401  (registers the algorithm)
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit import policy as dp, scheduler
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.dataset import create_indices, gather_index
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.normalizer import fit_limits, normalize, to_jax
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.unet import ConditionalUnet1D

HORIZON, TO, TA = 16, 2, 8


def test_save_and_load(tmp_path):
    """Train a few steps through the RL-X algorithm class on pizza_dr_mjx, save, load: identical weights and normaliser."""
    from ml_collections import config_dict
    from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.default_config import get_config as dp_config
    from trust_region_irl.environments.pizza_dr_mjx.create_env import create_train_and_eval_env
    from trust_region_irl.environments.pizza_dr_mjx.default_config import get_config as env_config

    rng = np.random.default_rng(0)
    data = tmp_path / "demos.npz"
    np.savez(data, states=rng.normal(size=(150, 212)).astype(np.float32), actions=rng.uniform(-0.8, 0.8, (150, 2)).astype(np.float32),
             episode_lengths=np.array([50, 50, 50]))
    config = config_dict.ConfigDict()
    config.algorithm, config.environment = dp_config("diffusion_policy.flax_full_jit"), env_config("pizza_dr_mjx")
    config.runner = config_dict.ConfigDict(dict(save_model=True, track_console=False, track_tb=False, track_wandb=False, load_model=""))
    config.environment.nr_envs, config.environment.impl = 2, "jax"
    a = config.algorithm
    a.data_path, a.batch_size, a.total_gradient_steps, a.evaluation_frequency = str(data), 32, 3, -1
    a.down_dims, a.diffusion_step_embed_dim, a.lr_warmup_steps = "32,64,128", 32, 1
    train_env, eval_env = create_train_and_eval_env(config)
    model = DiffusionPolicy(config, train_env, eval_env, str(tmp_path / "run"), None)
    model.train()
    config.runner.load_model = str(tmp_path / "run" / "models" / "latest.model")
    loaded = DiffusionPolicy.load(config, train_env, eval_env, str(tmp_path / "run2"), None, [])
    for name in ("params", "ema_params"):
        same = jax.tree.map(lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y))), model.state[name], loaded.state[name])
        assert all(jax.tree.leaves(same)), name
    for k in ("scale", "offset"):
        assert np.array_equal(np.asarray(model.norm_obs[k]), np.asarray(loaded.norm_obs[k]))
        assert np.array_equal(np.asarray(model.norm_action[k]), np.asarray(loaded.norm_action[k]))
    assert int(model.state["step"]) == 3


def one_episode(T=200):
    """Actions are a smooth function of the time, which the observation encodes."""
    t = np.arange(T, dtype=np.float32)
    obs = np.stack([np.sin(t / 10), np.cos(t / 10), t / T, np.sin(t / 23), np.cos(t / 31), np.ones_like(t)], -1)
    actions = 0.8 * np.stack([np.sin(t / 7), np.cos(t / 5)], -1)
    return obs.astype(np.float32), actions.astype(np.float32)


@pytest.mark.slow
def test_overfit_one_episode():
    """Train on one episode; the sampled actions must reproduce it (the official training and sampling path: loss,
    AdamW + cosine schedule, EMA, 100-step DDPM sampling, actions from index n_obs_steps - 1). The U-Net is smaller
    than the default (32/64/128 channels) so the test also runs on a CPU; the default network itself is checked
    against the official one in test_dp_parity.py."""
    obs, actions = one_episode()
    rows = jnp.asarray(gather_index(create_indices([len(obs)], HORIZON, TO - 1, TA - 1), HORIZON))
    p_obs, p_act = fit_limits(obs), fit_limits(actions)
    nobs, nact = jnp.asarray(normalize(p_obs, obs)), jnp.asarray(normalize(p_act, actions))
    model = ConditionalUnet1D(input_dim=2, down_dims=(32, 64, 128), kernel_size=5, n_groups=8,
                              diffusion_step_embed_dim=64, cond_predict_scale=True)
    ddpm, ts = scheduler.make_ddpm(100), jnp.asarray(scheduler.inference_timesteps(100, 100))
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, HORIZON, 2)), jnp.zeros((1,), jnp.int32), jnp.zeros((1, TO * 6)))
    steps = 1500
    tx = dp.make_optimizer(dp.cosine_with_warmup(1e-4, 100, steps))
    state = (params, tx.init(params), params, jnp.asarray(0))

    @jax.jit
    def train(state, key):
        def body(state, k):
            params, opt_state, ema, n = state
            k_b, k_n, k_t = jax.random.split(k, 3)
            r = rows[jax.random.randint(k_b, (64,), 0, rows.shape[0])]
            noise = jax.random.normal(k_n, (64, HORIZON, 2))
            t = jax.random.randint(k_t, (64,), 0, 100)
            loss, g = jax.value_and_grad(lambda p: dp.compute_loss(model, p, ddpm, nobs[r], nact[r], noise, t, TO))(params)
            upd, opt_state = tx.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
            ema = dp.ema_update(ema, params, dp.ema_decay(n))
            return (params, opt_state, ema, n + 1), loss
        return jax.lax.scan(body, state, jax.random.split(key, steps))

    def action_error(p):
        starts = np.arange(0, len(obs) - HORIZON, 6)                   # windows fully inside the episode
        history = jnp.asarray(np.stack([obs[s:s + TO] for s in starts]))
        target = np.stack([actions[s + TO - 1:s + TO - 1 + TA] for s in starts])
        pred = dp.predict_action(model, p, ddpm, ts, to_jax(p_obs), to_jax(p_act), history, jax.random.PRNGKey(1), HORIZON, TO, TA)
        return float(np.abs(np.asarray(pred) - target).mean())

    before = action_error(params)
    state, losses = train(state, jax.random.PRNGKey(2))
    after = action_error(state[2])
    print(f"loss {float(losses[:50].mean()):.3f} -> {float(losses[-50:].mean()):.4f}; action error {before:.3f} -> {after:.4f}")
    assert float(losses[-50:].mean()) < 0.1 * float(losses[:50].mean())
    assert after < 0.05 and after < 0.2 * before
