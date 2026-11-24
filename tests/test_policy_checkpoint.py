"""Tests for trust_region_irl.policy_checkpoint.load_actor, the actor interface of the evaluation scripts
(scripts/pizza/plot_policy_paths.py, scripts/balance/eval_policies.py): the Gaussian actor and the Diffusion Policy
actor (action chunks and observation history), and that the scripts' rollout loop gives the same result as the
evaluation inside Diffusion Policy training.

    python -m pytest tests/test_policy_checkpoint.py -q
"""
import numpy as np
import jax
import jax.numpy as jnp

from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.actor import DPActor
from trust_region_irl.policy_checkpoint import GaussianActor, load_actor


class RecordingDP(DPActor):
    """DPActor whose planner reports what it was given: action dim 0 = time of the older history frame + 1000 * index
    in the chunk, dim 1 = time of the newest frame (the observation's first entry is the step it was made at)."""
    def __init__(self):
        self.n_obs_steps, self.n_action_steps, self.action_dim = 2, 8, 2

    def plan(self, params, history, key):
        j = jnp.arange(self.n_action_steps)[None, :]
        older, newest = history[:, 0, 0][:, None], history[:, 1, 0][:, None]
        return jnp.stack([older + 1000 * j, newest + 0 * j], axis=-1)


def test_dp_chunking_and_history():
    actor, n, T = RecordingDP(), 2, 24
    obs_seq = jnp.stack([jnp.full((n, 3), float(t)) for t in range(T)])          # obs at step t holds t
    new_episode = jnp.zeros((T, n), bool).at[5, 1].set(True).at[16, 1].set(True)  # env 1 restarts at steps 5 and 16

    def step(state, inputs):
        t, obs, new = inputs
        action, state = actor.act(None, state, obs, t, new)
        return state, action
    _, actions = jax.lax.scan(step, actor.init(obs_seq[0], jax.random.PRNGKey(0)), (jnp.arange(T), obs_seq, new_episode))
    actions = np.asarray(actions)
    for t in range(T):
        p, j = 8 * (t // 8), t % 8                     # the chunk executed at t was planned at p
        for i in range(n):
            restarted_at_plan = bool(new_episode[p, i]) or p == 0
            older = p if restarted_at_plan else p - 1   # padded with the newest frame at an episode start
            assert actions[t, i, 1] == p, (t, i)
            assert actions[t, i, 0] == older + 1000 * j, (t, i, actions[t, i])


def test_gaussian_actor_is_the_mean_action():
    actor = GaussianActor(212, 2)
    obs = jax.random.normal(jax.random.PRNGKey(0), (5, 212))
    params = actor.policy.init(jax.random.PRNGKey(1), obs)
    action, state = actor.act(params, actor.init(obs, None), obs, 3, jnp.zeros(5, bool))
    assert np.array_equal(np.asarray(action), np.asarray(actor.policy.apply(params, obs)[0]))
    assert state == {}


def test_training_evaluation_matches_the_script_rollout(tmp_path):
    """A saved Diffusion Policy, evaluated by the algorithm class and by the evaluation scripts' loop (load_actor),
    on the same episodes and sampling noise: identical coverage."""
    from ml_collections import config_dict
    from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.diffusion_policy import DiffusionPolicy
    from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.default_config import get_config as dp_config
    from trust_region_irl.environments.pizza_dr_mjx.create_env import create_train_and_eval_env
    from trust_region_irl.environments.pizza_dr_mjx.default_config import get_config as env_config

    rng = np.random.default_rng(0)
    data = tmp_path / "demos.npz"
    np.savez(data, states=rng.normal(size=(150, 212)).astype(np.float32), actions=rng.uniform(-0.8, 0.8, (150, 2)).astype(np.float32),
             episode_lengths=np.array([50, 50, 50]))
    config = config_dict.ConfigDict()
    config.algorithm, config.environment = dp_config("diffusion_policy.flax_full_jit"), env_config("pizza_dr_mjx")
    config.runner = config_dict.ConfigDict(dict(save_model=True, track_console=False, track_tb=False, track_wandb=False))
    config.environment.nr_envs, config.environment.impl, config.environment.horizon = 3, "jax", 24
    a = config.algorithm
    a.data_path, a.batch_size, a.total_gradient_steps, a.evaluation_frequency = str(data), 32, 3, -1
    a.down_dims, a.diffusion_step_embed_dim, a.lr_warmup_steps, a.num_inference_steps = "32,64,128", 32, 1, 10
    a.num_train_timesteps = 10
    train_env, eval_env = create_train_and_eval_env(config)
    model = DiffusionPolicy(config, train_env, eval_env, str(tmp_path / "run"), None)
    model.train()
    key = jax.random.PRNGKey(7)
    from_class = jax.device_get(model._make_evaluate()(model.state["ema_params"], key))

    actor, params, name = load_actor(str(tmp_path / "run" / "models" / "latest.model"), 212, 2)
    assert name == "diffusion_policy.flax_full_jit" and isinstance(actor, DPActor)
    k, reset_key = jax.random.split(key)
    s = eval_env.reset(jax.random.split(reset_key, 3), True)
    state = actor.init(s.next_observation, k)
    for t in range(24):                                 # the scripts' loop, unrolled
        action, state = actor.act(params, state, s.next_observation, t, s.terminated | s.truncated)
        s = eval_env.step(s, action)
    assert np.isclose(float(np.mean(s.info["rollout/coverage"])), from_class["eval/coverage"], atol=1e-6)
