"""Tests for AIRL (trust_region_irl.algorithms.airl_ppo.flax_full_jit) on pizza_dr_mjx, and its PushT options (task-based
reward input, success-threshold curriculum) on pusht_mjx.

    python -m pytest tests/test_airl.py            # all, about 4 min on a laptop GPU
    python -m pytest tests/test_airl.py -m "not slow"
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from ml_collections import config_dict

from trust_region_irl.algorithms.airl_ppo.flax_full_jit import airl_ppo as airl_module
from trust_region_irl.algorithms.airl_ppo.flax_full_jit.airl_ppo import AIRL_PPO
from trust_region_irl.algorithms.airl_ppo.flax_full_jit.default_config import get_config as airl_config
from trust_region_irl.algorithms.airl_ppo.flax_full_jit.discriminator import get_discriminator
from trust_region_irl.environments.pizza_dr_mjx.create_env import create_train_and_eval_env
from trust_region_irl.environments.pizza_dr_mjx.default_config import get_config as env_config

OBS, ACT = 212, 2


def make_config(data_path, nr_envs=16, **algorithm):
    config = config_dict.ConfigDict()
    config.algorithm = airl_config("airl_ppo.flax_full_jit")
    config.environment = env_config("pizza_dr_mjx")
    config.runner = config_dict.ConfigDict(dict(save_model=True, track_console=False, track_tb=False, track_wandb=False))
    config.environment.nr_envs, config.environment.seed, config.environment.impl = nr_envs, 0, "jax"
    config.algorithm.data_path = str(data_path)
    for k, v in algorithm.items():
        config.algorithm[k] = v
    return config


def fake_expert_data(path, n=4000, seed=0, obs_dim=OBS):
    rng = np.random.default_rng(seed)
    ab = np.zeros(n, np.float32); ab[799::800] = 1.0
    np.savez(path, states=rng.normal(size=(n, obs_dim)).astype(np.float32), actions=rng.uniform(-1, 1, (n, ACT)).astype(np.float32),
             next_states=rng.normal(size=(n, obs_dim)).astype(np.float32), absorbing=ab, rewards=np.zeros(n, np.float32))
    return path


PUSHT_OBS = 24


def make_pusht_config(data_path, nr_envs=16, **algorithm):
    """make_config on pusht_mjx with random start poses within 0.2 m, as in the PushT comparison."""
    from trust_region_irl.environments.pusht_mjx.default_config import get_config as pusht_env_config
    config = make_config(data_path, nr_envs, **algorithm)
    config.environment = pusht_env_config("pusht_mjx")
    config.environment.nr_envs, config.environment.seed, config.environment.impl = nr_envs, 0, "jax"
    config.environment.random_TPos, config.environment.spawn_radius = True, 0.2
    return config


def create_pusht_envs(config):
    from trust_region_irl.environments.pusht_mjx.create_env import create_train_and_eval_env as create_pusht
    return create_pusht(config)


def flat(tree):
    return np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree.leaves(tree)])


def test_this_repository_provides_airl():
    """The RL-X runner falls back silently between packages; make sure AIRL comes from this repository."""
    assert AIRL_PPO.__module__ == "trust_region_irl.algorithms.airl_ppo.flax_full_jit.airl_ppo"
    assert "/trust_region_irl/algorithms/airl_ppo/flax_full_jit/" in airl_module.__file__


@pytest.mark.parametrize("reward_input", ["observation", "embodiment-free"])
def test_reward_input(tmp_path, reward_input):
    """embodiment-free: changing the arm joints (observation 6:20) leaves the discriminator output unchanged."""
    config = make_config(fake_expert_data(tmp_path / "data.npz"), reward_input=reward_input)
    env, _ = create_train_and_eval_env(config)
    disc = get_discriminator(config, env)
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    x, xn = jax.random.normal(k[0], (8, OBS)), jax.random.normal(k[1], (8, OBS))
    a, ab, logp = jnp.zeros((8, ACT)), jnp.zeros(8), jnp.zeros(8)
    params = disc.init(k[2], x[:1], a[:1], xn[:1], ab[:1], logp[:1])
    # per sample, as in the training loop (the discriminator takes one transition with a scalar log-prob)
    per_sample = jax.vmap(lambda s, act, sn, absorbing, lp: disc.apply(params, s, act, sn, absorbing, lp))
    out = lambda s, sn: per_sample(s, a, sn, ab, logp)
    base = out(x, xn)
    joints = jnp.zeros(OBS).at[6:20].set(1.0)
    task = jnp.zeros(OBS).at[0].set(1.0)
    joint_change = float(jnp.abs(out(x + joints, xn + joints) - base).max())
    task_change = float(jnp.abs(out(x + task, xn + task) - base).max())
    assert task_change > 1e-6
    if reward_input == "embodiment-free":
        assert tuple(env.encoder_input_indices) == tuple(range(0, 6)) + tuple(range(20, OBS))
        assert joint_change == 0.0
    else:
        assert joint_change > 1e-6


def train_and_capture(config, tmp_path):
    train_env, eval_env = create_train_and_eval_env(config)
    model = AIRL_PPO(config, train_env, eval_env, str(tmp_path / f"run_{config.algorithm.learning_rate_disc}"), None)
    init = {"policy": flat(model.policy_state.params), "disc": flat(model.discriminator_state.params)}
    saved = []
    model.save = lambda p, c, d, file_name=None: saved.append({"policy": flat(p.params), "disc": flat(d.params)})
    model.train()
    assert saved, "the training loop never saved"
    return init, saved[-1]


@pytest.mark.slow
def test_discriminator_updates_reach_the_policy(tmp_path):
    """Same seed, discriminator LR > 0 vs 0. If the PPO reward used the initial discriminator (the bug in the earlier
    copy of this file), both runs would end with identical policies."""
    data = fake_expert_data(tmp_path / "data.npz")
    batch = 16 * 10
    common = dict(total_timesteps=batch * 6, evaluation_and_save_frequency=batch * 3, evaluation_active=False,
                  minibatch_size=80, nr_epochs=2, nr_epochs_disc=2, std_dev=0.4,
                  reward_input="embodiment-free", handle_absorbing_states=False)
    init_on, on = train_and_capture(make_config(data, learning_rate_disc=1e-3, **common), tmp_path)
    init_off, off = train_and_capture(make_config(data, learning_rate_disc=0.0, **common), tmp_path)
    assert np.array_equal(init_on["policy"], init_off["policy"]) and np.array_equal(init_on["disc"], init_off["disc"])
    assert np.abs(on["disc"] - init_on["disc"]).max() > 1e-4, "the discriminator did not train"
    assert np.array_equal(off["disc"], init_off["disc"]), "discriminator LR 0 should leave it unchanged"
    assert np.abs(on["policy"] - off["policy"]).max() > 1e-4, "the discriminator updates do not reach the policy"


@pytest.mark.parametrize("uses_action", [False, True])
def test_discriminator_uses_action(tmp_path, uses_action):
    """discriminator_uses_action: g reads the action as well; off (default), the output does not depend on it."""
    config = make_config(fake_expert_data(tmp_path / "data.npz"), reward_input="embodiment-free", discriminator_uses_action=uses_action)
    env, _ = create_train_and_eval_env(config)
    disc = get_discriminator(config, env)
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    x, xn, a = jax.random.normal(k[0], (8, OBS)), jax.random.normal(k[1], (8, OBS)), jax.random.uniform(k[2], (8, ACT), minval=-1, maxval=1)
    ab, logp = jnp.zeros(8), jnp.zeros(8)
    params = disc.init(k[3], x[:1], a[:1], xn[:1], ab[:1], logp[:1])
    out = jax.vmap(lambda s, act, sn: disc.apply(params, s, act, sn, 0.0, 0.0))
    assert params["params"]["gnet_dense1"]["kernel"].shape[0] == 198 + (ACT if uses_action else 0)
    assert params["params"]["hnet_dense1"]["kernel"].shape[0] == 198
    action_change = float(jnp.abs(out(x, a + 0.5, xn) - out(x, a, xn)).max())
    joints = jnp.zeros(OBS).at[6:20].set(1.0)
    assert float(jnp.abs(out(x + joints, a, xn + joints) - out(x, a, xn)).max()) == 0.0
    assert (action_change > 1e-6) if uses_action else (action_change == 0.0)


def test_cap_std_dev():
    from trust_region_irl.algorithms.airl_ppo.flax_full_jit.policy import cap_std_dev
    params = {"params": {"policy_logstd": jnp.log(jnp.array([[0.2, 3.0]])), "Dense_0": {"kernel": jnp.ones((3, 2))}}}
    capped = cap_std_dev(params, 1.0)
    assert jax.tree.structure(capped) == jax.tree.structure(params)
    assert np.allclose(np.exp(capped["params"]["policy_logstd"]), [[0.2, 1.0]])
    assert np.array_equal(capped["params"]["Dense_0"]["kernel"], params["params"]["Dense_0"]["kernel"])
    assert cap_std_dev(params, -1.0) is params


@pytest.mark.slow
def test_std_cap_and_action_input_in_training(tmp_path):
    """A short training run with both options on: the std never leaves the cap, even with a large entropy bonus
    pushing it up, and the discriminator (with the action in its gradient penalty) trains."""
    data = fake_expert_data(tmp_path / "data.npz")
    batch = 16 * 10
    config = make_config(data, total_timesteps=batch * 6, evaluation_and_save_frequency=batch * 3, evaluation_active=False,
                         minibatch_size=80, nr_epochs=2, nr_epochs_disc=2, std_dev=0.4, entropy_coef=1.0, learning_rate=1e-2,
                         learning_rate_disc=1e-3, reward_input="embodiment-free", handle_absorbing_states=False,
                         discriminator_uses_action=True, max_std_dev=0.5)
    train_env, eval_env = create_train_and_eval_env(config)
    model = AIRL_PPO(config, train_env, eval_env, str(tmp_path / "run"), None)
    init_disc = flat(model.discriminator_state.params)
    saved = []
    model.save = lambda p, c, d, file_name=None: saved.append((np.exp(np.asarray(p.params["params"]["policy_logstd"])), flat(d.params)))
    model.train()
    std, disc = saved[-1]
    assert np.all(std <= 0.5 + 1e-6) and np.all(std > 0.4 + 1e-3), std   # pushed up by the entropy bonus, stopped at the cap
    assert np.abs(disc - init_disc).max() > 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("nr_disc_steps", [-1, 3])
def test_nr_disc_steps(tmp_path, nr_disc_steps):
    """nr_disc_steps > 0: exactly that many discriminator gradient steps per iteration; default: nr_epochs_disc passes
    over the batch (RL-X). The policy update is the same either way."""
    data = fake_expert_data(tmp_path / "data.npz")
    batch, iterations = 16 * 10, 4
    config = make_config(data, total_timesteps=batch * iterations, evaluation_and_save_frequency=batch * 2, evaluation_active=False,
                         minibatch_size=40, nr_epochs=2, nr_epochs_disc=2, std_dev=0.4, learning_rate_disc=1e-3,
                         reward_input="embodiment-free", handle_absorbing_states=False, nr_disc_steps=nr_disc_steps)
    train_env, eval_env = create_train_and_eval_env(config)
    model = AIRL_PPO(config, train_env, eval_env, str(tmp_path / "run"), None)
    saved = []
    model.save = lambda p, c, d, file_name=None: saved.append((int(p.step), int(d.step)))
    model.train()
    policy_steps, disc_steps = saved[-1]
    assert policy_steps == iterations * 2 * (batch // 40)
    assert disc_steps == iterations * (nr_disc_steps if nr_disc_steps > 0 else 2 * (batch // 40))


@pytest.mark.parametrize("log_pi_weight", [1.0, 0.1])
def test_log_pi_weight(tmp_path, log_pi_weight):
    """The policy reward is f - w log pi (default w = 1: the discriminator logits themselves)."""
    data = fake_expert_data(tmp_path / "data.npz")
    batch = 16 * 10
    config = make_config(data, total_timesteps=batch * 2, evaluation_and_save_frequency=batch * 2, evaluation_active=False,
                         minibatch_size=80, nr_epochs=1, nr_epochs_disc=1, std_dev=0.4, learning_rate_disc=1e-3,
                         reward_input="embodiment-free", handle_absorbing_states=False, log_pi_weight=log_pi_weight)
    train_env, eval_env = create_train_and_eval_env(config)
    model = AIRL_PPO(config, train_env, eval_env, str(tmp_path / "run"), None)
    logged = []
    model.log = lambda name, value, step: logged.append((step, name, float(value)))
    model.save = lambda *a, **k: None
    model.train()
    steps = sorted({s for s, n, v in logged if n == "reward/airl_reward"})
    assert len(steps) == 2
    for s in steps:
        m = {n: v for st, n, v in logged if st == s}
        assert abs(m["reward/airl_reward"] - (m["reward/f"] + log_pi_weight * m["reward/minus_log_pi"])) < 1e-4 * (1 + abs(m["reward/f"]))
        assert m["reward/minus_log_pi"] > 0.1   # std 0.4 on 2 actions: -log pi is about 1 on average, so the weight is visible


@pytest.mark.parametrize("terminal_tail", ["double", "single", "triple"])
def test_terminal_tail(tmp_path, terminal_tail):
    """Reward of the absorbing state after an episode ends in s' (GAE adds gamma/(1-gamma) times it plus entropy_coef H_T).
    double (default, RL-X / dev code): g(s')/(1-gamma) - h(s') + w H_T, so g(s') counts again on top of the
    gamma/(1-gamma) g(s') in the ending step's own f; single: w H_T only. Other values are refused."""
    batch = 16 * 10
    config = make_pusht_config(fake_expert_data(tmp_path / "data.npz", obs_dim=PUSHT_OBS), reward_input="task-based",
                               log_pi_weight=0.03, terminal_tail=terminal_tail, total_timesteps=batch,
                               evaluation_and_save_frequency=batch, evaluation_active=False, minibatch_size=80)
    train_env, eval_env = create_pusht_envs(config)
    if terminal_tail == "triple":
        with pytest.raises(ValueError):
            AIRL_PPO(config, train_env, eval_env, str(tmp_path / "run"), None)
        return
    model = AIRL_PPO(config, train_env, eval_env, str(tmp_path / "run"), None)
    params = model.discriminator_state.params
    s = jax.random.normal(jax.random.PRNGKey(1), (8, PUSHT_OBS))
    a = jnp.zeros((8, ACT))
    mlp = lambda d1, d2, d3, x: d3(jnp.tanh(d2(jnp.tanh(d1(x)))))[..., 0]
    g = model.discriminator.apply(params, s[:, :7], method=lambda m, x: mlp(m.gnet_dense1, m.gnet_dense2, m.gnet_dense3, x))
    h = model.discriminator.apply(params, s[:, :7], method=lambda m, x: mlp(m.hnet_dense1, m.hnet_dense2, m.hnet_dense3, x))
    gamma, w, H_T = config.algorithm.gamma, 0.03, 2 * np.log(2.0)
    expected = (g / (1 - gamma) - h + w * H_T) if terminal_tail == "double" else w * H_T * jnp.ones(8)
    assert float(jnp.abs(g / (1 - gamma) - h).max()) > 1e-3   # the two variants differ by this much here
    assert np.allclose(model.absorbing_state_reward(params, s, a), expected, rtol=1e-4, atol=1e-5)


def test_reward_input_task_based(tmp_path):
    """task-based (PushT): g and h read the T pose only (observation 0-6), as TRIRL-SPR's byol-task-based-observation; the
    end effector and the arm (7-23) do not change the output."""
    config = make_pusht_config(fake_expert_data(tmp_path / "data.npz", obs_dim=PUSHT_OBS), reward_input="task-based")
    env, _ = create_pusht_envs(config)
    disc = get_discriminator(config, env)
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    x, xn = jax.random.normal(k[0], (8, PUSHT_OBS)), jax.random.normal(k[1], (8, PUSHT_OBS))
    a, ab, logp = jnp.zeros((8, ACT)), jnp.zeros(8), jnp.zeros(8)
    params = disc.init(k[2], x[:1], a[:1], xn[:1], ab[:1], logp[:1])
    out = jax.vmap(lambda s, sn: disc.apply(params, s, a[0], sn, 0.0, 0.0))
    assert params["params"]["gnet_dense1"]["kernel"].shape[0] == 7
    robot = jnp.zeros(PUSHT_OBS).at[7:].set(1.0)
    base = out(x, xn)
    assert float(jnp.abs(out(x + robot, xn + robot) - base).max()) == 0.0
    for i in range(7):
        task = jnp.zeros(PUSHT_OBS).at[i].set(1.0)
        assert float(jnp.abs(out(x + task, xn + task) - base).max()) > 1e-6, i


@pytest.mark.slow
def test_success_threshold_curriculum(tmp_path):
    """PushT's success-threshold curriculum: each training rollout uses its stage's threshold (logged as
    env/success_threshold), the evaluation always the final one. Thresholds 0 then 10: no training episode can succeed in
    the first half, every one succeeds on its first step in the second half, and so does every evaluation episode."""
    data = fake_expert_data(tmp_path / "data.npz", obs_dim=PUSHT_OBS)
    batch = 16 * 10
    config = make_pusht_config(data, total_timesteps=batch * 4, evaluation_and_save_frequency=batch * 2, evaluation_active=True,
                               minibatch_size=80, nr_epochs=1, nr_epochs_disc=1, std_dev=0.4, learning_rate_disc=1e-3,
                               reward_input="task-based")
    config.environment.curriculum_success = True
    config.environment.success_threshold_values, config.environment.success_threshold_fractions = "0.0,10.0", "0.5,0.5"
    train_env, eval_env = create_pusht_envs(config)
    model = AIRL_PPO(config, train_env, eval_env, str(tmp_path / "run"), None)
    logged = []
    model.log = lambda name, value, step: logged.append((step, name, float(value)))
    model.save = lambda *a, **k: None
    model.train()
    series = lambda name: [v for s, v in sorted((s, v) for s, n, v in logged if n == name)]
    assert series("env/success_threshold") == [0.0, 0.0, 10.0, 10.0]
    train_success = series("env_info/is_success")
    assert train_success[:2] == [0.0, 0.0] and train_success[2:] == [1.0, 1.0]
    assert series("eval/is_success") == [1.0, 1.0]
