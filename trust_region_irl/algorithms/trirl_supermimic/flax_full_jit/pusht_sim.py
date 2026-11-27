"""The real PushT simulation (pusht_mjx) for the world-model checks of plan step 1 (scripts/supermimic): roll out a
policy and record what the world model needs, restore logged physics states, and replay fixed action sequences."""
import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx

from trust_region_irl.algorithms.trirl_supermimic.flax_full_jit import pusht_model as pm


def record_rollout(env, policy, policy_params, std, nr_envs, nr_steps, key):
    """Roll out the Gaussian policy (mean from policy_params, standard deviation std, or the policy's own if None) in
    nr_envs copies of env for nr_steps; returns numpy arrays, time-major (T, N, ...): state, next_state (30,
    pusht_model), qpos, qvel (physics before the step), action, done (episode ended after the step), terminated
    (success), diverged."""
    vel = pm.t_velocity_slice(env)
    key, reset_key = jax.random.split(key)

    @jax.jit
    def run(key):
        state = env.reset(jax.random.split(reset_key, nr_envs), False)

        def body(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            mean, logstd = policy.apply(policy_params, state.next_observation)
            s = jnp.exp(logstd) if std is None else std
            action = mean + s * jax.random.normal(sub, mean.shape)
            full = pm.full_state(state.next_observation, state.data.qvel, vel)
            qpos, qvel = state.data.qpos, state.data.qvel
            state = env.step(state, action)
            next_full = pm.full_state(state.actual_next_observation, state.data.qvel, vel)
            out = dict(state=full, next_state=next_full, qpos=qpos, qvel=qvel, action=action,
                       done=state.terminated | state.truncated, terminated=state.terminated,
                       diverged=state.info["env_info/diverged"] > 0.5)
            return (state, key), out

        _, out = jax.lax.scan(body, (state, key), None, nr_steps)
        return out

    out = run(key)
    return {k: np.asarray(v) for k, v in out.items()}


def states_from_physics(env, qpos, qvel, key):
    """Batched environment States whose physics is (qpos (N, nq), qvel (N, nv)), at the start of an episode and with
    success threshold -1, so the episode does not end at the goal and a fixed action sequence runs to its end."""
    n = qpos.shape[0]
    template = env.reset(jax.random.split(key, n), True)

    def set_one(st, qp, qv):
        data = st.data.replace(qpos=qp, qvel=qv, ctrl=jnp.zeros_like(st.data.ctrl))
        data = mjx.forward(env.mjx_model, data)
        obs = env.get_observation(data)
        store = {"episode_return": jnp.zeros((), jnp.float32), "episode_length": jnp.zeros((), jnp.int32)}
        return st.replace(data=data, next_observation=obs, actual_next_observation=obs, reward=jnp.zeros(()),
                          terminated=jnp.zeros((), bool), truncated=jnp.zeros((), bool),
                          success_threshold=jnp.float32(-1.0), info_episode_store=store)

    return jax.jit(jax.vmap(set_one))(template, jnp.asarray(qpos, jnp.float32), jnp.asarray(qvel, jnp.float32))


def replay_actions(env, state, actions):
    """state: batched States (N); actions (N, H, a) -> full states after every step (N, H, 30), done (N, H) (only a
    divergence or the horizon can end the episode here)."""
    vel = pm.t_velocity_slice(env)

    def body(st, a):
        st = env.step(st, a)
        return st, (pm.full_state(st.actual_next_observation, st.data.qvel, vel), st.terminated | st.truncated)

    _, (full, done) = jax.lax.scan(body, state, jnp.swapaxes(actions, 0, 1))
    return jnp.swapaxes(full, 0, 1), jnp.swapaxes(done, 0, 1)
