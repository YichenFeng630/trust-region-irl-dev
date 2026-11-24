"""Closed-loop Diffusion Policy for a step-by-step rollout loop (one action per call), so evaluation scripts written
for Gaussian policies can run it unchanged.

    state = actor.init(first_obs, key)
    action, state = actor.act(params, state, obs, t, new_episode)    # inside a lax.scan over t = 0, 1, ...

As the official MultiStepWrapper: the observation history holds the last n_obs_steps observations, padded with the
first one at the start of an episode; every n_action_steps steps the policy samples a new action chunk (100 DDPM
steps) and executes it one action at a time. Chunks are aligned to the rollout step t for all envs at once
(lax.cond on t % n_action_steps), so the steps in between cost nothing. An env that resets mid-chunk restarts its
history but finishes the current chunk; in fixed-horizon evaluations (no early ends) this never happens.
"""
import jax
import jax.numpy as jnp

from trust_region_irl.algorithms.diffusion_policy.flax_full_jit import policy as dp


class DPActor:
    def __init__(self, model, ddpm, timesteps_seq, norm_obs, norm_action, horizon, n_obs_steps, n_action_steps):
        self.model, self.ddpm, self.timesteps_seq = model, ddpm, timesteps_seq
        self.norm_obs, self.norm_action = norm_obs, norm_action
        self.horizon, self.n_obs_steps, self.n_action_steps = horizon, n_obs_steps, n_action_steps
        self.action_dim = int(norm_action["scale"].shape[-1])

    def plan(self, params, history, key):
        """(N, n_obs_steps, Do) unnormalised history -> (N, n_action_steps, Da) actions to execute."""
        return dp.predict_action(self.model, params, self.ddpm, self.timesteps_seq, self.norm_obs, self.norm_action,
                                 history, key, self.horizon, self.n_obs_steps, self.n_action_steps)

    def init(self, obs, key):
        return {"history": jnp.repeat(obs[:, None], self.n_obs_steps, axis=1),
                "chunk": jnp.zeros((obs.shape[0], self.n_action_steps, self.action_dim), obs.dtype), "key": key}

    def act(self, params, state, obs, t, new_episode):
        """obs (N, Do) is the current observation, new_episode (N,) marks envs whose episode starts at this step."""
        pushed = jnp.concatenate([state["history"][:, 1:], obs[:, None]], axis=1)
        history = jnp.where(new_episode[:, None, None], jnp.repeat(obs[:, None], self.n_obs_steps, axis=1), pushed)
        key, k = jax.random.split(state["key"])
        i = t % self.n_action_steps
        chunk = jax.lax.cond(i == 0, lambda: self.plan(params, history, k), lambda: state["chunk"])
        return chunk[:, i], {"history": history, "chunk": chunk, "key": key}
