"""Fixed-size replay buffer of SuperMimic (Sec. 3.1): the newest transitions replace the oldest.

Rows are time steps of all parallel environments, kept in time order, so a k-step window of one environment is k
consecutive rows. Per transition t it stores s_t, a_t, s_{t+1}, done_t (the episode ended after a_t: success, horizon or
divergence) and bad_t (diverged: s_{t+1} is unusable). At an episode end, s_{t+1} is the final observation but its
velocity entries are not known (the environment has already reset its physics state), see sample_windows.
"""
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ReplayBuffer:
    states: jax.Array        # (C, N, d)
    actions: jax.Array       # (C, N, a)
    next_states: jax.Array   # (C, N, d)
    done: jax.Array          # (C, N)
    bad: jax.Array           # (C, N)
    ptr: jax.Array           # next row to write
    filled: jax.Array        # number of rows written (<= C)

    @classmethod
    def create(cls, capacity, nr_envs, state_dim, action_dim):
        z = lambda *s: jnp.zeros(s, jnp.float32)
        return cls(z(capacity, nr_envs, state_dim), z(capacity, nr_envs, action_dim), z(capacity, nr_envs, state_dim),
                   z(capacity, nr_envs), z(capacity, nr_envs), jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))

    @property
    def capacity(self):
        return self.states.shape[0]

    def add(self, states, actions, next_states, done, bad):
        """Arrays of T rows (T, N, ...); the capacity must be a multiple of T so a write never wraps."""
        T = states.shape[0]
        assert self.capacity % T == 0, "replay capacity must be a multiple of the rollout length"
        upd = lambda buf, x: jax.lax.dynamic_update_slice_in_dim(buf, x.astype(buf.dtype), self.ptr, axis=0)
        return self.replace(states=upd(self.states, states), actions=upd(self.actions, actions),
                            next_states=upd(self.next_states, next_states), done=upd(self.done, done), bad=upd(self.bad, bad),
                            ptr=(self.ptr + T) % self.capacity, filled=jnp.minimum(self.filled + T, self.capacity))

    def _rows(self, logical):
        """Row of the logical time index (0 = oldest row still stored)."""
        oldest = (self.ptr - self.filled) % self.capacity
        return (oldest + logical) % self.capacity

    def sample_windows(self, key, batch_size, k):
        """k-step windows: s_0 (B, d), actions (B, k, a), targets s_1..s_k (B, k, d), valid (B, k), vel_valid (B, k).
        Step j is valid if no episode ended before it inside the window and its own target is usable; its velocity entries
        are valid if, in addition, the episode does not end at step j. Needs filled >= k."""
        k_t, k_e = jax.random.split(key)
        start = jax.random.randint(k_t, (batch_size,), 0, jnp.maximum(self.filled - k + 1, 1))
        env = jax.random.randint(k_e, (batch_size,), 0, self.states.shape[1])
        rows = self._rows(start[:, None] + jnp.arange(k)[None, :])          # (B, k)
        e = env[:, None]
        s0 = self.states[rows[:, 0], env]
        actions = self.actions[rows, e]
        targets = self.next_states[rows, e]
        done = self.done[rows, e]
        bad = self.bad[rows, e]
        ended_before = jnp.concatenate([jnp.zeros((batch_size, 1)), jnp.cumsum(done, axis=1)[:, :-1]], axis=1) > 0
        valid = (~ended_before) & (bad < 0.5)
        vel_valid = valid & (done < 0.5)
        return s0, actions, targets, valid.astype(jnp.float32), vel_valid.astype(jnp.float32)

    def sample_states(self, key, batch_size):
        """Start states s_t for imagination, uniform over all stored rows and environments."""
        k_t, k_e = jax.random.split(key)
        rows = self._rows(jax.random.randint(k_t, (batch_size,), 0, jnp.maximum(self.filled, 1)))
        env = jax.random.randint(k_e, (batch_size,), 0, self.states.shape[1])
        return self.states[rows, env]
