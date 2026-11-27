"""Deterministic world model of SuperMimic (Liao et al., 2026, arXiv 2610.28480, Sec. 3.2, Eq. 1) for flat states.

s_{t+1} = s_t + denorm(MLP(norm(s_t), clip(a_t))). The paper does not state the output parameterisation; we predict the
normalised state change (as MBPO) and start from a zero output layer, so an untrained model predicts the mean change.
Fit on k-step autoregressive windows: from the logged s_0 the model rolls forward on the logged actions, and every step's
prediction is compared with the logged state (Eq. 1 without the reward, observation and termination terms: in this
implementation those heads are analytic, see pusht_model.py, so they have no loss term).
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct


@struct.dataclass
class RunningStats:
    """Mean and variance of all rows added so far (parallel update of Chan et al.), per dimension."""
    count: jax.Array
    mean: jax.Array
    m2: jax.Array

    @classmethod
    def create(cls, dim):
        return cls(jnp.zeros((), jnp.float32), jnp.zeros((dim,), jnp.float32), jnp.zeros((dim,), jnp.float32))

    def update(self, x, weight):
        """x (n, d); weight (n,) in {0, 1}: rows with weight 0 are ignored."""
        w = weight.astype(jnp.float32)[:, None]
        n = jnp.sum(w)
        mean_b = jnp.sum(w * x, axis=0) / jnp.maximum(n, 1.0)
        m2_b = jnp.sum(w * (x - mean_b) ** 2, axis=0)
        total = self.count + n
        delta = mean_b - self.mean
        mean = self.mean + delta * n / jnp.maximum(total, 1.0)
        m2 = self.m2 + m2_b + delta ** 2 * self.count * n / jnp.maximum(total, 1.0)
        return RunningStats(total, jnp.where(n > 0, mean, self.mean), jnp.where(n > 0, m2, self.m2))

    def std(self, floor):
        return jnp.maximum(jnp.sqrt(self.m2 / jnp.maximum(self.count, 1.0)), floor)


@struct.dataclass
class Normalizer:
    state: RunningStats    # of s_t
    delta: RunningStats    # of s_{t+1} - s_t


class DynamicsMLP(nn.Module):
    out_dim: int
    hidden_dims: str = "512,512,512"

    @nn.compact
    def __call__(self, x):
        for width in [int(w) for w in self.hidden_dims.split(",")]:
            x = nn.Dense(width)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)


class WorldModel:
    """state_dim, action_dim; quat: (start, stop) of a unit quaternion inside the state, renormalised after every step;
    velocity_dims: indices whose target can be missing (masked by vel_valid in the loss); state_clip: the predicted
    state is clipped to +-state_clip (the environment clips its observation the same way)."""

    def __init__(self, state_dim, action_dim, hidden_dims="512,512,512", quat=None, velocity_dims=(),
                 state_std_floor=1e-3, delta_std_floor=1e-3, state_clip=10.0, action_low=-1.0, action_high=1.0):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.net = DynamicsMLP(state_dim, hidden_dims)
        self.quat = quat
        mask = np.zeros(state_dim, np.float32)
        mask[list(velocity_dims)] = 1.0
        self.velocity_mask = jnp.asarray(mask)
        self.state_std_floor, self.delta_std_floor = state_std_floor, delta_std_floor
        self.state_clip = state_clip
        self.action_low, self.action_high = action_low, action_high

    def init(self, key):
        return self.net.init(key, jnp.zeros((self.state_dim + self.action_dim,), jnp.float32))

    def init_normalizer(self):
        return Normalizer(RunningStats.create(self.state_dim), RunningStats.create(self.state_dim))

    def update_normalizer(self, norm, states, next_states, valid, vel_valid):
        """valid (n,): the transition's target is usable; vel_valid (n,): also its velocity entries (not at episode ends)."""
        delta_weight = valid * vel_valid   # deltas of episode-end steps miss the velocity entries, leave them out
        return Normalizer(norm.state.update(states, valid), norm.delta.update(next_states - states, delta_weight))

    def step(self, params, norm, state, action):
        """One step for a single state (state_dim,) and action (action_dim,); vmap for batches."""
        action = jnp.clip(action, self.action_low, self.action_high)
        x = jnp.concatenate([(state - norm.state.mean) / norm.state.std(self.state_std_floor), action])
        delta = self.net.apply(params, x) * norm.delta.std(self.delta_std_floor) + norm.delta.mean
        nxt = state + delta
        if self.quat is not None:
            a, b = self.quat
            q = nxt[a:b]
            nxt = nxt.at[a:b].set(q / jnp.maximum(jnp.linalg.norm(q), 1e-6))
        return jnp.clip(nxt, -self.state_clip, self.state_clip)

    def rollout(self, params, norm, state, actions):
        """state (state_dim,), actions (k, action_dim) -> predicted states (k, state_dim): s_1..s_k."""
        def body(s, a):
            s = self.step(params, norm, s, a)
            return s, s
        _, states = jax.lax.scan(body, state, actions)
        return states

    def window_errors(self, params, norm, state, actions, targets):
        """Per-step error (k, state_dim) divided by the std of the one-step change of each dimension."""
        pred = self.rollout(params, norm, state, actions)
        return (pred - targets) / norm.delta.std(self.delta_std_floor), pred

    def loss(self, params, norm, states, actions, targets, valid, vel_valid):
        """Eq. 1 (state term) on a batch of k-step windows: states (B, d), actions (B, k, a), targets (B, k, d),
        valid / vel_valid (B, k). Sum over the k steps, mean over the dimensions and the windows."""
        err, _ = jax.vmap(self.window_errors, in_axes=(None, None, 0, 0, 0))(params, norm, states, actions, targets)
        dim_mask = valid[..., None] * (1.0 - self.velocity_mask * (1.0 - vel_valid[..., None]))
        per_window = jnp.sum(dim_mask * err ** 2, axis=(1, 2)) / self.state_dim
        loss = jnp.mean(per_window)
        metrics = {"world_model/loss": loss,
                   "world_model/valid_steps": jnp.mean(valid)}
        return loss, metrics
