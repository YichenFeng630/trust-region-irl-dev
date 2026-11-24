from typing import Optional, Tuple

import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_discriminator(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    # reward_input: "observation" = the whole observation; "embodiment-free" = only the environment's
    # encoder_input_indices (no robot joints), the same input as TRIRL-SPR's byol-embodiment-free reward
    if config.algorithm.reward_input == "observation":
        input_indices = None
    elif config.algorithm.reward_input == "embodiment-free":
        input_indices = tuple(int(i) for i in env.encoder_input_indices)
    else:
        raise ValueError(f"Unknown reward_input {config.algorithm.reward_input}, expected 'observation' or 'embodiment-free'")

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return DiscriminatorShaped(config.algorithm.handle_absorbing_states, config.algorithm.gamma, input_indices,
                                   config.algorithm.discriminator_uses_action)


class DiscriminatorShaped(nn.Module):
    handle_absorbing_states: bool
    gamma: int
    input_indices: Optional[Tuple[int, ...]] = None   # observation entries g and h read; None = all
    use_action: bool = False                          # g reads (s, a) instead of s; h always reads the state only

    def setup(self):
        self.gnet_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense3 = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))

        self.hnet_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense3 = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))


    def __call__(self, x, a, x_n, absorbing, logp, shaping=1.0):
        """
        D(s) + gamma h(s') - h(s)
        Args:
            x : state
            a: action (not used)
            x_n: next state
            abs: bool for whether the next state is absorbing
        """
        if self.input_indices is not None:
            idx = jnp.asarray(self.input_indices)
            x, x_n = x[..., idx], x_n[..., idx]
        if self.use_action:
            # as DiscriminatorShapedSA in the paper code: g(s, a); g at the next state (absorbing case only) gets a zero action
            g_in, g_in_n = jnp.concatenate([x, a], axis=-1), jnp.concatenate([x_n, jnp.zeros_like(a)], axis=-1)
        else:
            g_in, g_in_n = x, x_n

        # g(x)
        r = self.gnet_dense1(g_in)
        r = nn.tanh(r)
        r = self.gnet_dense2(r)
        r = nn.tanh(r)
        r = self.gnet_dense3(r)

        # g(x_n)
        rx_n = self.gnet_dense1(g_in_n)
        rx_n = nn.tanh(rx_n)
        rx_n = self.gnet_dense2(rx_n)
        rx_n = nn.tanh(rx_n)
        rx_n = self.gnet_dense3(rx_n)

        # h(x)
        hx = self.hnet_dense1(x)
        hx = nn.tanh(hx)
        hx = self.hnet_dense2(hx)
        hx = nn.tanh(hx)
        hx = self.hnet_dense3(hx)

        # h(x_n)
        hx_n = self.hnet_dense1(x_n)
        hx_n = nn.tanh(hx_n)
        hx_n = self.hnet_dense2(hx_n)
        hx_n = nn.tanh(hx_n)
        hx_n = self.hnet_dense3(hx_n)

        absorbing = jnp.asarray(absorbing, dtype=r.dtype)
        absorbing = jnp.broadcast_to(absorbing[..., None], r.shape)

        # Shaped reward: step. 6 in alg 1 in https://arxiv.org/pdf/1710.11248v2
        if self.handle_absorbing_states:
            f = r + shaping * ((1 - absorbing) * self.gamma * hx_n + absorbing * ((self.gamma/(1 - self.gamma)) * rx_n) - hx)
        else:
            f = r + shaping * (self.gamma * hx_n - hx)
        reward = f - shaping * logp # D = sigmoid(reward)

        return reward.squeeze(-1)