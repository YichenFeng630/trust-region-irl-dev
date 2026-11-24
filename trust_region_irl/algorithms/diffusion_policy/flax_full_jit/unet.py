"""ConditionalUnet1D of Diffusion Policy (Chi et al. 2023), ported to Flax.

Source: real-stanford/diffusion_policy @ 5ba07ac
  diffusion_policy/model/diffusion/conditional_unet1d.py   ConditionalResidualBlock1D, ConditionalUnet1D
  diffusion_policy/model/diffusion/conv1d_components.py    Downsample1d, Upsample1d, Conv1dBlock
  diffusion_policy/model/diffusion/positional_embedding.py SinusoidalPosEmb

Differences from the PyTorch code:
  - Tensors are (batch, horizon, channels) instead of (batch, channels, horizon).
  - Only global conditioning (obs_as_global_cond=True, the lowdim default) is ported; local conditioning is not.
Parameters keep PyTorch's names, shapes and default initialisation (Conv1d / ConvTranspose1d / Linear weights and
biases ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)), GroupNorm weight 1 and bias 0), so an official state_dict maps onto
this module one to one (see tests/test_dp_parity.py).
"""
import math
from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def _torch_uniform(fan_in):
    bound = 1.0 / math.sqrt(fan_in)
    return lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -bound, bound)


class Linear(nn.Module):
    """torch.nn.Linear; weight (out, in)."""
    features: int

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        weight = self.param("weight", _torch_uniform(fan_in), (self.features, fan_in))
        bias = self.param("bias", _torch_uniform(fan_in), (self.features,))
        return x @ weight.T + bias


class Conv1d(nn.Module):
    """torch.nn.Conv1d on (batch, length, channels); weight (out, in, kernel)."""
    features: int
    kernel_size: int
    stride: int = 1
    padding: int = 0

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1] * self.kernel_size
        weight = self.param("weight", _torch_uniform(fan_in), (self.features, x.shape[-1], self.kernel_size))
        bias = self.param("bias", _torch_uniform(fan_in), (self.features,))
        y = jax.lax.conv_general_dilated(x, weight, window_strides=(self.stride,), padding=[(self.padding, self.padding)],
                                         dimension_numbers=("NWC", "OIW", "NWC"))
        return y + bias


class ConvTranspose1d(nn.Module):
    """torch.nn.ConvTranspose1d on (batch, length, channels); weight (in, out, kernel).

    Computed as a convolution of the stride-dilated input, padded by kernel - 1 - padding, with the flipped kernel."""
    features: int
    kernel_size: int
    stride: int = 1
    padding: int = 0

    @nn.compact
    def __call__(self, x):
        fan_in = self.features * self.kernel_size          # PyTorch's fan-in for a transposed-conv weight
        weight = self.param("weight", _torch_uniform(fan_in), (x.shape[-1], self.features, self.kernel_size))
        bias = self.param("bias", _torch_uniform(fan_in), (self.features,))
        rhs = jnp.flip(weight, axis=-1).transpose(1, 0, 2)  # (out, in, kernel)
        pad = self.kernel_size - 1 - self.padding
        y = jax.lax.conv_general_dilated(x, rhs, window_strides=(1,), padding=[(pad, pad)], lhs_dilation=(self.stride,),
                                         dimension_numbers=("NWC", "OIW", "NWC"))
        return y + bias


class GroupNorm(nn.Module):
    """torch.nn.GroupNorm(num_groups, channels) on (batch, length, channels): statistics over the length and the
    channels of each group (biased variance, eps 1e-5), then a per-channel affine map."""
    num_groups: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, t, c = x.shape
        g = x.reshape(b, t, self.num_groups, c // self.num_groups)
        mean = g.mean(axis=(1, 3), keepdims=True)
        var = jnp.square(g - mean).mean(axis=(1, 3), keepdims=True)
        x = ((g - mean) / jnp.sqrt(var + self.eps)).reshape(b, t, c)
        weight = self.param("weight", nn.initializers.ones, (c,))
        bias = self.param("bias", nn.initializers.zeros, (c,))
        return x * weight + bias


class SinusoidalPosEmb(nn.Module):
    dim: int

    def __call__(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""
    out_channels: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        x = Conv1d(self.out_channels, self.kernel_size, padding=self.kernel_size // 2, name="conv")(x)
        x = GroupNorm(self.n_groups, name="norm")(x)
        return mish(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning (per-channel scale and bias, or bias only) after the first."""
    out_channels: int
    kernel_size: int = 3
    n_groups: int = 8
    cond_predict_scale: bool = False

    @nn.compact
    def __call__(self, x, cond):
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups, name="blocks_0")(x)
        cond_channels = self.out_channels * 2 if self.cond_predict_scale else self.out_channels
        embed = Linear(cond_channels, name="cond_encoder")(mish(cond))
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels)
            out = embed[:, 0][:, None, :] * out + embed[:, 1][:, None, :]
        else:
            out = out + embed[:, None, :]
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups, name="blocks_1")(out)
        residual = x if x.shape[-1] == self.out_channels else Conv1d(self.out_channels, 1, name="residual_conv")(x)
        return out + residual


class ConditionalUnet1D(nn.Module):
    input_dim: int
    down_dims: Sequence[int] = (256, 512, 1024)
    kernel_size: int = 3
    n_groups: int = 8
    diffusion_step_embed_dim: int = 256
    cond_predict_scale: bool = False

    @nn.compact
    def __call__(self, sample, timestep, global_cond: Optional[jnp.ndarray] = None):
        """sample (B, T, input_dim); timestep scalar or (B,); global_cond (B, global_cond_dim) -> (B, T, input_dim)."""
        timesteps = jnp.broadcast_to(jnp.asarray(timestep), (sample.shape[0],))
        dsed = self.diffusion_step_embed_dim
        global_feature = SinusoidalPosEmb(dsed)(timesteps)
        global_feature = Linear(dsed * 4, name="diffusion_step_encoder_1")(global_feature)
        global_feature = Linear(dsed, name="diffusion_step_encoder_3")(mish(global_feature))
        if global_cond is not None:
            global_feature = jnp.concatenate([global_feature, global_cond], axis=-1)

        block = lambda channels, name: ConditionalResidualBlock1D(channels, self.kernel_size, self.n_groups,
                                                                  self.cond_predict_scale, name=name)
        all_dims = [self.input_dim] + list(self.down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        x, h = sample, []
        for ind, (_, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            x = block(dim_out, f"down_modules_{ind}_0")(x, global_feature)
            x = block(dim_out, f"down_modules_{ind}_1")(x, global_feature)
            h.append(x)
            if not is_last:
                x = Conv1d(dim_out, 3, stride=2, padding=1, name=f"down_modules_{ind}_2")(x)

        for ind in range(2):
            x = block(all_dims[-1], f"mid_modules_{ind}")(x, global_feature)

        for ind, (dim_in, _) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1   # as in the official code: never true for 3 down dims, both levels upsample
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = block(dim_in, f"up_modules_{ind}_0")(x, global_feature)
            x = block(dim_in, f"up_modules_{ind}_1")(x, global_feature)
            if not is_last:
                x = ConvTranspose1d(dim_in, 4, stride=2, padding=1, name=f"up_modules_{ind}_2")(x)

        x = Conv1dBlock(self.down_dims[0], self.kernel_size, name="final_conv_0")(x)   # official: default n_groups=8
        return Conv1d(self.input_dim, 1, name="final_conv_1")(x)
