"""DDPM noise scheduler as used by Diffusion Policy (diffusers DDPMScheduler, the version pinned by the official code:
diffusers 0.11.1, diffusers/schedulers/scheduling_ddpm.py).

Supported configuration (the lowdim U-Net default): beta_schedule "squaredcos_cap_v2" or "linear",
variance_type "fixed_small", prediction_type "epsilon", clip_sample True, and num_inference_steps equal to or a
divisor of num_train_timesteps. With num_inference_steps == num_train_timesteps (the default, 100) the step below is
identical to later diffusers versions as well.
"""
import math
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp


class DDPM(NamedTuple):
    betas: jnp.ndarray            # (T,)
    alphas: jnp.ndarray           # (T,)
    alphas_cumprod: jnp.ndarray   # (T,)
    clip_sample: bool


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """Glide cosine schedule ("squaredcos_cap_v2"), as diffusers."""
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


def make_ddpm(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", beta_start=1e-4, beta_end=0.02, clip_sample=True):
    if beta_schedule == "squaredcos_cap_v2":
        betas = betas_for_alpha_bar(num_train_timesteps)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    else:
        raise NotImplementedError(beta_schedule)
    alphas = (1.0 - betas).astype(np.float32)
    alphas_cumprod = np.cumprod(alphas, axis=0, dtype=np.float32)
    return DDPM(jnp.asarray(betas), jnp.asarray(alphas), jnp.asarray(alphas_cumprod), bool(clip_sample))


def inference_timesteps(num_train_timesteps, num_inference_steps):
    """diffusers set_timesteps: arange(0, T, T // n)[::-1]."""
    n = min(num_train_timesteps, num_inference_steps)
    return np.arange(0, num_train_timesteps, num_train_timesteps // n)[::-1].copy()


def add_noise(ddpm, original_samples, noise, timesteps):
    """q(x_t | x_0); timesteps (B,) int."""
    acp = ddpm.alphas_cumprod[timesteps]
    shape = (-1,) + (1,) * (original_samples.ndim - 1)
    return jnp.sqrt(acp).reshape(shape) * original_samples + jnp.sqrt(1.0 - acp).reshape(shape) * noise


def step(ddpm, model_output, t, sample, noise):
    """One reverse step x_t -> x_{t-1} (epsilon prediction, fixed_small variance); t is a scalar timestep and
    `noise` the standard-normal draw diffusers would take with torch.randn (ignored at t == 0)."""
    alpha_prod_t = ddpm.alphas_cumprod[t]
    alpha_prod_t_prev = jnp.where(t > 0, ddpm.alphas_cumprod[jnp.maximum(t - 1, 0)], 1.0)
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev

    pred_original_sample = (sample - jnp.sqrt(beta_prod_t) * model_output) / jnp.sqrt(alpha_prod_t)
    if ddpm.clip_sample:
        pred_original_sample = jnp.clip(pred_original_sample, -1.0, 1.0)

    pred_original_sample_coeff = (jnp.sqrt(alpha_prod_t_prev) * ddpm.betas[t]) / beta_prod_t
    current_sample_coeff = jnp.sqrt(ddpm.alphas[t]) * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    variance = jnp.maximum(beta_prod_t_prev / beta_prod_t * ddpm.betas[t], 1e-20)   # fixed_small
    return pred_prev_sample + jnp.where(t > 0, jnp.sqrt(variance), 0.0) * noise
