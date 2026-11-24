"""Diffusion Policy core functions: training loss, action sampling, EMA and the learning-rate schedule.

Sources (real-stanford/diffusion_policy @ 5ba07ac unless stated otherwise):
  compute_loss, predict_action, conditional_sample  diffusion_policy/policy/diffusion_unet_lowdim_policy.py
      (obs_as_global_cond=True, pred_action_steps_only=False, oa_step_convention=True: nothing is in-painted, the
      loss covers the whole action trajectory, and the executed actions start at index n_obs_steps - 1)
  ema_decay, ema_update                            diffusion_policy/model/diffusion/ema_model.py (EMAModel)
  cosine_with_warmup                               diffusers 0.11.1 optimization.get_cosine_schedule_with_warmup,
                                                   stepped once per optimizer step as in the official workspace
  make_optimizer                                   torch.optim.AdamW(lr, betas, eps, weight_decay) on all parameters
"""
import jax
import jax.numpy as jnp
import optax

from trust_region_irl.algorithms.diffusion_policy.flax_full_jit import scheduler as ddpm_lib
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.normalizer import normalize, unnormalize


def global_cond(nobs, n_obs_steps):
    """Flatten the first n_obs_steps (normalised) observations of each window into one conditioning vector."""
    return nobs[:, :n_obs_steps].reshape(nobs.shape[0], -1)


def compute_loss(model, params, ddpm, nobs, naction, noise, timesteps, n_obs_steps):
    """MSE between the predicted and the added noise. nobs (B, T, Do) and naction (B, T, Da) are normalised windows;
    noise (B, T, Da) ~ N(0, I) and timesteps (B,) ~ U{0, ..., num_train_timesteps - 1} are the random draws."""
    cond = global_cond(nobs, n_obs_steps)
    noisy = ddpm_lib.add_noise(ddpm, naction, noise, timesteps)
    pred = model.apply(params, noisy, timesteps, cond)
    return jnp.mean(jnp.square(pred - noise))


def conditional_sample(model, params, ddpm, timesteps_seq, cond, init_noise, step_noises):
    """Reverse diffusion from init_noise (B, T, Da); step_noises (n_steps, B, T, Da) are the per-step variance draws."""
    def body(x, inputs):
        t, noise = inputs
        eps = model.apply(params, x, t, cond)
        return ddpm_lib.step(ddpm, eps, t, x, noise), None
    x, _ = jax.lax.scan(body, init_noise, (timesteps_seq, step_noises))
    return x


def predict_action(model, params, ddpm, timesteps_seq, norm_obs, norm_action, obs_history, key,
                   horizon, n_obs_steps, n_action_steps):
    """obs_history (B, n_obs_steps, Do), unnormalised -> the n_action_steps actions to execute (B, n_action_steps, Da)."""
    batch, action_dim = obs_history.shape[0], norm_action["scale"].shape[-1]
    cond = global_cond(normalize(norm_obs, obs_history), n_obs_steps)
    k_init, k_steps = jax.random.split(key)
    init_noise = jax.random.normal(k_init, (batch, horizon, action_dim))
    step_noises = jax.random.normal(k_steps, (len(timesteps_seq), batch, horizon, action_dim))
    naction = conditional_sample(model, params, ddpm, timesteps_seq, cond, init_noise, step_noises)
    start = n_obs_steps - 1
    return unnormalize(norm_action, naction)[:, start:start + n_action_steps]


def ema_decay(optimization_step, update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999):
    """EMAModel.get_decay; optimization_step = number of EMA updates done before this one."""
    step = jnp.maximum(0, optimization_step - update_after_step - 1)
    value = 1.0 - (1.0 + step / inv_gamma) ** -power
    return jnp.where(step <= 0, 0.0, jnp.clip(value, min_value, max_value))


def ema_update(ema_params, params, decay):
    return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params)


def cosine_with_warmup(base_lr, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """lr at optimizer update `count` (0 for the first update, as torch LambdaLR stepped after each optimizer step)."""
    def schedule(count):
        count = jnp.asarray(count, jnp.float32)
        warm = count / max(1, num_warmup_steps)
        progress = (count - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * num_cycles * 2.0 * progress)))
        return base_lr * jnp.where(count < num_warmup_steps, warm, cosine)
    return schedule


def make_optimizer(learning_rate, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6):
    """torch AdamW: decoupled weight decay on every parameter, no gradient clipping."""
    return optax.adamw(learning_rate=learning_rate, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)
