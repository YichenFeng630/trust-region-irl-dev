"""Diffusion Policy (Chi et al. 2023, lowdim CNN U-Net) as an RL-X algorithm: offline training on the expert dataset,
with periodic closed-loop evaluation in the environment.

Training follows the official workspace (diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py):
the normaliser is fitted on the whole dataset, every epoch visits all windows in a random order, AdamW with a cosine
schedule and warm-up is stepped once per batch, and the EMA copy is updated after every step and used for evaluation.
Differences: a fixed number of gradient steps instead of epochs (the last batch of an epoch is dropped), no validation
split, and the checkpoint is the last one (no top-k selection on evaluation scores).
"""
import json
import logging
import os
import shutil
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
import wandb
from flax.training import orbax_utils

from trust_region_irl.algorithms.diffusion_policy.flax_full_jit import policy as dp
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.actor import DPActor
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit import scheduler as ddpm_lib
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.dataset import create_indices, gather_index, episode_ends_from
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.general_properties import GeneralProperties
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.normalizer import fit_limits, normalize, to_jax
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.unet import ConditionalUnet1D

rlx_logger = logging.getLogger("rl_x")

# task metrics logged after an evaluation, when the environment reports them
EVAL_METRICS = ("episode_return", "episode_length", "is_success", "success_clean", "coverage", "lifted", "wandered_off",
                "goal_dist", "goal_dist_2nd_half", "ball_lost")


def build_model(cfg, action_dim):
    return ConditionalUnet1D(input_dim=action_dim, down_dims=tuple(int(d) for d in str(cfg.down_dims).split(",")),
                             kernel_size=cfg.kernel_size, n_groups=cfg.n_groups,
                             diffusion_step_embed_dim=cfg.diffusion_step_embed_dim, cond_predict_scale=cfg.cond_predict_scale)


def build_ddpm(cfg):
    ddpm = ddpm_lib.make_ddpm(cfg.num_train_timesteps, cfg.beta_schedule, cfg.beta_start, cfg.beta_end, cfg.clip_sample)
    return ddpm, jnp.asarray(ddpm_lib.inference_timesteps(cfg.num_train_timesteps, cfg.num_inference_steps))


class DiffusionPolicy:
    def __init__(self, config, train_env, eval_env, run_path, writer):
        self.config = config
        self.train_env, self.eval_env, self.writer = train_env, eval_env, writer
        cfg = config.algorithm
        self.cfg = cfg
        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console, self.track_tb, self.track_wandb = config.runner.track_console, config.runner.track_tb, config.runner.track_wandb
        self.seed = config.environment.seed
        self.nr_envs = config.environment.nr_envs
        self.horizon, self.n_obs_steps, self.n_action_steps = cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps
        self.os_shape = train_env.single_observation_space.shape
        self.as_shape = train_env.single_action_space.shape
        self.env_horizon = train_env.horizon
        # PushT's success-threshold curriculum (environment option, used by the online methods): the evaluation uses the
        # final threshold, as TRIRL-SPR and AIRL do; None = keep the environment's own threshold
        env_cfg = config.environment
        self.eval_success_threshold = (float(env_cfg.success_threshold_values.split(",")[-1])
                                       if env_cfg.get("curriculum_success", False) else None)

        # dataset: windows over the demos, normaliser fitted on all of it
        data = np.load(cfg.data_path)
        obs = np.asarray(data["states"], np.float32).reshape(-1, self.os_shape[0])
        actions = np.asarray(data["actions"], np.float32).reshape(-1, self.as_shape[0])
        indices = create_indices(episode_ends_from(data), self.horizon, pad_before=self.n_obs_steps - 1, pad_after=self.n_action_steps - 1)
        self.rows = jnp.asarray(gather_index(indices, self.horizon))
        self.norm_obs, self.norm_action = to_jax(fit_limits(obs)), to_jax(fit_limits(actions))
        self.nobs = jnp.asarray(normalize(fit_limits(obs), obs))
        self.naction = jnp.asarray(normalize(fit_limits(actions), actions))
        self.nr_windows = len(indices)
        self.steps_per_epoch = self.nr_windows // cfg.batch_size
        if self.steps_per_epoch < 1:
            raise ValueError(f"batch_size {cfg.batch_size} is larger than the {self.nr_windows} training windows")
        rlx_logger.info(f"Diffusion Policy: {len(obs)} transitions, {self.nr_windows} windows, {self.steps_per_epoch} steps per epoch")

        self.model = build_model(cfg, self.as_shape[0])
        self.ddpm, self.timesteps_seq = build_ddpm(cfg)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, init_key = jax.random.split(self.key)
        params = self.model.init(init_key, jnp.zeros((1, self.horizon, self.as_shape[0])), jnp.zeros((1,), jnp.int32),
                                 jnp.zeros((1, self.n_obs_steps * self.os_shape[0])))
        self.lr_schedule = dp.cosine_with_warmup(cfg.learning_rate, cfg.lr_warmup_steps, cfg.total_gradient_steps)
        self.tx = dp.make_optimizer(self.lr_schedule, (cfg.adam_beta1, cfg.adam_beta2), cfg.adam_eps, cfg.weight_decay)
        self.state = {"params": params, "opt_state": self.tx.init(params), "ema_params": jax.tree.map(jnp.copy, params),
                      "step": jnp.asarray(0, jnp.int32)}
        rlx_logger.info(f"U-Net parameters: {sum(x.size for x in jax.tree.leaves(params)) / 1e6:.2f} M")

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.best_eval_return = -np.inf
            self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    # ------------------------------------------------------------------ training #
    def _make_train_epoch(self):
        cfg, model, ddpm = self.cfg, self.model, self.ddpm

        def train_step(state, data, inputs):
            batch_windows, key = inputs
            rows = data["rows"][batch_windows]                                # (B, T)
            nobs, naction = data["nobs"][rows], data["naction"][rows]
            k_noise, k_t = jax.random.split(key)
            noise = jax.random.normal(k_noise, naction.shape)
            timesteps = jax.random.randint(k_t, (naction.shape[0],), 0, cfg.num_train_timesteps)
            loss, grads = jax.value_and_grad(lambda p: dp.compute_loss(model, p, ddpm, nobs, naction, noise, timesteps, self.n_obs_steps))(state["params"])
            updates, opt_state = self.tx.update(grads, state["opt_state"], state["params"])
            params = optax.apply_updates(state["params"], updates)
            decay = dp.ema_decay(state["step"], cfg.ema_update_after_step, cfg.ema_inv_gamma, cfg.ema_power, cfg.ema_min_value, cfg.ema_max_value)
            ema_params = dp.ema_update(state["ema_params"], params, decay)
            new_state = {"params": params, "opt_state": opt_state, "ema_params": ema_params, "step": state["step"] + 1}
            return new_state, {"loss": loss, "lr": self.lr_schedule(state["step"]), "ema_decay": decay}

        @jax.jit
        def train_epoch(state, data, key, n_steps):
            k_perm, k_steps = jax.random.split(key)
            perm = jax.random.permutation(k_perm, self.nr_windows)[: self.steps_per_epoch * cfg.batch_size]
            batches = perm.reshape(self.steps_per_epoch, cfg.batch_size)
            keys = jax.random.split(k_steps, self.steps_per_epoch)
            active = jnp.arange(self.steps_per_epoch) < n_steps                # the last epoch may be cut short

            def guarded(state, inputs):
                batch, k, on = inputs
                new_state, metrics = train_step(state, data, (batch, k))
                state = jax.tree.map(lambda n, o: jnp.where(on, n, o), new_state, state)
                return state, metrics
            state, metrics = jax.lax.scan(guarded, state, (batches, keys, active))
            mean = lambda x: jnp.sum(jnp.where(active, x, 0.0)) / jnp.maximum(n_steps, 1)
            return state, {k: mean(v) for k, v in metrics.items()}
        return train_epoch

    def _make_evaluate(self):
        """Closed-loop evaluation with the same actor the evaluation scripts use (actor.DPActor)."""
        env = self.eval_env
        actor = DPActor(self.model, self.ddpm, self.timesteps_seq, self.norm_obs, self.norm_action,
                        self.horizon, self.n_obs_steps, self.n_action_steps)

        @jax.jit
        def evaluate(ema_params, key):
            key, reset_key = jax.random.split(key)
            env_state = env.reset(jax.random.split(reset_key, self.nr_envs), True)
            if self.eval_success_threshold is not None:
                env_state = env_state.replace(success_threshold=jnp.full_like(env_state.success_threshold, self.eval_success_threshold))
            actor_state = actor.init(env_state.next_observation, key)

            def step(carry, t):
                env_state, actor_state = carry
                action, actor_state = actor.act(ema_params, actor_state, env_state.next_observation, t,
                                                env_state.terminated | env_state.truncated)
                return (env.step(env_state, action), actor_state), None

            (env_state, _), _ = jax.lax.scan(step, (env_state, actor_state), jnp.arange(self.env_horizon))
            return {f"eval/{m}": jnp.mean(env_state.info[f"rollout/{m}"]) for m in EVAL_METRICS if f"rollout/{m}" in env_state.info}
        return evaluate

    def train(self):
        cfg = self.cfg
        train_epoch, evaluate = self._make_train_epoch(), self._make_evaluate()
        data = {"rows": self.rows, "nobs": self.nobs, "naction": self.naction}
        total, done = int(cfg.total_gradient_steps), 0
        eval_every = int(cfg.evaluation_frequency)
        next_eval = eval_every if eval_every > 0 else None
        start = time.time()
        while done < total:
            n = min(self.steps_per_epoch, total - done)
            self.key, k = jax.random.split(self.key)
            t0 = time.time()
            self.state, metrics = train_epoch(self.state, data, k, n)
            metrics = jax.device_get(metrics)
            done += n
            log = {f"train/{k}": float(v) for k, v in metrics.items()}
            log["time/gradient_steps_per_s"] = n / (time.time() - t0)
            log["steps/gradient_steps"] = done
            self._log_dict(log, done)
            if next_eval is not None and (done >= next_eval or done >= total):
                self._evaluate_and_save(evaluate, done)
                while next_eval <= done:
                    next_eval += eval_every
            elif done >= total and self.save_model:
                self.save()
        rlx_logger.info(f"Training time: {time.time() - start:.1f} s")

    def _evaluate_and_save(self, evaluate, step):
        """latest.model at every evaluation; best.model when eval/episode_return reaches a new high (as TRIRL-SPR and
        the PPO baseline do)."""
        metrics = {}
        if self.cfg.evaluation_active:
            self.key, k = jax.random.split(self.key)
            t0 = time.time()
            metrics = {name: float(v) for name, v in jax.device_get(evaluate(self.state["ema_params"], k)).items()}
            metrics["time/evaluation_s"] = time.time() - t0
            self._log_dict(metrics, step)
        if self.save_model:
            self.save()
            current = metrics.get("eval/episode_return")
            if current is not None and current > self.best_eval_return:
                self.best_eval_return = current
                self.save(file_name="best.model")
                rlx_logger.info(f"[save-best] new best eval/episode_return={current:.4f} -> best.model")

    # ------------------------------------------------------------------ logging #
    def _log_dict(self, metrics, step):
        if self.track_wandb:
            wandb.log({"global_step": int(step), **metrics})
        if self.track_tb:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐")
            for k, v in metrics.items():
                rlx_logger.info(f"│ {k.ljust(30)}│ {str(np.format_float_positional(v, trim='-')).ljust(14)[:14]} │")
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")
        else:
            rlx_logger.info(f"Step: {step}")

    # ------------------------------------------------------------------ checkpoints #
    def checkpoint_tree(self):
        return {"model": self.state["params"], "ema": self.state["ema_params"],
                "normalizer": {"obs": self.norm_obs, "action": self.norm_action}}

    def save(self, file_name=None):
        file_name = file_name or self.latest_model_file_name
        checkpoint = self.checkpoint_tree()
        tmp = f"{self.save_path}/tmp"
        self.checkpointer.save(tmp, checkpoint, save_args=orbax_utils.save_args_from_target(checkpoint))
        with open(f"{tmp}/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{file_name}", "zip", tmp)
        os.replace(f"{self.save_path}/{file_name}.zip", f"{self.save_path}/{file_name}")
        shutil.rmtree(tmp)
        if self.track_wandb:
            wandb.save(f"{self.save_path}/{file_name}", base_path=self.save_path)

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        path = os.path.abspath(config.runner.load_model)
        tmp = os.path.join(os.path.dirname(path), "tmp")
        shutil.unpack_archive(path, tmp, "zip")
        loaded = json.load(open(f"{tmp}/config_algorithm.json"))
        for key, value in loaded.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = DiffusionPolicy(config, train_env, eval_env, run_path, writer)
        target = model.checkpoint_tree()
        checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(tmp, item=target, restore_args=orbax_utils.restore_args_from_target(target))
        model.state["params"], model.state["ema_params"] = checkpoint["model"], checkpoint["ema"]
        model.norm_obs, model.norm_action = checkpoint["normalizer"]["obs"], checkpoint["normalizer"]["action"]
        shutil.rmtree(tmp)
        return model

    def test(self, episodes):
        metrics = jax.device_get(self._make_evaluate()(self.state["ema_params"], jax.random.PRNGKey(self.seed)))
        for k, v in metrics.items():
            rlx_logger.info(f"{k}: {float(v):.4f}")

    def general_properties():
        return GeneralProperties
