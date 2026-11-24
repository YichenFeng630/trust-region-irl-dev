"""Load the policy of a saved model (TRIRL best/latest.model.zip, PPO retraining or PPO best/latest.model, AIRL, Diffusion
Policy) for evaluation and rendering scripts that run outside the training loop.

Checkpoints are orbax PyTrees zipped together with config_algorithm.json; they are saved with CPU sharding, so the
arrays are restored as numpy arrays.

load_policy_params(path)    the Gaussian policy's params (TRIRL, AIRL, PPO), ready for Policy.apply
load_actor(path, obs, act)  (actor, params, algorithm name) for any of the saved models, where actor has
                            init(first_obs, key) -> state and act(params, state, obs, t, new_episode) -> (action, state):
                            the mean action for Gaussian policies, closed-loop action chunks for Diffusion Policy
"""
import json
import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint


def _read_checkpoint(path):
    """(checkpoint tree as numpy arrays, config_algorithm.json as a dict or {})."""
    tmp = tempfile.mkdtemp()
    try:
        shutil.unpack_archive(path, tmp, "zip")
        config_file = os.path.join(tmp, "config_algorithm.json")
        config = json.load(open(config_file)) if os.path.exists(config_file) else {}
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        meta = checkpointer.metadata(tmp)
        meta = getattr(meta, "item_metadata", meta)
        meta = getattr(meta, "tree", meta)
        restore_args = jax.tree.map(lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray), meta)
        checkpoint = checkpointer.restore(tmp, restore_args=restore_args)
    finally:
        shutil.rmtree(tmp)
    return checkpoint, config


def load_policy_params(path):
    """Policy params of the checkpoint archive at `path`, ready for Policy.apply."""
    params = _read_checkpoint(path)[0]["policy"]["params"]
    return params if "params" in params else {"params": params}


class GaussianActor:
    """The mean action of a Gaussian PPO policy (TRIRL, AIRL, PPO), as the evaluation scripts always used it."""
    def __init__(self, obs_dim, action_dim):
        from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy
        self.policy = Policy((action_dim,), 1.0, jnp.arange(obs_dim))

    def init(self, obs, key):
        return {}

    def act(self, params, state, obs, t, new_episode):
        return self.policy.apply(params, obs)[0], state


def load_actor(path, obs_dim, action_dim):
    """(actor, params, algorithm name) for the saved model at `path`; see the module docstring."""
    checkpoint, config = _read_checkpoint(path)
    name = config.get("name", "")
    if name.startswith("diffusion_policy"):
        from ml_collections import config_dict
        from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.actor import DPActor
        from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.diffusion_policy import build_model, build_ddpm
        cfg = config_dict.ConfigDict(config)
        ddpm, timesteps_seq = build_ddpm(cfg)
        to_jax = lambda tree: jax.tree.map(jnp.asarray, tree)
        norm = to_jax(checkpoint["normalizer"])
        actor = DPActor(build_model(cfg, action_dim), ddpm, timesteps_seq, norm["obs"], norm["action"],
                        cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps)
        return actor, to_jax(checkpoint["ema"]), name
    params = checkpoint["policy"]["params"]
    return GaussianActor(obs_dim, action_dim), (params if "params" in params else {"params": params}), name
