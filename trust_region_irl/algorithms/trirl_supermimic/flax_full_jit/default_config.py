from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.default_config import get_config as get_spr_config


def get_config(algorithm_name):
    """trirl_ppo_spr's options (same defaults) plus SuperMimic's. Defaults of the SuperMimic options follow the paper's
    Table A.5 unless noted; see ../README.md for the deviations."""
    config = get_spr_config(algorithm_name)

    config.policy_update = "bptt"            # bptt: SuperMimic; ppo: trirl_ppo_spr's PPO + KL update, unchanged
    config.reward_source = "trirl"           # trirl | env (PushT's true reward, no discriminator) | zero (IRL reward 0)

    # world model (Sec. 3.2)
    config.wm_hidden_dims = "512,512,512"    # paper: 2048 x 3 for a humanoid
    config.wm_window = 8                     # k, autoregressive window of the fit
    config.wm_batch_size = 2048              # windows per gradient step
    config.wm_minibatches_per_update = 10    # gradient steps per world-model update ("10 batches of 2048 windows")
    config.wm_grad_clip_norm = 1.0
    config.replay_capacity = 1024            # steps per environment (paper 1000; a multiple of nr_steps here)

    # imagination, policy and critic (Sec. 3.3)
    config.imagination_horizon = 32          # H
    config.imagination_batch_size = 4096     # imagined rollouts per policy update
    config.nr_model_updates = -1             # world-model + policy updates per TRIRL iteration; -1: update_to_data_ratio * nr_steps
    config.update_to_data_ratio = 0.4
    config.bptt_learning_rate = 2e-4         # world model, policy and critic
    config.lr_decay_fraction = 0.78          # cosine decay over this share of the updates (250k of 320k), then constant
    config.lr_final_fraction = 0.03
    config.policy_grad_clip_norm = 1.0       # global norm, then ...
    config.policy_grad_clip_value = 0.1      # ... element-wise
    config.critic_grad_clip_norm = 1.0
    config.critic_epochs = 5
    config.critic_minibatches = 4
    config.td_lambda = 0.95
    config.return_normalization = "advantage_std"   # advantage_std: divide the return by the advantage std (as PPO's normalisation); none
    config.straight_through_termination = False     # True: the success test passes a sigmoid surrogate's gradient
    config.st_temperature = 0.1              # surrogate's temperature, as a fraction of the success threshold
    config.remat = True                      # recompute the imagination steps in the backward pass (memory)

    # plan step 2.5 (single-iteration comparison from a checkpoint) and the control runs
    config.save_at_fractions = ""            # e.g. "0.25,0.5,0.75": also keep latest.model as progress_<f>.model.zip there
    config.init_checkpoint = ""              # TRIRL-SPR model archive: start from its policy, discriminator and corrected
                                             # reward (and critic with ppo); optimisers start fresh
    config.freeze_reward = False             # no discriminator update and no reward correction: the reward stays fixed
    config.init_world_model = ""             # bptt: directory written by scripts/supermimic/wm_offline.py (params + normaliser)
    config.bptt_warmup_updates = 0           # bptt: world-model + critic updates without policy step before the first iteration's update

    return config
