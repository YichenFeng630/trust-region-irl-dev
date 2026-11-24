from ml_collections import config_dict


def get_config(algorithm_name):
    """Defaults of the official lowdim U-Net config (diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml)."""
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.device = "gpu"  # cpu, gpu
    config.data_path = "../trirl_dataset/rl_expert/pizza_dr_demos_100.npz"

    # windows: observation history, prediction horizon, executed actions
    config.horizon = 16
    config.n_obs_steps = 2
    config.n_action_steps = 8

    # ConditionalUnet1D
    config.down_dims = "256,512,1024"
    config.kernel_size = 5
    config.n_groups = 8
    config.diffusion_step_embed_dim = 256
    config.cond_predict_scale = True

    # DDPM
    config.num_train_timesteps = 100
    config.num_inference_steps = 100
    config.beta_schedule = "squaredcos_cap_v2"
    config.beta_start = 0.0001
    config.beta_end = 0.02
    config.clip_sample = True

    # optimisation (AdamW, cosine schedule with warm-up, stepped once per batch)
    config.learning_rate = 1e-4
    config.adam_beta1 = 0.95
    config.adam_beta2 = 0.999
    config.adam_eps = 1e-8
    config.weight_decay = 1e-6
    config.lr_warmup_steps = 500
    config.batch_size = 256
    config.total_gradient_steps = 200000

    # EMA of the weights, used for evaluation
    config.ema_update_after_step = 0
    config.ema_inv_gamma = 1.0
    config.ema_power = 0.75
    config.ema_min_value = 0.0
    config.ema_max_value = 0.9999

    # closed-loop evaluation in the environment (nr_envs episodes), in gradient steps; -1 to disable
    config.evaluation_frequency = 20000
    config.evaluation_active = True

    return config
