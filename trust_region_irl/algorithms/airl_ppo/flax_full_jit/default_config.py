from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 100e6
    config.learning_rate = 1e-4
    config.anneal_learning_rate = False
    config.nr_steps = 10
    config.nr_epochs = 10
    config.minibatch_size = 512
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.entropy_coef = 0.001
    config.critic_coef = 1.0
    config.max_grad_norm = 10.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = -1  # -1 to disable
    config.evaluation_active = True
    config.learning_rate_disc = 1e-5
    config.nr_epochs_disc = 10
    config.nr_disc_steps = -1  # > 0: this many discriminator gradient steps per iteration, each on minibatch_size agent + expert samples (n_disc_epochs in the TRIRL paper code); <= 0: nr_epochs_disc passes over the whole batch
    config.env_reward_frac = 0.0
    config.data_path = "../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz"

    config.gp_lambda = 0.05
    config.gp_alpha = 0.5
    config.handle_absorbing_states = True
    config.reward_input = "observation"  # observation (whole observation) | embodiment-free (env.encoder_input_indices, no robot joints) | task-based (observation 0-6: PushT's T pose)
    config.discriminator_uses_action = False  # True: g(s, a) + gamma h(s') - h(s) (shaped-sa in the TRIRL paper code), False: g(s)
    config.max_std_dev = -1.0  # > 0: keep the learnable policy std at or below this value (projected after every update); <= 0: no cap
    config.log_pi_weight = 1.0  # w in the policy reward f - w log pi; the discriminator always uses f - log pi. 1.0 = RL-X / TRIRL paper code; the original AIRL code (Fu et al. 2018, App. D.2) uses an entropy weight of 0.1 on Ant
    config.terminal_tail = "double"  # reward of the absorbing state after an episode ends (handle_absorbing_states): "double" = RL-X / dev code, the discriminator's absorbing branch at (s', s'), so g(s') counts again on top of the gamma/(1-gamma) g(s') already in the step's own f (about 10,000 g(s') in all at gamma 0.99); "single" = only the entropy of the uniform absorbing policy, so g(s') counts once, as in the discriminator's loss (diagnostic, PushT 2026-11-26)

    return config
