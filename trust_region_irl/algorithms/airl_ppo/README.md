# AIRL

`flax_full_jit/` is synced from RL-X [`rl_x/algorithms/airl_ppo/flax_full_jit`](https://github.com/nico-bohlinger/RL-X/tree/0ff04fe/rl_x/algorithms/airl_ppo/flax_full_jit) at commit `0ff04fe`. The only change is the module path in the imports (`rl_x.algorithms.airl_ppo.flax_full_jit` to `trust_region_irl.algorithms.airl_ppo.flax_full_jit`).

It replaces the earlier `flax_full_jit/` copy in this repository, which trained the policy on the initial discriminator and could not be constructed for action spaces with more than one dimension.

Added on top of RL-X:
- `reward_input`: `observation` (default, as in RL-X), `embodiment-free`, where g and h read only the environment's `encoder_input_indices` (no robot joints), the same input as TRIRL-SPR's `byol-embodiment-free` reward, or `task-based`, observation entries 0-6 (PushT's T position and quaternion), as TRIRL-SPR's `byol-task-based-observation`.
- PushT's success-threshold curriculum (environment option `curriculum_success`): each rollout uses the threshold of its stage of training, logged as `env/success_threshold`, and the evaluation uses the last one, as in TRIRL-SPR.
- `discriminator_uses_action` (default False): g reads the state and the action, g(s, a) + gamma h(s') - h(s), as `shaped-sa` in the TRIRL paper code; the gradient penalty then includes the action gradient. h still reads the state only.
- `nr_disc_steps` (default -1, off): this many discriminator gradient steps per iteration, each on `minibatch_size` agent and expert samples, as `n_disc_epochs` in the TRIRL paper code (3 to 10 there). Off, the discriminator makes `nr_epochs_disc` passes over the whole batch, as in RL-X: 800 steps per iteration with 4096 envs, 10 steps and minibatch 512.
- `max_std_dev` (default -1, off): after every policy update the learnable std is projected back to at most this value.
- `log_pi_weight` (default 1.0, as in RL-X): the PPO reward is f - w log pi with w = `log_pi_weight`; the discriminator still uses f - log pi. The original AIRL code (Fu et al. 2018, App. D.2) uses an entropy weight of 0.1 on Ant. On pizza, w = 1 lets the policy std grow without bound (actions are clipped to [-1, 1], so a larger std raises -log pi at no cost); w = 0.1 or less keeps it below about 1. The pizza comparison uses w = 0.03 ([docs/usage.md](../../../docs/usage.md)).
- The PPO reward is logged in its two parts: `reward/f` (discriminator) and `reward/minus_log_pi` (entropy term), with `reward/airl_reward` = f - w log pi the reward PPO receives.
- `best.model` is saved whenever `eval/episode_return` reaches a new high, next to `latest.model`.
- The evaluation also logs the task metrics an environment reports: `eval/is_success`, `eval/success_clean`, `eval/coverage`, `eval/lifted`, `eval/wandered_off` (pizza), `eval/goal_dist`, `eval/goal_dist_2nd_half`, `eval/ball_lost` (ball balancing); PushT reports `eval/is_success`.

For fixed-horizon tasks without terminations (pizza, ball balancing) set `--algorithm.handle_absorbing_states=False`: the demos mark their last step as absorbing only as an episode boundary.

`flax_loco_mjx/` is the TRIRL paper's AIRL and is unchanged.
