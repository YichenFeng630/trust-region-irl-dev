# TRIRL + SuperMimic (`trirl_supermimic.flax_full_jit`)

TRIRL-SPR (`trirl_ppo_spr`) with the policy update of SuperMimic (Liao et al., *SuperMimic: Scaling Humanoid Whole-Body
Control with One Differentiable World Model*, CoRL 2026, [arXiv 2610.28480](https://arxiv.org/abs/2610.28480)). The
encoder pretraining, the discriminator, the reward correction, the evaluation and the saving are `trirl_ppo_spr`'s,
unchanged. Only the policy step of every TRIRL iteration changes:

| | `policy_update=ppo` | `policy_update=bptt` (default) |
|---|---|---|
| data | the iteration's rollout, 10 epochs | the rollout also goes into a replay buffer (newest replace oldest) |
| model | none | deterministic world model, fitted on k-step autoregressive windows of the buffer (Eq. 1) |
| policy | PPO clip + eta KL(pi_theta \|\| pi_k) | imagined H-step return through the frozen model, differentiated (BPTT), + eta KL(pi_theta \|\| pi_k) (Eq. 2) |
| critic | GAE returns of the rollout | lambda-returns of the imagined rollouts (Eq. 3), no target network |

`policy_update=ppo` runs the PPO code of `trirl_ppo_spr` unchanged; `tests/test_supermimic.py` checks that both give
the same training. `reward_source` selects the reward of the imagined rollouts: `trirl` (TRIRL's), `env` (PushT's true
reward, no discriminator), `zero` (the IRL reward set to 0 in every step and in the terminal tail).

Only PushT (`pusht_mjx`) is supported for now: the world model's state and its analytic heads are in `pusht_model.py`.

## Files

| File | Content |
|---|---|
| `flax_full_jit/trirl_supermimic.py` | training loop (copy of `trirl_ppo_spr/flax_full_jit/trirl_ppo.py` up to `load`, plus the BPTT branch) |
| `flax_full_jit/world_model.py` | world model, running normaliser, Eq. 1 |
| `flax_full_jit/replay_buffer.py` | replay buffer in time order, k-step windows that do not cross episode ends |
| `flax_full_jit/imagination.py` | policy loss (Eq. 2 with TRIRL's KL and entropy), lambda-returns (Eq. 3), critic loss |
| `flax_full_jit/pusht_model.py` | PushT's world-model state (observation + the T's velocity), success, true reward |
| `flax_full_jit/pusht_sim.py` | real PushT rollouts, restoring logged physics states, replaying action sequences (scripts, tests) |
| `flax_full_jit/default_config.py` | `trirl_ppo_spr`'s options plus the SuperMimic options below |

Scripts for the offline checks: `scripts/supermimic/` (`collect_pusht.py`, `wm_offline.py`, `grad_direction.py`; `one_iteration_eval.py` measures a single iteration's KL and frozen-reward return change, plan step 2.5).

## The policy loss

Per imagined trajectory from a buffer state $s_0$, with $a_t = \mu_\theta(o_t) + \sigma_\theta \epsilon_t$, $\hat s_{t+1} = f_\omega(\hat s_t, a_t)$:

$$G = \sum_{t=0}^{H-1} \gamma^t c_t \big(r_k(\hat s_t) + \gamma\, d_{t+1} T_k(\hat s_{t+1})\big) + \gamma^H c_H V_\psi(\hat s_H), \qquad W = \sum_{t=0}^{H-1}\gamma^t c_t$$

$$L = -\frac{G}{W\,\sigma_A} + \frac{1}{W}\sum_{t=0}^{H-1}\gamma^t c_t \Big[\eta\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_k\big)(\hat s_t) - \alpha^2 \mathcal H(\pi_\theta)\Big]$$

- $r_k = \alpha[(1-\epsilon) D_{corr} + \epsilon\beta D_k]$: the reward `trirl_ppo_spr` gives PPO in iteration k, on the T pose.
- $d_{t+1}$: success of the predicted state at the current curriculum threshold; $c_t$: not ended before t. Hard, no
  gradient (as SHAC); `straight_through_termination=True` passes a sigmoid surrogate's gradient instead.
- $T_k(s) = (r_k(s) + \alpha H_{terminal})/(1-\gamma)$: value of the absorbing state, so $\gamma T_k$ is
  `trirl_ppo_spr`'s `terminal_tail` (`handle_absorbing_states=True`).
- $\sigma_A$: standard deviation of the lambda-return advantages of the batch (stop-gradient), the counterpart of PPO's
  per-minibatch advantage normalisation, so that eta and the entropy weight act on a similar scale as with PPO.
- KL and entropy are evaluated at the stop-gradient imagined states; the KL is $\mathrm{KL}(\pi_\theta\|\pi_k)$ as in
  the TRIRL paper and `trirl_ppo_spr`; the entropy weight is $\alpha^2$ because `trirl_ppo_spr` multiplies by
  `entropy_coef` twice.

## References read

| Source | What we took |
|---|---|
| SuperMimic paper, Sec. 3 and Appendix A-D (no code released) | Eq. 1-3, Algorithm A.1, Table A.5 hyperparameters |
| SHAC, [NVlabs/DiffRL](https://github.com/NVlabs/DiffRL) `a4c0dd1`, `algorithms/shac.py` | reparameterised Gaussian actor through the dynamics; hard early termination with no value after it; TD-lambda targets (`compute_target_values`, checked against our `lambda_returns` in the tests through `trirl_ppo_spr`'s GAE); critic minibatches with global-norm clipping |
| PWM, [imgeorgiev/PWM](https://github.com/imgeorgiev/PWM) `9816252`, `algorithms/pwm.py` | actor loss scaled per rollout (PWM divides by H or a running return std; we divide by W and the advantage std) |

## Deviations from the paper

| | Paper | Here | Why |
|---|---|---|---|
| reward | fixed hand-written tracking reward | TRIRL's learned reward, changes every iteration | the point of the combination |
| policy | deterministic; data collection adds N(0, 0.1^2) | Gaussian, reparameterised; data collection uses the policy's own noise | TRIRL's entropy and KL need it; the discriminator must see the policy's own distribution |
| trust region | none (gradient clipping only) | eta KL(pi_theta \|\| pi_k), eta's schedule as `trirl_ppo_spr` | TRIRL's reward-correction step assumes it |
| termination | learned probability, differentiable; termination = failure | analytic success test, hard, no gradient; termination = success, with the terminal tail | the success rule is known and its threshold follows a curriculum |
| return scale | not stated | divided by W and the advantage std | eta comparable with PPO's; checked by the measured per-iteration KL |
| world model size | MLP 2048 x 3 | MLP 512 x 3 (`wm_hidden_dims`), LayerNorm + SiLU | 30-dim state |
| world model output | not stated | normalised state change, zero-initialised output layer; quaternion renormalised; state clipped to +-10 like the observation | usual for state-space models (MBPO) |
| world model loss | squared error of state, reward, observation, termination | squared error of the state, each dimension divided by the std of its one-step change; velocity targets masked at episode ends | reward, observation and termination heads are analytic here; the environment resets its physics at an episode end |
| update schedule | 0.4 updates per environment step, continuous | per TRIRL iteration `nr_model_updates` (default round(0.4 x nr_steps) = 51) world-model updates (10 gradient steps of 2048 windows each) and policy updates | TRIRL works in iterations; "10 batches of 2048 windows per update" read as 10 gradient steps |
| learning-rate schedule | cosine to 0.03x over 250k of 320k iterations | the same fractions of this run's update count | |
| action bound | not stated; SHAC and PWM squash with tanh before the simulator (`shac.py` l. 198, `pwm.py` l. 324) | clip(a, -1, 1) inside the model, as PushT does | keeps the environment's semantics and the PPO policy; the gradient is zero beyond the bound, so policy means can stay saturated (step 2: 57% of imagined actions clipped). Open question for step 3 |
| critic input | state, goal, sim context | the world model's state | PushT has no goal input or randomised parameters |
| imagination start | buffer state with its sim context | buffer state | no sim context |

## Options

| Option | Default | Meaning |
|---|---|---|
| `policy_update` | bptt | `bptt` (SuperMimic) or `ppo` (`trirl_ppo_spr` unchanged) |
| `reward_source` | trirl | `trirl`, `env` (true reward, no discriminator, no pretraining), `zero` (IRL reward 0) |
| `wm_hidden_dims` | 512,512,512 | world model layers |
| `wm_window`, `wm_batch_size`, `wm_minibatches_per_update` | 8, 2048, 10 | k, windows per gradient step, gradient steps per update |
| `replay_capacity` | 1024 | steps per environment (multiple of `nr_steps`) |
| `imagination_horizon`, `imagination_batch_size` | 32, 4096 | H, imagined rollouts per policy update |
| `nr_model_updates`, `update_to_data_ratio` | -1, 0.4 | updates per TRIRL iteration (-1: ratio x `nr_steps`) |
| `bptt_learning_rate`, `lr_decay_fraction`, `lr_final_fraction` | 2e-4, 0.78, 0.03 | Adam rate of model, policy and critic; cosine decay |
| `policy_grad_clip_norm`, `policy_grad_clip_value` | 1.0, 0.1 | global norm, then element-wise |
| `critic_epochs`, `critic_minibatches`, `td_lambda` | 5, 4, 0.95 | critic fit on the imagined rollouts |
| `return_normalization` | advantage_std | or `none` |
| `straight_through_termination`, `st_temperature` | False, 0.1 | surrogate gradient of the success test; temperature as a share of the threshold |
| `remat` | True | recompute the imagination in the backward pass |
| `save_at_fractions` | "" | e.g. `0.25,0.5,0.75`: keep a copy of `latest.model` as `progress_<f>.model.zip` at those points of the run (does not change training) |
| `init_checkpoint` | "" | start from a TRIRL-SPR archive: policy, discriminator, corrected reward, and the critic with ppo; optimisers fresh; needs `pretrain_env_steps=0` |
| `freeze_reward` | False | no discriminator update and no reward correction (single-iteration comparisons, plan step 2.5) |
| `init_world_model` | "" | bptt: start the world model from `scripts/supermimic/wm_offline.py`'s output directory |
| `bptt_warmup_updates` | 0 | bptt: in the first iteration, this many world-model + critic updates (policy fixed) before the policy update |

## Logged values (bptt)

`world_model/loss`, `world_model/new_data_T_pos_err_{1,k}`, `world_model/new_data_T_angle_err_k` (error on the newest
rollout before fitting on it); `imagination/return`, `imagination/kl`, `imagination/ended_within_horizon`,
`imagination/action_clipped`, `imagination/advantage_std`; `imagination/{model,real}_return_H` (H-step return from the
rollout's first states, imagined vs real, environments cut by the horizon left out); `policy/real_kl` (also with ppo:
KL(pi_new || pi_k) on the states pi_k collected).
