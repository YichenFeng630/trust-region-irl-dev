# Diffusion Policy

`flax_full_jit/` is a Flax port of the state-based Diffusion Policy with the CNN U-Net (Chi et al. 2023, [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)), from the official code [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) at `5ba07ac`, configured as `train_diffusion_unet_lowdim_workspace.yaml`. It trains offline on an expert dataset and evaluates closed-loop in the environment.

Files in `flax_full_jit/`:

| File | Ported from |
|---|---|
| `unet.py` | `model/diffusion/conditional_unet1d.py`, `conv1d_components.py`, `positional_embedding.py` |
| `scheduler.py` | diffusers 0.11.1 `DDPMScheduler` (epsilon prediction, `fixed_small`, `clip_sample`) |
| `normalizer.py` | `model/common/normalizer.py` (`limits` mode, [-1, 1]) |
| `dataset.py` | `common/sampler.py` (`create_indices`, `sample_sequence`) |
| `policy.py` | `policy/diffusion_unet_lowdim_policy.py`, `model/diffusion/ema_model.py`, diffusers cosine schedule with warm-up, `torch.optim.AdamW` |
| `diffusion_policy.py` | `workspace/train_diffusion_unet_lowdim_workspace.py` (training loop), RL-X algorithm interface |

Parameters keep PyTorch's names, shapes and default initialisation, so an official `state_dict` maps onto them one to one. `tests/test_dp_parity.py` checks every module against the official code on CPU with the same weights and random draws: the full U-Net agrees to 2e-6 and the full 100-step inference to 2e-5 (float32).

**Defaults** (as the official lowdim config): observation history 2, prediction horizon 16, 8 executed actions; U-Net 256/512/1024, kernel 5, 8 groups, step embedding 256, FiLM with scale; DDPM with 100 steps, `squaredcos_cap_v2`; AdamW (lr 1e-4, betas 0.95/0.999, weight decay 1e-6), cosine schedule with 500 warm-up steps, batch 256; EMA with power 0.75 used for evaluation.

**Differences from the official training loop:** a fixed number of gradient steps (`total_gradient_steps`, default 200k) instead of epochs, with the last incomplete batch of each epoch dropped; no validation split; checkpoints are `latest.model`, written at every evaluation, and `best.model`, written when the evaluation return reaches a new high, as in TRIRL-SPR (no top-k selection); only global observation conditioning (no local conditioning).

```bash
cd experiments
python experiment.py --algorithm.name=diffusion_policy.flax_full_jit \
  --algorithm.data_path=../trirl_dataset/rl_expert/pizza_dr_demos_100.npz \
  --environment.name=pizza_dr_mjx --environment.impl=jax --environment.nr_envs=512 \
  --runner.mode=train --runner.save_model=True --runner.track_console=True
```
`environment.nr_envs` is the number of evaluation episodes. Evaluation logs the environment's task metrics (`eval/coverage`, `eval/is_success`, ... for pizza; `eval/goal_dist`, `eval/ball_lost` for ball balancing).
