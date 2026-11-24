# AIRL

`flax_full_jit/` is synced from RL-X [`rl_x/algorithms/airl_ppo/flax_full_jit`](https://github.com/nico-bohlinger/RL-X/tree/0ff04fe/rl_x/algorithms/airl_ppo/flax_full_jit) at commit `0ff04fe`. The only change is the module path in the imports (`rl_x.algorithms.airl_ppo.flax_full_jit` to `trust_region_irl.algorithms.airl_ppo.flax_full_jit`).

It replaces the earlier `flax_full_jit/` copy in this repository, which trained the policy on the initial discriminator and could not be constructed for action spaces with more than one dimension.

`flax_loco_mjx/` is the TRIRL paper's AIRL and is unchanged.
