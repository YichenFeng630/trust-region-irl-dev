#!/bin/bash
# Balance "set1hist": set 1 (300 nominal varied-goal demos, nominal dynamics, no pushes, set 1's algorithm config) with
# three changes: obs_mode=positions_history (no ball / plate velocities), lost_terminates=True (losing the ball ends the
# episode), and the reward/* logging. Data: scripts/balance/set1hist/build_data.sh. Run from the repo root:
#   bash scripts/balance/set1hist/launch.sh             -> 3 seeds, 25M SPR pretraining + 500M steps (as set 1)
#   SMOKE=1 bash scripts/balance/set1hist/launch.sh     -> one short smoke test (no W&B, no checkpoints)
ALGO_ARGS=$(python3 -c "import json; c=json.load(open('scripts/balance/set1hist/algo_config.json')); print(' '.join(f'--algorithm.{k}={v}' for k, v in sorted(c.items()) if k != 'name'))")
ENV_ARGS="--environment.obs_mode=positions_history --environment.lost_terminates=True"
EXCL="dgx-station,cn[01-09],cn[11-12]"   # no V100s; the 10-11 GB cards run out of memory
LOGS=${LOGS:-$PWD/logs}; mkdir -p "$LOGS"
if [ -n "$SMOKE" ]; then
  ENV_NAME=balance_fr3_mjx SEED=0 NR_ENVS=512 WANDB=False SAVE=${SAVE:-False} RUN_NAME=balance_set1hist_smoke PROJECT=balance_fr3_smoke EXP_NAME=smoke \
  ALGO_ARGS="$ALGO_ARGS --algorithm.total_timesteps=4e6 --algorithm.pretrain_env_steps=1e6 --algorithm.evaluation_and_save_frequency=1048576" ENV_ARGS="$ENV_ARGS" \
    sbatch --parsable --qos=student --exclude=$EXCL -t 01:00:00 ${DEP:+--dependency=afterok:$DEP} -J bal_s1h_smoke -o "$LOGS/balance_set1hist_smoke_%j.out" scripts/slurm/train_job.sh
  exit
fi
for s in 0 1 2; do
  ENV_NAME=balance_fr3_mjx SEED=$s RUN_NAME=balance_set1hist_lostterm_embfree_varied300_k5_pre25M_train500M_seed$s PROJECT=balance_fr3_spr_mjwarp36 EXP_NAME=balance_spr \
  ALGO_ARGS="$ALGO_ARGS" ENV_ARGS="$ENV_ARGS" \
    sbatch --parsable --qos=student --exclude=$EXCL -t 20:00:00 -J bal_s1h_s$s -o "$LOGS/balance_set1hist_s${s}_%j.out" scripts/slurm/train_job.sh
done
