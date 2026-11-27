#!/bin/bash
#SBATCH -p main
#SBATCH --exclude=dgx-station
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J bal_s1h_data
# Expert data for the balance "set1hist" runs: set 1's 300 nominal varied-goal BaMPC demos with obs_mode=positions_history.
# Submit from the repo root:  RAW=<bampc_collect>/balance_fr3_varied_300.npz sbatch scripts/balance/set1hist/build_data.sh
set -eo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/common.sh"; cd "$REPO"
RAW=${RAW:-$HOME/sant/trirl_workspace/bampc_collect/balance_fr3_varied_300.npz}
OUT=${OUT:-trirl_dataset/rl_expert/balance_fr3_varied_300_hist.npz}
python -u scripts/balance/make_expert_dataset.py "$RAW" "$OUT" positions_history 2>&1 | quiet
