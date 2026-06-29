#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

PYTHON=/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python
US_CKPT=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/us-sharpen-stage1-gpu3-20260602-181841/phase2-current-ussr-10k/checkpoints/steps-0000735844-episodes-00007003
CURRENT_USSR_CKPT=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/multiagent-warmstart-10k-forcedsetup-gpu3-20260601-103415
STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/multiagent-us7003-currentussr-terminal2-nuke1-100k-gpu3-${STAMP}

mkdir -p "${RUN_DIR}/game_logs" "${RUN_DIR}/metrics"

export CUDA_VISIBLE_DEVICES=3

{
  echo "run_dir=${RUN_DIR}"
  echo "us_ckpt=${US_CKPT}"
  echo "current_ussr_ckpt=${CURRENT_USSR_CKPT}"
  echo "started=$(date -Is)"
} | tee "${RUN_DIR}/run.info"

"${PYTHON}" struggle_ai/train_rllib.py \
  --multi-agent \
  --num-env-runners 8 \
  --num-gpus 1 \
  --num-gpus-per-env-runner 0 \
  --train-batch-size 2048 \
  --minibatch-size 256 \
  --batch-mode complete_episodes \
  --stop-episodes 100000 \
  --checkpoint-eval-every-steps 0 \
  --checkpoint-eval-every-episodes 1000 \
  --eval-games 20 \
  --eval-max-episode-steps 1200 \
  --log-games-every 100 \
  --max-episode-steps 1200 \
  --terminal-reward-scale 2.0 \
  --turn-vp-reward-scale 0.01 \
  --nuke-death-penalty 1.0 \
  --high-stability-coup-penalty 0.03 \
  --low-stability-warzone-coup-reward 0.03 \
  --headline-opponent-event-penalty 0.03 \
  --empty-country-influence-reward 0.01 \
  --control-battleground-reward 0.02 \
  --control-non-battleground-reward 0.005 \
  --restore-from "${CURRENT_USSR_CKPT}" \
  --eval-initial-from "${CURRENT_USSR_CKPT}" \
  --load-us-policy-from "${US_CKPT}" \
  --load-ussr-policy-from "${CURRENT_USSR_CKPT}" \
  --checkpoint-dir "${RUN_DIR}" \
  --log-games-dir "${RUN_DIR}/game_logs" \
  --metrics-dir "${RUN_DIR}/metrics" \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "finished=$(date -Is)" | tee -a "${RUN_DIR}/run.info"
