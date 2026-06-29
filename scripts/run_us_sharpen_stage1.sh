#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

PYTHON=/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python
CURRENT_CKPT=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/multiagent-warmstart-10k-forcedsetup-gpu3-20260601-103415
WEAK_USSR_CKPT=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/multiagent-warmstart-10k-forcedsetup-gpu3-20260601-103415/checkpoints/steps-0000002451-episodes-00000022
STAMP=$(date +%Y%m%d-%H%M%S)
ROOT=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/us-sharpen-stage1-gpu3-${STAMP}
PHASE1=${ROOT}/phase1-weak-ussr-10k
PHASE2=${ROOT}/phase2-current-ussr-10k

mkdir -p "${PHASE1}/game_logs" "${PHASE1}/metrics" "${PHASE2}/game_logs" "${PHASE2}/metrics"

COMMON_ARGS=(
  --multi-agent
  --policies-to-train us
  --num-env-runners 8
  --num-gpus 1
  --num-gpus-per-env-runner 0
  --train-batch-size 2048
  --minibatch-size 256
  --batch-mode complete_episodes
  --stop-episodes 10000
  --checkpoint-eval-every-steps 0
  --checkpoint-eval-every-episodes 1000
  --eval-games 20
  --eval-max-episode-steps 1200
  --log-games-every 100
  --max-episode-steps 1200
  --turn-vp-reward-scale 0.01
  --nuke-death-penalty 0.5
  --high-stability-coup-penalty 0.03
  --low-stability-warzone-coup-reward 0.03
  --headline-opponent-event-penalty 0.03
  --empty-country-influence-reward 0.01
  --control-battleground-reward 0.02
  --control-non-battleground-reward 0.005
)

export CUDA_VISIBLE_DEVICES=3

{
  echo "root=${ROOT}"
  echo "phase1=${PHASE1}"
  echo "phase2=${PHASE2}"
  echo "current_ckpt=${CURRENT_CKPT}"
  echo "weak_ussr_ckpt=${WEAK_USSR_CKPT}"
  echo "started=$(date -Is)"
} | tee "${ROOT}/run.info"

"${PYTHON}" struggle_ai/train_rllib.py \
  "${COMMON_ARGS[@]}" \
  --restore-from "${CURRENT_CKPT}" \
  --eval-initial-from "${WEAK_USSR_CKPT}" \
  --load-ussr-policy-from "${WEAK_USSR_CKPT}" \
  --checkpoint-dir "${PHASE1}" \
  --log-games-dir "${PHASE1}/game_logs" \
  --metrics-dir "${PHASE1}/metrics" \
  2>&1 | tee "${PHASE1}/train.log"

"${PYTHON}" struggle_ai/train_rllib.py \
  "${COMMON_ARGS[@]}" \
  --restore-from "${PHASE1}/checkpoints/final" \
  --eval-initial-from "${CURRENT_CKPT}" \
  --load-ussr-policy-from "${CURRENT_CKPT}" \
  --checkpoint-dir "${PHASE2}" \
  --log-games-dir "${PHASE2}/game_logs" \
  --metrics-dir "${PHASE2}/metrics" \
  2>&1 | tee "${PHASE2}/train.log"

echo "finished=$(date -Is)" | tee -a "${ROOT}/run.info"
