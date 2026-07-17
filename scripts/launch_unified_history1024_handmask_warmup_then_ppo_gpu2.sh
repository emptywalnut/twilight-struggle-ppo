#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="$GPU"

STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="${RUN_DIR:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-handmask-warmup-ppo-gpu${GPU}-${STAMP}}"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"

DISTILL_RUN="${DISTILL_RUN:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-distill-defconfilter5-gpu2-20260703-042419}"
DISTILL_MANIFEST="${DISTILL_MANIFEST:-$DISTILL_RUN/distill/manifest.json}"
OLD_BC_CKPT="${OLD_BC_CKPT:-$DISTILL_RUN/warmup_bc/checkpoints/bc-samples-00503754}"

SOURCE_RUN="${SOURCE_RUN:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/transformer-history-elo-history3-defconfilter5-300-gpu2-20260627-091331}"
US_TEACHER="${US_TEACHER:-$SOURCE_RUN/checkpoints/checkpoints/steps-0000043084-episodes-00000100}"
USSR_TEACHER="${USSR_TEACHER:-$SOURCE_RUN/checkpoints/checkpoints/steps-0000090780-episodes-00000200}"

BC_EPOCHS="${BC_EPOCHS:-3}"
PPO_EPISODES="${PPO_EPISODES:-10000}"
RUN_PPO="${RUN_PPO:-1}"

mkdir -p "$RUN_DIR"/{warmup_bc,ppo_checkpoints,game_logs,metrics}
{
  echo "run_dir=$RUN_DIR"
  echo "gpu=$GPU"
  echo "distill_manifest=$DISTILL_MANIFEST"
  echo "old_bc_ckpt=$OLD_BC_CKPT"
  echo "us_teacher=$US_TEACHER"
  echo "ussr_teacher=$USSR_TEACHER"
  echo "bc_epochs=$BC_EPOCHS"
  echo "ppo_episodes=$PPO_EPISODES"
  echo "run_ppo=$RUN_PPO"
  echo "history_length=1024"
  echo "card_history_length=160"
  echo "absolute_visible_hand_masks=1"
} | tee "$RUN_DIR/launch_config.txt"
echo "$RUN_DIR" > /tmp/struggle_unified_history1024_latest.txt

"$PYTHON" -m struggle_ai.warmup_bc \
  --input "$DISTILL_MANIFEST" \
  --output-dir "$RUN_DIR/warmup_bc" \
  --multi-agent \
  --policy-sharing unified \
  --policies-to-train shared \
  --restore-from "$OLD_BC_CKPT" \
  --partial-warmstart \
  --epochs "$BC_EPOCHS" \
  --batch-size 512 \
  --lr 1e-4 \
  --num-gpus 1 \
  --model-arch transformer_history \
  --hidden 256 \
  --history-layers 2 \
  --card-history-layers 2 \
  --history-attention-heads 4 \
  --history-dropout 0.05 \
  --graph-layers 2 \
  --graph-neighbor-hops 2 \
  2>&1 | tee "$RUN_DIR/warmup_bc/train.log"

BC_CKPT=$("$PYTHON" - <<'PY' "$RUN_DIR/warmup_bc/warmup_report.json"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text())
print(payload["checkpoint"])
PY
)
echo "$BC_CKPT" | tee "$RUN_DIR/warmup_bc/checkpoint_path.txt"

if [[ "$RUN_PPO" != "1" ]]; then
  echo "RUN_PPO=$RUN_PPO; stopping after warmup BC. BC checkpoint: $BC_CKPT"
  exit 0
fi

"$PYTHON" -m struggle_ai.train_rllib \
  --multi-agent \
  --policy-sharing unified \
  --model-arch transformer_history \
  --hidden 256 \
  --history-layers 2 \
  --card-history-layers 2 \
  --history-attention-heads 4 \
  --history-dropout 0.05 \
  --graph-layers 2 \
  --graph-neighbor-hops 2 \
  --restore-from "$BC_CKPT" \
  --partial-warmstart \
  --load-report-path "$RUN_DIR/ppo_load_report.json" \
  --benchmark-label defconfilter5_teacher \
  --benchmark-us-policy-from "$US_TEACHER" \
  --benchmark-ussr-policy-from "$USSR_TEACHER" \
  --stop-episodes "$PPO_EPISODES" \
  --num-env-runners 8 \
  --num-gpus 1 \
  --num-gpus-per-env-runner 0 \
  --train-batch-size 4096 \
  --minibatch-size 512 \
  --rollout-fragment-length 512 \
  --batch-mode complete_episodes \
  --sample-timeout-s 1800 \
  --checkpoint-dir "$RUN_DIR/ppo_checkpoints" \
  --checkpoint-eval-every-steps 0 \
  --checkpoint-eval-every-episodes 1000 \
  --eval-games 10 \
  --eval-min-games-per-side 10 \
  --eval-history-opponents 2 \
  --eval-max-episode-steps 1200 \
  --eval-progress-every-games 10 \
  --elo-path "$RUN_DIR/elo_unified.json" \
  --elo-k-factor 32 \
  --wandb-mode disabled \
  --log-games-dir "$RUN_DIR/game_logs" \
  --log-games-every 20 \
  --metrics-dir "$RUN_DIR/metrics" \
  --terminal-reward-scale 2.0 \
  --reward-shaping-scale 0.9 \
  --reward-shaping-final-scale 0.7 \
  --reward-shaping-phaseout-start-episodes 2000 \
  --reward-shaping-phaseout-end-episodes 8000 \
  --turn-vp-reward-scale 0.01 \
  --nuke-death-penalty 1.0 \
  --scoring-card-held-penalty 1.0 \
  --high-stability-coup-penalty 0.03 \
  --low-stability-warzone-coup-reward 0.03 \
  --headline-opponent-event-penalty 0.03 \
  --defcon-risk-pick-penalty 0.05 \
  --defcon-risk-commit-penalty 0.15 \
  --empty-country-influence-reward 0.01 \
  --control-battleground-reward 0.02 \
  --control-non-battleground-reward 0.005 \
  --max-episode-step-penalty 0.25 \
  --defcon-suicide-mode hard_filter \
  --max-episode-steps 1200 \
  --persistent-random-prob 0.0 \
  --heuristic-override-prob 0.0 \
  --scripted-side-prob 0.0 \
  --force-setup-heuristic \
  2>&1 | tee "$RUN_DIR/ppo_train.log"
