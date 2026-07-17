#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="${RUN_DIR:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-mixedopponents-explore-gpu${GPU}-${STAMP}}"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"

DEFAULT_RESTORE_FROM="/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-handmask-warmup-ppo-gpu2-20260704-100115/warmup_bc/checkpoints/bc-samples-00503754"
LATEST_SOURCE_FILE="${LATEST_SOURCE_FILE:-/tmp/struggle_unified_mix_explore_latest.txt}"
if [[ -f "$LATEST_SOURCE_FILE" ]]; then
  SOURCE_RUN_DIR=$(cat "$LATEST_SOURCE_FILE")
  if [[ -e "$SOURCE_RUN_DIR/best_checkpoint" ]]; then
    DEFAULT_RESTORE_FROM=$(readlink -f "$SOURCE_RUN_DIR/best_checkpoint")
  elif [[ -f "$SOURCE_RUN_DIR/best_checkpoint_current.json" ]]; then
    DEFAULT_RESTORE_FROM=$("$PYTHON" - <<'PY' "$SOURCE_RUN_DIR/best_checkpoint_current.json" "$DEFAULT_RESTORE_FROM"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
fallback = sys.argv[2]
try:
    checkpoint = Path(json.loads(manifest.read_text()).get("checkpoint") or "")
except Exception:
    checkpoint = Path()
print(checkpoint if checkpoint.exists() else fallback)
PY
)
  fi
fi
RESTORE_FROM="${RESTORE_FROM:-$DEFAULT_RESTORE_FROM}"
US_TEACHER="${US_TEACHER:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/transformer-history-elo-history3-defconfilter5-300-gpu2-20260627-091331/checkpoints/checkpoints/steps-0000043084-episodes-00000100}"
USSR_TEACHER="${USSR_TEACHER:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/transformer-history-elo-history3-defconfilter5-300-gpu2-20260627-091331/checkpoints/checkpoints/steps-0000090780-episodes-00000200}"

ANCHORS="${ANCHORS:-[\"/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-mixedopponents-explore-gpu2-20260711-073656/ppo_checkpoints/checkpoints/steps-0001107061-episodes-00005002\",\"/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-mixedopponents-explore-gpu2-20260711-073656/ppo_checkpoints/checkpoints/steps-0001338487-episodes-00006000\",\"/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-mixedopponents-explore-gpu2-20260711-073656/ppo_checkpoints/checkpoints/steps-0001565171-episodes-00007006\"]}"
if [[ -z "${MIX+x}" ]]; then
  MIX='{"current_self":0.30,"teacher":0.30,"anchor":0.30,"random_legal":0.10,"heuristic":0.00}'
fi
LR="${LR:-0.00002}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
KL_COEFF="${KL_COEFF:-0.4}"
KL_TARGET="${KL_TARGET:-0.006}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.006}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-1.15}"
PERSISTENT_RANDOM_PROB="${PERSISTENT_RANDOM_PROB:-0.01}"
HEURISTIC_OVERRIDE_PROB="${HEURISTIC_OVERRIDE_PROB:-0.0}"
US_FOCUS_ELO_LAG_THRESHOLD="${US_FOCUS_ELO_LAG_THRESHOLD:-75}"
US_FOCUS_PROB="${US_FOCUS_PROB:-0.70}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-128}"
ROLLOUT_FRAGMENT_LENGTH="${ROLLOUT_FRAGMENT_LENGTH:-256}"
PPO_EPISODES="${PPO_EPISODES:-10000}"
CHECKPOINT_EVAL_EVERY_EPISODES="${CHECKPOINT_EVAL_EVERY_EPISODES:-1000}"
EVAL_GAMES="${EVAL_GAMES:-30}"
EVAL_MIN_GAMES_PER_SIDE="${EVAL_MIN_GAMES_PER_SIDE:-30}"
EVAL_HISTORY_OPPONENTS="${EVAL_HISTORY_OPPONENTS:-2}"
EARLY_STOP_PATIENCE_EVALS="${EARLY_STOP_PATIENCE_EVALS:-3}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-5}"

mkdir -p "$RUN_DIR"/{ppo_checkpoints,game_logs,metrics}
{
  echo "run_dir=$RUN_DIR"
  echo "restore_from=$RESTORE_FROM"
  echo "gpu=$GPU"
  echo "history_length=1024"
  echo "card_history_length=160"
  echo "absolute_visible_hand_masks=1"
  echo "opponent_mix=$MIX"
  echo "anchors=$ANCHORS"
  echo "lr=$LR"
  echo "num_epochs=$NUM_EPOCHS"
  echo "kl_coeff=$KL_COEFF"
  echo "kl_target=$KL_TARGET"
  echo "entropy_coeff=$ENTROPY_COEFF"
  echo "policy_temperature=$POLICY_TEMPERATURE"
  echo "persistent_random_prob=$PERSISTENT_RANDOM_PROB"
  echo "heuristic_override_prob=$HEURISTIC_OVERRIDE_PROB"
  echo "us_focus_elo_lag_threshold=$US_FOCUS_ELO_LAG_THRESHOLD"
  echo "us_focus_prob=$US_FOCUS_PROB"
  echo "train_batch_size=$TRAIN_BATCH_SIZE"
  echo "minibatch_size=$MINIBATCH_SIZE"
  echo "rollout_fragment_length=$ROLLOUT_FRAGMENT_LENGTH"
  echo "ppo_episodes=$PPO_EPISODES"
  echo "checkpoint_eval_every_episodes=$CHECKPOINT_EVAL_EVERY_EPISODES"
  echo "eval_games=$EVAL_GAMES"
  echo "eval_min_games_per_side=$EVAL_MIN_GAMES_PER_SIDE"
  echo "eval_history_opponents=$EVAL_HISTORY_OPPONENTS"
  echo "early_stop_patience_evals=$EARLY_STOP_PATIENCE_EVALS"
  echo "early_stop_min_delta=$EARLY_STOP_MIN_DELTA"
} | tee "$RUN_DIR/launch_config.txt"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

echo "$RUN_DIR" > /tmp/struggle_unified_mix_explore_latest.txt

"$PYTHON" -m struggle_ai.train_rllib \
  --multi-agent \
  --policy-sharing unified \
  --unified-opponent-mix-json "$MIX" \
  --unified-anchor-policy-from "$ANCHORS" \
  --unified-opponent-mix-seed 260704 \
  --adaptive-us-focus \
  --us-focus-elo-lag-threshold "$US_FOCUS_ELO_LAG_THRESHOLD" \
  --us-focus-prob "$US_FOCUS_PROB" \
  --model-arch transformer_history \
  --hidden 256 \
  --history-layers 2 \
  --card-history-layers 2 \
  --history-attention-heads 4 \
  --history-dropout 0.05 \
  --graph-layers 2 \
  --graph-neighbor-hops 2 \
  --policy-temperature "$POLICY_TEMPERATURE" \
  --lr "$LR" \
  --num-epochs "$NUM_EPOCHS" \
  --kl-coeff "$KL_COEFF" \
  --kl-target "$KL_TARGET" \
  --entropy-coeff "$ENTROPY_COEFF" \
  --restore-from "$RESTORE_FROM" \
  --partial-warmstart \
  --load-report-path "$RUN_DIR/ppo_load_report.json" \
  --benchmark-label defconfilter5_teacher \
  --benchmark-us-policy-from "$US_TEACHER" \
  --benchmark-ussr-policy-from "$USSR_TEACHER" \
  --stop-episodes "$PPO_EPISODES" \
  --num-env-runners 8 \
  --num-gpus 1 \
  --num-gpus-per-env-runner 0 \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --minibatch-size "$MINIBATCH_SIZE" \
  --rollout-fragment-length "$ROLLOUT_FRAGMENT_LENGTH" \
  --batch-mode complete_episodes \
  --sample-timeout-s 1800 \
  --checkpoint-dir "$RUN_DIR/ppo_checkpoints" \
  --checkpoint-eval-every-steps 0 \
  --checkpoint-eval-every-episodes "$CHECKPOINT_EVAL_EVERY_EPISODES" \
  --eval-games "$EVAL_GAMES" \
  --eval-min-games-per-side "$EVAL_MIN_GAMES_PER_SIDE" \
  --eval-history-opponents "$EVAL_HISTORY_OPPONENTS" \
  --eval-max-episode-steps 1200 \
  --eval-progress-every-games 10 \
  --elo-path "$RUN_DIR/elo_unified.json" \
  --elo-k-factor 32 \
  --best-checkpoint-manifest "$RUN_DIR/best_checkpoint.json" \
  --best-score-side-weight 0.25 \
  --best-score-balance-penalty 0.25 \
  --best-score-nuke-penalty 100 \
  --best-score-scoring-card-held-penalty 100 \
  --best-score-random-side-floor 0.80 \
  --best-score-random-side-penalty 200 \
  --best-score-benchmark-side-floor 0.50 \
  --best-score-benchmark-side-penalty 250 \
  --early-stop-patience-evals "$EARLY_STOP_PATIENCE_EVALS" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
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
  --persistent-random-prob "$PERSISTENT_RANDOM_PROB" \
  --heuristic-override-prob "$HEURISTIC_OVERRIDE_PROB" \
  --scripted-side-prob 0.0 \
  --force-setup-heuristic \
  2>&1 | tee "$RUN_DIR/ppo_train.log"
