#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="${RUN_DIR:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-noforcesetup-coolshape-lowmem-both-gpu${GPU}-${STAMP}}"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"

RESTORE_FROM="${RESTORE_FROM:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-antidrift-from3002-gpu2-20260714-221128/ppo_checkpoints/checkpoints/steps-0000217905-episodes-00001010}"
ANCHORS="${ANCHORS:-[\"$RESTORE_FROM\"]}"
MIX="${MIX:-{\"current_self\":0.55,\"teacher\":0.00,\"anchor\":0.35,\"random_legal\":0.10,\"heuristic\":0.00}}"

LR="${LR:-0.00001}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
KL_COEFF="${KL_COEFF:-0.8}"
KL_TARGET="${KL_TARGET:-0.004}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.008}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-1.20}"
PERSISTENT_RANDOM_PROB="${PERSISTENT_RANDOM_PROB:-0.02}"
HEURISTIC_OVERRIDE_PROB="${HEURISTIC_OVERRIDE_PROB:-0.0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1020}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-60}"
ROLLOUT_FRAGMENT_LENGTH="${ROLLOUT_FRAGMENT_LENGTH:-170}"
NUM_ENV_RUNNERS="${NUM_ENV_RUNNERS:-6}"
PPO_EPISODES="${PPO_EPISODES:-1000000}"
CHECKPOINT_EVAL_EVERY_EPISODES="${CHECKPOINT_EVAL_EVERY_EPISODES:-5000}"
EVAL_GAMES="${EVAL_GAMES:-30}"
EVAL_MIN_GAMES_PER_SIDE="${EVAL_MIN_GAMES_PER_SIDE:-30}"
EVAL_HISTORY_OPPONENTS="${EVAL_HISTORY_OPPONENTS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-260000}"

if [[ ! -d "$RESTORE_FROM" ]]; then
  echo "missing checkpoint: $RESTORE_FROM" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"/{ppo_checkpoints,game_logs,metrics}
{
  echo "run_dir=$RUN_DIR"
  echo "restore_from=$RESTORE_FROM"
  echo "gpu=$GPU"
  echo "timeout_seconds=$TIMEOUT_SECONDS"
  echo "policy_sharing=unified"
  echo "unified_train_focus_side=both"
  echo "force_setup_heuristic=0"
  echo "low_memory=1"
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
  echo "reward_shaping_scale=0.9"
  echo "reward_shaping_final_scale=0.25"
  echo "reward_shaping_phaseout_start_episodes=5000"
  echo "reward_shaping_phaseout_end_episodes=100000"
  echo "num_env_runners=$NUM_ENV_RUNNERS"
  echo "train_batch_size=$TRAIN_BATCH_SIZE"
  echo "minibatch_size=$MINIBATCH_SIZE"
  echo "rollout_fragment_length=$ROLLOUT_FRAGMENT_LENGTH"
  echo "ppo_episodes=$PPO_EPISODES"
  echo "checkpoint_eval_every_episodes=$CHECKPOINT_EVAL_EVERY_EPISODES"
  echo "eval_games=$EVAL_GAMES"
  echo "eval_min_games_per_side=$EVAL_MIN_GAMES_PER_SIDE"
  echo "eval_history_opponents=$EVAL_HISTORY_OPPONENTS"
} | tee "$RUN_DIR/launch_config.txt"

echo "$RUN_DIR" > /tmp/struggle_unified_noforcesetup_coolshape_latest.txt

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

timeout "$TIMEOUT_SECONDS" "$PYTHON" -m struggle_ai.train_rllib \
  --multi-agent \
  --policy-sharing unified \
  --unified-train-focus-side both \
  --unified-opponent-mix-json "$MIX" \
  --unified-anchor-policy-from "$ANCHORS" \
  --unified-opponent-mix-seed 260718 \
  --no-adaptive-us-focus \
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
  --eval-initial-from "$RESTORE_FROM" \
  --stop-episodes "$PPO_EPISODES" \
  --num-env-runners "$NUM_ENV_RUNNERS" \
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
  --wandb-mode disabled \
  --log-games-dir "$RUN_DIR/game_logs" \
  --log-games-every 20 \
  --metrics-dir "$RUN_DIR/metrics" \
  --terminal-reward-scale 2.0 \
  --reward-shaping-scale 0.9 \
  --reward-shaping-final-scale 0.25 \
  --reward-shaping-phaseout-start-episodes 5000 \
  --reward-shaping-phaseout-end-episodes 100000 \
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
  --no-force-setup-heuristic \
  2>&1 | tee "$RUN_DIR/ppo_train.log"
