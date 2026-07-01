#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="$GPU"

STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-distill-splitbest-gpu${GPU}-${STAMP}
PYTHON=/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python

US_TEACHER=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/selected_policies/best_us57005_ussr66001/full_checkpoints/us_source_steps-0012650523-episodes-00057005
USSR_TEACHER=/mnt2/users/kaile/hantao/game/struggle/runs/ppo/selected_policies/best_us57005_ussr66001/full_checkpoints/ussr_source_steps-0014919451-episodes-00066001

DISTILL_EPISODES="${DISTILL_EPISODES:-1000}"
BC_EPOCHS="${BC_EPOCHS:-3}"
PPO_EPISODES="${PPO_EPISODES:-10000}"
RUN_PPO="${RUN_PPO:-1}"

mkdir -p "$RUN_DIR"/{distill,warmup_bc,ppo_checkpoints,game_logs,metrics}
echo "$RUN_DIR" > /tmp/struggle_unified_distill_latest.txt

"$PYTHON" scripts/generate_policy_distill_warmup.py \
  --us-policy-from "$US_TEACHER" \
  --ussr-policy-from "$USSR_TEACHER" \
  --output-jsonl "$RUN_DIR/distill/split_teacher_games.jsonl" \
  --output-manifest "$RUN_DIR/distill/manifest.json" \
  --episodes "$DISTILL_EPISODES" \
  --max-episode-steps 1200 \
  --num-gpus 0 \
  --model-arch feedforward \
  --hidden 256 \
  --graph-layers 2 \
  --graph-neighbor-hops 2 \
  2>&1 | tee "$RUN_DIR/distill/generate.log"

"$PYTHON" -m struggle_ai.warmup_bc \
  --input "$RUN_DIR/distill/manifest.json" \
  --output-dir "$RUN_DIR/warmup_bc" \
  --multi-agent \
  --policy-sharing unified \
  --policies-to-train shared \
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
  --partial-warmstart \
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
echo "$BC_CKPT" > "$RUN_DIR/warmup_bc/checkpoint_path.txt"

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
  --benchmark-label split_teacher_best \
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
