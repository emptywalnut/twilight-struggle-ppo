#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

SESSION="${SESSION:-struggle_unified_mix_explore}"
LATEST_FILE="${LATEST_FILE:-/tmp/struggle_unified_mix_explore_latest.txt}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3600}"
MIN_NUKE_SAMPLE="${MIN_NUKE_SAMPLE:-100}"
NUKE_THRESHOLD="${NUKE_THRESHOLD:-0.70}"
MECH_THRESHOLD="${MECH_THRESHOLD:-0.05}"
STOP_EPISODES="${STOP_EPISODES:-10000}"
STALL_SECONDS="${STALL_SECONDS:-3600}"
AUTO_START_ON_COMPLETE="${AUTO_START_ON_COMPLETE:-1}"
WATCHDOG_LOG="${WATCHDOG_LOG:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified_mix_watchdog.log}"
BC_CKPT="${BC_CKPT:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-handmask-warmup-ppo-gpu2-20260704-100115/warmup_bc/checkpoints/bc-samples-00503754}"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"

mkdir -p "$(dirname "$WATCHDOG_LOG")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"
}

latest_checkpoint_or_bc() {
  local run_dir="$1"
  "$PYTHON" - <<'PY' "$run_dir" "$BC_CKPT"
import sys
import json
from pathlib import Path

run = Path(sys.argv[1])
fallback = Path(sys.argv[2])
for manifest_name in ("best_checkpoint.json", "best_checkpoint_current.json"):
    manifest = run / manifest_name
    if manifest.exists():
        try:
            checkpoint = Path(json.loads(manifest.read_text()).get("checkpoint") or "")
        except Exception:
            checkpoint = Path()
        if checkpoint.exists():
            print(checkpoint)
            raise SystemExit
best_link = run / "best_checkpoint"
if best_link.exists():
    print(best_link.resolve())
    raise SystemExit
candidates = []
for marker in run.glob("ppo_checkpoints/**/rllib_checkpoint.json"):
    try:
        candidates.append((marker.stat().st_mtime, marker.parent))
    except OSError:
        pass
if candidates:
    print(max(candidates, key=lambda item: item[0])[1])
else:
    print(fallback)
PY
}

rank_run_best() {
  local run_dir="$1"
  if [[ -f "$run_dir/ppo_train.log" || -f "$run_dir/train.log" ]]; then
    "$PYTHON" scripts/rank_ppo_checkpoints.py "$run_dir" \
      --out "$run_dir/ranked_checkpoints_current.json" \
      --best-manifest "$run_dir/best_checkpoint_current.json" \
      --best-score-scoring-card-held-penalty 100 \
      >>"$WATCHDOG_LOG" 2>&1 || log "checkpoint ranking failed for $run_dir"
  fi
}

summarize_run() {
  local run_dir="$1"
  "$PYTHON" - <<'PY' "$run_dir"
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
train_log = run / "ppo_train.log"
latest_iter = None
latest_artifact_mtime = train_log.stat().st_mtime if train_log.exists() else 0.0
if train_log.exists():
    for line in train_log.read_text(errors="replace").splitlines():
        if line.startswith("{'iter'"):
            try:
                latest_iter = ast.literal_eval(line.replace("np.float64(", "").replace(")", ""))
            except Exception:
                latest_iter = line

rows = []
for path in sorted((run / "metrics").glob("terminal-*.jsonl")):
    try:
        latest_artifact_mtime = max(latest_artifact_mtime, path.stat().st_mtime)
    except OSError:
        pass
    try:
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except OSError:
        pass
last = rows[-100:]
counter = Counter(str(row.get("terminal_reason") or row.get("reason") or "unknown") for row in last)
mech = sum(1 for row in last if row.get("bridge_error") or row.get("no_legal_actions") or str(row.get("terminal_reason")) == "timeout")
nuke = sum(1 for row in last if str(row.get("terminal_reason")) in {"nuclear_war", "thermonuclear war", "Cuban Missile Crisis"})
scoring_card_held = sum(1 for row in last if str(row.get("terminal_reason")) == "scoring card held")
payload = {
    "run": str(run),
    "terminal_rows": len(rows),
    "last_window": len(last),
    "last100_nuke_rate": nuke / len(last) if last else 0.0,
    "last100_scoring_card_held_rate": scoring_card_held / len(last) if last else 0.0,
    "last100_mechanical_rate": mech / len(last) if last else 0.0,
    "last100_reasons": dict(counter),
    "latest_iter": latest_iter,
    "has_traceback": train_log.exists() and "Traceback" in train_log.read_text(errors="replace")[-20000:],
    "has_oom": train_log.exists() and "OutOfMemory" in train_log.read_text(errors="replace")[-20000:],
    "latest_artifact_age_seconds": max(0.0, __import__("time").time() - latest_artifact_mtime) if latest_artifact_mtime else None,
}
print(json.dumps(payload, sort_keys=True))
PY
}

start_training_session() {
  local mode="$1"
  local restore_from="$2"
  local stamp
  stamp=$(date +%Y%m%d-%H%M%S)
  local run_dir="/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-mixedopponents-${mode}-gpu2-${stamp}"
  local mix='{"current_self":0.30,"teacher":0.30,"anchor":0.30,"random_legal":0.10,"heuristic":0.00}'
  local entropy="0.006"
  local temp="1.15"
  local random_prob="0.01"
  local train_batch="2048"
  local minibatch="128"
  local fragment="256"
  if [[ "$mode" == "autostable" ]]; then
    mix='{"current_self":0.25,"teacher":0.35,"anchor":0.35,"random_legal":0.05,"heuristic":0.00}'
    entropy="0.004"
    random_prob="0.0"
  elif [[ "$mode" == "autooom" ]]; then
    train_batch="1024"
    minibatch="64"
    fragment="128"
  fi
  tmux new-session -d -s "$SESSION" \
    "cd /mnt2/users/kaile/hantao/game/struggle && RUN_DIR='$run_dir' RESTORE_FROM='$restore_from' MIX='$mix' ENTROPY_COEFF='$entropy' POLICY_TEMPERATURE='$temp' PERSISTENT_RANDOM_PROB='$random_prob' TRAIN_BATCH_SIZE='$train_batch' MINIBATCH_SIZE='$minibatch' ROLLOUT_FRAGMENT_LENGTH='$fragment' scripts/launch_unified_mix_explore_gpu2.sh"
  echo "$run_dir" > "$LATEST_FILE"
  log "started training session=$SESSION mode=$mode restore_from=$restore_from run_dir=$run_dir"
}

while true; do
  if [[ ! -f "$LATEST_FILE" ]]; then
    log "latest file missing: $LATEST_FILE"
    sleep "$SLEEP_SECONDS"
    continue
  fi
  RUN_DIR=$(cat "$LATEST_FILE")
  SUMMARY=$(summarize_run "$RUN_DIR" || echo '{}')
  log "status $SUMMARY"

  if ! pgrep -af "struggle_ai.train_rllib" | grep -F "$RUN_DIR" >/dev/null; then
    if echo "$SUMMARY" | grep -q '"has_oom": true'; then
      RESTORE=$(latest_checkpoint_or_bc "$RUN_DIR")
      log "detected dead OOM run; restarting with smaller PPO batches from $RESTORE"
      tmux kill-session -t "$SESSION" 2>/dev/null || true
      start_training_session "autooom" "$RESTORE"
    elif [[ "$AUTO_START_ON_COMPLETE" == "1" ]] && "$PYTHON" - <<'PY' "$SUMMARY" "$STOP_EPISODES"
import json
import sys
payload = json.loads(sys.argv[1])
stop = int(sys.argv[2])
latest_iter = payload.get("latest_iter")
episodes = 0
if isinstance(latest_iter, dict):
    episodes = int(latest_iter.get("episodes_seen") or 0)
terminal_rows = int(payload.get("terminal_rows") or 0)
raise SystemExit(0 if max(episodes, terminal_rows) >= stop else 1)
PY
    then
      log "completed run detected; ranking checkpoints before stabilized restart"
      rank_run_best "$RUN_DIR"
      RESTORE=$(latest_checkpoint_or_bc "$RUN_DIR")
      log "completed run best restore is $RESTORE"
      tmux kill-session -t "$SESSION" 2>/dev/null || true
      start_training_session "autostable" "$RESTORE"
    else
      log "training process is not alive and no OOM signature was found; leaving stopped for manual inspection"
    fi
    sleep "$SLEEP_SECONDS"
    continue
  fi

  DECISION=$("$PYTHON" - <<'PY' "$SUMMARY" "$MIN_NUKE_SAMPLE" "$NUKE_THRESHOLD" "$MECH_THRESHOLD" "$STALL_SECONDS"
import json
import sys
payload = json.loads(sys.argv[1])
min_sample = int(sys.argv[2])
nuke_threshold = float(sys.argv[3])
mech_threshold = float(sys.argv[4])
stall_seconds = float(sys.argv[5])
if payload.get("last_window", 0) >= min_sample and payload.get("last100_mechanical_rate", 0.0) > mech_threshold:
    print("mechanical")
elif payload.get("last_window", 0) >= min_sample and payload.get("last100_nuke_rate", 0.0) > nuke_threshold:
    print("nuke")
elif (payload.get("latest_artifact_age_seconds") or 0.0) > stall_seconds:
    print("stall")
else:
    print("ok")
PY
)
  if [[ "$DECISION" == "mechanical" ]]; then
    log "mechanical anomaly exceeded threshold; stopping run for manual inspection"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  elif [[ "$DECISION" == "nuke" ]]; then
    RESTORE=$(latest_checkpoint_or_bc "$RUN_DIR")
    log "nuke rate exceeded threshold over last window; restarting with conservative exploration from $RESTORE"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    start_training_session "autostable" "$RESTORE"
  elif [[ "$DECISION" == "stall" ]]; then
    RESTORE=$(latest_checkpoint_or_bc "$RUN_DIR")
    log "progress artifacts stale for more than ${STALL_SECONDS}s; restarting from $RESTORE"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    start_training_session "autostable" "$RESTORE"
  fi

  sleep "$SLEEP_SECONDS"
done
