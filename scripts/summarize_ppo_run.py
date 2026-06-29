from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


NUKE_REASONS = {"nuclear_war", "thermonuclear war", "Cuban Missile Crisis"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Twilight Struggle PPO run health.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--last", type=int, default=200, help="Only summarize the latest N terminal games; 0 means all.")
    return parser.parse_args()


def read_terminal_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "metrics").glob("terminal-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    records.sort(key=lambda row: float(row.get("time") or 0.0))
    return records


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    winners = Counter(str(row.get("winner") or "") for row in records)
    reasons = Counter(str(row.get("terminal_reason") or "") for row in records)
    unsafe_filter_skipped_reasons: Counter[str] = Counter()
    for row in records:
        unsafe_filter_skipped_reasons.update(row.get("unsafe_filter_skipped_reasons") or {})
    return {
        "games": total,
        "us_win_rate": rate(winners["us"], total),
        "ussr_win_rate": rate(winners["ussr"], total),
        "tie_rate": rate(winners["tie"], total),
        "nuke_terminal_rate": rate(sum(reasons[reason] for reason in NUKE_REASONS), total),
        "timeout_rate": rate(reasons["max_episode_steps"] + reasons["eval_max_episode_steps"], total),
        "no_legal_actions_rate": rate(sum(1 for row in records if row.get("no_legal_actions") or row.get("terminal_reason") == "no_legal_actions"), total),
        "bridge_error_rate": rate(sum(1 for row in records if row.get("bridge_error") or row.get("terminal_reason") == "bridge_error"), total),
        "unsafe_filter_skipped_rate": rate(sum(1 for row in records if float(row.get("unsafe_filter_skipped_count") or 0.0) > 0.0), total),
        "unsafe_filter_skipped_mean": mean(float(row.get("unsafe_filter_skipped_count") or 0.0) for row in records) if records else 0.0,
        "avg_steps": mean(float(row.get("steps") or 0.0) for row in records) if records else 0.0,
        "us_reward_mean": mean(float(row.get("us_reward") or 0.0) for row in records) if records else 0.0,
        "ussr_reward_mean": mean(float(row.get("ussr_reward") or 0.0) for row in records) if records else 0.0,
        "terminal_reasons": dict(reasons),
        "unsafe_filter_skipped_reasons": dict(unsafe_filter_skipped_reasons),
    }


def latest_checkpoint_evals(run_dir: Path, limit: int = 5) -> list[str]:
    train_log = run_dir / "train.log"
    if not train_log.exists():
        return []
    lines = []
    pattern = re.compile(r"checkpoint_eval_done|final_eval_done")
    with train_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if pattern.search(line):
                lines.append(line.strip())
    return lines[-limit:]


def main() -> None:
    args = parse_args()
    records = read_terminal_records(args.run_dir)
    selected = records[-args.last :] if args.last > 0 else records
    payload = {
        "run_dir": str(args.run_dir),
        "all": summarize_records(records),
        "latest": summarize_records(selected),
        "latest_window": args.last if args.last > 0 else "all",
        "latest_checkpoint_evals": latest_checkpoint_evals(args.run_dir),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
