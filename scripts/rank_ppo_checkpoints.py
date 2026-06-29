from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank PPO checkpoints from train.log checkpoint_eval_done records.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-nuke-rate", type=float, default=0.60)
    return parser.parse_args()


def checkpoint_records(train_log: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not train_log.exists():
        return records
    with train_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "checkpoint_eval_done" not in line and "final_eval_done" not in line:
                continue
            try:
                payload = ast.literal_eval(line.strip())
            except (SyntaxError, ValueError):
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def metric(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def side_score(record: dict[str, Any], side: str, max_nuke_rate: float) -> float:
    initial = metric(record, f"eval_vs_initial/{side}_side_win_rate")
    random = metric(record, f"eval_vs_random/{side}_side_win_rate")
    overall_initial = metric(record, "eval_vs_initial/win_rate")
    overall_random = metric(record, "eval_vs_random/win_rate")
    nuke_initial = metric(record, "eval_vs_initial/nuke_terminal_rate")
    nuke_random = metric(record, "eval_vs_random/nuke_terminal_rate")
    nuke_penalty = max(0.0, max(nuke_initial, nuke_random) - max_nuke_rate)
    return 2.0 * initial + random + 0.25 * (overall_initial + overall_random) - nuke_penalty


def rank_side(records: list[dict[str, Any]], side: str, max_nuke_rate: float) -> list[dict[str, Any]]:
    ranked = []
    for record in records:
        checkpoint = record.get("checkpoint")
        if not checkpoint:
            continue
        ranked.append(
            {
                "checkpoint": checkpoint,
                "episodes": record.get("episodes_seen") or record.get("checkpoint/episodes"),
                "train_steps": record.get("train_steps_seen") or record.get("checkpoint/train_steps"),
                "score": side_score(record, side, max_nuke_rate),
                "eval_vs_initial_side_win_rate": record.get(f"eval_vs_initial/{side}_side_win_rate"),
                "eval_vs_random_side_win_rate": record.get(f"eval_vs_random/{side}_side_win_rate"),
                "eval_vs_initial_win_rate": record.get("eval_vs_initial/win_rate"),
                "eval_vs_random_win_rate": record.get("eval_vs_random/win_rate"),
                "eval_vs_initial_nuke_terminal_rate": record.get("eval_vs_initial/nuke_terminal_rate"),
                "eval_vs_random_nuke_terminal_rate": record.get("eval_vs_random/nuke_terminal_rate"),
            }
        )
    return sorted(ranked, key=lambda item: float(item["score"]), reverse=True)


def main() -> None:
    args = parse_args()
    records = checkpoint_records(args.run_dir / "train.log")
    payload = {
        "run_dir": str(args.run_dir),
        "eval_records": len(records),
        "best_us": rank_side(records, "us", args.max_nuke_rate)[:10],
        "best_ussr": rank_side(records, "ussr", args.max_nuke_rate)[:10],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
