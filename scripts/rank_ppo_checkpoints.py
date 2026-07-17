from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank PPO checkpoints from train.log checkpoint_eval_done records.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--best-manifest", type=Path, default=None)
    parser.add_argument("--max-nuke-rate", type=float, default=0.60)
    parser.add_argument("--best-score-side-weight", type=float, default=0.25)
    parser.add_argument("--best-score-balance-penalty", type=float, default=0.25)
    parser.add_argument("--best-score-nuke-penalty", type=float, default=100.0)
    parser.add_argument("--best-score-scoring-card-held-penalty", type=float, default=100.0)
    parser.add_argument("--best-score-random-side-floor", type=float, default=0.80)
    parser.add_argument("--best-score-random-side-penalty", type=float, default=200.0)
    parser.add_argument("--best-score-benchmark-side-floor", type=float, default=0.50)
    parser.add_argument("--best-score-benchmark-side-penalty", type=float, default=250.0)
    return parser.parse_args()


def default_train_log(run_dir: Path) -> Path:
    for name in ("train.log", "ppo_train.log"):
        path = run_dir / name
        if path.exists():
            return path
    return run_dir / "train.log"


def checkpoint_records(train_log: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not train_log.exists():
        return records
    with train_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "checkpoint_eval_done" not in line and "final_eval_done" not in line:
                continue
            text = line.strip()
            text = re.sub(r"np\.(?:float64|float32|int64|int32|bool_)\(([^()]*)\)", r"\1", text)
            try:
                payload = ast.literal_eval(text)
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


def unified_score(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    bot_elo = metric(record, "elo/bot/current_rating") or 1500.0
    us_elo = metric(record, "elo/us/current_rating") or 1500.0
    ussr_elo = metric(record, "elo/ussr/current_rating") or 1500.0
    side_floor = min(us_elo, ussr_elo)
    side_gap = abs(us_elo - ussr_elo)
    nuke_values = [
        metric(record, key)
        for key in record
        if str(key).startswith("eval_vs_") and str(key).endswith("/nuke_terminal_rate")
    ]
    mean_nuke = sum(nuke_values) / len(nuke_values) if nuke_values else 0.0
    scoring_card_held_values = [
        metric(record, key)
        for key in record
        if str(key).startswith("eval_vs_") and str(key).endswith("/scoring_card_held_terminal_rate")
    ]
    mean_scoring_card_held = (
        sum(scoring_card_held_values) / len(scoring_card_held_values) if scoring_card_held_values else 0.0
    )
    random_us = metric(record, "eval_vs_random/us_side_win_rate")
    random_ussr = metric(record, "eval_vs_random/ussr_side_win_rate")
    random_shortfall = max(0.0, float(args.best_score_random_side_floor) - min(random_us, random_ussr))
    benchmark_side_rates = []
    for key in record:
        if not (
            str(key).startswith("eval_vs_")
            and (str(key).endswith("/us_side_win_rate") or str(key).endswith("/ussr_side_win_rate"))
        ):
            continue
        opponent = str(key).split("/", 1)[0].removeprefix("eval_vs_")
        if opponent == "random" or opponent.startswith("steps-"):
            continue
        benchmark_side_rates.append(metric(record, key))
    benchmark_side_floor = min(benchmark_side_rates) if benchmark_side_rates else 1.0
    benchmark_shortfall = max(0.0, float(args.best_score_benchmark_side_floor) - benchmark_side_floor)
    score = (
        bot_elo
        + float(args.best_score_side_weight) * side_floor
        - float(args.best_score_balance_penalty) * side_gap
        - float(args.best_score_nuke_penalty) * mean_nuke
        - float(args.best_score_scoring_card_held_penalty) * mean_scoring_card_held
        - float(args.best_score_random_side_penalty) * random_shortfall
        - float(args.best_score_benchmark_side_penalty) * benchmark_shortfall
    )
    return {
        "checkpoint": record.get("checkpoint"),
        "episodes": record.get("episodes_seen") or record.get("checkpoint/episodes"),
        "train_steps": record.get("train_steps_seen") or record.get("checkpoint/train_steps"),
        "score": score,
        "bot_elo": bot_elo,
        "us_elo": us_elo,
        "ussr_elo": ussr_elo,
        "side_floor_elo": side_floor,
        "side_gap_elo": side_gap,
        "mean_eval_nuke_rate": mean_nuke,
        "mean_eval_scoring_card_held_rate": mean_scoring_card_held,
        "random_us_side_win_rate": random_us,
        "random_ussr_side_win_rate": random_ussr,
        "random_side_shortfall": random_shortfall,
        "benchmark_side_floor_win_rate": benchmark_side_floor,
        "benchmark_side_shortfall": benchmark_shortfall,
        "eval_vs_initial_win_rate": record.get("eval_vs_initial/win_rate"),
        "eval_vs_initial_us_side_win_rate": record.get("eval_vs_initial/us_side_win_rate"),
        "eval_vs_initial_ussr_side_win_rate": record.get("eval_vs_initial/ussr_side_win_rate"),
        "eval_vs_random_win_rate": record.get("eval_vs_random/win_rate"),
        "eval_vs_random_nuke_terminal_rate": record.get("eval_vs_random/nuke_terminal_rate"),
    }


def rank_unified(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ranked = [unified_score(record, args) for record in records if record.get("checkpoint")]
    return sorted(ranked, key=lambda item: float(item["score"]), reverse=True)


def write_best_manifest(path: Path, run_dir: Path, best: dict[str, Any], records: list[dict[str, Any]]) -> None:
    checkpoint = str(best.get("checkpoint") or "")
    matching_record = next((record for record in records if str(record.get("checkpoint") or "") == checkpoint), {})
    payload = {
        "source_run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "eval_label": Path(checkpoint).name if checkpoint else None,
        "episodes_seen": best.get("episodes"),
        "train_steps_seen": best.get("train_steps"),
        "score": best.get("score"),
        "score_details": best,
        "eval_metrics": {
            key: value
            for key, value in matching_record.items()
            if isinstance(value, (int, float, str, bool))
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    link_path = path.parent / "best_checkpoint"
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        if checkpoint:
            link_path.symlink_to(Path(checkpoint))
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    train_log = args.train_log.resolve() if args.train_log else default_train_log(run_dir)
    records = checkpoint_records(train_log)
    best_unified = rank_unified(records, args)
    payload = {
        "run_dir": str(run_dir),
        "train_log": str(train_log),
        "eval_records": len(records),
        "best_unified": best_unified[:10],
        "best_us": rank_side(records, "us", args.max_nuke_rate)[:10],
        "best_ussr": rank_side(records, "ussr", args.max_nuke_rate)[:10],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.best_manifest and best_unified:
        write_best_manifest(args.best_manifest.resolve(), run_dir, best_unified[0], records)


if __name__ == "__main__":
    main()
