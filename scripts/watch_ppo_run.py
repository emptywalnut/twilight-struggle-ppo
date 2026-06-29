from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


NUKE_REASONS = {"nuclear_war", "thermonuclear war", "Cuban Missile Crisis"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a PPO run and optionally stop it when health gates fail.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tmux-session", default=None)
    parser.add_argument("--interval-s", type=float, default=300.0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--min-games", type=int, default=50)
    parser.add_argument("--max-nuke-rate", type=float, default=0.60)
    parser.add_argument("--max-timeout-rate", type=float, default=0.20)
    parser.add_argument("--max-no-legal-rate", type=float, default=0.0)
    parser.add_argument("--max-bridge-error-rate", type=float, default=0.0)
    parser.add_argument("--min-side-win-rate", type=float, default=0.05)
    parser.add_argument("--max-side-win-rate", type=float, default=0.95)
    parser.add_argument("--stop-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def terminal_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "metrics").glob("terminal-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    records.sort(key=lambda item: float(item.get("time") or 0.0))
    return records


def ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    winners = Counter(str(record.get("winner") or "") for record in records)
    reasons = Counter(str(record.get("terminal_reason") or "") for record in records)
    return {
        "games": count,
        "us_win_rate": ratio(winners["us"], count),
        "ussr_win_rate": ratio(winners["ussr"], count),
        "nuke_terminal_rate": ratio(sum(reasons[reason] for reason in NUKE_REASONS), count),
        "timeout_rate": ratio(reasons["max_episode_steps"] + reasons["eval_max_episode_steps"], count),
        "no_legal_actions_rate": ratio(sum(1 for record in records if record.get("no_legal_actions") or record.get("terminal_reason") == "no_legal_actions"), count),
        "bridge_error_rate": ratio(sum(1 for record in records if record.get("bridge_error") or record.get("terminal_reason") == "bridge_error"), count),
        "avg_steps": mean(float(record.get("steps") or 0.0) for record in records) if records else 0.0,
        "terminal_reasons": dict(reasons),
    }


def failed_gates(summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if int(summary["games"]) < args.min_games:
        return []
    failures: list[str] = []
    if float(summary["nuke_terminal_rate"]) > args.max_nuke_rate:
        failures.append("nuke_terminal_rate")
    if float(summary["timeout_rate"]) > args.max_timeout_rate:
        failures.append("timeout_rate")
    if float(summary["no_legal_actions_rate"]) > args.max_no_legal_rate:
        failures.append("no_legal_actions_rate")
    if float(summary["bridge_error_rate"]) > args.max_bridge_error_rate:
        failures.append("bridge_error_rate")
    us_rate = float(summary["us_win_rate"])
    ussr_rate = float(summary["ussr_win_rate"])
    if us_rate < args.min_side_win_rate or us_rate > args.max_side_win_rate:
        failures.append("us_win_rate")
    if ussr_rate < args.min_side_win_rate or ussr_rate > args.max_side_win_rate:
        failures.append("ussr_win_rate")
    return failures


def append_status(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "watchdog.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def stop_tmux(session: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], check=False)


def check_once(args: argparse.Namespace) -> dict[str, Any]:
    records = terminal_records(args.run_dir)
    window_records = records[-args.window :] if args.window > 0 else records
    summary = summarize(window_records)
    failures = failed_gates(summary, args)
    payload = {
        "time": time.time(),
        "run_dir": str(args.run_dir),
        "window": args.window,
        "min_games": args.min_games,
        "summary": summary,
        "failures": failures,
        "stopped": False,
    }
    if failures and args.stop_on_fail and args.tmux_session:
        stop_tmux(args.tmux_session)
        payload["stopped"] = True
        payload["tmux_session"] = args.tmux_session
    append_status(args.run_dir, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def main() -> None:
    args = parse_args()
    while True:
        check_once(args)
        if args.once:
            return
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
