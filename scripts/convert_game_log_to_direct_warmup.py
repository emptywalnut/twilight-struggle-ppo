#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.warmup_bc import find_action_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert existing raw game_logs/games-*.jsonl records into no-seed "
            "direct warmup samples. The converter uses the logged seed only to "
            "reconstruct missing pre-action observations; the output record has "
            "start.seed = null and trains directly from observations/legal_actions."
        )
    )
    parser.add_argument("--input", action="append", required=True, help="Raw games-*.jsonl file or directory.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--terminal-reason", default=None)
    parser.add_argument("--prefer-terminal-reason", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def iter_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("games-*.jsonl")))
        else:
            files.append(path)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    return files


def iter_records(inputs: list[str]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in iter_input_files(inputs):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield path, json.loads(line)


def record_matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    info = record.get("info") or {}
    if record.get("kind") != "terminal":
        return False
    if args.episode_index is not None and int(record.get("episode_index") or -1) != args.episode_index:
        return False
    if args.seed is not None and int(record.get("seed") or -1) != args.seed:
        return False
    if args.terminal_reason is not None and str(info.get("terminal_reason")) != args.terminal_reason:
        return False
    return True


def select_record(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    scanned = 0
    for path, record in iter_records(args.input):
        scanned += 1
        if record_matches(record, args):
            matches.append((path, record))
            if args.max_records and len(matches) >= args.max_records:
                break
        if args.max_records and scanned >= args.max_records and not matches:
            break
    if not matches:
        raise SystemExit("no matching terminal game record found")
    if args.prefer_terminal_reason:
        for item in matches:
            if str((item[1].get("info") or {}).get("terminal_reason")) == args.prefer_terminal_reason:
                return item
    return matches[0]


def compact_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: action.get(key)
        for key in ("type", "decision", "value", "label", "selector", "event", "card", "country", "region", "prompt")
        if key in action
    }


def action_index_from_log(log_action: dict[str, Any], legal_actions: list[dict[str, Any]]) -> int:
    action = log_action.get("action") or {}
    idx = find_action_index(action, legal_actions)
    if idx is not None:
        return idx
    raw_idx = log_action.get("action_index")
    if isinstance(raw_idx, int) and 0 <= raw_idx < len(legal_actions):
        candidate = legal_actions[raw_idx]
        if str(candidate.get("value")) == str(action.get("value")):
            return raw_idx
    legal_brief = [{k: a.get(k) for k in ("type", "decision", "value", "label")} for a in legal_actions[:40]]
    raise ValueError(f"cannot match logged action step={log_action.get('step')} action={action} legal={legal_brief}")


def convert_record(source_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    seed = int(record["seed"])
    env = TwilightStruggleEnv(
        {
            "force_setup_heuristic": False,
            "max_episode_steps": max(5000, int(record.get("steps") or 0) + 100),
            "log_games_dir": None,
            "metrics_dir": None,
            "log_action_details": False,
        }
    )
    samples: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    try:
        env.reset(seed=seed)
        for log_action in record.get("actions") or []:
            before_obs = env.last_obs
            legal_actions = list(env.legal_actions)
            idx = action_index_from_log(log_action, legal_actions)
            selected = legal_actions[idx]
            log_before = env.bridge.log()
            next_obs, reward, terminated, truncated, final_info = env.step(idx)
            after_obs = env.last_obs
            log_after = env.bridge.log()
            before_log = list((log_before or {}).get("log") or [])
            after_log = list((log_after or {}).get("log") or [])
            samples.append(
                {
                    "step": int(log_action.get("step") or len(samples)),
                    "turn": before_obs.get("turn"),
                    "action_round": before_obs.get("action_round"),
                    "side": before_obs.get("side"),
                    "phase": before_obs.get("phase"),
                    "prompt_class": (
                        "setup"
                        if str(before_obs.get("phase", "")).startswith("setup")
                        else "headline"
                        if "headline" in str(before_obs.get("phase", "")).lower()
                        or "headline" in str(before_obs.get("prompt", "")).lower()
                        else "decision"
                    ),
                    "observation": {key: value for key, value in before_obs.items() if key != "legal_actions"},
                    "legal_actions": legal_actions,
                    "expert_action_index": int(idx),
                    "choice": compact_action(selected),
                    "source_log_action": {
                        "policy_action_index": log_action.get("policy_action_index"),
                        "action_index": log_action.get("action_index"),
                        "override_kind": log_action.get("override_kind"),
                        "heuristic_override": log_action.get("heuristic_override"),
                        "random_override": log_action.get("random_override"),
                    },
                    "result_after": {
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "winner": final_info.get("winner"),
                        "terminal_reason": final_info.get("terminal_reason"),
                        "before": env.action_context(before_obs),
                        "after": env.action_context(after_obs),
                        "state_delta": env.state_delta(before_obs, after_obs),
                        "saito_log_delta": [str(item) for item in after_log[len(before_log) :]],
                    },
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    info = record.get("info") or final_info or {}
    last_after = samples[-1]["result_after"]["after"] if samples else {}
    output = {
        "format": "ts_warmup_game_v1",
        "game_id": f"converted-log-episode-{record.get('episode_index')}",
        "ruleset": "optional_us_plus_2",
        "partial": not bool(terminated) or bool(truncated),
        "source": {
            "kind": "converted_existing_game_log",
            "name": source_path.name,
            "source_path": str(source_path),
            "source_episode_index": record.get("episode_index"),
            "source_seed_used_for_conversion_only": seed,
        },
        "start": {
            "seed": None,
            "hands": ((record.get("start") or {}).get("hands") or {}),
            "initial_influence": {
                "us": {
                    "canada": 2,
                    "uk": 5,
                    "israel": 1,
                    "iran": 1,
                    "australia": 4,
                    "philippines": 1,
                    "japan": 1,
                    "southkorea": 1,
                    "panama": 1,
                    "southafrica": 1,
                },
                "ussr": {"eastgermany": 3, "finland": 1, "syria": 1, "iraq": 1, "northkorea": 3},
            },
        },
        "samples": samples,
        "result": {
            "winner": info.get("winner"),
            "terminal_reason": info.get("terminal_reason"),
            "vp": info.get("vp", last_after.get("vp")),
            "defcon": info.get("defcon", last_after.get("defcon")),
            "turn": last_after.get("turn"),
            "action_round": last_after.get("action_round"),
            "steps": len(samples),
        },
        "quality": {
            "status": "ok" if terminated and not truncated else "partial",
            "notes": [
                "Converted from existing raw game log into direct no-seed warmup samples.",
                "The source seed was used only during conversion to reconstruct full observations and legal action lists.",
            ],
        },
    }
    return output


def main() -> None:
    args = parse_args()
    source_path, record = select_record(args)
    converted = convert_record(source_path, record)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(converted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(json.dumps(converted, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source": str(source_path),
                "episode_index": record.get("episode_index"),
                "source_seed_used_for_conversion_only": record.get("seed"),
                "output_json": str(args.output_json),
                "output_jsonl": str(args.output_jsonl) if args.output_jsonl else None,
                "samples": len(converted.get("samples") or []),
                "partial": converted.get("partial"),
                "result": converted.get("result"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
