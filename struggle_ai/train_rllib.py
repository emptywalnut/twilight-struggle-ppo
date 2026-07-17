from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.baseline_policies import HeuristicPolicy, RandomLegalPolicy
from struggle_ai.env import TwilightStruggleEnv, TwilightStruggleMultiAgentEnv
from struggle_ai.rllib_masked_model import register_masked_model


NUKE_TERMINAL_REASONS = {"nuclear_war", "thermonuclear war", "Cuban Missile Crisis"}
SHARED_POLICY_ID = "shared_policy"
US_POLICY_ID = "us_policy"
USSR_POLICY_ID = "ussr_policy"
RANDOM_LEGAL_POLICY_ID = "random_legal_policy"
HEURISTIC_POLICY_ID = "heuristic_policy"
LEAGUE_DEFAULT_SAMPLING = {
    "current_main": 0.40,
    "best_historical": 0.20,
    "random_history": 0.20,
    "exploiter": 0.10,
    "random_legal": 0.05,
    "heuristic": 0.05,
}
UNIFIED_OPPONENT_MIX_DEFAULT = {
    "current_self": 0.45,
    "teacher": 0.25,
    "anchor": 0.20,
    "random_legal": 0.10,
    "heuristic": 0.00,
}


def side_to_policy_id(side: str, policy_sharing: str = "split") -> str:
    if policy_sharing == "unified":
        return SHARED_POLICY_ID
    return US_POLICY_ID if side == "us" else USSR_POLICY_ID


def parse_policies_to_train(raw: str | None, multi_agent: bool, policy_sharing: str = "split") -> list[str] | None:
    if not multi_agent:
        return None
    if policy_sharing == "unified":
        if not raw:
            return [SHARED_POLICY_ID]
        aliases = {
            "shared": SHARED_POLICY_ID,
            "shared_policy": SHARED_POLICY_ID,
            "unified": SHARED_POLICY_ID,
            "both": SHARED_POLICY_ID,
            "us": SHARED_POLICY_ID,
            "ussr": SHARED_POLICY_ID,
            "us_policy": SHARED_POLICY_ID,
            "ussr_policy": SHARED_POLICY_ID,
        }
        policies = []
        for item in raw.split(","):
            key = item.strip().lower()
            if not key:
                continue
            if key not in aliases:
                raise ValueError(f"unknown policy in --policies-to-train: {item}")
            policies.append(aliases[key])
        return sorted(set(policies))
    if not raw:
        return [US_POLICY_ID, USSR_POLICY_ID]
    aliases = {
        "us": US_POLICY_ID,
        "ussr": USSR_POLICY_ID,
        "us_policy": US_POLICY_ID,
        "ussr_policy": USSR_POLICY_ID,
    }
    policies: list[str] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unknown policy in --policies-to-train: {item}")
        policies.append(aliases[key])
    return policies


def league_fixed_policy_id(side: str, pool: str, index: int | str = 0) -> str:
    return f"league_{side}_{pool}_{index}"


def league_train_sides(role: str, train_side: str) -> list[str]:
    if train_side in {"us", "ussr"}:
        return [train_side]
    if role in {"main_us", "exploiter_us"}:
        return ["us"]
    if role in {"main_ussr", "exploiter_ussr"}:
        return ["ussr"]
    return ["us", "ussr"]


def league_train_policy_ids(role: str, train_side: str) -> list[str]:
    return [side_to_policy_id(side) for side in league_train_sides(role, train_side)]


def normalize_sampling(raw: dict[str, Any] | None) -> dict[str, float]:
    defaults = LEAGUE_DEFAULT_SAMPLING if raw is None else {key: 0.0 for key in LEAGUE_DEFAULT_SAMPLING}
    values = {key: float((raw or {}).get(key, default)) for key, default in defaults.items()}
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        return {"current_main": 1.0, **{key: 0.0 for key in values if key != "current_main"}}
    return {key: max(0.0, value) / total for key, value in values.items()}


def normalize_unified_opponent_mix(raw: dict[str, Any] | None) -> dict[str, float]:
    defaults = UNIFIED_OPPONENT_MIX_DEFAULT if raw is None else {key: 0.0 for key in UNIFIED_OPPONENT_MIX_DEFAULT}
    values = {key: float((raw or {}).get(key, default)) for key, default in defaults.items()}
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        return {"current_self": 1.0, **{key: 0.0 for key in values if key != "current_self"}}
    return {key: max(0.0, value) / total for key, value in values.items()}


def parse_path_list(raw: str | None) -> list[Path]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON list or comma-separated list of paths")
    return [Path(str(item)).expanduser().resolve() for item in parsed if str(item).strip()]


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    threshold = rng.random()
    cumulative = 0.0
    last_key = "current_main"
    for key, weight in weights.items():
        last_key = key
        cumulative += weight
        if threshold <= cumulative:
            return key
    return last_key


def league_policy_sources(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    seed = manifest.get("seed_policies") or {}
    for side in ("us", "ussr"):
        if seed.get(side):
            sources[league_fixed_policy_id(side, "current")] = {"side": side, "source": str(seed[side])}
        for idx, entry in enumerate(manifest.get("best_pool", {}).get(side, []) or []):
            checkpoint = entry.get("checkpoint") or entry.get("path")
            if checkpoint:
                sources[league_fixed_policy_id(side, "best", idx)] = {"side": side, "source": str(checkpoint)}
        for idx, entry in enumerate(manifest.get("snapshots", {}).get(side, []) or manifest.get("history_pool", {}).get(side, []) or []):
            checkpoint = entry.get("checkpoint") or entry.get("path")
            if checkpoint:
                sources[league_fixed_policy_id(side, "history", idx)] = {"side": side, "source": str(checkpoint)}
        exploiter = (manifest.get("exploiters") or {}).get(side)
        if isinstance(exploiter, dict):
            exploiter = exploiter.get("checkpoint") or exploiter.get("path")
        if exploiter:
            sources[league_fixed_policy_id(side, "exploiter")] = {"side": side, "source": str(exploiter)}
    return sources


def league_ids_by_side_and_pool(policy_sources: dict[str, dict[str, str]]) -> dict[str, dict[str, list[str]]]:
    pools = {side: {"current": [], "best": [], "history": [], "exploiter": []} for side in ("us", "ussr")}
    for policy_id, item in policy_sources.items():
        side = item["side"]
        parts = policy_id.split("_")
        pool = parts[2] if len(parts) >= 4 else ""
        if side in pools and pool in pools[side]:
            pools[side][pool].append(policy_id)
    return pools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for the Twilight Struggle bridge environment.")
    parser.add_argument("--stop-iters", type=int, default=1)
    parser.add_argument("--stop-timesteps", type=int, default=None)
    parser.add_argument("--stop-episodes", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--num-env-runners", type=int, default=8)
    parser.add_argument("--framework", choices=["torch"], default="torch")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--model-arch", choices=["feedforward", "transformer_history"], default="feedforward")
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--card-history-layers", type=int, default=2)
    parser.add_argument("--history-attention-heads", type=int, default=4)
    parser.add_argument("--history-dropout", type=float, default=0.1)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbor-hops", type=int, default=2)
    parser.add_argument("--heuristic-prior-scale", type=float, default=2.0)
    parser.add_argument("--setup-heuristic-prior-scale", type=float, default=0.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--kl-coeff", type=float, default=0.2)
    parser.add_argument("--kl-target", type=float, default=0.01)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--num-gpus-per-env-runner", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--rollout-fragment-length", type=parse_rollout_fragment_length, default="auto")
    parser.add_argument("--batch-mode", choices=["complete_episodes", "truncate_episodes"], default="complete_episodes")
    parser.add_argument("--sample-timeout-s", type=float, default=1800.0)
    parser.add_argument("--restore-from", default=None)
    parser.add_argument("--eval-initial-from", default=None)
    parser.add_argument("--policy-sharing", choices=["split", "unified"], default="split")
    parser.add_argument("--unified-train-focus-side", choices=["both", "us", "ussr"], default="both")
    parser.add_argument(
        "--unified-opponent-mix-json",
        default=None,
        help=(
            "JSON weights for unified-policy rollout opponent sampling. "
            "Supported keys: current_self, teacher, anchor, random_legal, heuristic."
        ),
    )
    parser.add_argument("--unified-opponent-mix-seed", type=int, default=260704)
    parser.add_argument(
        "--unified-anchor-policy-from",
        default=None,
        help="JSON list or comma-separated list of fixed shared-policy checkpoints for unified mixed-opponent training.",
    )
    parser.add_argument("--adaptive-us-focus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--us-focus-elo-lag-threshold", type=float, default=75.0)
    parser.add_argument("--us-focus-prob", type=float, default=0.70)
    parser.add_argument("--focus-opponent-policy-from", default=None)
    parser.add_argument("--load-shared-policy-from", default=None)
    parser.add_argument("--load-shared-source-side", choices=["us", "ussr"], default="us")
    parser.add_argument("--load-us-policy-from", default=None)
    parser.add_argument("--load-ussr-policy-from", default=None)
    parser.add_argument("--benchmark-label", default="side_selected_best")
    parser.add_argument("--benchmark-us-policy-from", default=None)
    parser.add_argument("--benchmark-ussr-policy-from", default=None)
    parser.add_argument("--partial-warmstart", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--load-report-path", default=None)
    parser.add_argument("--policies-to-train", default=None)
    parser.add_argument("--checkpoint-dir", default="runs/ppo")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--log-games-dir", default=None)
    parser.add_argument("--log-games-every", type=int, default=1)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--log-debug-snapshots", action="store_true")
    parser.add_argument("--no-log-action-details", action="store_true")
    parser.add_argument("--terminal-reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-shaping-scale", type=float, default=1.0)
    parser.add_argument("--reward-shaping-final-scale", type=float, default=None)
    parser.add_argument("--reward-shaping-phaseout-start-episodes", type=int, default=0)
    parser.add_argument("--reward-shaping-phaseout-end-episodes", type=int, default=0)
    parser.add_argument("--turn-vp-reward-scale", type=float, default=0.05)
    parser.add_argument("--nuke-death-penalty", type=float, default=0.0)
    parser.add_argument("--scoring-card-held-penalty", type=float, default=0.0)
    parser.add_argument("--high-stability-coup-penalty", type=float, default=0.0)
    parser.add_argument("--low-stability-warzone-coup-reward", type=float, default=0.0)
    parser.add_argument("--headline-opponent-event-penalty", type=float, default=0.0)
    parser.add_argument("--defcon-risk-pick-penalty", type=float, default=0.0)
    parser.add_argument("--defcon-risk-commit-penalty", type=float, default=0.0)
    parser.add_argument("--empty-country-influence-reward", type=float, default=0.0)
    parser.add_argument("--control-battleground-reward", type=float, default=0.0)
    parser.add_argument("--control-non-battleground-reward", type=float, default=0.0)
    parser.add_argument("--max-episode-step-penalty", type=float, default=0.0)
    parser.add_argument("--defcon-suicide-mode", choices=["none", "hard_filter", "penalty_only"], default="none")
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--warmup-random-steps", type=int, default=0)
    parser.add_argument("--warmup-random-prob", type=float, default=0.0)
    parser.add_argument("--persistent-random-prob", type=float, default=0.0)
    parser.add_argument("--heuristic-override-prob", type=float, default=0.0)
    parser.add_argument("--scripted-side-prob", type=float, default=0.0)
    parser.add_argument("--force-setup-heuristic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-agent", action="store_true")
    parser.add_argument("--league-manifest", type=Path, default=None)
    parser.add_argument("--league-role", choices=["main", "main_us", "main_ussr", "exploiter_us", "exploiter_ussr"], default="main")
    parser.add_argument("--league-train-side", choices=["both", "us", "ussr"], default="both")
    parser.add_argument("--league-opponent-seed", type=int, default=230624)
    parser.add_argument("--league-sampling-json", default=None)
    parser.add_argument("--checkpoint-eval-every-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-eval-every-episodes", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-min-games-per-side", type=int, default=30)
    parser.add_argument("--eval-history-opponents", type=int, default=0)
    parser.add_argument("--eval-max-episode-steps", type=int, default=None)
    parser.add_argument("--eval-progress-every-games", type=int, default=10)
    parser.add_argument("--elo-path", default=None)
    parser.add_argument("--elo-k-factor", type=float, default=32.0)
    parser.add_argument("--best-checkpoint-manifest", default=None)
    parser.add_argument("--best-score-side-weight", type=float, default=0.25)
    parser.add_argument("--best-score-balance-penalty", type=float, default=0.25)
    parser.add_argument("--best-score-nuke-penalty", type=float, default=100.0)
    parser.add_argument("--best-score-scoring-card-held-penalty", type=float, default=100.0)
    parser.add_argument("--best-score-random-side-floor", type=float, default=0.80)
    parser.add_argument("--best-score-random-side-penalty", type=float, default=200.0)
    parser.add_argument("--best-score-benchmark-side-floor", type=float, default=0.50)
    parser.add_argument("--best-score-benchmark-side-penalty", type=float, default=250.0)
    parser.add_argument("--early-stop-patience-evals", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    return parser.parse_args()


def parse_rollout_fragment_length(value: str) -> int | str:
    if str(value).lower() == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("rollout fragment length must be > 0 or 'auto'")
    return parsed


def flatten_scalars(payload: dict[str, Any], prefix: str = "") -> dict[str, int | float | bool]:
    scalars: dict[str, int | float | bool] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            scalars[name] = value
        elif isinstance(value, (int, float)):
            scalars[name] = value
        elif isinstance(value, dict):
            scalars.update(flatten_scalars(value, name))
    return scalars


def eval_games_per_side(eval_games: int, min_games_per_side: int) -> int:
    return max(0, int(eval_games), int(min_games_per_side))


def metric_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def new_elo_state(k_factor: float) -> dict[str, Any]:
    return {
        "version": 1,
        "k_factor": float(k_factor),
        "leaderboards": {
            "us": {"ratings": {}, "games": {}},
            "ussr": {"ratings": {}, "games": {}},
            "bot": {"ratings": {}, "games": {}},
        },
        "matches": [],
    }


def load_elo_state(path: Path, k_factor: float) -> dict[str, Any]:
    if not path.exists():
        return new_elo_state(k_factor)
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("version", 1)
    state["k_factor"] = float(k_factor)
    leaderboards = state.setdefault("leaderboards", {})
    for side in ("us", "ussr", "bot"):
        board = leaderboards.setdefault(side, {})
        board.setdefault("ratings", {})
        board.setdefault("games", {})
    state.setdefault("matches", [])
    return state


def save_elo_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def evaluated_checkpoint_score(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    bot_elo = float(metrics.get("elo/bot/current_rating", 1500.0) or 1500.0)
    us_elo = float(metrics.get("elo/us/current_rating", 1500.0) or 1500.0)
    ussr_elo = float(metrics.get("elo/ussr/current_rating", 1500.0) or 1500.0)
    side_floor = min(us_elo, ussr_elo)
    side_gap = abs(us_elo - ussr_elo)
    nuke_rates = [
        float(value)
        for key, value in metrics.items()
        if key.startswith("eval_vs_") and key.endswith("/nuke_terminal_rate") and isinstance(value, (int, float))
    ]
    mean_nuke = float(np.mean(nuke_rates)) if nuke_rates else 0.0
    scoring_card_held_rates = [
        float(value)
        for key, value in metrics.items()
        if key.startswith("eval_vs_")
        and key.endswith("/scoring_card_held_terminal_rate")
        and isinstance(value, (int, float))
    ]
    mean_scoring_card_held = float(np.mean(scoring_card_held_rates)) if scoring_card_held_rates else 0.0
    random_us = float(metrics.get("eval_vs_random/us_side_win_rate", 0.0) or 0.0)
    random_ussr = float(metrics.get("eval_vs_random/ussr_side_win_rate", 0.0) or 0.0)
    random_shortfall = max(0.0, float(args.best_score_random_side_floor) - min(random_us, random_ussr))
    benchmark_side_rates: list[float] = []
    for key, value in metrics.items():
        if not (
            key.startswith("eval_vs_")
            and (key.endswith("/us_side_win_rate") or key.endswith("/ussr_side_win_rate"))
        ):
            continue
        opponent = key.split("/", 1)[0].removeprefix("eval_vs_")
        if opponent == "random" or opponent.startswith("steps-"):
            continue
        try:
            benchmark_side_rates.append(float(value))
        except (TypeError, ValueError):
            pass
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
        "score": float(score),
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
    }


def write_best_checkpoint_manifest(
    path: Path,
    *,
    checkpoint: str,
    eval_label: str,
    episodes_seen: int,
    train_steps_seen: int,
    score_details: dict[str, float],
    eval_metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": checkpoint,
        "eval_label": eval_label,
        "episodes_seen": int(episodes_seen),
        "train_steps_seen": int(train_steps_seen),
        "score": float(score_details["score"]),
        "score_details": score_details,
        "eval_metrics": {
            key: value
            for key, value in eval_metrics.items()
            if isinstance(value, (int, float, str, bool))
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    link_path = path.parent / "best_checkpoint"
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(Path(checkpoint))
    except OSError:
        pass


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo_match(
    state: dict[str, Any],
    side: str,
    player: str,
    opponent: str,
    score: float,
    *,
    k_factor: float,
    meta: dict[str, Any] | None = None,
) -> tuple[float, float]:
    if side not in {"us", "ussr", "bot"}:
        raise ValueError(f"unknown Elo side leaderboard: {side}")
    board = state.setdefault("leaderboards", {}).setdefault(side, {"ratings": {}, "games": {}})
    ratings = board.setdefault("ratings", {})
    games = board.setdefault("games", {})
    rating_a = float(ratings.get(player, 1500.0))
    rating_b = float(ratings.get(opponent, 1500.0))
    expected_a = elo_expected(rating_a, rating_b)
    score_a = float(score)
    score_b = 1.0 - score_a
    new_a = rating_a + float(k_factor) * (score_a - expected_a)
    new_b = rating_b + float(k_factor) * (score_b - (1.0 - expected_a))
    ratings[player] = new_a
    ratings[opponent] = new_b
    games[player] = int(games.get(player, 0)) + 1
    games[opponent] = int(games.get(opponent, 0)) + 1
    state.setdefault("matches", []).append(
        {
            "side": side,
            "player": player,
            "opponent": opponent,
            "score": score_a,
            "player_rating_before": rating_a,
            "opponent_rating_before": rating_b,
            "player_rating_after": new_a,
            "opponent_rating_after": new_b,
            **(meta or {}),
        }
    )
    return new_a, new_b


def read_new_terminal_metrics(metrics_dir: Path | None, offsets: dict[str, int]) -> list[dict[str, Any]]:
    if metrics_dir is None or not metrics_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("terminal-*.jsonl")):
        key = str(path)
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offsets.get(key, 0))
            for line in handle:
                if not line.strip():
                    continue
                records.append(json.loads(line))
            offsets[key] = handle.tell()
    return records


def summarize_terminal_metrics(records: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    if not records:
        return {}
    count = len(records)
    winners = [str(record.get("winner") or "").lower() for record in records]
    reasons = [str(record.get("terminal_reason") or "") for record in records]
    metrics: dict[str, float | int] = {
        f"{prefix}/episodes": count,
        f"{prefix}/ussr_win_rate": sum(1 for winner in winners if winner == "ussr") / count,
        f"{prefix}/us_win_rate": sum(1 for winner in winners if winner == "us") / count,
        f"{prefix}/tie_rate": sum(1 for winner in winners if winner == "tie") / count,
        f"{prefix}/nuke_terminal_rate": sum(1 for reason in reasons if reason in NUKE_TERMINAL_REASONS) / count,
        f"{prefix}/nuke_terminal_ratio": sum(1 for reason in reasons if reason in NUKE_TERMINAL_REASONS) / count,
        f"{prefix}/vp_threshold_rate": sum(1 for reason in reasons if reason == "vp_threshold") / count,
        f"{prefix}/final_scoring_rate": sum(1 for reason in reasons if reason == "final scoring") / count,
        f"{prefix}/scoring_card_held_terminal_rate": sum(1 for reason in reasons if reason == "scoring card held") / count,
        f"{prefix}/avg_steps": float(np.mean([float(record.get("steps") or 0.0) for record in records])),
        f"{prefix}/us_reward": float(np.mean([float(record.get("us_reward") or 0.0) for record in records])),
        f"{prefix}/ussr_reward": float(np.mean([float(record.get("ussr_reward") or 0.0) for record in records])),
        f"{prefix}/us_reward_mean": float(np.mean([float(record.get("us_reward") or 0.0) for record in records])),
        f"{prefix}/ussr_reward_mean": float(np.mean([float(record.get("ussr_reward") or 0.0) for record in records])),
        f"{prefix}/no_legal_actions_rate": sum(1 for record in records if record.get("kind") == "no_legal_actions" or record.get("no_legal_actions")) / count,
        f"{prefix}/bridge_error_rate": sum(1 for record in records if record.get("kind") == "bridge_error" or record.get("bridge_error")) / count,
        f"{prefix}/timeout_rate": sum(1 for record in records if record.get("kind") == "truncated" or record.get("terminal_reason") in {"max_episode_steps", "eval_max_episode_steps"}) / count,
        f"{prefix}/filtered_action_mean": float(np.mean([float(record.get("filtered_action_count") or 0.0) for record in records])),
        f"{prefix}/unsafe_filter_skipped_rate": sum(1 for record in records if float(record.get("unsafe_filter_skipped_count") or 0.0) > 0.0) / count,
        f"{prefix}/unsafe_filter_skipped_mean": float(np.mean([float(record.get("unsafe_filter_skipped_count") or 0.0) for record in records])),
        f"{prefix}/us_nuke_loss_rate": sum(1 for record in records if record.get("nuke_loser") == "us") / count,
        f"{prefix}/ussr_nuke_loss_rate": sum(1 for record in records if record.get("nuke_loser") == "ussr") / count,
    }
    component_keys = sorted(
        {
            key
            for record in records
            for key, value in (record.get("episode_reward_components") or {}).items()
            if isinstance(value, (int, float))
        }
    )
    for key in component_keys:
        values = [float((record.get("episode_reward_components") or {}).get(key, 0.0)) for record in records]
        metrics[f"{prefix}/reward_{key}_mean"] = float(np.mean(values))
    return metrics


def compute_algo_action(algo: Any, obs: dict[str, Any], policy_id: str | None = None) -> int:
    try:
        action = algo.compute_single_action(obs, policy_id=policy_id, explore=False)
    except Exception:
        action = algo.compute_single_action(obs, explore=False)
    if isinstance(action, tuple):
        action = action[0]
    return int(action)


def random_legal_action(obs: dict[str, Any], rng: np.random.Generator) -> int:
    legal = np.flatnonzero(np.asarray(obs["action_mask"]) > 0.5)
    if len(legal) == 0:
        return 0
    return int(rng.choice(legal))


def eval_benchmark_policy_id(which: str, side: str, side_policy_ids: dict[str, str], switcher: "EvalPolicySwitcher") -> str:
    weights = switcher.weights.get(which, {})
    candidates = [
        side_to_policy_id(side, "split"),
        side_policy_ids.get(side, ""),
        SHARED_POLICY_ID,
        "default_policy",
    ]
    for candidate in candidates:
        if candidate and candidate in weights:
            return candidate
    raise KeyError(f"missing {which} weights for {side}; available policies: {sorted(weights)}")


class EvalPolicySwitcher:
    def __init__(
        self,
        algo: Any,
        current_weights: dict[str, dict[str, Any]],
        benchmark_weights: dict[str, dict[str, dict[str, Any]]],
        *,
        partial_compatible_load: bool = False,
    ):
        self.algo = algo
        self.weights = {"current": current_weights, **benchmark_weights}
        self.active: dict[str, str] = {}
        self.partial_compatible_load = partial_compatible_load

    def action(self, policy_id: str, which: str, obs: dict[str, Any]) -> int:
        policy = self.algo.get_policy(policy_id)
        if self.active.get(policy_id) != which:
            if which in self.weights and policy_id in self.weights[which]:
                weights = self.weights[which][policy_id]
                if self.partial_compatible_load and which != "current":
                    current = policy.get_weights()
                    current, _matched, _widened, _skipped_shape, _missing_source, _unexpected_source = (
                        copy_compatible_or_widened_weights(current, weights)
                    )
                    policy.set_weights(current)
                else:
                    policy.set_weights(weights)
            else:
                raise KeyError(f"missing {which} weights for policy {policy_id}")
            self.active[policy_id] = which
        return compute_algo_action(self.algo, obs, policy_id)

    def restore_current(self) -> None:
        for policy_id, weights in self.weights["current"].items():
            self.algo.get_policy(policy_id).set_weights(weights)
        self.active = {}


def make_eval_env_config(env_config: dict[str, Any], max_episode_steps: int | None) -> dict[str, Any]:
    config = {
        **env_config,
        "log_games_dir": None,
        "metrics_dir": None,
        "log_debug_snapshots": False,
        "log_action_details": False,
        "warmup_random_steps": 0,
        "warmup_random_prob": 0.0,
        "persistent_random_prob": 0.0,
        "heuristic_override_prob": 0.0,
        "scripted_side_prob": 0.0,
    }
    if max_episode_steps is not None:
        config["max_episode_steps"] = max_episode_steps
    return config


def reward_shaping_scale_for_episodes(args: argparse.Namespace, episodes_seen: int) -> float:
    start_scale = float(args.reward_shaping_scale)
    if args.reward_shaping_final_scale is None:
        return start_scale
    final_scale = float(args.reward_shaping_final_scale)
    start_episode = int(args.reward_shaping_phaseout_start_episodes)
    end_episode = int(args.reward_shaping_phaseout_end_episodes)
    if end_episode <= start_episode:
        return final_scale if episodes_seen >= start_episode else start_scale
    if episodes_seen <= start_episode:
        return start_scale
    if episodes_seen >= end_episode:
        return final_scale
    progress = (episodes_seen - start_episode) / float(end_episode - start_episode)
    return start_scale + progress * (final_scale - start_scale)


def apply_reward_shaping_scale(algo: Any, scale: float) -> None:
    def _apply(env) -> None:
        if hasattr(env, "set_reward_shaping_scale"):
            env.set_reward_shaping_scale(scale)
            return
        base_env = getattr(env, "base_env", None)
        if base_env is not None and hasattr(base_env, "set_reward_shaping_scale"):
            base_env.set_reward_shaping_scale(scale)

    applied = False
    try:
        env_runner_attr = getattr(algo, "env_runner", None)
        env_runner = env_runner_attr() if callable(env_runner_attr) else env_runner_attr
        if env_runner is not None and hasattr(env_runner, "foreach_env"):
            env_runner.foreach_env(_apply)
            applied = True
    except Exception as exc:
        print({"event": "reward_shaping_scale_apply_warning", "target": "env_runner", "error": str(exc)}, flush=True)
    if applied:
        return
    try:
        workers_attr = getattr(algo, "workers", None)
        workers = workers_attr() if callable(workers_attr) else workers_attr
        if workers is not None:
            workers.foreach_worker(lambda worker: worker.foreach_env(_apply))
            applied = True
    except Exception as exc:
        print({"event": "reward_shaping_scale_apply_warning", "target": "workers", "error": str(exc)}, flush=True)
    if not applied:
        print({"event": "reward_shaping_scale_apply_warning", "target": "none", "scale": scale}, flush=True)


def play_eval_game(
    algo: Any,
    env_config: dict[str, Any],
    seed: int,
    current_side: str,
    opponent: str,
    rng: np.random.Generator,
    side_policy_ids: dict[str, str],
    switcher: EvalPolicySwitcher | None = None,
) -> dict[str, Any]:
    env = TwilightStruggleEnv(env_config)
    try:
        obs, _info = env.reset(seed=seed)
        done = False
        final_info: dict[str, Any] = {}
        while not done:
            side = str((env.last_obs or {}).get("side") or "ussr")
            if side == current_side:
                policy_id = side_policy_ids[current_side]
                action = switcher.action(policy_id, "current", obs) if switcher else compute_algo_action(algo, obs, policy_id)
            elif opponent == "initial":
                if switcher is None:
                    raise RuntimeError("initial-opponent evaluation requires an EvalPolicySwitcher")
                policy_id = eval_benchmark_policy_id("initial", side, side_policy_ids, switcher)
                action = switcher.action(policy_id, "initial", obs)
            elif opponent == "random":
                action = random_legal_action(obs, rng)
            else:
                if switcher is None:
                    raise RuntimeError("benchmark-opponent evaluation requires an EvalPolicySwitcher")
                policy_id = eval_benchmark_policy_id(opponent, side, side_policy_ids, switcher)
                action = switcher.action(policy_id, opponent, obs)
            try:
                obs, _reward, terminated, truncated, final_info = env.step(action)
            except RuntimeError as exc:
                if "episode exceeded max_episode_steps" not in str(exc):
                    raise
                final_info = {
                    "winner": "timeout",
                    "terminal_reason": "eval_max_episode_steps",
                    "error": str(exc),
                }
                break
            done = bool(terminated or truncated)
        return {
            "winner": final_info.get("winner"),
            "terminal_reason": final_info.get("terminal_reason"),
            "steps": env.episode_step,
            "current_side": current_side,
        }
    finally:
        env.close()


def evaluate_policy(
    algo: Any,
    benchmark_weights: dict[str, dict[str, dict[str, Any]]],
    env_config: dict[str, Any],
    eval_games: int,
    min_games_per_side: int,
    max_episode_steps: int | None,
    seed_base: int,
    side_policy_ids: dict[str, str],
    eval_label: str,
    progress_every_games: int = 10,
    elo_state: dict[str, Any] | None = None,
    elo_k_factor: float = 32.0,
) -> dict[str, float | int]:
    games_per_side = eval_games_per_side(eval_games, min_games_per_side)
    if games_per_side <= 0:
        return {}
    eval_env_config = make_eval_env_config(env_config, max_episode_steps)
    unique_policy_ids = sorted(set(side_policy_ids.values()))
    unified_policy_eval = len(unique_policy_ids) == 1 and unique_policy_ids[0] == SHARED_POLICY_ID
    current_weights = {policy_id: algo.get_policy(policy_id).get_weights() for policy_id in unique_policy_ids}
    switcher = EvalPolicySwitcher(
        algo,
        current_weights,
        benchmark_weights,
        partial_compatible_load=True,
    )
    rng = np.random.default_rng(seed_base)
    metrics: dict[str, float | int] = {}
    try:
        opponents = [*benchmark_weights.keys(), "random"]
        metrics["eval/opponent_count"] = len(opponents)
        metrics["eval/games_per_side"] = games_per_side
        metrics["eval/total_games"] = len(opponents) * 2 * games_per_side
        for opponent_index, opponent in enumerate(opponents):
            opponent_metric = metric_label(opponent)
            records = []
            for side_index, current_side in enumerate(("us", "ussr")):
                for side_game_index in range(games_per_side):
                    records.append(
                        play_eval_game(
                            algo=algo,
                            env_config=eval_env_config,
                            seed=seed_base
                            + opponent_index * 10000
                            + side_index * games_per_side
                            + side_game_index,
                            current_side=current_side,
                            opponent=opponent,
                            rng=rng,
                            side_policy_ids=side_policy_ids,
                            switcher=switcher,
                        )
                    )
                    if progress_every_games > 0 and len(records) % progress_every_games == 0:
                        recent = records[-progress_every_games:]
                        print(
                            {
                                "event": "checkpoint_eval_progress",
                                "eval_label": eval_label,
                                "opponent": opponent,
                                "completed_games": len(records),
                                "total_games": 2 * games_per_side,
                                "current_side": current_side,
                                "recent_current_win_rate": sum(
                                    1 for record in recent if record.get("winner") == record.get("current_side")
                                )
                                / len(recent),
                                "recent_nuke_terminal_rate": sum(
                                    1 for record in recent if record.get("terminal_reason") in NUKE_TERMINAL_REASONS
                                )
                                / len(recent),
                                "recent_timeout_rate": sum(
                                    1 for record in recent if record.get("terminal_reason") == "eval_max_episode_steps"
                                )
                                / len(recent),
                            },
                            flush=True,
                        )
            count = len(records)
            current_wins = sum(1 for record in records if record.get("winner") == record.get("current_side"))
            us_games = [record for record in records if record.get("current_side") == "us"]
            ussr_games = [record for record in records if record.get("current_side") == "ussr"]
            nuke_count = sum(1 for record in records if record.get("terminal_reason") in NUKE_TERMINAL_REASONS)
            scoring_card_held_count = sum(1 for record in records if record.get("terminal_reason") == "scoring card held")
            timeout_count = sum(1 for record in records if record.get("terminal_reason") == "eval_max_episode_steps")
            metrics[f"eval_vs_{opponent_metric}/games"] = count
            metrics[f"eval_vs_{opponent_metric}/games_per_side"] = games_per_side
            metrics[f"eval_vs_{opponent_metric}/win_rate"] = current_wins / count if count else 0.0
            metrics[f"eval_vs_{opponent_metric}/us_side_win_rate"] = (
                sum(1 for record in us_games if record.get("winner") == "us") / len(us_games) if us_games else 0.0
            )
            metrics[f"eval_vs_{opponent_metric}/ussr_side_win_rate"] = (
                sum(1 for record in ussr_games if record.get("winner") == "ussr") / len(ussr_games) if ussr_games else 0.0
            )
            metrics[f"eval_vs_{opponent_metric}/nuke_terminal_rate"] = nuke_count / count if count else 0.0
            metrics[f"eval_vs_{opponent_metric}/scoring_card_held_terminal_rate"] = (
                scoring_card_held_count / count if count else 0.0
            )
            metrics[f"eval_vs_{opponent_metric}/timeout_rate"] = timeout_count / count if count else 0.0
            metrics[f"eval_vs_{opponent_metric}/avg_steps"] = float(np.mean([float(record.get("steps") or 0.0) for record in records])) if records else 0.0
            print(
                {
                    "event": "checkpoint_eval_opponent_done",
                    "eval_label": eval_label,
                    "opponent": opponent,
                    "games": count,
                    "games_per_side": games_per_side,
                    "win_rate": metrics[f"eval_vs_{opponent_metric}/win_rate"],
                    "us_side_win_rate": metrics[f"eval_vs_{opponent_metric}/us_side_win_rate"],
                    "ussr_side_win_rate": metrics[f"eval_vs_{opponent_metric}/ussr_side_win_rate"],
                    "nuke_terminal_rate": metrics[f"eval_vs_{opponent_metric}/nuke_terminal_rate"],
                    "scoring_card_held_terminal_rate": metrics[f"eval_vs_{opponent_metric}/scoring_card_held_terminal_rate"],
                    "timeout_rate": metrics[f"eval_vs_{opponent_metric}/timeout_rate"],
                    "avg_steps": metrics[f"eval_vs_{opponent_metric}/avg_steps"],
                },
                flush=True,
            )
            if elo_state is not None:
                if unified_policy_eval:
                    bot_score = current_wins / count if count else 0.5
                    bot_player = f"{eval_label}:{SHARED_POLICY_ID}"
                    opponent_weights = benchmark_weights.get(opponent, {})
                    if opponent == "random":
                        bot_opponent_policy = RANDOM_LEGAL_POLICY_ID
                    elif US_POLICY_ID in opponent_weights or USSR_POLICY_ID in opponent_weights:
                        bot_opponent_policy = "split_policy_pair"
                    else:
                        bot_opponent_policy = SHARED_POLICY_ID
                    bot_opponent = f"{opponent}:{bot_opponent_policy}"
                    bot_rating, bot_opponent_rating = update_elo_match(
                        elo_state,
                        "bot",
                        bot_player,
                        bot_opponent,
                        bot_score,
                        k_factor=elo_k_factor,
                        meta={
                            "eval_label": eval_label,
                            "opponent_family": opponent,
                            "games": count,
                            "us_side_win_rate": metrics[f"eval_vs_{opponent_metric}/us_side_win_rate"],
                            "ussr_side_win_rate": metrics[f"eval_vs_{opponent_metric}/ussr_side_win_rate"],
                            "nuke_terminal_rate": metrics[f"eval_vs_{opponent_metric}/nuke_terminal_rate"],
                            "scoring_card_held_terminal_rate": metrics[
                                f"eval_vs_{opponent_metric}/scoring_card_held_terminal_rate"
                            ],
                            "timeout_rate": metrics[f"eval_vs_{opponent_metric}/timeout_rate"],
                            "aggregate": True,
                        },
                    )
                    metrics[f"elo/bot/{opponent_metric}_score"] = bot_score
                    metrics[f"elo/bot/current_rating_vs_{opponent_metric}"] = float(bot_rating)
                    metrics[f"elo/bot/{opponent_metric}_opponent_rating"] = float(bot_opponent_rating)
                for record in records:
                    current_side = str(record.get("current_side"))
                    opponent_side = "ussr" if current_side == "us" else "us"
                    winner = record.get("winner")
                    if winner == current_side:
                        score = 1.0
                    elif winner == opponent_side:
                        score = 0.0
                    else:
                        score = 0.5
                    update_elo_match(
                        elo_state,
                        current_side,
                        f"{eval_label}:{current_side}_policy",
                        f"{opponent}:{opponent_side}_policy",
                        score,
                        k_factor=elo_k_factor,
                        meta={
                            "eval_label": eval_label,
                            "opponent_family": opponent,
                            "winner": winner,
                            "terminal_reason": record.get("terminal_reason"),
                            "steps": record.get("steps"),
                        },
                    )
        if elo_state is not None:
            for side in ("us", "ussr"):
                ratings = (elo_state.get("leaderboards", {}).get(side, {}) or {}).get("ratings", {})
                games = (elo_state.get("leaderboards", {}).get(side, {}) or {}).get("games", {})
                current_player = f"{eval_label}:{side}_policy"
                metrics[f"elo/{side}/current_rating"] = float(ratings.get(current_player, 1500.0))
                metrics[f"elo/{side}/current_games"] = int(games.get(current_player, 0))
                metrics[f"elo/{side}/leaderboard_size"] = len(ratings)
                for opponent_side in (("ussr",) if side == "us" else ("us",)):
                    for opponent in opponents:
                        opponent_metric = metric_label(opponent)
                        opponent_player = f"{opponent}:{opponent_side}_policy"
                        metrics[f"elo/{side}/{opponent_metric}_opponent_rating"] = float(ratings.get(opponent_player, 1500.0))
                        metrics[f"elo/{side}/{opponent_metric}_opponent_games"] = int(games.get(opponent_player, 0))
            if unified_policy_eval:
                ratings = (elo_state.get("leaderboards", {}).get("bot", {}) or {}).get("ratings", {})
                games = (elo_state.get("leaderboards", {}).get("bot", {}) or {}).get("games", {})
                current_player = f"{eval_label}:{SHARED_POLICY_ID}"
                metrics["elo/bot/current_rating"] = float(ratings.get(current_player, 1500.0))
                metrics["elo/bot/current_games"] = int(games.get(current_player, 0))
                metrics["elo/bot/leaderboard_size"] = len(ratings)
                for opponent in opponents:
                    opponent_metric = metric_label(opponent)
                    opponent_weights = benchmark_weights.get(opponent, {})
                    if opponent == "random":
                        opponent_policy = RANDOM_LEGAL_POLICY_ID
                    elif US_POLICY_ID in opponent_weights or USSR_POLICY_ID in opponent_weights:
                        opponent_policy = "split_policy_pair"
                    else:
                        opponent_policy = SHARED_POLICY_ID
                    opponent_player = f"{opponent}:{opponent_policy}"
                    metrics[f"elo/bot/{opponent_metric}_opponent_games"] = int(games.get(opponent_player, 0))
    finally:
        switcher.restore_current()
    return metrics


def save_checkpoint(algo: Any, checkpoint_root: Path, label: str) -> str:
    target = checkpoint_root / "checkpoints" / label
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = algo.save(str(target))
    checkpoint_path = getattr(getattr(checkpoint, "checkpoint", checkpoint), "path", checkpoint)
    checkpoint_path = str(checkpoint_path)
    strip_stateless_baseline_policies(Path(checkpoint_path))
    return checkpoint_path


def strip_stateless_baseline_policies(checkpoint_path: Path) -> None:
    metadata_path = checkpoint_path / "rllib_checkpoint.json"
    if not metadata_path.exists():
        return
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    policy_ids = list(metadata.get("policy_ids") or [])
    stripped = [policy_id for policy_id in (RANDOM_LEGAL_POLICY_ID, HEURISTIC_POLICY_ID) if policy_id in policy_ids]
    if not stripped:
        return
    metadata["policy_ids"] = [policy_id for policy_id in policy_ids if policy_id not in stripped]
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    for policy_id in stripped:
        shutil.rmtree(checkpoint_path / "policies" / policy_id, ignore_errors=True)
    marker_path = checkpoint_path / "stripped_stateless_policies.json"
    marker_path.write_text(json.dumps({"stripped_policy_ids": stripped}, indent=2, sort_keys=True), encoding="utf-8")


def checkpoint_policy_ids(checkpoint_path: Path) -> set[str]:
    metadata_path = checkpoint_path / "rllib_checkpoint.json"
    if not metadata_path.exists():
        return set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {str(policy_id) for policy_id in metadata.get("policy_ids", [])}


def load_policy_weights(checkpoint_path: Path, policy_id: str) -> dict[str, Any]:
    import pickle

    from ray.rllib.policy.policy import Policy

    register_masked_model()
    policy_path = checkpoint_path / "policies" / policy_id
    if not policy_path.exists():
        policy_path = checkpoint_path
    policy_state_path = policy_path / "policy_state.pkl"
    if policy_state_path.exists():
        with policy_state_path.open("rb") as handle:
            state = pickle.load(handle)
        weights = state.get("weights") if isinstance(state, dict) else None
        if isinstance(weights, dict):
            return weights
    policy = Policy.from_checkpoint(str(policy_path.resolve()))
    if isinstance(policy, dict):
        if policy_id in policy:
            policy = policy[policy_id]
        elif len(policy) == 1:
            policy = next(iter(policy.values()))
        else:
            raise RuntimeError(f"checkpoint {checkpoint_path} has ambiguous policies: {sorted(policy)}")
    return policy.get_weights()


def copy_compatible_or_widened_weights(
    current: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    updated = dict(current)
    matched: list[str] = []
    widened: list[dict[str, Any]] = []
    skipped_shape: list[dict[str, Any]] = []
    missing_source: list[str] = []
    unexpected_source: list[str] = []
    for key, value in current.items():
        if key not in source:
            missing_source.append(key)
            continue
        current_array = np.asarray(value)
        source_array = np.asarray(source[key])
        if current_array.shape == source_array.shape:
            updated[key] = source[key]
            matched.append(key)
            continue
        if (
            current_array.ndim == 2
            and source_array.ndim == 2
            and current_array.shape[0] == source_array.shape[0]
            and current_array.shape[1] > source_array.shape[1]
        ):
            widened_value = np.zeros_like(current_array)
            widened_value[:, : source_array.shape[1]] = source_array
            updated[key] = widened_value
            widened.append(
                {
                    "key": key,
                    "current_shape": list(current_array.shape),
                    "source_shape": list(source_array.shape),
                    "copied_columns": int(source_array.shape[1]),
                    "zeroed_new_columns": int(current_array.shape[1] - source_array.shape[1]),
                }
            )
            continue
        skipped_shape.append(
            {
                "key": key,
                "current_shape": list(current_array.shape),
                "source_shape": list(source_array.shape),
            }
        )
    for key in source:
        if key not in current:
            unexpected_source.append(key)
    return updated, matched, widened, skipped_shape, missing_source, unexpected_source


def main() -> None:
    args = parse_args()
    has_split_benchmark = bool(args.benchmark_us_policy_from or args.benchmark_ussr_policy_from)
    if has_split_benchmark and not (args.benchmark_us_policy_from and args.benchmark_ussr_policy_from):
        raise SystemExit("--benchmark-us-policy-from and --benchmark-ussr-policy-from must be provided together")
    if args.unified_train_focus_side != "both" and args.policy_sharing != "unified":
        raise SystemExit("--unified-train-focus-side requires --policy-sharing unified")
    if args.unified_opponent_mix_json and args.policy_sharing != "unified":
        raise SystemExit("--unified-opponent-mix-json requires --policy-sharing unified")
    if args.unified_opponent_mix_json and args.unified_train_focus_side != "both":
        raise SystemExit("--unified-opponent-mix-json cannot be combined with --unified-train-focus-side")
    unified_opponent_mix = (
        normalize_unified_opponent_mix(json.loads(args.unified_opponent_mix_json))
        if args.unified_opponent_mix_json
        else None
    )
    unified_anchor_paths = parse_path_list(args.unified_anchor_policy_from)
    if unified_opponent_mix and unified_opponent_mix.get("teacher", 0.0) > 0 and not has_split_benchmark:
        raise SystemExit("teacher weight in --unified-opponent-mix-json requires --benchmark-us-policy-from and --benchmark-ussr-policy-from")
    if unified_opponent_mix and unified_opponent_mix.get("anchor", 0.0) > 0 and not unified_anchor_paths:
        raise SystemExit("anchor weight in --unified-opponent-mix-json requires --unified-anchor-policy-from")
    unified_anchor_policy_ids = [
        f"unified_anchor_policy_{index:02d}"
        for index, _path in enumerate(unified_anchor_paths)
    ]
    focus_fixed_policy_id: str | None = None
    if args.policy_sharing == "unified" and args.unified_train_focus_side == "us":
        focus_fixed_policy_id = USSR_POLICY_ID
    elif args.policy_sharing == "unified" and args.unified_train_focus_side == "ussr":
        focus_fixed_policy_id = US_POLICY_ID
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    register_masked_model()
    league_manifest: dict[str, Any] | None = None
    league_sources: dict[str, dict[str, str]] = {}
    league_pools: dict[str, dict[str, list[str]]] = {}
    league_sampling = normalize_sampling(None)
    if args.league_manifest:
        if args.policy_sharing == "unified":
            raise SystemExit("--policy-sharing unified is not wired for --league-manifest yet")
        with Path(args.league_manifest).expanduser().open("r", encoding="utf-8") as handle:
            league_manifest = json.load(handle)
        league_sources = league_policy_sources(league_manifest)
        league_pools = league_ids_by_side_and_pool(league_sources)
        manifest_sampling = league_manifest.get("opponent_sampling") if isinstance(league_manifest, dict) else None
        override_sampling = json.loads(args.league_sampling_json) if args.league_sampling_json else None
        league_sampling = normalize_sampling(override_sampling or manifest_sampling)
        args.multi_agent = True
        print(
            {
                "event": "league_manifest_loaded",
                "league_manifest": str(args.league_manifest),
                "league_role": args.league_role,
                "league_train_side": args.league_train_side,
                "fixed_policy_count": len(league_sources),
                "opponent_sampling": league_sampling,
            },
            flush=True,
        )

    env_config = {
        "log_games_dir": args.log_games_dir,
        "log_games_every": args.log_games_every,
        "log_debug_snapshots": args.log_debug_snapshots,
        "log_action_details": not args.no_log_action_details,
        "terminal_reward_scale": args.terminal_reward_scale,
        "reward_shaping_scale": args.reward_shaping_scale,
        "turn_vp_reward_scale": args.turn_vp_reward_scale,
        "nuke_death_penalty": args.nuke_death_penalty,
        "scoring_card_held_penalty": args.scoring_card_held_penalty,
        "high_stability_coup_penalty": args.high_stability_coup_penalty,
        "low_stability_warzone_coup_reward": args.low_stability_warzone_coup_reward,
        "headline_opponent_event_penalty": args.headline_opponent_event_penalty,
        "defcon_risk_pick_penalty": args.defcon_risk_pick_penalty,
        "defcon_risk_commit_penalty": args.defcon_risk_commit_penalty,
        "empty_country_influence_reward": args.empty_country_influence_reward,
        "control_battleground_reward": args.control_battleground_reward,
        "control_non_battleground_reward": args.control_non_battleground_reward,
        "max_episode_step_penalty": args.max_episode_step_penalty,
        "defcon_suicide_mode": args.defcon_suicide_mode,
        "max_episode_steps": args.max_episode_steps,
        "warmup_random_steps": args.warmup_random_steps,
        "warmup_random_prob": args.warmup_random_prob,
        "persistent_random_prob": args.persistent_random_prob,
        "heuristic_override_prob": args.heuristic_override_prob,
        "scripted_side_prob": args.scripted_side_prob,
        "force_setup_heuristic": args.force_setup_heuristic,
    }
    env_name = "twilight_struggle_bridge_multi" if args.multi_agent else "twilight_struggle_bridge"
    register_env(
        env_name,
        lambda cfg: TwilightStruggleMultiAgentEnv(cfg) if args.multi_agent else TwilightStruggleEnv(cfg),
    )
    probe_env = TwilightStruggleEnv({**env_config, "log_games_dir": None})
    probe_observation_space = probe_env.observation_space
    probe_action_space = probe_env.action_space
    policy_spec = (None, probe_observation_space, probe_action_space, {})
    probe_env.close()
    checkpoint_root = Path(args.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else checkpoint_root / "metrics"
    env_config["metrics_dir"] = str(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    default_elo_name = "elo_unified.json" if args.policy_sharing == "unified" else "elo_ratings.json"
    elo_path = Path(args.elo_path).expanduser().resolve() if args.elo_path else checkpoint_root / default_elo_name
    elo_state = load_elo_state(elo_path, args.elo_k_factor)
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("Install W&B support with: pip install -e '.[train]'") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            dir=str(checkpoint_root),
            config=vars(args),
        )

    ray.init(ignore_reinit_error=True)
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env_name, env_config=env_config)
        .framework(args.framework)
        .resources(num_gpus=args.num_gpus)
        .env_runners(
            num_env_runners=args.num_env_runners,
            rollout_fragment_length=args.rollout_fragment_length,
            batch_mode=args.batch_mode,
            sample_timeout_s=args.sample_timeout_s,
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
        )
        .training(
            gamma=0.997,
            lambda_=0.95,
            lr=args.lr,
            clip_param=0.2,
            use_kl_loss=True,
            kl_coeff=args.kl_coeff,
            kl_target=args.kl_target,
            entropy_coeff=args.entropy_coeff,
            num_epochs=args.num_epochs,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            model={
                "custom_model": "twilight_masked_model",
                "custom_model_config": {
                    "hidden": args.hidden,
                    "model_arch": args.model_arch,
                    "history_layers": args.history_layers,
                    "card_history_layers": args.card_history_layers,
                    "history_attention_heads": args.history_attention_heads,
                    "history_dropout": args.history_dropout,
                    "graph_layers": args.graph_layers,
                    "graph_neighbor_hops": args.graph_neighbor_hops,
                    "heuristic_prior_scale": args.heuristic_prior_scale,
                    "setup_heuristic_prior_scale": args.setup_heuristic_prior_scale,
                    "policy_temperature": args.policy_temperature,
                },
            },
        )
    )
    if args.multi_agent:
        if args.policy_sharing == "unified":
            policies = {SHARED_POLICY_ID: policy_spec}
            if focus_fixed_policy_id:
                policies[focus_fixed_policy_id] = policy_spec
            if has_split_benchmark:
                policies[US_POLICY_ID] = policy_spec
                policies[USSR_POLICY_ID] = policy_spec
            for policy_id in unified_anchor_policy_ids:
                policies[policy_id] = policy_spec
            if unified_opponent_mix:
                policies[RANDOM_LEGAL_POLICY_ID] = (RandomLegalPolicy, probe_observation_space, probe_action_space, {})
                policies[HEURISTIC_POLICY_ID] = (HeuristicPolicy, probe_observation_space, probe_action_space, {})
        else:
            policies = {
                US_POLICY_ID: policy_spec,
                USSR_POLICY_ID: policy_spec,
            }
        for policy_id in league_sources:
            policies[policy_id] = policy_spec
        if args.league_manifest:
            policies[RANDOM_LEGAL_POLICY_ID] = (RandomLegalPolicy, probe_observation_space, probe_action_space, {})
            policies[HEURISTIC_POLICY_ID] = (HeuristicPolicy, probe_observation_space, probe_action_space, {})
        train_policy_ids = (
            league_train_policy_ids(args.league_role, args.league_train_side)
            if args.league_manifest
            else parse_policies_to_train(args.policies_to_train, multi_agent=True, policy_sharing=args.policy_sharing)
        )
        league_rng = random.Random(args.league_opponent_seed)
        unified_mix_rng = random.Random(args.unified_opponent_mix_seed)
        unified_train_us_probability = 0.5

        def _sample_unified_opponent_mix_mapping() -> dict[str, str]:
            assert unified_opponent_mix is not None
            train_side = "us" if unified_mix_rng.random() < unified_train_us_probability else "ussr"
            opponent_side = "ussr" if train_side == "us" else "us"
            category = weighted_choice(unified_mix_rng, unified_opponent_mix)
            if category == "teacher":
                opponent_policy = US_POLICY_ID if opponent_side == "us" else USSR_POLICY_ID
            elif category == "anchor" and unified_anchor_policy_ids:
                opponent_policy = unified_mix_rng.choice(unified_anchor_policy_ids)
            elif category == "random_legal":
                opponent_policy = RANDOM_LEGAL_POLICY_ID
            elif category == "heuristic":
                opponent_policy = HEURISTIC_POLICY_ID
            else:
                category = "current_self"
                opponent_policy = SHARED_POLICY_ID
            return {
                train_side: SHARED_POLICY_ID,
                opponent_side: opponent_policy,
                "_unified_train_side": train_side,
                "_unified_opponent_category": category,
            }

        def _sample_league_mapping() -> dict[str, str]:
            train_sides = league_train_sides(args.league_role, args.league_train_side)
            train_side = train_sides[0] if len(train_sides) == 1 else league_rng.choice(train_sides)
            opponent_side = "ussr" if train_side == "us" else "us"
            category = weighted_choice(league_rng, league_sampling)
            opponent_policy = None
            if category == "current_main":
                candidates = league_pools.get(opponent_side, {}).get("current", [])
                opponent_policy = candidates[0] if candidates else side_to_policy_id(opponent_side)
            elif category == "best_historical":
                candidates = league_pools.get(opponent_side, {}).get("best", []) or league_pools.get(opponent_side, {}).get("current", [])
                opponent_policy = league_rng.choice(candidates) if candidates else side_to_policy_id(opponent_side)
            elif category == "random_history":
                candidates = league_pools.get(opponent_side, {}).get("history", []) or league_pools.get(opponent_side, {}).get("current", [])
                opponent_policy = league_rng.choice(candidates) if candidates else side_to_policy_id(opponent_side)
            elif category == "exploiter":
                candidates = league_pools.get(opponent_side, {}).get("exploiter", []) or league_pools.get(opponent_side, {}).get("current", [])
                opponent_policy = candidates[0] if candidates else side_to_policy_id(opponent_side)
            elif category == "random_legal":
                opponent_policy = RANDOM_LEGAL_POLICY_ID
            elif category == "heuristic":
                opponent_policy = HEURISTIC_POLICY_ID
            else:
                opponent_policy = side_to_policy_id(opponent_side)
            mapping = {
                train_side: side_to_policy_id(train_side),
                opponent_side: opponent_policy,
                "_league_train_side": train_side,
                "_league_opponent_category": category,
            }
            return mapping

        def _league_policy_mapping(agent_id, episode, worker, **kwargs):
            if not args.league_manifest:
                if args.policy_sharing == "unified":
                    if unified_opponent_mix:
                        if "_unified_opponent_mix_mapping" not in episode.user_data:
                            episode.user_data["_unified_opponent_mix_mapping"] = _sample_unified_opponent_mix_mapping()
                        return episode.user_data["_unified_opponent_mix_mapping"].get(str(agent_id), SHARED_POLICY_ID)
                    if args.unified_train_focus_side == "us":
                        return SHARED_POLICY_ID if str(agent_id) == "us" else USSR_POLICY_ID
                    if args.unified_train_focus_side == "ussr":
                        return SHARED_POLICY_ID if str(agent_id) == "ussr" else US_POLICY_ID
                    return SHARED_POLICY_ID
                return US_POLICY_ID if agent_id == "us" else USSR_POLICY_ID
            if "_league_policy_mapping" not in episode.user_data:
                episode.user_data["_league_policy_mapping"] = _sample_league_mapping()
            return episode.user_data["_league_policy_mapping"].get(str(agent_id), side_to_policy_id(str(agent_id)))

        config = config.multi_agent(
            policies=policies,
            policy_mapping_fn=_league_policy_mapping,
            policies_to_train=train_policy_ids,
        )
    algo = config.build()
    side_policy_ids = {
        "us": side_to_policy_id("us", args.policy_sharing) if args.multi_agent else "default_policy",
        "ussr": side_to_policy_id("ussr", args.policy_sharing) if args.multi_agent else "default_policy",
    }
    def _current_weights() -> dict[str, dict[str, Any]]:
        return copy.deepcopy({
            policy_id: algo.get_policy(policy_id).get_weights()
            for policy_id in set(side_policy_ids.values())
        })

    def _weights_from_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
        policy_ids = checkpoint_policy_ids(path)
        if args.multi_agent:
            if args.policy_sharing == "unified":
                if SHARED_POLICY_ID in policy_ids or (path / "policies" / SHARED_POLICY_ID).exists() or path.name == SHARED_POLICY_ID:
                    return {SHARED_POLICY_ID: load_policy_weights(path, SHARED_POLICY_ID)}
                source_policy_id = US_POLICY_ID if args.load_shared_source_side == "us" else USSR_POLICY_ID
                if source_policy_id in policy_ids or path.name == source_policy_id:
                    return {SHARED_POLICY_ID: load_policy_weights(path, source_policy_id)}
                if "default_policy" in policy_ids or (path / "policies" / "default_policy").exists() or path.name == "default_policy":
                    return {SHARED_POLICY_ID: load_policy_weights(path, "default_policy")}
                other_policy_id = USSR_POLICY_ID if source_policy_id == US_POLICY_ID else US_POLICY_ID
                if other_policy_id in policy_ids or path.name == other_policy_id:
                    return {SHARED_POLICY_ID: load_policy_weights(path, other_policy_id)}
            if US_POLICY_ID in policy_ids and USSR_POLICY_ID in policy_ids:
                return {
                    US_POLICY_ID: load_policy_weights(path, US_POLICY_ID),
                    USSR_POLICY_ID: load_policy_weights(path, USSR_POLICY_ID),
                }
            if US_POLICY_ID in policy_ids or path.name == US_POLICY_ID:
                return {US_POLICY_ID: load_policy_weights(path, US_POLICY_ID)}
            if USSR_POLICY_ID in policy_ids or path.name == USSR_POLICY_ID:
                return {USSR_POLICY_ID: load_policy_weights(path, USSR_POLICY_ID)}
            for fallback_id in ("shared_policy", "default_policy"):
                if fallback_id in policy_ids or (path / "policies" / fallback_id).exists():
                    weights = load_policy_weights(path, fallback_id)
                    return {US_POLICY_ID: weights, USSR_POLICY_ID: weights}
        if "default_policy" in policy_ids or (path / "policies" / "default_policy").exists():
            return {"default_policy": load_policy_weights(path, "default_policy")}
        if "shared_policy" in policy_ids or (path / "policies" / "shared_policy").exists():
            return {"default_policy": load_policy_weights(path, "shared_policy")}
        raise RuntimeError(f"cannot load policy weights from checkpoint {path}")

    def _restore_training_checkpoint(path: Path) -> None:
        policy_ids = checkpoint_policy_ids(path)
        if args.multi_agent and args.policy_sharing == "unified":
            if SHARED_POLICY_ID in policy_ids:
                if (path / "stripped_stateless_policies.json").exists():
                    weights_by_policy = _weights_from_checkpoint(path)
                    algo.get_policy(SHARED_POLICY_ID).set_weights(weights_by_policy[SHARED_POLICY_ID])
                    print(f"initialized_shared_policy_from_stripped_checkpoint={path}")
                    return
                algo.restore(str(path))
                print(f"restored_from={path}")
                return
            weights_by_policy = _weights_from_checkpoint(path)
            algo.get_policy(SHARED_POLICY_ID).set_weights(weights_by_policy[SHARED_POLICY_ID])
            print(f"initialized_shared_policy_from={path}")
            return
        if args.multi_agent and not ({US_POLICY_ID, USSR_POLICY_ID} <= policy_ids):
            weights_by_policy = _weights_from_checkpoint(path)
            algo.get_policy(US_POLICY_ID).set_weights(weights_by_policy[US_POLICY_ID])
            algo.get_policy(USSR_POLICY_ID).set_weights(weights_by_policy[USSR_POLICY_ID])
            print(f"initialized_separate_policies_from={path}")
            return
        algo.restore(str(path))
        print(f"restored_from={path}")

    load_reports: list[dict[str, Any]] = []

    def _load_weights_for_policy(path: Path, target_policy_id: str, preferred_source_policy_id: str | None = None) -> dict[str, Any]:
        policy_ids = checkpoint_policy_ids(path)
        candidate_policy_ids = [
            preferred_source_policy_id,
            target_policy_id,
            SHARED_POLICY_ID,
            "default_policy",
            US_POLICY_ID,
            USSR_POLICY_ID,
        ]
        for candidate_policy_id in candidate_policy_ids:
            if not candidate_policy_id:
                continue
            if (
                candidate_policy_id in policy_ids
                or path.name == candidate_policy_id
                or (path / "policies" / candidate_policy_id).exists()
            ):
                return load_policy_weights(path, candidate_policy_id)
        raise RuntimeError(f"cannot load weights for {target_policy_id} from checkpoint {path}")

    def _set_or_warmstart_policy(policy_id: str, path: Path, side: str) -> None:
        source_weights = _weights_from_checkpoint(path)[policy_id]
        _set_or_warmstart_policy_weights(policy_id, source_weights, path, side)

    def _set_or_warmstart_policy_weights(policy_id: str, source_weights: dict[str, Any], path: Path, side: str) -> None:
        policy = algo.get_policy(policy_id)
        use_partial = args.partial_warmstart if args.partial_warmstart is not None else args.model_arch == "transformer_history"
        if not use_partial:
            policy.set_weights(source_weights)
            print(f"loaded_{side}_policy_from={path}")
            return
        current = policy.get_weights()
        current, matched, widened, skipped_shape, missing_source, unexpected_source = copy_compatible_or_widened_weights(current, source_weights)
        policy.set_weights(current)
        report = {
            "side": side,
            "policy_id": policy_id,
            "source": str(path),
            "matched_count": len(matched),
            "widened_count": len(widened),
            "matched_keys": matched,
            "widened_keys": widened,
            "missing_source_keys": missing_source,
            "unexpected_source_keys": unexpected_source,
            "shape_mismatch": skipped_shape,
        }
        load_reports.append(report)
        print({"event": "partial_warmstart", **{k: v for k, v in report.items() if k != "matched_keys"}}, flush=True)

    if args.eval_initial_from:
        eval_initial_path = Path(args.eval_initial_from).expanduser().resolve()
        initial_weights = _weights_from_checkpoint(eval_initial_path)
        print(f"eval_initial_from={eval_initial_path}")
    else:
        initial_weights = None
    if args.restore_from:
        restore_path = Path(args.restore_from).expanduser().resolve()
        _restore_training_checkpoint(restore_path)
    if args.multi_agent and args.policy_sharing == "unified":
        shared_source: str | None = args.load_shared_policy_from
        shared_label = "shared"
        if shared_source is None:
            if args.load_shared_source_side == "ussr" and args.load_ussr_policy_from:
                shared_source = args.load_ussr_policy_from
                shared_label = "shared_from_ussr"
            elif args.load_us_policy_from:
                shared_source = args.load_us_policy_from
                shared_label = "shared_from_us"
            elif args.load_ussr_policy_from:
                shared_source = args.load_ussr_policy_from
                shared_label = "shared_from_ussr"
        if shared_source:
            shared_path = Path(shared_source).expanduser().resolve()
            _set_or_warmstart_policy(SHARED_POLICY_ID, shared_path, shared_label)
        if focus_fixed_policy_id:
            if args.focus_opponent_policy_from:
                focus_source = args.focus_opponent_policy_from
            elif focus_fixed_policy_id == USSR_POLICY_ID and args.benchmark_ussr_policy_from:
                focus_source = args.benchmark_ussr_policy_from
            elif focus_fixed_policy_id == US_POLICY_ID and args.benchmark_us_policy_from:
                focus_source = args.benchmark_us_policy_from
            elif focus_fixed_policy_id == USSR_POLICY_ID and args.load_ussr_policy_from:
                focus_source = args.load_ussr_policy_from
            elif focus_fixed_policy_id == US_POLICY_ID and args.load_us_policy_from:
                focus_source = args.load_us_policy_from
            else:
                raise SystemExit(
                    "--unified-train-focus-side requires --focus-opponent-policy-from "
                    "or the matching benchmark/load side policy path"
                )
            focus_path = Path(focus_source).expanduser().resolve()
            focus_weights = _load_weights_for_policy(
                focus_path,
                focus_fixed_policy_id,
                preferred_source_policy_id=focus_fixed_policy_id,
            )
            _set_or_warmstart_policy_weights(focus_fixed_policy_id, focus_weights, focus_path, f"fixed_{args.unified_train_focus_side}_opponent")
    elif args.multi_agent and args.load_us_policy_from:
        us_path = Path(args.load_us_policy_from).expanduser().resolve()
        _set_or_warmstart_policy(US_POLICY_ID, us_path, "us")
    elif args.league_manifest and not args.restore_from and league_manifest and (league_manifest.get("seed_policies") or {}).get("us"):
        us_path = Path((league_manifest.get("seed_policies") or {})["us"]).expanduser().resolve()
        _set_or_warmstart_policy(US_POLICY_ID, us_path, "us")
    if args.multi_agent and args.policy_sharing != "unified" and args.load_ussr_policy_from:
        ussr_path = Path(args.load_ussr_policy_from).expanduser().resolve()
        _set_or_warmstart_policy(USSR_POLICY_ID, ussr_path, "ussr")
    elif args.league_manifest and not args.restore_from and league_manifest and (league_manifest.get("seed_policies") or {}).get("ussr"):
        ussr_path = Path((league_manifest.get("seed_policies") or {})["ussr"]).expanduser().resolve()
        _set_or_warmstart_policy(USSR_POLICY_ID, ussr_path, "ussr")
    static_benchmark_weights: dict[str, dict[str, dict[str, Any]]] = {}
    if has_split_benchmark:
        benchmark_label = metric_label(str(args.benchmark_label or "side_selected_best"))
        benchmark_us_path = Path(args.benchmark_us_policy_from).expanduser().resolve()
        benchmark_ussr_path = Path(args.benchmark_ussr_policy_from).expanduser().resolve()
        static_benchmark_weights[benchmark_label] = {
            US_POLICY_ID: load_policy_weights(benchmark_us_path, US_POLICY_ID),
            USSR_POLICY_ID: load_policy_weights(benchmark_ussr_path, USSR_POLICY_ID),
        }
        print(
            {
                "event": "split_benchmark_loaded",
                "benchmark_label": benchmark_label,
                "us_policy": str(benchmark_us_path),
                "ussr_policy": str(benchmark_ussr_path),
            },
            flush=True,
        )
        if args.policy_sharing == "unified" and unified_opponent_mix:
            benchmark_us_weights = _load_weights_for_policy(
                benchmark_us_path,
                US_POLICY_ID,
                preferred_source_policy_id=US_POLICY_ID,
            )
            benchmark_ussr_weights = _load_weights_for_policy(
                benchmark_ussr_path,
                USSR_POLICY_ID,
                preferred_source_policy_id=USSR_POLICY_ID,
            )
            _set_or_warmstart_policy_weights(US_POLICY_ID, benchmark_us_weights, benchmark_us_path, "unified_mix_teacher_us")
            _set_or_warmstart_policy_weights(USSR_POLICY_ID, benchmark_ussr_weights, benchmark_ussr_path, "unified_mix_teacher_ussr")
            print(
                {
                    "event": "unified_opponent_mix_loaded",
                    "mix": unified_opponent_mix,
                    "teacher_us_policy": str(benchmark_us_path),
                    "teacher_ussr_policy": str(benchmark_ussr_path),
                },
                flush=True,
            )
    if args.multi_agent and args.policy_sharing == "unified" and unified_anchor_paths:
        for anchor_policy_id, anchor_path in zip(unified_anchor_policy_ids, unified_anchor_paths, strict=True):
            source_weights = _weights_from_checkpoint(anchor_path)
            weights = source_weights.get(SHARED_POLICY_ID)
            if weights is None and len(source_weights) == 1:
                weights = next(iter(source_weights.values()))
            if weights is None:
                raise RuntimeError(f"cannot load unified anchor policy {anchor_policy_id} from {anchor_path}")
            _set_or_warmstart_policy_weights(anchor_policy_id, weights, anchor_path, anchor_policy_id)
        print(
            {
                "event": "unified_anchor_policies_loaded",
                "anchor_policy_count": len(unified_anchor_policy_ids),
                "anchor_policies": dict(zip(unified_anchor_policy_ids, [str(path) for path in unified_anchor_paths], strict=True)),
            },
            flush=True,
        )
    for fixed_policy_id, item in league_sources.items():
        source_path = Path(item["source"]).expanduser().resolve()
        source_policy_id = side_to_policy_id(item["side"])
        source_weights = _weights_from_checkpoint(source_path)
        weights = source_weights.get(source_policy_id)
        if weights is None and len(source_weights) == 1:
            weights = next(iter(source_weights.values()))
        if weights is None:
            raise RuntimeError(f"cannot load fixed league policy {fixed_policy_id} from {source_path}")
        policy = algo.get_policy(fixed_policy_id)
        current = policy.get_weights()
        if args.model_arch == "transformer_history":
            current, matched, widened, skipped_shape, _missing_source, _unexpected_source = copy_compatible_or_widened_weights(current, weights)
            policy.set_weights(current)
            print(
                {
                    "event": "league_fixed_partial_load",
                    "policy_id": fixed_policy_id,
                    "source": str(source_path),
                    "matched_count": len(matched),
                    "widened_count": len(widened),
                    "shape_mismatch_count": len(skipped_shape),
                },
                flush=True,
            )
        else:
            policy.set_weights(weights)
            print({"event": "league_fixed_load", "policy_id": fixed_policy_id, "source": str(source_path)}, flush=True)
    if load_reports:
        report_path = Path(args.load_report_path) if args.load_report_path else checkpoint_root / "load_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump({"model_arch": args.model_arch, "reports": load_reports}, handle, indent=2, sort_keys=True)
        print(f"load_report={report_path}")
    if initial_weights is None:
        initial_weights = _current_weights()
    if args.eval_only:
        eval_label = "eval_only"
        eval_checkpoint = str(Path(args.restore_from).expanduser().resolve()) if args.restore_from else "current_weights"
        benchmark_weights = {"initial": initial_weights, **static_benchmark_weights}
        print(
            {
                "event": "eval_only_start",
                "checkpoint": eval_checkpoint,
                "eval_games_per_opponent": args.eval_games,
                "eval_games_per_side": eval_games_per_side(args.eval_games, args.eval_min_games_per_side),
                "eval_opponents": [*benchmark_weights.keys(), "random"],
                "eval_max_episode_steps": args.eval_max_episode_steps,
                "eval_progress_every_games": args.eval_progress_every_games,
                "elo_path": str(elo_path),
            },
            flush=True,
        )
        eval_metrics = evaluate_policy(
            algo=algo,
            benchmark_weights=benchmark_weights,
            env_config=env_config,
            eval_games=args.eval_games,
            min_games_per_side=args.eval_min_games_per_side,
            max_episode_steps=args.eval_max_episode_steps,
            seed_base=30_000_000,
            side_policy_ids=side_policy_ids,
            eval_label=eval_label,
            progress_every_games=args.eval_progress_every_games,
            elo_state=elo_state,
            elo_k_factor=args.elo_k_factor,
        )
        eval_metrics["checkpoint/path"] = eval_checkpoint
        eval_metrics["checkpoint/episodes"] = 0
        eval_metrics["checkpoint/train_steps"] = 0
        score_details = evaluated_checkpoint_score(eval_metrics, args)
        eval_metrics.update({f"best_checkpoint/{key}": value for key, value in score_details.items()})
        save_elo_state(elo_path, elo_state)
        if args.best_checkpoint_manifest:
            write_best_checkpoint_manifest(
                Path(args.best_checkpoint_manifest).expanduser().resolve(),
                checkpoint=eval_checkpoint,
                eval_label=eval_label,
                episodes_seen=0,
                train_steps_seen=0,
                score_details=score_details,
                eval_metrics=eval_metrics,
            )
        print(
            {
                "event": "eval_only_done",
                "checkpoint": eval_checkpoint,
                "elo_path": str(elo_path),
                **eval_metrics,
            },
            flush=True,
        )
        if wandb_run:
            wandb_run.log({"iter": 0, **eval_metrics}, step=0)
            wandb_run.finish()
        ray.shutdown()
        return
    i = 0
    episodes_seen = 0
    train_steps_seen = 0
    last_sampled_steps: int | None = None
    metric_offsets: dict[str, int] = {}
    next_checkpoint_eval_step = args.checkpoint_eval_every_steps
    next_checkpoint_eval_episode = args.checkpoint_eval_every_episodes
    checkpoint_paths: list[str] = []
    historical_eval_weights: list[tuple[str, dict[str, dict[str, Any]]]] = []
    active_reward_shaping_scale: float | None = None
    best_manifest_path = (
        Path(args.best_checkpoint_manifest).expanduser().resolve()
        if args.best_checkpoint_manifest
        else checkpoint_root / "best_checkpoint.json"
    )
    best_score = float("-inf")
    stale_eval_count = 0
    while True:
        reward_shaping_scale = reward_shaping_scale_for_episodes(args, episodes_seen)
        if active_reward_shaping_scale is None or abs(active_reward_shaping_scale - reward_shaping_scale) > 1e-9:
            apply_reward_shaping_scale(algo, reward_shaping_scale)
            active_reward_shaping_scale = reward_shaping_scale
            print(
                {
                    "event": "reward_shaping_scale_set",
                    "episodes_seen": episodes_seen,
                    "reward_shaping_scale": reward_shaping_scale,
                },
                flush=True,
            )
        result = algo.train()
        i += 1
        episodes_this_iter = int((result.get("env_runners") or {}).get("num_episodes") or result.get("episodes_this_iter") or 0)
        episodes_seen += episodes_this_iter
        new_terminal_records = read_new_terminal_metrics(metrics_dir, metric_offsets)
        rollout_metrics = summarize_terminal_metrics(new_terminal_records, "rollout")
        sampled_steps = int(result.get("num_env_steps_sampled_lifetime") or result.get("num_env_steps_sampled") or 0)
        if last_sampled_steps is None:
            last_sampled_steps = sampled_steps
        else:
            train_steps_seen += max(0, sampled_steps - last_sampled_steps)
            last_sampled_steps = sampled_steps
        print(
            {
                "iter": i,
                "episode_reward_mean": (result.get("env_runners") or {}).get("episode_reward_mean") or result.get("episode_reward_mean"),
                "episodes_this_iter": episodes_this_iter,
                "episodes_seen": episodes_seen,
                "train_steps_seen": train_steps_seen,
                "num_env_steps_sampled_lifetime": sampled_steps,
                "reward_shaping_scale": active_reward_shaping_scale,
                **rollout_metrics,
            },
            flush=True,
        )
        eval_metrics: dict[str, float | int] = {}
        checkpoint_path: str | None = None
        step_eval_due = args.checkpoint_eval_every_steps > 0 and train_steps_seen >= next_checkpoint_eval_step
        episode_eval_due = args.checkpoint_eval_every_episodes > 0 and episodes_seen >= next_checkpoint_eval_episode
        if step_eval_due or episode_eval_due:
            checkpoint_label = f"steps-{train_steps_seen:010d}-episodes-{episodes_seen:08d}"
            checkpoint_path = save_checkpoint(algo, checkpoint_root, checkpoint_label)
            checkpoint_paths.append(checkpoint_path)
            benchmark_weights = {"initial": initial_weights, **static_benchmark_weights}
            history_limit = max(0, int(args.eval_history_opponents))
            if history_limit > 0:
                benchmark_weights.update(
                    {label: weights for label, weights in historical_eval_weights[-history_limit:]}
                )
            print(
                {
                    "event": "checkpoint_eval_start",
                    "checkpoint": checkpoint_path,
                    "train_steps_seen": train_steps_seen,
                    "episodes_seen": episodes_seen,
                    "eval_games_per_opponent": args.eval_games,
                    "eval_games_per_side": eval_games_per_side(args.eval_games, args.eval_min_games_per_side),
                    "eval_opponents": [*benchmark_weights.keys(), "random"],
                    "eval_history_opponents": min(history_limit, len(historical_eval_weights)),
                    "eval_max_episode_steps": args.eval_max_episode_steps,
                    "eval_progress_every_games": args.eval_progress_every_games,
                    "elo_path": str(elo_path),
                },
                flush=True,
            )
            eval_metrics = evaluate_policy(
                algo=algo,
                benchmark_weights=benchmark_weights,
                env_config=env_config,
                eval_games=args.eval_games,
                min_games_per_side=args.eval_min_games_per_side,
                max_episode_steps=args.eval_max_episode_steps,
                seed_base=10_000_000 + episodes_seen,
                side_policy_ids=side_policy_ids,
                eval_label=checkpoint_label,
                progress_every_games=args.eval_progress_every_games,
                elo_state=elo_state,
                elo_k_factor=args.elo_k_factor,
            )
            eval_metrics["checkpoint/train_steps"] = train_steps_seen
            eval_metrics["checkpoint/episodes"] = episodes_seen
            eval_metrics["checkpoint/path"] = checkpoint_path
            save_elo_state(elo_path, elo_state)
            score_details = evaluated_checkpoint_score(eval_metrics, args)
            eval_metrics.update({f"best_checkpoint/{key}": value for key, value in score_details.items()})
            if (
                args.adaptive_us_focus
                and args.policy_sharing == "unified"
                and unified_opponent_mix
            ):
                previous_probability = unified_train_us_probability
                if score_details["us_elo"] + float(args.us_focus_elo_lag_threshold) < score_details["ussr_elo"]:
                    unified_train_us_probability = float(args.us_focus_prob)
                else:
                    unified_train_us_probability = 0.5
                eval_metrics["adaptive_us_focus/train_us_probability"] = unified_train_us_probability
                if abs(previous_probability - unified_train_us_probability) > 1e-9:
                    print(
                        {
                            "event": "adaptive_us_focus_update",
                            "previous_train_us_probability": previous_probability,
                            "train_us_probability": unified_train_us_probability,
                            "us_elo": score_details["us_elo"],
                            "ussr_elo": score_details["ussr_elo"],
                            "threshold": args.us_focus_elo_lag_threshold,
                        },
                        flush=True,
                    )
            previous_best_score = best_score
            score_improved = score_details["score"] > previous_best_score
            score_improved_by_min_delta = score_details["score"] > previous_best_score + float(args.early_stop_min_delta)
            if score_improved:
                best_score = score_details["score"]
                if score_improved_by_min_delta:
                    stale_eval_count = 0
                else:
                    stale_eval_count += 1
                write_best_checkpoint_manifest(
                    best_manifest_path,
                    checkpoint=checkpoint_path,
                    eval_label=checkpoint_label,
                    episodes_seen=episodes_seen,
                    train_steps_seen=train_steps_seen,
                    score_details=score_details,
                    eval_metrics=eval_metrics,
                )
                print(
                    {
                        "event": "best_checkpoint_updated",
                        "best_checkpoint": checkpoint_path,
                        "best_manifest": str(best_manifest_path),
                        "previous_best_score": previous_best_score,
                        "stale_eval_count": stale_eval_count,
                        "early_stop_min_delta_met": score_improved_by_min_delta,
                        **{f"score/{key}": value for key, value in score_details.items()},
                    },
                    flush=True,
                )
            else:
                stale_eval_count += 1
                print(
                    {
                        "event": "best_checkpoint_not_improved",
                        "checkpoint": checkpoint_path,
                        "best_score": best_score,
                        "stale_eval_count": stale_eval_count,
                        **{f"score/{key}": value for key, value in score_details.items()},
                    },
                    flush=True,
                )
            if history_limit > 0:
                historical_eval_weights.append((checkpoint_label, _current_weights()))
                if len(historical_eval_weights) > history_limit:
                    historical_eval_weights = historical_eval_weights[-history_limit:]
            print(
                {
                    "event": "checkpoint_eval_done",
                    "checkpoint": checkpoint_path,
                    "train_steps_seen": train_steps_seen,
                    "episodes_seen": episodes_seen,
                    "elo_path": str(elo_path),
                    **eval_metrics,
                },
                flush=True,
            )
            while args.checkpoint_eval_every_steps > 0 and train_steps_seen >= next_checkpoint_eval_step:
                next_checkpoint_eval_step += args.checkpoint_eval_every_steps
            while args.checkpoint_eval_every_episodes > 0 and episodes_seen >= next_checkpoint_eval_episode:
                next_checkpoint_eval_episode += args.checkpoint_eval_every_episodes
            if args.early_stop_patience_evals > 0 and stale_eval_count >= args.early_stop_patience_evals:
                print(
                    {
                        "event": "early_stop_triggered",
                        "stale_eval_count": stale_eval_count,
                        "patience": args.early_stop_patience_evals,
                        "best_score": best_score,
                        "best_manifest": str(best_manifest_path),
                    },
                    flush=True,
                )
                break
        if wandb_run:
            wandb_payload = {
                "iter": i,
                "episodes_seen": episodes_seen,
                "train_steps_seen": train_steps_seen,
                "sampled_steps": sampled_steps,
                "reward_shaping_scale": active_reward_shaping_scale,
                **flatten_scalars(result),
                **rollout_metrics,
                **eval_metrics,
            }
            wandb_run.log(wandb_payload, step=i)
            if checkpoint_path:
                wandb_run.summary["latest_checkpoint"] = checkpoint_path
        if args.stop_episodes is not None and episodes_seen >= args.stop_episodes:
            break
        if args.stop_timesteps is not None and sampled_steps >= args.stop_timesteps:
            break
        if args.stop_timesteps is None and args.stop_episodes is None and i >= args.stop_iters:
            break
    checkpoint_path = save_checkpoint(algo, checkpoint_root, "final")
    checkpoint_paths.append(checkpoint_path)
    final_benchmark_weights = {"initial": initial_weights, **static_benchmark_weights}
    history_limit = max(0, int(args.eval_history_opponents))
    if history_limit > 0:
        final_benchmark_weights.update({label: weights for label, weights in historical_eval_weights[-history_limit:]})
    print(
        {
            "event": "final_eval_start",
            "checkpoint": checkpoint_path,
            "episodes_seen": episodes_seen,
            "eval_games_per_opponent": args.eval_games,
            "eval_games_per_side": eval_games_per_side(args.eval_games, args.eval_min_games_per_side),
            "eval_opponents": [*final_benchmark_weights.keys(), "random"],
            "eval_history_opponents": min(history_limit, len(historical_eval_weights)),
            "eval_max_episode_steps": args.eval_max_episode_steps,
            "eval_progress_every_games": args.eval_progress_every_games,
            "elo_path": str(elo_path),
        },
        flush=True,
    )
    final_eval_metrics = evaluate_policy(
        algo=algo,
        benchmark_weights=final_benchmark_weights,
        env_config=env_config,
        eval_games=args.eval_games,
        min_games_per_side=args.eval_min_games_per_side,
        max_episode_steps=args.eval_max_episode_steps,
        seed_base=20_000_000 + episodes_seen,
        side_policy_ids=side_policy_ids,
        eval_label="final",
        progress_every_games=args.eval_progress_every_games,
        elo_state=elo_state,
        elo_k_factor=args.elo_k_factor,
    )
    save_elo_state(elo_path, elo_state)
    final_score_details = evaluated_checkpoint_score(final_eval_metrics, args)
    final_eval_metrics.update({f"best_checkpoint/{key}": value for key, value in final_score_details.items()})
    if final_score_details["score"] > best_score:
        previous_best_score = best_score
        best_score = final_score_details["score"]
        write_best_checkpoint_manifest(
            best_manifest_path,
            checkpoint=checkpoint_path,
            eval_label="final",
            episodes_seen=episodes_seen,
            train_steps_seen=train_steps_seen,
            score_details=final_score_details,
            eval_metrics=final_eval_metrics,
        )
        print(
            {
                "event": "best_checkpoint_updated",
                "best_checkpoint": checkpoint_path,
                "best_manifest": str(best_manifest_path),
                "previous_best_score": previous_best_score,
                **{f"score/{key}": value for key, value in final_score_details.items()},
            },
            flush=True,
        )
    print({"event": "final_eval_done", "checkpoint": checkpoint_path, "elo_path": str(elo_path), **final_eval_metrics}, flush=True)
    print(f"checkpoint={checkpoint_path}")
    if wandb_run:
        wandb_run.summary["checkpoint"] = str(checkpoint_path)
        wandb_run.summary["checkpoints"] = checkpoint_paths
        wandb_run.summary["best_checkpoint_manifest"] = str(best_manifest_path)
        wandb_run.summary["best_checkpoint_score"] = best_score
        if final_eval_metrics:
            wandb_run.log({"iter": i, "episodes_seen": episodes_seen, **{f"final/{k}": v for k, v in final_eval_metrics.items()}}, step=i + 1)
        wandb_run.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
