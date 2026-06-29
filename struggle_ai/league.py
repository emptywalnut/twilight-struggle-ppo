from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SELECTED = Path("runs/ppo/selected_policies/best_us57005_ussr66001/policies")
DEFAULT_REWARD_FLAGS = {
    "terminal-reward-scale": 2.0,
    "reward-shaping-scale": 1.0,
    "reward-shaping-final-scale": 0.9,
    "reward-shaping-phaseout-start-episodes": 100,
    "reward-shaping-phaseout-end-episodes": 300,
    "turn-vp-reward-scale": 0.01,
    "nuke-death-penalty": 1.0,
    "high-stability-coup-penalty": 0.03,
    "low-stability-warzone-coup-reward": 0.03,
    "headline-opponent-event-penalty": 0.03,
    "defcon-risk-pick-penalty": 0.05,
    "defcon-risk-commit-penalty": 0.15,
    "empty-country-influence-reward": 0.01,
    "control-battleground-reward": 0.02,
    "control-non-battleground-reward": 0.005,
    "max-episode-step-penalty": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manifest-driven league launcher for Twilight Struggle PPO.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create a league directory and seed manifest.")
    init.add_argument("--league-dir", type=Path, required=True)
    init.add_argument("--us-policy", type=Path, default=DEFAULT_SELECTED / "us_policy")
    init.add_argument("--ussr-policy", type=Path, default=DEFAULT_SELECTED / "ussr_policy")
    init.add_argument("--max-history-per-side", type=int, default=20)

    launch = sub.add_parser("launch", help="Run or print one controlled league training job.")
    launch.add_argument("--league-dir", type=Path, required=True)
    launch.add_argument("--role", choices=["main", "exploiter_us", "exploiter_ussr"], required=True)
    launch.add_argument("--branch", choices=["hard_filter", "penalty_only"], default="hard_filter")
    launch.add_argument("--model-arch", choices=["feedforward", "transformer_history"], default="feedforward")
    launch.add_argument("--hidden", type=int, default=256)
    launch.add_argument("--history-layers", type=int, default=2)
    launch.add_argument("--history-attention-heads", type=int, default=4)
    launch.add_argument("--history-dropout", type=float, default=0.05)
    launch.add_argument("--graph-layers", type=int, default=2)
    launch.add_argument("--graph-neighbor-hops", type=int, default=5)
    launch.add_argument("--heuristic-prior-scale", type=float, default=2.0)
    launch.add_argument("--setup-heuristic-prior-scale", type=float, default=0.0)
    launch.add_argument("--policy-temperature", type=float, default=1.0)
    launch.add_argument("--entropy-coeff", type=float, default=0.0)
    launch.add_argument("--train-batch-size", type=int, default=1000)
    launch.add_argument("--minibatch-size", type=int, default=256)
    launch.add_argument("--rollout-fragment-length", default="256")
    launch.add_argument("--batch-mode", choices=["complete_episodes", "truncate_episodes"], default="complete_episodes")
    launch.add_argument("--sample-timeout-s", type=float, default=300.0)
    launch.add_argument("--max-episode-steps", type=int, default=1200)
    launch.add_argument("--log-games-every", type=int, default=25)
    launch.add_argument("--checkpoint-eval-every-episodes", type=int, default=None)
    launch.add_argument("--train-side", choices=["both", "us", "ussr"], default="both")
    launch.add_argument("--episodes", type=int, default=1000)
    launch.add_argument("--eval-games", type=int, default=50)
    launch.add_argument("--eval-min-games-per-side", type=int, default=10)
    launch.add_argument("--eval-history-opponents", type=int, default=3)
    launch.add_argument("--eval-max-episode-steps", type=int, default=1200)
    launch.add_argument("--gpu", type=float, default=1.0)
    launch.add_argument("--cuda-visible-devices", default="2")
    launch.add_argument("--python", default=sys.executable)
    launch.add_argument("--num-env-runners", type=int, default=0)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--extra", action="append", default=[], help="Additional raw train_rllib.py flags.")

    promote = sub.add_parser("promote", help="Record a promoted snapshot in best/history pools.")
    promote.add_argument("--league-dir", type=Path, required=True)
    promote.add_argument("--side", choices=["us", "ussr"], required=True)
    promote.add_argument("--checkpoint", type=Path, required=True)
    promote.add_argument("--win-rate", type=float, required=True)
    promote.add_argument("--nuke-rate", type=float, required=True)
    promote.add_argument("--label", default=None)

    payoff = sub.add_parser("record-payoff", help="Append one payoff-matrix row.")
    payoff.add_argument("--league-dir", type=Path, required=True)
    payoff.add_argument("--policy-a", required=True)
    payoff.add_argument("--policy-b", required=True)
    payoff.add_argument("--side-a", choices=["us", "ussr"], required=True)
    payoff.add_argument("--seed-start", type=int, required=True)
    payoff.add_argument("--seed-end", type=int, required=True)
    payoff.add_argument("--win-rate", type=float, required=True)
    payoff.add_argument("--nuke-rate", type=float, required=True)
    payoff.add_argument("--avg-steps", type=float, required=True)

    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def manifest_path(league_dir: Path) -> Path:
    return league_dir / "league_manifest.json"


def init_league(args: argparse.Namespace) -> None:
    league_dir = args.league_dir
    for rel in [
        "main_snapshot_pool/us",
        "main_snapshot_pool/ussr",
        "best_pool/us",
        "best_pool/ussr",
        "history_pool/us",
        "history_pool/ussr",
        "exploiters/us",
        "exploiters/ussr",
        "runs",
        "archive",
    ]:
        (league_dir / rel).mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.time(),
        "seed_policies": {"us": str(args.us_policy), "ussr": str(args.ussr_policy)},
        "roles": ["main_us", "main_ussr", "exploiter_us", "exploiter_ussr"],
        "opponent_sampling": {
            "current_main": 0.40,
            "best_historical": 0.20,
            "random_history": 0.20,
            "exploiter": 0.10,
            "random_legal": 0.05,
            "heuristic": 0.05,
        },
        "promotion": {
            "min_win_rate_delta": 0.05,
            "same_win_rate_nuke_rate_delta": 0.10,
            "diagnostic_eval_games": 50,
            "promotion_eval_games": 100,
        },
        "max_history_per_side": args.max_history_per_side,
        "snapshots": {"us": [], "ussr": []},
        "best_pool": {"us": [], "ussr": []},
        "notes": [
            "PSRO-lite is intentionally disabled until at least 8 viable frozen policies per side exist.",
            "Use launch --role main/exploiter_* for main-plus-exploiters diagnostics before long runs.",
        ],
    }
    write_json(manifest_path(league_dir), manifest)
    (league_dir / "payoff_matrix.jsonl").touch()
    print(f"league_manifest={manifest_path(league_dir)}")


def train_command(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    run_dir = args.league_dir / "runs" / f"{args.role}-{args.branch}-{int(time.time())}"
    seed = manifest["seed_policies"]
    cmd = [
        args.python,
        "-m",
        "struggle_ai.train_rllib",
        "--multi-agent",
        "--league-manifest",
        str(args.league_dir / "league_manifest.json"),
        "--league-role",
        args.role,
        "--league-train-side",
        args.train_side,
        "--checkpoint-dir",
        str(run_dir),
        "--metrics-dir",
        str(run_dir / "metrics"),
        "--log-games-dir",
        str(run_dir / "game_logs"),
        "--log-games-every",
        str(args.log_games_every),
        "--load-us-policy-from",
        seed["us"],
        "--load-ussr-policy-from",
        seed["ussr"],
        "--stop-episodes",
        str(args.episodes),
        "--checkpoint-eval-every-episodes",
        str(args.checkpoint_eval_every_episodes or (2000 if args.episodes <= 10_000 else 5000)),
        "--checkpoint-eval-every-steps",
        "0",
        "--eval-games",
        str(args.eval_games),
        "--eval-min-games-per-side",
        str(args.eval_min_games_per_side),
        "--eval-history-opponents",
        str(args.eval_history_opponents),
        "--eval-max-episode-steps",
        str(args.eval_max_episode_steps),
        "--num-gpus",
        str(args.gpu),
        "--num-env-runners",
        str(args.num_env_runners),
        "--train-batch-size",
        str(args.train_batch_size),
        "--minibatch-size",
        str(args.minibatch_size),
        "--rollout-fragment-length",
        str(args.rollout_fragment_length),
        "--batch-mode",
        args.batch_mode,
        "--sample-timeout-s",
        str(args.sample_timeout_s),
        "--defcon-suicide-mode",
        args.branch,
        "--model-arch",
        args.model_arch,
        "--hidden",
        str(args.hidden),
        "--history-layers",
        str(args.history_layers),
        "--history-attention-heads",
        str(args.history_attention_heads),
        "--history-dropout",
        str(args.history_dropout),
        "--graph-layers",
        str(args.graph_layers),
        "--graph-neighbor-hops",
        str(args.graph_neighbor_hops),
        "--heuristic-prior-scale",
        str(args.heuristic_prior_scale),
        "--setup-heuristic-prior-scale",
        str(args.setup_heuristic_prior_scale),
        "--policy-temperature",
        str(args.policy_temperature),
        "--entropy-coeff",
        str(args.entropy_coeff),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--force-setup-heuristic",
    ]
    if args.model_arch == "transformer_history":
        cmd.append("--partial-warmstart")
    if args.role == "exploiter_us":
        cmd.extend(["--policies-to-train", "us"])
    elif args.role == "exploiter_ussr":
        cmd.extend(["--policies-to-train", "ussr"])
    for key, value in DEFAULT_REWARD_FLAGS.items():
        cmd.extend([f"--{key}", str(value)])
    for extra in args.extra:
        cmd.extend(shlex.split(extra))
    return cmd


def launch(args: argparse.Namespace) -> None:
    manifest = read_json(manifest_path(args.league_dir))
    cmd = train_command(args, manifest)
    command_text = " ".join(shlex.quote(part) for part in cmd)
    env_prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(args.cuda_visible_devices))}"
    launch_dir = args.league_dir / "launches"
    launch_dir.mkdir(parents=True, exist_ok=True)
    launch_path = launch_dir / f"{args.role}-{args.branch}-{int(time.time())}.sh"
    launch_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(args.cuda_visible_devices))}\n"
        + command_text
        + "\n",
        encoding="utf-8",
    )
    launch_path.chmod(0o755)
    print(f"launch_script={launch_path}")
    print(f"{env_prefix} {command_text}")
    if not args.dry_run:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        subprocess.run(cmd, check=True, env=env)


def promote(args: argparse.Namespace) -> None:
    manifest = read_json(manifest_path(args.league_dir))
    side = args.side
    label = args.label or args.checkpoint.name
    entry = {
        "label": label,
        "checkpoint": str(args.checkpoint),
        "win_rate": args.win_rate,
        "nuke_rate": args.nuke_rate,
        "promoted_at": time.time(),
    }
    manifest["snapshots"].setdefault(side, []).append(entry)
    manifest["best_pool"].setdefault(side, []).append(entry)
    max_history = int(manifest.get("max_history_per_side") or 20)
    if len(manifest["snapshots"][side]) > max_history:
        manifest["snapshots"][side] = manifest["snapshots"][side][-max_history:]
    write_json(manifest_path(args.league_dir), manifest)
    print(json.dumps(entry, sort_keys=True))


def record_payoff(args: argparse.Namespace) -> None:
    row = {
        "time": time.time(),
        "policy_a": args.policy_a,
        "policy_b": args.policy_b,
        "side_a": args.side_a,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "win_rate": args.win_rate,
        "nuke_rate": args.nuke_rate,
        "avg_steps": args.avg_steps,
    }
    path = args.league_dir / "payoff_matrix.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.cmd == "init":
        init_league(args)
    elif args.cmd == "launch":
        launch(args)
    elif args.cmd == "promote":
        promote(args)
    elif args.cmd == "record-payoff":
        record_payoff(args)


if __name__ == "__main__":
    main()
