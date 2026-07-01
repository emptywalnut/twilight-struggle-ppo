from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv, TwilightStruggleMultiAgentEnv
from struggle_ai.train_rllib import (
    NUKE_TERMINAL_REASONS,
    US_POLICY_ID,
    USSR_POLICY_ID,
    compute_algo_action,
    load_policy_weights,
    random_legal_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Twilight Struggle policy evaluations.")
    parser.add_argument("--us-policy", type=Path, required=True)
    parser.add_argument("--ussr-policy", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=30_000_000)
    parser.add_argument("--opponent", choices=["mixed", "random", "heuristic"], default="mixed")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--defcon-suicide-mode", choices=["none", "hard_filter", "penalty_only"], default="none")
    parser.add_argument("--terminal-reward-scale", type=float, default=2.0)
    parser.add_argument("--nuke-death-penalty", type=float, default=1.0)
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
    return parser.parse_args()


def build_algo(args: argparse.Namespace) -> Any:
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc
    from struggle_ai.rllib_masked_model import register_masked_model

    register_masked_model()
    register_env("twilight_eval_probe", lambda cfg: TwilightStruggleMultiAgentEnv(cfg))
    probe = TwilightStruggleEnv({"log_games_dir": None, "metrics_dir": None})
    policy_spec = (None, probe.observation_space, probe.action_space, {})
    probe.close()
    ray.init(ignore_reinit_error=True)
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment("twilight_eval_probe", env_config={})
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(
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
            }
        )
        .multi_agent(
            policies={US_POLICY_ID: policy_spec, USSR_POLICY_ID: policy_spec},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: US_POLICY_ID if agent_id == "us" else USSR_POLICY_ID,
        )
    )
    algo = config.build()
    algo.get_policy(US_POLICY_ID).set_weights(load_policy_weights(args.us_policy.expanduser().resolve(), US_POLICY_ID))
    algo.get_policy(USSR_POLICY_ID).set_weights(load_policy_weights(args.ussr_policy.expanduser().resolve(), USSR_POLICY_ID))
    return algo


def play_game(algo: Any, args: argparse.Namespace, seed: int, rng: np.random.Generator) -> dict[str, Any]:
    log_dir = args.out_dir / "game_logs"
    env = TwilightStruggleEnv(
        {
            "seed": seed,
            "log_games_dir": str(log_dir),
            "log_games_every": 1,
            "metrics_dir": str(args.out_dir / "metrics"),
            "log_action_details": True,
            "max_episode_steps": args.max_episode_steps,
            "defcon_suicide_mode": args.defcon_suicide_mode,
            "terminal_reward_scale": args.terminal_reward_scale,
            "nuke_death_penalty": args.nuke_death_penalty,
        }
    )
    try:
        obs, _info = env.reset(seed=seed)
        final_info: dict[str, Any] = {}
        terminated = truncated = False
        while not (terminated or truncated):
            side = str((env.last_obs or {}).get("side") or "ussr")
            if args.opponent == "random" and side == "ussr":
                action = random_legal_action(obs, rng)
            elif args.opponent == "heuristic" and side == "ussr":
                action = env.heuristic_action_index()
            else:
                action = compute_algo_action(algo, obs, US_POLICY_ID if side == "us" else USSR_POLICY_ID)
            obs, _reward, terminated, truncated, final_info = env.step(action)
        return {
            "seed": seed,
            "winner": final_info.get("winner"),
            "terminal_reason": final_info.get("terminal_reason"),
            "steps": env.episode_step,
            "defcon_suicide_mode": args.defcon_suicide_mode,
            "filtered_actions": len(env.episode_filtered_actions),
        }
    finally:
        env.close()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {"games": 0}
    return {
        "games": count,
        "us_wins": sum(1 for record in records if record.get("winner") == "us"),
        "ussr_wins": sum(1 for record in records if record.get("winner") == "ussr"),
        "ties": sum(1 for record in records if record.get("winner") == "tie"),
        "timeouts": sum(1 for record in records if record.get("terminal_reason") in {"max_episode_steps", "eval_max_episode_steps"}),
        "nuke_terminals": sum(1 for record in records if record.get("terminal_reason") in NUKE_TERMINAL_REASONS),
        "nuke_terminal_rate": sum(1 for record in records if record.get("terminal_reason") in NUKE_TERMINAL_REASONS) / count,
        "avg_steps": float(np.mean([float(record.get("steps") or 0.0) for record in records])),
        "filtered_actions": sum(int(record.get("filtered_actions") or 0) for record in records),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    algo = build_algo(args)
    rng = np.random.default_rng(args.seed_base)
    records = [play_game(algo, args, args.seed_base + idx, rng) for idx in range(args.games)]
    payload = {"args": vars(args) | {"us_policy": str(args.us_policy), "ussr_policy": str(args.ussr_policy), "out_dir": str(args.out_dir)}, "summary": summarize(records), "games": records}
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with (args.out_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload["summary"], sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
