#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv, TwilightStruggleMultiAgentEnv
from struggle_ai.rllib_masked_model import register_masked_model
from struggle_ai.train_rllib import (
    US_POLICY_ID,
    USSR_POLICY_ID,
    compute_algo_action,
    copy_compatible_or_widened_weights,
    load_policy_weights,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate direct warmup samples by rolling out teacher policies. "
            "Use this to distill split US/USSR checkpoints into later BC/RL runs."
        )
    )
    parser.add_argument("--us-policy-from", type=Path, required=True)
    parser.add_argument("--ussr-policy-from", type=Path, required=True)
    parser.add_argument("--us-source-policy-id", default=US_POLICY_ID)
    parser.add_argument("--ussr-source-policy-id", default=USSR_POLICY_ID)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--reject-terminal-reason",
        action="append",
        default=[],
        help="Terminal reason to discard from the distillation dataset; may be repeated.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum rollout attempts. Defaults to 10x --episodes when rejection is enabled, else --episodes.",
    )
    parser.add_argument("--seed", type=int, default=9100000000)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
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
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--checkpoint-algo-dir", type=Path, default=None)
    return parser.parse_args()


def compact_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: action.get(key)
        for key in ("type", "decision", "value", "label", "selector", "event", "card", "country", "region", "prompt")
        if key in action
    }


def build_teacher_algo(args: argparse.Namespace, env: TwilightStruggleEnv):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    register_masked_model()
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    policy_spec = (None, env.observation_space, env.action_space, {})
    env_name = "twilight_struggle_distill_probe"
    register_env(env_name, lambda cfg: TwilightStruggleMultiAgentEnv(cfg))
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env_name, env_config={"log_games_dir": None, "metrics_dir": None})
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
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
            policies_to_train=[],
        )
    )
    algo = config.build()
    load_teacher_policy(algo, US_POLICY_ID, args.us_policy_from.expanduser().resolve(), args.us_source_policy_id)
    load_teacher_policy(algo, USSR_POLICY_ID, args.ussr_policy_from.expanduser().resolve(), args.ussr_source_policy_id)
    return algo


def load_teacher_policy(algo: Any, target_policy_id: str, checkpoint_path: Path, source_policy_id: str) -> None:
    policy = algo.get_policy(target_policy_id)
    source = load_policy_weights(checkpoint_path, source_policy_id)
    current, matched, widened, skipped_shape, missing_source, unexpected_source = copy_compatible_or_widened_weights(
        policy.get_weights(),
        source,
    )
    policy.set_weights(current)
    print(
        {
            "event": "teacher_policy_loaded",
            "target_policy_id": target_policy_id,
            "source_policy_id": source_policy_id,
            "checkpoint": str(checkpoint_path),
            "matched_count": len(matched),
            "widened_count": len(widened),
            "shape_mismatch_count": len(skipped_shape),
            "missing_source_count": len(missing_source),
            "unexpected_source_count": len(unexpected_source),
            "widened_keys": widened,
        },
        flush=True,
    )


def prompt_class(obs: dict[str, Any]) -> str:
    phase = str(obs.get("phase") or "").lower()
    prompt = str(obs.get("prompt") or "").lower()
    if phase.startswith("setup"):
        return "setup"
    if "headline" in phase or "headline" in prompt:
        return "headline"
    return "decision"


def run_episode(env: TwilightStruggleEnv, algo: Any, seed: int, episode_index: int, args: argparse.Namespace) -> dict[str, Any]:
    encoded_obs, _info = env.reset(seed=seed)
    samples: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    while not terminated and not truncated and len(samples) < args.max_episode_steps:
        raw_obs = env.last_obs or {}
        side = str(raw_obs.get("side") or "").lower()
        policy_id = US_POLICY_ID if side == "us" else USSR_POLICY_ID
        legal_actions = list(env.legal_actions)
        action_index = compute_algo_action(algo, encoded_obs, policy_id)
        if action_index < 0 or action_index >= len(legal_actions):
            legal = np.flatnonzero(np.asarray(encoded_obs["action_mask"]) > 0.5)
            action_index = int(legal[0]) if len(legal) else 0
        selected = legal_actions[action_index]
        before_log = env.bridge.log()
        next_obs, reward, terminated, truncated, final_info = env.step(action_index)
        after_obs = env.last_obs or {}
        after_log = env.bridge.log()
        before_lines = list((before_log or {}).get("log") or [])
        after_lines = list((after_log or {}).get("log") or [])
        samples.append(
            {
                "step": len(samples),
                "turn": raw_obs.get("turn"),
                "action_round": raw_obs.get("action_round"),
                "side": side,
                "phase": raw_obs.get("phase"),
                "prompt_class": prompt_class(raw_obs),
                "observation": {key: value for key, value in raw_obs.items() if key != "legal_actions"},
                "legal_actions": legal_actions,
                "expert_action_index": int(action_index),
                "choice": compact_action(selected),
                "teacher": {
                    "policy_id": policy_id,
                    "checkpoint": str(args.us_policy_from if policy_id == US_POLICY_ID else args.ussr_policy_from),
                },
                "result_after": {
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "winner": final_info.get("winner"),
                    "terminal_reason": final_info.get("terminal_reason"),
                    "before": env.action_context(raw_obs),
                    "after": env.action_context(after_obs),
                    "state_delta": env.state_delta(raw_obs, after_obs),
                    "saito_log_delta": [str(item) for item in after_lines[len(before_lines) :]],
                },
            }
        )
        encoded_obs = next_obs
    if not terminated and not truncated and len(samples) >= args.max_episode_steps:
        truncated = True
        final_info = {"winner": "timeout", "terminal_reason": "max_episode_steps"}
    last_after = samples[-1]["result_after"]["after"] if samples else {}
    return {
        "format": "ts_warmup_game_v1",
        "game_id": f"distill-teacher-episode-{episode_index:06d}",
        "ruleset": "optional_us_plus_2",
        "partial": bool(truncated),
        "source": {
            "kind": "policy_distillation_rollout",
            "us_policy_from": str(args.us_policy_from),
            "ussr_policy_from": str(args.ussr_policy_from),
        },
        "start": {"seed": None},
        "samples": samples,
        "result": {
            "winner": final_info.get("winner"),
            "terminal_reason": final_info.get("terminal_reason"),
            "vp": final_info.get("vp", last_after.get("vp")),
            "defcon": final_info.get("defcon", last_after.get("defcon")),
            "turn": last_after.get("turn"),
            "action_round": last_after.get("action_round"),
            "steps": len(samples),
        },
        "quality": {
            "status": "partial" if truncated else "ok",
            "notes": ["Generated from split-policy teacher rollouts for direct warmup BC."],
        },
    }


def main() -> None:
    args = parse_args()
    reject_terminal_reasons = {str(item).lower() for item in (args.reject_terminal_reason or [])}
    max_attempts = args.max_attempts or (args.episodes * 10 if reject_terminal_reasons else args.episodes)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    env = TwilightStruggleEnv(
        {
            "force_setup_heuristic": False,
            "max_episode_steps": args.max_episode_steps,
            "log_games_dir": None,
            "metrics_dir": None,
            "log_action_details": False,
        }
    )
    algo = build_teacher_algo(args, env)
    records_written = 0
    attempts = 0
    samples_written = 0
    result_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    try:
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            while records_written < args.episodes and attempts < max_attempts:
                attempts += 1
                record = run_episode(env, algo, args.seed + attempts, attempts, args)
                reason = str((record.get("result") or {}).get("terminal_reason") or "unknown")
                if reason.lower() in reject_terminal_reasons:
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                    if attempts % 10 == 0:
                        print(
                            {
                                "event": "distill_progress",
                                "attempts": attempts,
                                "accepted_episodes": records_written,
                                "samples": samples_written,
                                "terminal_reasons": result_counts,
                                "rejected_terminal_reasons": rejected_counts,
                            },
                            flush=True,
                        )
                    continue
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                records_written += 1
                samples_written += len(record.get("samples") or [])
                result_counts[reason] = result_counts.get(reason, 0) + 1
                if records_written % 10 == 0 or records_written == args.episodes:
                    print(
                        {
                            "event": "distill_progress",
                            "attempts": attempts,
                            "accepted_episodes": records_written,
                            "samples": samples_written,
                            "terminal_reasons": result_counts,
                            "rejected_terminal_reasons": rejected_counts,
                        },
                        flush=True,
                    )
        if records_written < args.episodes:
            raise RuntimeError(
                f"only accepted {records_written} / {args.episodes} requested records after {attempts} attempts; "
                f"rejected={rejected_counts}"
            )
        manifest = {
            "format": "ts_warmup_manifest_v1",
            "game_files": [args.output_jsonl.name],
            "records": records_written,
            "attempts": attempts,
            "samples": samples_written,
            "terminal_reasons": result_counts,
            "rejected_terminal_reasons": rejected_counts,
            "source": {
                "us_policy_from": str(args.us_policy_from),
                "ussr_policy_from": str(args.ussr_policy_from),
            },
        }
        output_manifest = args.output_manifest or (args.output_jsonl.parent / "manifest.json")
        output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.checkpoint_algo_dir:
            checkpoint_path = save_checkpoint(algo, args.checkpoint_algo_dir, "teacher-loaded")
            manifest["teacher_loaded_checkpoint"] = checkpoint_path
            output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output_jsonl": str(args.output_jsonl), "manifest": str(output_manifest), **manifest}, indent=2), flush=True)
    finally:
        env.close()
        algo.stop()


if __name__ == "__main__":
    main()
