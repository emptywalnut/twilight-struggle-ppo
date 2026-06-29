from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from struggle_ai.env import TwilightStruggleEnv, TwilightStruggleMultiAgentEnv
from struggle_ai.features import encode_observation
from struggle_ai.rllib_masked_model import register_masked_model
from struggle_ai.train_rllib import (
    US_POLICY_ID,
    USSR_POLICY_ID,
    checkpoint_policy_ids,
    copy_compatible_or_widened_weights,
    load_policy_weights,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Behavior-clone expert Twilight Struggle decisions from warmup JSONL. "
            "Records may contain direct observation/legal-action samples; seed-based "
            "bridge replay is only a legacy fallback."
        )
    )
    parser.add_argument("--input", action="append", required=True, help="Warmup JSONL file, manifest.json, or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/warmup_bc"))
    parser.add_argument("--restore-from", type=Path, default=None, help="Optional RLlib checkpoint to initialize from.")
    parser.add_argument("--load-us-policy-from", type=Path, default=None)
    parser.add_argument("--load-ussr-policy-from", type=Path, default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policies-to-train", default="us,ussr", help="Comma list: us,ussr.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--model-arch", choices=["feedforward", "transformer_history"], default="feedforward")
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--history-attention-heads", type=int, default=4)
    parser.add_argument("--history-dropout", type=float, default=0.05)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbor-hops", type=int, default=5)
    parser.add_argument("--heuristic-prior-scale", type=float, default=2.0)
    parser.add_argument("--setup-heuristic-prior-scale", type=float, default=0.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--partial-warmstart", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--setup-weight", type=float, default=0.5)
    parser.add_argument("--headline-weight", type=float, default=1.25)
    parser.add_argument("--default-weight", type=float, default=1.0)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--strict-result", action="store_true", help="Require replayed terminal result to match supplied result.")
    parser.add_argument("--allow-partial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Only collect/report matched samples; do not train.")
    return parser.parse_args()


@dataclass
class WarmupSample:
    obs: dict[str, np.ndarray]
    target: int
    side: str
    policy_id: str
    weight: float
    game_id: str
    step: int


def policy_id_for_side(side: str, multi_agent: bool) -> str:
    if not multi_agent:
        return "default_policy"
    return US_POLICY_ID if side == "us" else USSR_POLICY_ID


def parse_policies(raw: str, multi_agent: bool) -> set[str]:
    if not multi_agent:
        return {"default_policy"}
    aliases = {
        "us": US_POLICY_ID,
        "ussr": USSR_POLICY_ID,
        "us_policy": US_POLICY_ID,
        "ussr_policy": USSR_POLICY_ID,
    }
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in items if item not in aliases]
    if unknown:
        raise ValueError(f"unknown --policies-to-train entries: {unknown}")
    out = {aliases[item] for item in items}
    return out or {US_POLICY_ID, USSR_POLICY_ID}


def iter_game_file(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for record in payload:
                if isinstance(record, dict):
                    yield record
            return
        if isinstance(payload, dict):
            yield payload
            return
        raise ValueError(f"{path}: expected JSON object or array")

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def discover_game_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            manifest = path / "manifest.json"
            if manifest.exists():
                files.extend(discover_game_files([str(manifest)]))
            else:
                files.extend(sorted(path.glob("*.jsonl")))
                files.extend(sorted(path.glob("*.json")))
            continue
        if path.name == "manifest.json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for rel in manifest.get("game_files", []):
                files.append((path.parent / rel).resolve())
            continue
        files.append(path)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing warmup input files: {missing}")
    return files


def iter_game_records(inputs: list[str], max_games: int | None = None) -> Iterable[dict[str, Any]]:
    seen = 0
    for file in discover_game_files(inputs):
        for record in iter_game_file(file):
            yield record
            seen += 1
            if max_games is not None and seen >= max_games:
                return


def normalize_action_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if "choice" in entry:
        return entry
    if "action" in entry:
        before = entry.get("before") or {}
        return {
            "step": entry.get("step"),
            "turn": entry.get("turn") or before.get("turn"),
            "action_round": entry.get("action_round") or before.get("action_round"),
            "side": entry.get("side") or before.get("side"),
            "phase": before.get("phase"),
            "choice": entry.get("action") or {},
            "weight": entry.get("weight"),
        }
    return entry


def direct_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    entries = record.get("samples")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    entries = record.get("actions")
    if isinstance(entries, list) and any(
        isinstance(entry, dict)
        and (
            ("observation" in entry and ("legal_actions" in entry or "legal_actions" in entry.get("observation", {})))
            or ("before_observation" in entry and ("legal_actions" in entry or "legal_actions" in entry.get("before_observation", {})))
        )
        for entry in entries
    ):
        return [entry for entry in entries if isinstance(entry, dict)]
    return []


def selected_choice(entry: dict[str, Any]) -> dict[str, Any]:
    for key in ("choice", "action", "selected_action", "expert_action"):
        value = entry.get(key)
        if isinstance(value, dict):
            return value
    return {}


def observation_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("observation", "before_observation", "obs", "state", "before_state"):
        value = entry.get(key)
        if isinstance(value, dict):
            return value
    before = entry.get("before")
    if isinstance(before, dict) and "countries" in before:
        return before
    return None


def legal_actions_from_entry(entry: dict[str, Any], obs: dict[str, Any]) -> list[dict[str, Any]]:
    value = entry.get("legal_actions")
    if isinstance(value, list):
        return [action for action in value if isinstance(action, dict)]
    value = obs.get("legal_actions")
    if isinstance(value, list):
        return [action for action in value if isinstance(action, dict)]
    value = entry.get("legal")
    if isinstance(value, list):
        return [action for action in value if isinstance(action, dict)]
    return []


def action_match_score(choice: dict[str, Any], legal_action: dict[str, Any]) -> int:
    score = 0
    for key in ("type", "decision", "value"):
        if str(choice.get(key, "")) and str(choice.get(key, "")) == str(legal_action.get(key, "")):
            score += 4
        elif str(choice.get(key, "")):
            return -1
    for key in ("card", "country", "region", "selector", "event"):
        if str(choice.get(key, "")) and str(choice.get(key, "")) == str(legal_action.get(key, "")):
            score += 1
        elif str(choice.get(key, "")) and key in legal_action:
            return -1
    return score


def find_action_index(choice: dict[str, Any], legal_actions: list[dict[str, Any]]) -> int | None:
    if choice.get("type") == "setup":
        choice = {
            **choice,
            "type": "saito_dom",
            "decision": "country_click",
        }

    scored = [(idx, action_match_score(choice, action)) for idx, action in enumerate(legal_actions)]
    scored = [(idx, score) for idx, score in scored if score >= 0]
    if not scored:
        value = str(choice.get("value", ""))
        value_matches = [idx for idx, action in enumerate(legal_actions) if str(action.get("value", "")) == value]
        return value_matches[0] if len(value_matches) == 1 else None

    best_score = max(score for _idx, score in scored)
    best = [idx for idx, score in scored if score == best_score]
    return best[0] if len(best) == 1 else None


def sample_weight(entry: dict[str, Any], args: argparse.Namespace) -> float:
    if isinstance(entry.get("weight"), (int, float)):
        return float(entry["weight"])
    phase = str(entry.get("phase") or "").lower()
    prompt_class = str(entry.get("prompt_class") or "").lower()
    if phase.startswith("setup") or prompt_class == "setup":
        return float(args.setup_weight)
    if "headline" in phase or prompt_class == "headline":
        return float(args.headline_weight)
    return float(args.default_weight)


def validate_seed(record: dict[str, Any]) -> int | None:
    seed = (record.get("start") or {}).get("seed", record.get("seed"))
    if seed is None:
        return None
    return int(seed)


def collect_direct_record_samples(
    record: dict[str, Any],
    feature_spec: Any,
    args: argparse.Namespace,
) -> tuple[list[WarmupSample], list[dict[str, Any]]]:
    game_id = str(record.get("game_id") or record.get("episode_index") or record.get("source_id") or "unknown")
    samples: list[WarmupSample] = []
    failures: list[dict[str, Any]] = []
    for offset, raw_entry in enumerate(direct_entries(record)):
        entry = normalize_action_entry(raw_entry)
        obs = observation_from_entry(entry)
        if obs is None:
            failures.append({"game_id": game_id, "step": entry.get("step", offset), "error": "missing observation"})
            continue
        legal_actions = legal_actions_from_entry(entry, obs)
        if not legal_actions:
            failures.append({"game_id": game_id, "step": entry.get("step", offset), "error": "missing legal_actions"})
            continue
        choice = selected_choice(entry)
        idx = entry.get("expert_action_index", entry.get("action_index"))
        if isinstance(idx, int):
            if idx < 0 or idx >= len(legal_actions):
                failures.append({"game_id": game_id, "step": entry.get("step", offset), "error": f"expert index out of range: {idx}"})
                continue
        else:
            idx = find_action_index(choice, legal_actions)
            if idx is None:
                legal_brief = [
                    {key: action.get(key) for key in ("type", "decision", "value", "label")}
                    for action in legal_actions[:30]
                ]
                failures.append(
                    {
                        "game_id": game_id,
                        "step": entry.get("step", offset),
                        "error": f"cannot match direct action: choice={choice} legal={legal_brief}",
                    }
                )
                continue
        side = str(entry.get("side") or obs.get("side") or "").lower()
        if side not in {"us", "ussr"}:
            failures.append({"game_id": game_id, "step": entry.get("step", offset), "error": f"missing/invalid side: {side}"})
            continue
        obs_without_legal = {key: value for key, value in obs.items() if key != "legal_actions"}
        try:
            encoded = encode_observation(obs_without_legal, legal_actions, feature_spec)
        except Exception as exc:
            failures.append({"game_id": game_id, "step": entry.get("step", offset), "error": f"encode failed: {exc}"})
            continue
        samples.append(
            WarmupSample(
                obs={key: np.asarray(value).copy() for key, value in encoded.items()},
                target=int(idx),
                side=side,
                policy_id=policy_id_for_side(side, args.multi_agent),
                weight=sample_weight(entry, args),
                game_id=game_id,
                step=int(entry.get("step") or offset),
            )
        )
        if args.max_samples is not None and len(samples) >= args.max_samples:
            break
    return samples, failures


def collect_samples(args: argparse.Namespace) -> tuple[list[WarmupSample], list[dict[str, Any]]]:
    samples: list[WarmupSample] = []
    failures: list[dict[str, Any]] = []
    env = TwilightStruggleEnv(
        {
            "force_setup_heuristic": False,
            "max_episode_steps": 5000,
            "log_games_dir": None,
            "metrics_dir": None,
            "log_action_details": False,
        }
    )
    try:
        for record in iter_game_records(args.input, args.max_games):
            game_id = str(record.get("game_id") or record.get("episode_index") or record.get("seed") or "unknown")
            direct = direct_entries(record)
            if direct:
                direct_samples, direct_failures = collect_direct_record_samples(record, env.feature_spec, args)
                samples.extend(direct_samples)
                failures.extend(direct_failures)
                if args.max_samples is not None and len(samples) >= args.max_samples:
                    return samples[: args.max_samples], failures
                continue
            seed = validate_seed(record)
            if seed is None:
                failures.append(
                    {
                        "game_id": game_id,
                        "error": (
                            "missing direct observation/legal_actions samples and missing start.seed for legacy replay; "
                            "no-seed warmup requires per-decision observation_before plus legal_actions"
                        ),
                    }
                )
                continue
            try:
                obs, _info = env.reset(seed=seed)
                final_info: dict[str, Any] = {}
                for raw_entry in record.get("actions", []):
                    entry = normalize_action_entry(raw_entry)
                    phase = str(entry.get("phase") or "").lower()
                    if args.skip_setup and phase.startswith("setup"):
                        idx = find_action_index(entry.get("choice") or {}, env.legal_actions)
                        if idx is None:
                            raise ValueError(f"cannot match skipped setup action: {entry}")
                        obs, _reward, terminated, truncated, final_info = env.step(idx)
                        if terminated or truncated:
                            break
                        continue

                    expected_side = str(entry.get("side") or "").lower()
                    actual_side = str((env.last_obs or {}).get("side") or "").lower()
                    if expected_side and expected_side != actual_side:
                        raise ValueError(f"side mismatch at step {entry.get('step')}: expected {expected_side}, bridge has {actual_side}")

                    idx = find_action_index(entry.get("choice") or {}, env.legal_actions)
                    if idx is None:
                        legal_brief = [
                            {key: action.get(key) for key in ("type", "decision", "value", "label")}
                            for action in env.legal_actions[:30]
                        ]
                        raise ValueError(
                            f"cannot match action at step {entry.get('step')}: choice={entry.get('choice')} legal={legal_brief}"
                        )

                    side = actual_side or expected_side
                    policy_id = policy_id_for_side(side, args.multi_agent)
                    samples.append(
                        WarmupSample(
                            obs={key: np.asarray(value).copy() for key, value in obs.items()},
                            target=int(idx),
                            side=side,
                            policy_id=policy_id,
                            weight=sample_weight(entry, args),
                            game_id=game_id,
                            step=int(entry.get("step") or len(samples)),
                        )
                    )
                    obs, _reward, terminated, truncated, final_info = env.step(idx)
                    if args.max_samples is not None and len(samples) >= args.max_samples:
                        return samples, failures
                    if terminated or truncated:
                        break

                if args.strict_result and not record.get("partial"):
                    expected = record.get("result") or {}
                    if expected:
                        expected_winner = expected.get("winner")
                        expected_reason = expected.get("terminal_reason")
                        actual_winner = final_info.get("winner")
                        actual_reason = final_info.get("terminal_reason")
                        if expected_winner and expected_winner != actual_winner:
                            raise ValueError(f"winner mismatch: expected {expected_winner}, got {actual_winner}")
                        if expected_reason and expected_reason != actual_reason:
                            raise ValueError(f"terminal reason mismatch: expected {expected_reason}, got {actual_reason}")
            except Exception as exc:
                failures.append({"game_id": game_id, "error": str(exc)})
                continue
    finally:
        env.close()
    return samples, failures


def build_algorithm(args: argparse.Namespace):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    register_masked_model()
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    probe_env = TwilightStruggleEnv({"force_setup_heuristic": False, "log_games_dir": None, "metrics_dir": None})
    policy_spec = (None, probe_env.observation_space, probe_env.action_space, {})
    probe_env.close()

    env_name = "twilight_struggle_warmup_multi" if args.multi_agent else "twilight_struggle_warmup"
    register_env(
        env_name,
        lambda cfg: TwilightStruggleMultiAgentEnv(cfg) if args.multi_agent else TwilightStruggleEnv(cfg),
    )
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env_name, env_config={"force_setup_heuristic": False, "log_games_dir": None, "metrics_dir": None})
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
    )
    if args.multi_agent:
        config = config.multi_agent(
            policies={US_POLICY_ID: policy_spec, USSR_POLICY_ID: policy_spec},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: US_POLICY_ID if agent_id == "us" else USSR_POLICY_ID,
            policies_to_train=sorted(parse_policies(args.policies_to_train, True)),
        )
    algo = config.build()
    return algo


def checkpoint_weights(path: Path, multi_agent: bool) -> dict[str, dict[str, Any]]:
    policy_ids = checkpoint_policy_ids(path)
    if multi_agent:
        if US_POLICY_ID in policy_ids and USSR_POLICY_ID in policy_ids:
            return {
                US_POLICY_ID: load_policy_weights(path, US_POLICY_ID),
                USSR_POLICY_ID: load_policy_weights(path, USSR_POLICY_ID),
            }
        for fallback in ("shared_policy", "default_policy"):
            if fallback in policy_ids or (path / "policies" / fallback).exists():
                weights = load_policy_weights(path, fallback)
                return {US_POLICY_ID: weights, USSR_POLICY_ID: weights}
    return {"default_policy": load_policy_weights(path, "default_policy")}


def initialize_algorithm(algo: Any, args: argparse.Namespace) -> None:
    def use_partial_warmstart() -> bool:
        if args.partial_warmstart is not None:
            return bool(args.partial_warmstart)
        return args.model_arch == "transformer_history"

    def set_policy_weights(policy_id: str, weights: dict[str, Any], source: Path, label: str) -> None:
        policy = algo.get_policy(policy_id)
        if not use_partial_warmstart():
            policy.set_weights(weights)
            print(f"loaded_{label}_from={source}", flush=True)
            return
        current, matched, widened, skipped_shape, missing_source, unexpected_source = copy_compatible_or_widened_weights(
            policy.get_weights(),
            weights,
        )
        policy.set_weights(current)
        print(
            {
                "event": "warmup_partial_warmstart",
                "label": label,
                "policy_id": policy_id,
                "source": str(source),
                "matched_count": len(matched),
                "widened_count": len(widened),
                "widened_keys": widened,
                "missing_source_count": len(missing_source),
                "unexpected_source_count": len(unexpected_source),
                "shape_mismatch_count": len(skipped_shape),
            },
            flush=True,
        )

    if args.restore_from:
        path = args.restore_from.expanduser().resolve()
        policy_ids = checkpoint_policy_ids(path)
        if args.multi_agent and use_partial_warmstart():
            weights = checkpoint_weights(path, True)
            if US_POLICY_ID in weights:
                set_policy_weights(US_POLICY_ID, weights[US_POLICY_ID], path, "restore_us_policy")
            if USSR_POLICY_ID in weights:
                set_policy_weights(USSR_POLICY_ID, weights[USSR_POLICY_ID], path, "restore_ussr_policy")
            if US_POLICY_ID not in weights and USSR_POLICY_ID not in weights:
                raise RuntimeError(f"no warmstartable policies found in {path}")
            print(f"partial_restored_from={path}", flush=True)
        elif args.multi_agent and not ({US_POLICY_ID, USSR_POLICY_ID} <= policy_ids):
            weights = checkpoint_weights(path, True)
            set_policy_weights(US_POLICY_ID, weights[US_POLICY_ID], path, "restore_us_policy")
            set_policy_weights(USSR_POLICY_ID, weights[USSR_POLICY_ID], path, "restore_ussr_policy")
        else:
            algo.restore(str(path))
        print(f"restored_from={path}", flush=True)
    if args.multi_agent and args.load_us_policy_from:
        path = args.load_us_policy_from.expanduser().resolve()
        set_policy_weights(US_POLICY_ID, checkpoint_weights(path, True)[US_POLICY_ID], path, "us_policy")
    if args.multi_agent and args.load_ussr_policy_from:
        path = args.load_ussr_policy_from.expanduser().resolve()
        set_policy_weights(USSR_POLICY_ID, checkpoint_weights(path, True)[USSR_POLICY_ID], path, "ussr_policy")


def to_torch_batch(policy: Any, samples: list[WarmupSample]):
    import torch

    device = next(policy.model.parameters()).device
    keys = samples[0].obs.keys()
    obs = {
        key: torch.as_tensor(np.stack([sample.obs[key] for sample in samples]), dtype=torch.float32, device=device)
        for key in keys
    }
    targets = torch.as_tensor([sample.target for sample in samples], dtype=torch.long, device=device)
    weights = torch.as_tensor([sample.weight for sample in samples], dtype=torch.float32, device=device)
    return obs, targets, weights


def train_bc(algo: Any, samples: list[WarmupSample], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    trainable = parse_policies(args.policies_to_train, args.multi_agent)
    grouped: dict[str, list[WarmupSample]] = {}
    for sample in samples:
        if sample.policy_id in trainable:
            grouped.setdefault(sample.policy_id, []).append(sample)
    optimizers = {
        policy_id: torch.optim.Adam(algo.get_policy(policy_id).model.parameters(), lr=args.lr)
        for policy_id in grouped
    }
    rng = random.Random(args.seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_stats: dict[str, Any] = {"epoch": epoch}
        for policy_id, policy_samples in grouped.items():
            rng.shuffle(policy_samples)
            policy = algo.get_policy(policy_id)
            model = policy.model
            model.train()
            total_loss = 0.0
            total_weight = 0.0
            correct = 0.0
            seen = 0
            for start in range(0, len(policy_samples), args.batch_size):
                batch = policy_samples[start : start + args.batch_size]
                obs, targets, weights = to_torch_batch(policy, batch)
                logits, _state = model({"obs": obs}, [], None)
                per_sample = F.cross_entropy(logits, targets, reduction="none")
                loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
                optimizers[policy_id].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[policy_id].step()
                total_loss += float((per_sample * weights).sum().detach().cpu())
                total_weight += float(weights.sum().detach().cpu())
                pred = torch.argmax(logits.detach(), dim=-1)
                correct += float(((pred == targets).float() * weights).sum().cpu())
                seen += len(batch)
            epoch_stats[f"{policy_id}/samples"] = seen
            epoch_stats[f"{policy_id}/loss"] = total_loss / max(1.0, total_weight)
            epoch_stats[f"{policy_id}/weighted_acc"] = correct / max(1.0, total_weight)
        history.append(epoch_stats)
        print(epoch_stats, flush=True)
    return {"epochs": history, "samples_by_policy": {key: len(value) for key, value in grouped.items()}}


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "warmup_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    samples, failures = collect_samples(args)
    by_policy: dict[str, int] = {}
    for sample in samples:
        by_policy[sample.policy_id] = by_policy.get(sample.policy_id, 0) + 1
    print(
        {
            "matched_samples": len(samples),
            "samples_by_policy": by_policy,
            "failed_games": len(failures),
            "first_failures": failures[:5],
        },
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "warmup_failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")
    if not samples:
        raise SystemExit("no warmup samples matched; see warmup_failures.jsonl")
    if args.dry_run:
        write_report(
            args.output_dir,
            {
                "dry_run": True,
                "matched_samples": len(samples),
                "samples_by_policy": by_policy,
                "failed_games": failures,
            },
        )
        return

    algo = build_algorithm(args)
    try:
        initialize_algorithm(algo, args)
        train_report = train_bc(algo, samples, args)
        checkpoint_path = save_checkpoint(algo, args.output_dir, f"bc-samples-{len(samples):08d}")
        report = {
            "dry_run": False,
            "matched_samples": len(samples),
            "samples_by_policy": by_policy,
            "failed_games": failures,
            "train": train_report,
            "checkpoint": checkpoint_path,
        }
        write_report(args.output_dir, report)
        print({"checkpoint": checkpoint_path, **train_report}, flush=True)
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
