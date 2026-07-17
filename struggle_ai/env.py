from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    class MultiAgentEnv:  # type: ignore[no-redef]
        pass

from struggle_ai.bridge_client import BridgeError, TwilightBridgeClient
from struggle_ai.features import (
    ACTION_FEATURES,
    CARD_HISTORY_FEATURES,
    CARD_HISTORY_LENGTH,
    CARD_FEATURES,
    COUNTRY_FEATURES,
    EVENT_FEATURES,
    GLOBAL_FEATURES,
    HISTORY_LENGTH,
    MAX_ACTIONS,
    REGION_FEATURES,
    FeatureSpec,
    encode_observation,
    make_spec,
)
from struggle_ai.log_format import format_record
from struggle_ai.policies import score_action


NUKE_DEATH_REASONS = {"nuclear_war", "thermonuclear war", "Cuban Missile Crisis"}
DEFCON_LOWERING_EVENT_CARDS = {"duckandcover", "kal007", "wwby"}
DEFCON_FORCED_COUP_EVENT_CARDS = {"che", "ortega"}
DEFCON_RANDOM_EVENT_TRIGGER_CARDS = {"fiveyearplan"}
DEFCON_RISK_EVENT_CARDS = (
    DEFCON_LOWERING_EVENT_CARDS
    | DEFCON_FORCED_COUP_EVENT_CARDS
    | DEFCON_RANDOM_EVENT_TRIGGER_CARDS
    | {"missileenvy"}
)
SHOWCARD_ID_RE = re.compile(r'id=["\']([^"\']+)["\']', re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")


class TwilightStruggleEnv(gym.Env):
    """Single-policy current-player environment for PPO self-play.

    Each step presents the side to move as the observation perspective. The
    action is an index into the padded legal-action candidate list.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        self.log_games_dir = Path(self.config["log_games_dir"]) if self.config.get("log_games_dir") else None
        self.metrics_dir = Path(self.config["metrics_dir"]) if self.config.get("metrics_dir") else None
        self.log_games_every = int(self.config.get("log_games_every", 1))
        self.log_debug_snapshots = bool(self.config.get("log_debug_snapshots", False))
        self.log_action_details = bool(self.config.get("log_action_details", True))
        self.terminal_reward_scale = float(self.config.get("terminal_reward_scale", 1.0))
        self.reward_shaping_scale = float(self.config.get("reward_shaping_scale", 1.0))
        self.turn_vp_reward_scale = float(self.config.get("turn_vp_reward_scale", 0.05))
        self.nuke_death_penalty = float(self.config.get("nuke_death_penalty", 0.0))
        self.scoring_card_held_penalty = float(self.config.get("scoring_card_held_penalty", 0.0))
        self.high_stability_coup_penalty = float(self.config.get("high_stability_coup_penalty", 0.0))
        self.low_stability_warzone_coup_reward = float(self.config.get("low_stability_warzone_coup_reward", 0.0))
        self.headline_opponent_event_penalty = float(self.config.get("headline_opponent_event_penalty", 0.0))
        self.defcon_risk_pick_penalty = float(self.config.get("defcon_risk_pick_penalty", 0.0))
        self.defcon_risk_commit_penalty = float(self.config.get("defcon_risk_commit_penalty", 0.0))
        self.empty_country_influence_reward = float(self.config.get("empty_country_influence_reward", 0.0))
        self.control_battleground_reward = float(self.config.get("control_battleground_reward", 0.0))
        self.control_non_battleground_reward = float(self.config.get("control_non_battleground_reward", 0.0))
        self.max_episode_step_penalty = float(self.config.get("max_episode_step_penalty", 0.0))
        self.defcon_suicide_mode = str(self.config.get("defcon_suicide_mode", "none"))
        self.max_episode_steps = int(self.config.get("max_episode_steps", 1200))
        self.warmup_random_steps = int(self.config.get("warmup_random_steps", 0))
        self.warmup_random_prob = float(self.config.get("warmup_random_prob", 0.0))
        self.persistent_random_prob = float(self.config.get("persistent_random_prob", 0.0))
        self.heuristic_override_prob = float(self.config.get("heuristic_override_prob", 0.0))
        self.scripted_side_prob = float(self.config.get("scripted_side_prob", 0.0))
        self.force_setup_heuristic = bool(self.config.get("force_setup_heuristic", True))
        self.total_env_steps = 0
        self.random_overrides = 0
        self.heuristic_overrides = 0
        self.scripted_side_overrides = 0
        self.base_seed = int(self.config.get("seed", 1))
        self.worker_seed_offset = (os.getpid() % 100000) * 100000
        self.action_rng = np.random.default_rng(int(self.config.get("action_seed", os.getpid())))
        self.episode_index = 0
        self.episode_seed: int | None = None
        self.scripted_side: str | None = None
        self.episode_step = 0
        self.episode_actions: list[dict[str, Any]] = []
        self.episode_filtered_actions: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.card_history: list[dict[str, Any]] = []
        self.episode_start: dict[str, Any] | None = None
        self.episode_reward_components: dict[str, float] = {}
        self.episode_side_rewards: dict[str, float] = {"us": 0.0, "ussr": 0.0}
        self.saito_log_len = 0
        self.bridge = TwilightBridgeClient(self.config.get("bridge_path"))
        self.cards = self.bridge.cards()
        self.cards_by_id = {str(card.get("id")): card for card in self.cards}
        self.countries = self.bridge.countries()
        self.feature_spec: FeatureSpec = make_spec(self.cards, self.countries)
        self.legal_actions: list[dict[str, Any]] = []
        self.last_obs: dict[str, Any] | None = None
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(-np.inf, np.inf, shape=(GLOBAL_FEATURES,), dtype=np.float32),
                "events": spaces.Box(-np.inf, np.inf, shape=(EVENT_FEATURES,), dtype=np.float32),
                "countries": spaces.Box(-np.inf, np.inf, shape=(self.feature_spec.country_count, COUNTRY_FEATURES), dtype=np.float32),
                "regions": spaces.Box(-np.inf, np.inf, shape=(self.feature_spec.region_count, REGION_FEATURES), dtype=np.float32),
                "cards": spaces.Box(-np.inf, np.inf, shape=(self.feature_spec.card_count, CARD_FEATURES), dtype=np.float32),
                "us_hand": spaces.Box(0.0, 1.0, shape=(self.feature_spec.card_count,), dtype=np.float32),
                "ussr_hand": spaces.Box(0.0, 1.0, shape=(self.feature_spec.card_count,), dtype=np.float32),
                "country_adjacency": spaces.Box(0.0, 1.0, shape=(self.feature_spec.country_count, self.feature_spec.country_count), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(MAX_ACTIONS,), dtype=np.float32),
                "action_features": spaces.Box(-np.inf, np.inf, shape=(MAX_ACTIONS, ACTION_FEATURES), dtype=np.float32),
                "history_actions": spaces.Box(-np.inf, np.inf, shape=(HISTORY_LENGTH, ACTION_FEATURES), dtype=np.float32),
                "history_sides": spaces.Box(-1.0, 1.0, shape=(HISTORY_LENGTH,), dtype=np.float32),
                "history_turn_ar": spaces.Box(-np.inf, np.inf, shape=(HISTORY_LENGTH, 2), dtype=np.float32),
                "history_vp_defcon": spaces.Box(-np.inf, np.inf, shape=(HISTORY_LENGTH, 2), dtype=np.float32),
                "history_mask": spaces.Box(0.0, 1.0, shape=(HISTORY_LENGTH,), dtype=np.float32),
                "card_history": spaces.Box(-np.inf, np.inf, shape=(CARD_HISTORY_LENGTH, CARD_HISTORY_FEATURES), dtype=np.float32),
                "card_history_mask": spaces.Box(0.0, 1.0, shape=(CARD_HISTORY_LENGTH,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.episode_index += 1
        self.episode_step = 0
        self.episode_actions = []
        self.episode_filtered_actions = []
        self.history = []
        self.card_history = []
        self.episode_reward_components = {}
        self.episode_side_rewards = {"us": 0.0, "ussr": 0.0}
        self.episode_seed = int(seed) if seed is not None else self.base_seed + self.worker_seed_offset + self.episode_index
        self.scripted_side = self.sample_scripted_side()
        obs = self.bridge.reset(self.episode_seed)
        self.episode_start = self.bridge.log() if self.log_games_dir is not None else None
        self.saito_log_len = len((self.episode_start or {}).get("log", []))
        self.last_obs = obs
        self.legal_actions = self.filtered_legal_actions(obs, obs["legal_actions"])
        self.record_no_legal_actions_if_needed(obs, "reset")
        return self.encode_current_observation(obs), {"side": obs["side"]}

    def step(self, action: int):
        if not self.legal_actions:
            if self.last_obs and self.last_obs.get("terminal"):
                info = {
                    "acting_side": self.last_obs.get("side", "ussr"),
                    "terminal_reward": 0.0,
                    "vp_delta_reward": 0.0,
                    "terminal_noop": True,
                }
                return encode_observation(self.obs_with_history(self.last_obs), [], self.feature_spec), 0.0, True, False, info
            raise RuntimeError(f"step called with no legal actions: {self.action_context(self.last_obs)}")
        if action < 0 or action >= min(len(self.legal_actions), MAX_ACTIONS):
            raise ValueError(f"illegal action index {action}")
        acting_side = self.last_obs["side"] if self.last_obs else "ussr"
        before = self.action_context(self.last_obs)
        debug_before = self.bridge.log() if self.log_debug_snapshots else None
        policy_action_index = int(action)
        override_kind = self.exploration_override_kind(acting_side)
        if override_kind in {"warmup_random", "persistent_random", "scripted_random"}:
            action = int(self.action_rng.integers(0, min(len(self.legal_actions), MAX_ACTIONS)))
            self.random_overrides += 1
            if override_kind == "scripted_random":
                self.scripted_side_overrides += 1
        elif override_kind in {"heuristic", "scripted_heuristic", "forced_setup_heuristic"}:
            action = self.heuristic_action_index()
            self.heuristic_overrides += 1
            if override_kind == "scripted_heuristic":
                self.scripted_side_overrides += 1
        random_override = override_kind in {"warmup_random", "persistent_random", "scripted_random"}
        heuristic_override = override_kind in {"heuristic", "scripted_heuristic", "forced_setup_heuristic"}
        selected_action = self.legal_actions[action]
        selected_action_features = self.selected_action_features(action)
        previous_obs = self.last_obs
        try:
            result = self.bridge.step(selected_action)
        except BridgeError as exc:
            error_log = self.safe_bridge_log()
            self.episode_actions.append(
                {
                    "step": self.episode_step,
                    "side": acting_side,
                    "turn": before["turn"],
                    "action_round": before["action_round"],
                    "before": before,
                    "after": None,
                    "action_index": int(action),
                    "policy_action_index": policy_action_index,
                    "random_override": random_override,
                    "heuristic_override": heuristic_override,
                    "override_kind": override_kind,
                    "action": selected_action,
                    "debug_before": debug_before,
                    "debug_after": error_log,
                    "saito_log_delta": self.saito_log_delta(error_log),
                    "bridge_error": str(exc),
                }
            )
            self.write_terminal_metrics(
                "bridge_error",
                {"bridge_error": str(exc), "acting_side": acting_side, "terminal_reason": "bridge_error"},
            )
            self.write_game_log("bridge_error", {"bridge_error": str(exc), "acting_side": acting_side})
            raise
        obs = result["observation"]
        after = self.action_context(obs)
        detail_snapshot = self.bridge.log() if self.log_action_details or self.log_debug_snapshots else None
        debug_after = detail_snapshot if self.log_debug_snapshots else None
        saito_log_delta = self.saito_log_delta(detail_snapshot)
        self.episode_actions.append(
            {
                "step": self.episode_step,
                "side": acting_side,
                "turn": before["turn"],
                "action_round": before["action_round"],
                "before": before,
                "after": after,
                "action_index": int(action),
                "policy_action_index": policy_action_index,
                "random_override": random_override,
                "heuristic_override": heuristic_override,
                "override_kind": override_kind,
                "action": selected_action,
                "state_delta": self.state_delta(previous_obs, obs),
                "saito_log_delta": saito_log_delta,
                "debug_before": debug_before,
                "debug_after": debug_after,
            }
        )
        self.append_history_entry(before, acting_side, selected_action_features)
        self.append_or_update_card_history_entry(previous_obs, before, after, acting_side, selected_action)
        raw_rewards = result.get("reward") or {}
        terminal_rewards = {
            "us": self.terminal_reward_scale * float(raw_rewards.get("us", 0.0)),
            "ussr": self.terminal_reward_scale * float(raw_rewards.get("ussr", 0.0)),
        }
        terminal_reward = terminal_rewards.get(acting_side, 0.0)
        nuke_death_penalty = self.terminal_nuke_death_penalty(terminal_reward, result)
        scoring_card_held_penalty = self.terminal_scoring_card_held_penalty(terminal_reward, result)
        nuke_death_penalties = {
            "us": self.terminal_nuke_death_penalty(terminal_rewards["us"], result),
            "ussr": self.terminal_nuke_death_penalty(terminal_rewards["ussr"], result),
        }
        scoring_card_held_penalties = {
            "us": self.terminal_scoring_card_held_penalty(terminal_rewards["us"], result),
            "ussr": self.terminal_scoring_card_held_penalty(terminal_rewards["ussr"], result),
        }
        vp_delta_rewards = self.vp_delta_rewards(before, after)
        vp_delta_reward = vp_delta_rewards.get(acting_side, 0.0)
        coup_target_reward = self.coup_target_reward(selected_action)
        headline_opponent_event_reward = self.headline_opponent_event_reward(selected_action, acting_side)
        defcon_risk_action_reward, defcon_risk_action_reason = self.defcon_risk_action_reward(before, selected_action)
        empty_country_influence_rewards = self.empty_country_influence_rewards(previous_obs, obs, before)
        control_country_rewards = self.control_country_rewards(previous_obs, obs, before)
        side_delta_rewards = {
            "us": (
                vp_delta_rewards.get("us", 0.0)
                + empty_country_influence_rewards.get("us", 0.0)
                + control_country_rewards.get("us", 0.0)
            ),
            "ussr": (
                vp_delta_rewards.get("ussr", 0.0)
                + empty_country_influence_rewards.get("ussr", 0.0)
                + control_country_rewards.get("ussr", 0.0)
            ),
        }
        empty_country_influence_reward = empty_country_influence_rewards.get(acting_side, 0.0)
        control_country_reward = control_country_rewards.get(acting_side, 0.0)
        reward = (
            terminal_reward
            + nuke_death_penalty
            + scoring_card_held_penalty
            + vp_delta_reward
            + coup_target_reward
            + headline_opponent_event_reward
            + defcon_risk_action_reward
            + empty_country_influence_reward
            + control_country_reward
        )
        step_reward_components = {
            "terminal_reward": terminal_reward,
            "nuke_death_penalty": nuke_death_penalty,
            "scoring_card_held_penalty": scoring_card_held_penalty,
            "vp_delta_reward": vp_delta_reward,
            "coup_target_reward": coup_target_reward,
            "headline_opponent_event_reward": headline_opponent_event_reward,
            "defcon_risk_action_reward": defcon_risk_action_reward,
            "empty_country_influence_reward": empty_country_influence_reward,
            "control_country_reward": control_country_reward,
            "total_reward": reward,
        }
        for key, value in step_reward_components.items():
            self.episode_reward_components[key] = self.episode_reward_components.get(key, 0.0) + float(value)
        side_step_rewards = {
            "us": (
                terminal_rewards["us"]
                + nuke_death_penalties["us"]
                + scoring_card_held_penalties["us"]
                + side_delta_rewards.get("us", 0.0)
            ),
            "ussr": (
                terminal_rewards["ussr"]
                + nuke_death_penalties["ussr"]
                + scoring_card_held_penalties["ussr"]
                + side_delta_rewards.get("ussr", 0.0)
            ),
        }
        side_step_rewards[acting_side] += coup_target_reward + headline_opponent_event_reward + defcon_risk_action_reward
        for side, side_reward in side_step_rewards.items():
            self.episode_side_rewards[side] = self.episode_side_rewards.get(side, 0.0) + float(side_reward)
        self.last_obs = obs
        self.legal_actions = self.filtered_legal_actions(obs, obs["legal_actions"])
        self.episode_step += 1
        no_legal_actions = not self.legal_actions and not bool(obs.get("terminal"))
        terminated = bool(result["done"]) or no_legal_actions
        truncated = False
        info = {
            **result["info"],
            "acting_side": acting_side,
            "terminal_reward": terminal_reward,
            "terminal_rewards": terminal_rewards,
            "nuke_death_penalty": nuke_death_penalty,
            "nuke_death_penalties": nuke_death_penalties,
            "scoring_card_held_penalty": scoring_card_held_penalty,
            "scoring_card_held_penalties": scoring_card_held_penalties,
            "vp_delta_reward": vp_delta_reward,
            "vp_delta_rewards": vp_delta_rewards,
            "coup_target_reward": coup_target_reward,
            "headline_opponent_event_reward": headline_opponent_event_reward,
            "defcon_risk_action_reward": defcon_risk_action_reward,
            "defcon_risk_action_reason": defcon_risk_action_reason,
            "empty_country_influence_reward": empty_country_influence_reward,
            "empty_country_influence_rewards": empty_country_influence_rewards,
            "control_country_reward": control_country_reward,
            "control_country_rewards": control_country_rewards,
            "side_delta_rewards": side_delta_rewards,
            "side_step_rewards": side_step_rewards,
            "random_override": random_override,
            "heuristic_override": heuristic_override,
            "override_kind": override_kind,
            "random_overrides": self.random_overrides,
            "heuristic_overrides": self.heuristic_overrides,
            "scripted_side": self.scripted_side,
            "scripted_side_overrides": self.scripted_side_overrides,
            "reward_shaping_scale": self.reward_shaping_scale,
            "total_env_steps": self.total_env_steps + 1,
        }
        if no_legal_actions:
            info = {
                **info,
                "winner": "invalid",
                "terminal_reason": "no_legal_actions",
                "no_legal_actions_source": "step",
                "no_legal_actions_context": self.action_context(obs),
            }
        self.total_env_steps += 1
        if no_legal_actions:
            self.write_terminal_metrics("no_legal_actions", info)
            self.write_game_log("no_legal_actions", info)
        elif terminated:
            self.write_terminal_metrics("terminal", info)
            self.write_game_log("terminal", info)
        elif self.max_episode_steps and self.episode_step >= self.max_episode_steps:
            truncated = True
            max_episode_step_penalty = -self.max_episode_step_penalty if self.max_episode_step_penalty else 0.0
            if max_episode_step_penalty:
                reward += max_episode_step_penalty
                self.episode_reward_components["max_episode_step_penalty"] = (
                    self.episode_reward_components.get("max_episode_step_penalty", 0.0)
                    + float(max_episode_step_penalty)
                )
                self.episode_reward_components["total_reward"] = (
                    self.episode_reward_components.get("total_reward", 0.0)
                    + float(max_episode_step_penalty)
                )
                self.episode_side_rewards[acting_side] = self.episode_side_rewards.get(acting_side, 0.0) + float(max_episode_step_penalty)
                side_step_rewards[acting_side] = side_step_rewards.get(acting_side, 0.0) + float(max_episode_step_penalty)
            info = {
                **info,
                "winner": "timeout",
                "terminal_reason": "max_episode_steps",
                "max_episode_steps": self.max_episode_steps,
                "max_episode_step_penalty": max_episode_step_penalty,
                "timeout_context": self.action_context(obs),
                "side_step_rewards": side_step_rewards,
            }
            self.write_terminal_metrics("truncated", info)
            self.write_game_log("truncated", info)
        return self.encode_current_observation(obs), reward, terminated, truncated, info

    def encode_current_observation(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return encode_observation(self.obs_with_history(obs), self.legal_actions, self.feature_spec)

    def set_reward_shaping_scale(self, scale: float) -> None:
        self.reward_shaping_scale = float(scale)

    def obs_with_history(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {**obs, "history": self.history, "card_history": self.card_history}

    def selected_action_features(self, action_index: int) -> list[float]:
        if not self.last_obs:
            return [0.0] * ACTION_FEATURES
        encoded = encode_observation(self.obs_with_history(self.last_obs), self.legal_actions, self.feature_spec)
        return encoded["action_features"][action_index].astype(float).tolist()

    def append_history_entry(self, before: dict[str, Any], acting_side: str, action_features: list[float]) -> None:
        self.history.append(
            {
                "side": acting_side,
                "turn": before.get("turn", 0),
                "action_round": before.get("action_round", 0),
                "vp": before.get("vp", 0),
                "defcon": before.get("defcon", 0),
                "action_features": action_features,
            }
        )
        if len(self.history) > HISTORY_LENGTH:
            self.history = self.history[-HISTORY_LENGTH:]

    def append_or_update_card_history_entry(
        self,
        obs: dict[str, Any] | None,
        before: dict[str, Any],
        after: dict[str, Any],
        acting_side: str,
        action: dict[str, Any],
    ) -> None:
        if not obs:
            return
        decision = str(action.get("decision") or "").lower()
        value = str(action.get("value") or "").lower()
        if decision in {"country_click", "country_mouseup"}:
            return
        card_id = self.action_card_id(obs, action)
        if not card_id:
            return

        mode = self.card_history_mode(obs, action, card_id)
        phase = str(obs.get("phase") or "")
        target_idx = self.find_card_history_entry(acting_side, card_id, before)
        if target_idx is None:
            target_idx = len(self.card_history)
            self.card_history.append(
                {
                    "side": acting_side,
                    "turn": before.get("turn", 0),
                    "action_round": before.get("action_round", 0),
                    "phase": phase,
                    "card": card_id,
                    "vp_before": before.get("vp", 0),
                    "defcon_before": before.get("defcon", 0),
                    "vp_after": after.get("vp", before.get("vp", 0)),
                    "defcon_after": after.get("defcon", before.get("defcon", 0)),
                    "modes": [],
                    "completed": False,
                    "removed": False,
                }
            )
        entry = self.card_history[target_idx]
        modes = list(entry.get("modes") or [])
        if mode and mode not in modes:
            modes.append(mode)
        if value in {"before_ops", "after_ops"} and value not in modes:
            modes.append(value)
        if "headline" in str(action.get("prompt") or obs.get("prompt") or "").lower() and "headline" not in modes:
            modes.append("headline")
        entry["modes"] = modes
        entry["vp_after"] = after.get("vp", entry.get("vp_after", entry.get("vp_before", 0)))
        entry["defcon_after"] = after.get("defcon", entry.get("defcon_after", entry.get("defcon_before", 0)))
        if value in {"event", "ops", "space", "discard"}:
            entry["completed"] = True
        if card_id in set((obs.get("removed") or [])):
            entry["removed"] = True
        if len(self.card_history) > CARD_HISTORY_LENGTH:
            self.card_history = self.card_history[-CARD_HISTORY_LENGTH:]

    def find_card_history_entry(self, acting_side: str, card_id: str, before: dict[str, Any]) -> int | None:
        turn = before.get("turn", 0)
        action_round = before.get("action_round", 0)
        for idx in range(len(self.card_history) - 1, -1, -1):
            entry = self.card_history[idx]
            if (
                entry.get("side") == acting_side
                and entry.get("card") == card_id
                and entry.get("turn") == turn
                and entry.get("action_round") == action_round
            ):
                return idx
        return None

    def card_history_mode(self, obs: dict[str, Any], action: dict[str, Any], card_id: str) -> str:
        value = str(action.get("value") or "").lower()
        prompt = str(action.get("prompt") or obs.get("prompt") or "").lower()
        if value in {"event", "ops", "space", "before_ops", "after_ops", "discard"}:
            return value
        if "headline" in prompt:
            return "headline"
        if "space" in prompt:
            return "space"
        if "discard" in prompt:
            return "discard"
        if card_id:
            return "selected"
        return ""

    def filtered_legal_actions(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.defcon_suicide_mode != "hard_filter":
            return list(legal_actions)
        kept: list[dict[str, Any]] = []
        filtered: list[dict[str, Any]] = []
        for idx, action in enumerate(legal_actions):
            reason = self.deterministic_defcon_suicide_reason(obs, action)
            if reason:
                filtered.append({"index": idx, "reason": reason, "action": action, "context": self.action_context(obs)})
            else:
                kept.append(action)
        if not filtered:
            return list(legal_actions)
        if not kept:
            self.episode_filtered_actions.extend({**item, "filter_skipped": True} for item in filtered)
            return list(legal_actions)
        self.episode_filtered_actions.extend(filtered)
        return kept

    def deterministic_defcon_suicide_reason(self, obs: dict[str, Any], action: dict[str, Any]) -> str | None:
        defcon = int(obs.get("defcon") or 5)
        prompt_raw = str(action.get("prompt") or obs.get("prompt") or "")
        prompt = prompt_raw.lower()
        decision = str(action.get("decision") or "").lower()
        value = str(action.get("value") or "")
        value_lower = value.lower()
        active_card = self.action_card_id(obs, action)
        active_card_key = active_card.lower() if active_card else None
        if "set defcon to" in prompt:
            try:
                target_defcon = int(value)
                if target_defcon < 2:
                    return "defcon_set_below_two"
                if target_defcon <= 2 and self.headline_has_defcon_lowering_card(obs):
                    return "howilearned_set_two_before_defcon_lowering_headline"
            except ValueError:
                pass
        if defcon <= 2 and value_lower == "boycott" and "olympic" in self.normalize_card_text(prompt):
            return "defcon_2_olympic_boycott"
        if (
            decision == "list"
            and "headline" in prompt
            and defcon <= 2
            and active_card_key in DEFCON_LOWERING_EVENT_CARDS
        ):
            return f"headline_defcon_{defcon}_{active_card_key}"
        if (
            decision == "list"
            and defcon <= 2
            and active_card_key in DEFCON_RISK_EVENT_CARDS
            and self.is_immediate_card_play_prompt(prompt)
        ):
            return f"defcon_{defcon}_immediate_card_{active_card_key}"
        if (
            decision == "list"
            and defcon <= 2
            and active_card_key in DEFCON_RISK_EVENT_CARDS
            and self.is_action_round_card_selection_prompt(obs, prompt)
        ):
            return f"defcon_{defcon}_pick_card_{active_card_key}"
        if (
            decision == "list"
            and "headline" in prompt
            and active_card_key in DEFCON_LOWERING_EVENT_CARDS
            and self.headline_has_card(obs, "howilearned")
        ):
            return f"headline_defcon_{defcon}_{active_card_key}_with_howilearned"
        if (
            decision == "list"
            and "headline" in prompt
            and active_card_key == "howilearned"
            and self.headline_has_defcon_lowering_card(obs)
        ):
            return f"headline_defcon_{defcon}_howilearned_with_defcon_lowering_card"
        if self.cuban_missile_crisis_blocks_coup(obs):
            if value_lower == "coup" and "ops" in prompt:
                return "cuban_missile_crisis_coup_mode"
            if "coup" in prompt and decision in {"country_click", "country_mouseup"}:
                return "cuban_missile_crisis_coup_target"
        delayed_defcon_event = self.active_event_is_defcon_risk(obs)
        if (
            defcon == 3
            and value_lower == "coup"
            and "ops" in prompt
            and delayed_defcon_event
            and not (obs.get("side") == "us" and (obs.get("events") or {}).get("nuclearsubs"))
        ):
            return "defcon_3_coup_before_defcon_lowering_event"
        if defcon > 2:
            return None
        if value_lower == "lower" and "summit" in prompt:
            return "defcon_2_summit_lower"
        if active_card_key in DEFCON_LOWERING_EVENT_CARDS and value_lower in {"event", "ops", "before_ops", "after_ops"}:
            return f"defcon_2_{active_card_key}_event"
        if active_card_key == "missileenvy" and value_lower == "event":
            return "defcon_2_missileenvy_event"
        if active_card_key in DEFCON_FORCED_COUP_EVENT_CARDS and value_lower in {"event", "ops", "before_ops", "after_ops"}:
            return f"defcon_2_{active_card_key}_forced_coup_event"
        if active_card_key in DEFCON_RANDOM_EVENT_TRIGGER_CARDS and value_lower in {"event", "ops", "before_ops", "after_ops", "place", "realign"}:
            return f"defcon_2_{active_card_key}_random_event_trigger"
        if value_lower == "before_ops" and delayed_defcon_event:
            return "defcon_2_defcon_lowering_opponent_event"
        if value_lower == "event" and active_card_key == "howilearned":
            return None
        if value_lower == "coup" and "ops" in prompt:
            if obs.get("side") == "us" and (obs.get("events") or {}).get("nuclearsubs"):
                return None
            return "defcon_2_coup_mode"
        if "coup" not in prompt and value_lower != "coup":
            return None
        if decision not in {"country_click", "country_mouseup"}:
            return None
        country = next((c for c in obs.get("countries", []) if c.get("id") == value), None)
        if not country or not bool(country.get("bg")):
            return None
        if obs.get("side") == "us" and (obs.get("events") or {}).get("nuclearsubs"):
            return None
        return "defcon_2_battleground_coup"

    def action_card_id(self, obs: dict[str, Any], action: dict[str, Any]) -> str | None:
        for candidate in (
            action.get("card"),
            action.get("value"),
            obs.get("active_card"),
            obs.get("event_card"),
            obs.get("event_name"),
            self.queue_card_candidate(obs.get("queue_top")),
            obs.get("queue_top"),
        ):
            card_id = self.normalize_card_identifier(candidate)
            if card_id:
                return card_id
        prompt = str(action.get("prompt") or obs.get("prompt") or "")
        match = SHOWCARD_ID_RE.search(prompt)
        if match:
            card_id = self.normalize_card_identifier(match.group(1))
            if card_id:
                return card_id
        return self.card_id_from_text(prompt)

    @staticmethod
    def queue_card_candidate(queue_top: Any) -> str | None:
        parts = str(queue_top or "").split("\t")
        if len(parts) >= 3 and parts[0] in {"ops", "event", "discard", "space"}:
            return parts[2]
        return None

    def active_event_is_defcon_risk(self, obs: dict[str, Any]) -> bool:
        for candidate in (obs.get("active_card"), obs.get("event_card"), obs.get("event_name"), obs.get("queue_top")):
            card_id = self.normalize_card_identifier(candidate) or self.card_id_from_text(str(candidate or ""))
            if card_id and card_id.lower() in DEFCON_RISK_EVENT_CARDS:
                return True
        prompt = str(obs.get("prompt") or "")
        card_id = self.card_id_from_text(prompt)
        return bool(card_id and card_id.lower() in DEFCON_RISK_EVENT_CARDS)

    @staticmethod
    def is_action_round_card_selection_prompt(obs: dict[str, Any], prompt: str) -> bool:
        if str(obs.get("phase") or "").lower() != "action":
            return False
        if "pick a card" not in prompt:
            return False
        return all(term not in prompt for term in ("discard", "headline", "select headline"))

    @staticmethod
    def is_immediate_card_play_prompt(prompt: str) -> bool:
        return "choose card to play immediately" in prompt

    @staticmethod
    def cuban_missile_crisis_blocks_coup(obs: dict[str, Any]) -> bool:
        events = obs.get("events") or {}
        if events.get("cubanmissilecrisis_cancelled"):
            return False
        return bool(events.get("cubanmissilecrisis"))

    def headline_has_card(self, obs: dict[str, Any], target_card: str) -> bool:
        headline = obs.get("headline") or {}
        target = target_card.lower()
        for value in headline.values():
            card_id = self.normalize_card_identifier(value)
            if card_id and card_id.lower() == target:
                return True
        return False

    def headline_has_defcon_lowering_card(self, obs: dict[str, Any]) -> bool:
        headline = obs.get("headline") or {}
        for value in headline.values():
            card_id = self.normalize_card_identifier(value)
            if card_id and card_id.lower() in DEFCON_LOWERING_EVENT_CARDS:
                return True
        return False

    def normalize_card_identifier(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text in self.cards_by_id:
            return text
        normalized = self.normalize_card_text(text)
        for card_id, card in self.cards_by_id.items():
            if normalized == self.normalize_card_text(card_id) or normalized == self.normalize_card_text(card.get("name", "")):
                return card_id
        return None

    def card_id_from_text(self, text: str) -> str | None:
        stripped = HTML_TAG_RE.sub(" ", str(text or ""))
        normalized = self.normalize_card_text(stripped)
        if not normalized:
            return None
        for card_id, card in self.cards_by_id.items():
            if self.normalize_card_text(card_id) in normalized or self.normalize_card_text(card.get("name", "")) in normalized:
                return card_id
        return None

    @staticmethod
    def normalize_card_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    def record_no_legal_actions_if_needed(self, obs: dict[str, Any], source: str) -> None:
        if self.legal_actions or obs.get("terminal"):
            return
        info = {
            "acting_side": obs.get("side"),
            "terminal_reason": "no_legal_actions",
            "no_legal_actions_source": source,
            "no_legal_actions_context": self.action_context(obs),
        }
        self.write_terminal_metrics("no_legal_actions", info)
        self.write_game_log("no_legal_actions", info)

    def sample_scripted_side(self) -> str | None:
        if self.scripted_side_prob <= 0 or self.action_rng.random() >= self.scripted_side_prob:
            return None
        return "us" if self.action_rng.random() < 0.5 else "ussr"

    def exploration_override_kind(self, acting_side: str) -> str | None:
        phase = str((self.last_obs or {}).get("phase", "")).lower()
        if self.force_setup_heuristic and phase.startswith("setup_"):
            return "forced_setup_heuristic"
        if (
            self.total_env_steps < self.warmup_random_steps
            and self.warmup_random_prob > 0
            and self.action_rng.random() < self.warmup_random_prob
        ):
            return "warmup_random"
        if self.persistent_random_prob > 0 and self.action_rng.random() < self.persistent_random_prob:
            return "persistent_random"
        if self.heuristic_override_prob > 0 and self.action_rng.random() < self.heuristic_override_prob:
            return "heuristic"
        if self.scripted_side and acting_side == self.scripted_side:
            return "scripted_heuristic" if self.action_rng.random() < 0.85 else "scripted_random"
        return None

    def heuristic_action_index(self) -> int:
        scores = [score_action(action, obs=self.last_obs) for action in self.legal_actions[:MAX_ACTIONS]]
        best_score = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best_score]
        return int(self.action_rng.choice(candidates))

    def render(self):
        side = self.last_obs["side"] if self.last_obs else None
        return self.bridge.render_text(side)

    def close(self):
        self.bridge.close()

    def write_game_log(self, kind: str, info: dict[str, Any]) -> None:
        if self.log_games_dir is None:
            return
        if self.log_games_every > 1 and self.episode_index % self.log_games_every != 0:
            return
        self.log_games_dir.mkdir(parents=True, exist_ok=True)
        worker = os.getpid()
        payload = {
            "kind": kind,
            "time": time.time(),
            "episode_index": self.episode_index,
            "seed": self.episode_seed,
            "steps": self.episode_step,
            "info": info,
            "cards": self.cards,
            "start": self.episode_start,
            "actions": self.episode_actions,
            "saito": self.bridge.log(),
            "filtered_actions": self.episode_filtered_actions,
        }
        path = self.log_games_dir / f"games-{worker}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        sorted_path = self.log_games_dir / f"games-{worker}.sorted.txt"
        with sorted_path.open("a", encoding="utf-8") as handle:
            handle.write(format_record(payload) + "\n\n")

    def write_terminal_metrics(self, kind: str, info: dict[str, Any]) -> None:
        if self.metrics_dir is None:
            return
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        worker = os.getpid()
        payload = {
            "kind": kind,
            "time": time.time(),
            "episode_index": self.episode_index,
            "seed": self.episode_seed,
            "steps": self.episode_step,
            "winner": info.get("winner"),
            "terminal_reason": info.get("terminal_reason"),
            "acting_side": info.get("acting_side"),
            "bridge_error": info.get("bridge_error"),
            "no_legal_actions": kind == "no_legal_actions",
            "filtered_action_count": len(self.episode_filtered_actions),
            "unsafe_filter_skipped_count": self.unsafe_filter_skipped_count(),
            "unsafe_filter_skipped_reasons": self.unsafe_filter_skipped_reasons(),
            "nuke_loser": self.nuke_loser(info),
            "nuke_suicide_side": self.nuke_loser(info),
            "reward": info.get("terminal_reward"),
            "us_reward": self.episode_side_rewards.get("us", 0.0),
            "ussr_reward": self.episode_side_rewards.get("ussr", 0.0),
            "episode_side_rewards": self.episode_side_rewards,
            "episode_reward_components": self.episode_reward_components,
            "reward_shaping_scale": self.reward_shaping_scale,
        }
        path = self.metrics_dir / f"terminal-{worker}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def unsafe_filter_skipped_count(self) -> int:
        return sum(1 for item in self.episode_filtered_actions if item.get("filter_skipped"))

    def unsafe_filter_skipped_reasons(self) -> dict[str, int]:
        reasons: dict[str, int] = {}
        for item in self.episode_filtered_actions:
            if not item.get("filter_skipped"):
                continue
            reason = str(item.get("reason") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    @staticmethod
    def nuke_loser(info: dict[str, Any]) -> str | None:
        if str(info.get("terminal_reason") or "") not in NUKE_DEATH_REASONS:
            return None
        winner = str(info.get("winner") or "")
        if winner == "us":
            return "ussr"
        if winner == "ussr":
            return "us"
        return None

    def safe_bridge_log(self) -> dict[str, Any] | None:
        try:
            return self.bridge.log()
        except BridgeError:
            return None

    def saito_log_delta(self, snapshot: dict[str, Any] | None) -> list[str]:
        if not snapshot:
            return []
        log = list(snapshot.get("log") or [])
        delta = log[self.saito_log_len :]
        self.saito_log_len = len(log)
        return [str(item) for item in delta]

    @staticmethod
    def state_delta(before: dict[str, Any] | None, after: dict[str, Any]) -> dict[str, Any]:
        if before is None:
            return {}
        delta: dict[str, Any] = {}
        before_vp = before.get("vp")
        after_vp = after.get("vp")
        if before_vp != after_vp:
            delta["vp"] = {"before": before_vp, "after": after_vp}
        before_defcon = before.get("defcon")
        after_defcon = after.get("defcon")
        if before_defcon != after_defcon:
            delta["defcon"] = {"before": before_defcon, "after": after_defcon}
        before_countries = {country["id"]: country for country in before.get("countries", []) if "id" in country}
        country_changes = []
        for country in after.get("countries", []):
            country_id = country.get("id")
            previous = before_countries.get(country_id)
            if not country_id or not previous:
                continue
            if previous.get("us") == country.get("us") and previous.get("ussr") == country.get("ussr"):
                continue
            country_changes.append(
                {
                    "id": country_id,
                    "name": country.get("name") or country_id,
                    "us_before": previous.get("us"),
                    "us_after": country.get("us"),
                    "ussr_before": previous.get("ussr"),
                    "ussr_after": country.get("ussr"),
                    "control_before": previous.get("control"),
                    "control_after": country.get("control"),
                }
            )
        if country_changes:
            delta["countries"] = country_changes
        return delta

    @staticmethod
    def action_context(obs: dict[str, Any] | None) -> dict[str, Any]:
        if obs is None:
            return {}
        return {
            "side": obs.get("side"),
            "current_player": obs.get("current_player"),
            "phase": obs.get("phase"),
            "turn": obs.get("turn"),
            "action_round": obs.get("action_round"),
            "vp": obs.get("vp"),
            "defcon": obs.get("defcon"),
            "event_name": obs.get("event_name"),
            "queue_top": obs.get("queue_top"),
            "prompt": obs.get("prompt", ""),
        }

    def vp_delta_rewards(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
        if self.turn_vp_reward_scale == 0:
            return {"us": 0.0, "ussr": 0.0}
        if not before or not after:
            return {"us": 0.0, "ussr": 0.0}
        before_vp = self.bounded_vp(before.get("vp"))
        after_vp = self.bounded_vp(after.get("vp"))
        vp_delta = max(-40.0, min(40.0, after_vp - before_vp))
        reward = self.reward_shaping_scale * self.turn_vp_reward_scale * (vp_delta / 20.0)
        return {"us": reward, "ussr": -reward}

    @staticmethod
    def bounded_vp(value: Any) -> float:
        try:
            vp = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(vp):
            return 0.0
        return max(-20.0, min(20.0, vp))

    def coup_target_reward(self, action: dict[str, Any]) -> float:
        if self.high_stability_coup_penalty == 0.0 and self.low_stability_warzone_coup_reward == 0.0:
            return 0.0
        prompt = str(action.get("prompt", "")).lower()
        decision = str(action.get("decision", "")).lower()
        if "coup" not in prompt and "coup" not in decision:
            return 0.0
        country_id = str(action.get("value", ""))
        country = next((c for c in (self.last_obs or {}).get("countries", []) if c.get("id") == country_id), None)
        if not country:
            return 0.0
        stability = int(country.get("stability") or 0)
        if stability >= 3:
            return -self.reward_shaping_scale * self.high_stability_coup_penalty
        if stability in {1, 2} and bool(country.get("bg")):
            return self.reward_shaping_scale * self.low_stability_warzone_coup_reward
        return 0.0

    def headline_opponent_event_reward(self, action: dict[str, Any], acting_side: str) -> float:
        if self.headline_opponent_event_penalty == 0.0:
            return 0.0
        prompt = str(action.get("prompt", "")).lower()
        if "headline" not in prompt:
            return 0.0
        card_id = str(action.get("value", ""))
        card = self.cards_by_id.get(card_id)
        if not card or card.get("scoring"):
            return 0.0
        owner = str(card.get("player") or card.get("side") or "").lower()
        if owner in {"", acting_side, "both", "neutral"}:
            return 0.0
        return -self.reward_shaping_scale * self.headline_opponent_event_penalty

    def defcon_risk_action_reward(self, obs: dict[str, Any], action: dict[str, Any]) -> tuple[float, str | None]:
        if self.defcon_risk_pick_penalty == 0.0 and self.defcon_risk_commit_penalty == 0.0:
            return 0.0, None
        reason = self.deterministic_defcon_suicide_reason(obs, action)
        if not reason:
            return 0.0, None
        if "pick_card" in reason:
            penalty = self.defcon_risk_pick_penalty
        else:
            penalty = self.defcon_risk_commit_penalty
        if penalty == 0.0:
            return 0.0, reason
        return -self.reward_shaping_scale * penalty, reason

    def empty_country_influence_rewards(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any],
        before_context: dict[str, Any],
    ) -> dict[str, float]:
        if self.empty_country_influence_reward == 0.0:
            return {"us": 0.0, "ussr": 0.0}
        if not before_obs:
            return {"us": 0.0, "ussr": 0.0}

        before_countries = {country.get("id"): country for country in before_obs.get("countries", [])}
        rewards = {"us": 0.0, "ussr": 0.0}
        for country in after_obs.get("countries", []):
            if not bool(country.get("bg")):
                continue
            country_id = country.get("id")
            previous = before_countries.get(country_id)
            if not previous:
                continue
            if int(previous.get("us") or 0) != 0 or int(previous.get("ussr") or 0) != 0:
                continue
            us_delta = int(country.get("us") or 0) - int(previous.get("us") or 0)
            ussr_delta = int(country.get("ussr") or 0) - int(previous.get("ussr") or 0)
            if us_delta > 0:
                rewards["us"] += self.reward_shaping_scale * self.empty_country_influence_reward
                rewards["ussr"] -= self.reward_shaping_scale * self.empty_country_influence_reward
            if ussr_delta > 0:
                rewards["ussr"] += self.reward_shaping_scale * self.empty_country_influence_reward
                rewards["us"] -= self.reward_shaping_scale * self.empty_country_influence_reward
        return rewards

    def control_country_rewards(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any],
        before_context: dict[str, Any],
    ) -> dict[str, float]:
        if self.control_battleground_reward == 0.0 and self.control_non_battleground_reward == 0.0:
            return {"us": 0.0, "ussr": 0.0}
        if not before_obs:
            return {"us": 0.0, "ussr": 0.0}

        before_countries = {country.get("id"): country for country in before_obs.get("countries", [])}
        rewards = {"us": 0.0, "ussr": 0.0}
        for country in after_obs.get("countries", []):
            scale = self.reward_shaping_scale * (self.control_battleground_reward if bool(country.get("bg")) else self.control_non_battleground_reward)
            if scale == 0.0:
                continue
            country_id = country.get("id")
            previous = before_countries.get(country_id)
            if not previous:
                continue
            old_owner = str(previous.get("control") or "none").lower()
            new_owner = str(country.get("control") or "none").lower()
            if old_owner == new_owner:
                continue
            if old_owner in {"us", "ussr"}:
                rewards[old_owner] -= scale
            if new_owner in {"us", "ussr"}:
                rewards[new_owner] += scale
        return rewards

    def terminal_nuke_death_penalty(self, terminal_reward: float, result: dict[str, Any]) -> float:
        if self.nuke_death_penalty == 0.0:
            return 0.0
        if not result.get("done"):
            return 0.0
        reason = str((result.get("info") or {}).get("terminal_reason") or "")
        if reason not in NUKE_DEATH_REASONS:
            return 0.0
        if terminal_reward >= 0:
            return 0.0
        return -self.nuke_death_penalty

    def terminal_scoring_card_held_penalty(self, terminal_reward: float, result: dict[str, Any]) -> float:
        if self.scoring_card_held_penalty == 0.0:
            return 0.0
        if not result.get("done"):
            return 0.0
        reason = str((result.get("info") or {}).get("terminal_reason") or "")
        if reason != "scoring card held":
            return 0.0
        if terminal_reward >= 0:
            return 0.0
        return -self.scoring_card_held_penalty


class TwilightStruggleMultiAgentEnv(MultiAgentEnv):
    """Two-agent wrapper with one active side per bridge decision.

    The wrapped bridge env still owns the full game state, but RLlib sees US
    and USSR as separate agents. This keeps return/advantage computation from
    mixing rewards across sides while allowing both agents to map to one shared
    policy.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.base_env = TwilightStruggleEnv(config)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.possible_agents = ["us", "ussr"]
        self.agents: list[str] = []
        self.current_agent: str | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.current_agent = str(info.get("side") or self.base_env.last_obs.get("side") or "ussr")
        self.agents = [self.current_agent]
        return {self.current_agent: obs}, {self.current_agent: info}

    def step(self, action_dict: dict[str, int]):
        if not self.current_agent:
            raise RuntimeError("step called before reset")
        if self.current_agent not in action_dict:
            raise KeyError(f"missing action for current agent {self.current_agent}")
        previous_agent = self.current_agent
        obs, reward, terminated, truncated, info = self.base_env.step(action_dict[previous_agent])
        terminated_all = bool(terminated)
        truncated_all = bool(truncated)
        rewards = {previous_agent: reward}
        other_agent = "ussr" if previous_agent == "us" else "us"
        other_side_step_reward = float((info.get("side_step_rewards") or {}).get(other_agent, 0.0))
        if other_side_step_reward != 0.0:
            rewards[other_agent] = other_side_step_reward
        terminations = {agent_id: terminated_all for agent_id in rewards}
        terminations["__all__"] = terminated_all
        truncations = {agent_id: truncated_all for agent_id in rewards}
        truncations["__all__"] = truncated_all
        if terminated_all or truncated_all:
            self.current_agent = None
            self.agents = []
            infos = {"__common__": {"last_agent": previous_agent, **info}}
            return {}, rewards, terminations, truncations, infos

        next_agent = str((self.base_env.last_obs or {}).get("side") or previous_agent)
        self.current_agent = next_agent
        self.agents = [next_agent]
        infos = {next_agent: {"previous_agent": previous_agent, **info}}
        return {next_agent: obs}, rewards, terminations, truncations, infos

    def render(self):
        return self.base_env.render()

    def set_reward_shaping_scale(self, scale: float) -> None:
        self.base_env.set_reward_shaping_scale(scale)

    def close(self):
        self.base_env.close()
