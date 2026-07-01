from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from struggle_ai.policies import score_action


MAX_ACTIONS = 512
HISTORY_LENGTH = 512
CARD_HISTORY_LENGTH = 160

REGIONS = ("europe", "asia", "seasia", "mideast", "africa", "camerica", "samerica")
EVENT_KEYS = (
    "nato",
    "nato_westgermany",
    "nato_france",
    "formosan",
    "norad",
    "vietnam_revolts",
    "vietnam_revolts_eligible",
    "china_card_eligible",
    "redscare_player1",
    "redscare_player2",
    "containment",
    "brezhnev",
    "quagmire",
    "beartrap",
    "cubanmissilecrisis",
    "cubanmissilecrisis_cancelled",
    "flowerpower",
    "nuclearsubs",
    "shuttlediplomacy",
    "usjapan",
    "iranianhostage",
    "ironlady",
    "reformer",
    "northseaoil",
    "wwby",
    "deathsquads",
    "campdavid",
    "johnpaul",
    "opec",
    "saltnegotiations",
    "missileenvy",
    "unintervention",
    "aldrich",
    "cia",
    "lonegunman",
    "space_station",
    "eagle_has_landed",
    "yuri",
    "awacs",
    "muslimrevolution",
)

GLOBAL_FEATURES = 24
COUNTRY_FEATURES = 34
REGION_FEATURES = 20
CARD_FEATURES = 30
ACTION_FEATURES = 45
CARD_HISTORY_FEATURES = 26
EVENT_FEATURES = len(EVENT_KEYS)
HEURISTIC_PRIOR_FEATURE = 40

DEFCON_LOWERING_EVENT_CARDS = frozenset({"duckandcover", "kal007", "wwby"})
DEFCON_FORCED_COUP_EVENT_CARDS = frozenset({"ortega"})
DEFCON_RANDOM_EVENT_TRIGGER_CARDS = frozenset({"fiveyearplan"})
DEFCON_RISK_EVENT_CARDS = (
    DEFCON_LOWERING_EVENT_CARDS
    | DEFCON_FORCED_COUP_EVENT_CARDS
    | DEFCON_RANDOM_EVENT_TRIGGER_CARDS
    | frozenset({"missileenvy"})
)


@dataclass(frozen=True)
class FeatureSpec:
    card_ids: tuple[str, ...]
    country_ids: tuple[str, ...]
    card_meta: dict[str, dict[str, Any]]
    country_meta: dict[str, dict[str, Any]]
    country_adjacency: np.ndarray

    @property
    def card_count(self) -> int:
        return len(self.card_ids)

    @property
    def country_count(self) -> int:
        return len(self.country_ids)

    @property
    def region_count(self) -> int:
        return len(REGIONS)


def make_spec(cards: list[dict[str, Any]], countries: list[dict[str, Any]]) -> FeatureSpec:
    country_ids = tuple(country["id"] for country in countries)
    country_index = {country_id: idx for idx, country_id in enumerate(country_ids)}
    adjacency = np.zeros((len(country_ids), len(country_ids)), dtype=np.float32)
    for country in countries:
        src = country_index[country["id"]]
        for neighbour in country.get("neighbours", []) or []:
            if neighbour in country_index:
                dst = country_index[neighbour]
                adjacency[src, dst] = 1.0
                adjacency[dst, src] = 1.0
    return FeatureSpec(
        card_ids=tuple(card["id"] for card in cards),
        country_ids=country_ids,
        card_meta={card["id"]: card for card in cards},
        country_meta={country["id"]: country for country in countries},
        country_adjacency=adjacency,
    )


def encode_observation(obs: dict[str, Any], legal_actions: list[dict[str, Any]], spec: FeatureSpec) -> dict[str, np.ndarray]:
    side = obs["side"]
    card_index = {card_id: idx for idx, card_id in enumerate(spec.card_ids)}
    country_index = {country_id: idx for idx, country_id in enumerate(spec.country_ids)}
    by_country = {country["id"]: country for country in obs["countries"]}

    country_features = encode_countries(obs, by_country, country_index, spec)
    region_features = encode_regions(obs, by_country, spec)
    card_features = encode_cards(obs, legal_actions, card_index, spec)
    event_features = encode_events(obs, side)
    global_features = encode_global(obs, side, spec)

    action_mask = np.zeros((MAX_ACTIONS,), dtype=np.float32)
    action_features = np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.float32)
    for idx, action in enumerate(legal_actions[:MAX_ACTIONS]):
        action_mask[idx] = 1.0
        action_features[idx] = encode_action(action, obs, card_index, country_index, by_country, spec)

    return {
        "global": global_features,
        "events": event_features,
        "countries": country_features,
        "regions": region_features,
        "cards": card_features,
        "country_adjacency": spec.country_adjacency.copy(),
        "action_mask": action_mask,
        "action_features": action_features,
        **encode_history(obs.get("history", [])),
        **encode_card_history(obs.get("card_history", []), spec),
    }


def encode_history(history: list[dict[str, Any]], length: int = HISTORY_LENGTH) -> dict[str, np.ndarray]:
    history_actions = np.zeros((length, ACTION_FEATURES), dtype=np.float32)
    history_sides = np.zeros((length,), dtype=np.float32)
    history_turn_ar = np.zeros((length, 2), dtype=np.float32)
    history_vp_defcon = np.zeros((length, 2), dtype=np.float32)
    history_mask = np.zeros((length,), dtype=np.float32)
    recent = list(history or [])[-length:]
    offset = length - len(recent)
    for idx, item in enumerate(recent, start=offset):
        action_features = item.get("action_features")
        if action_features is not None:
            arr = np.asarray(action_features, dtype=np.float32)
            history_actions[idx, : min(ACTION_FEATURES, arr.shape[0])] = arr[:ACTION_FEATURES]
        side = str(item.get("side") or "")
        history_sides[idx] = side_sign(side) if side in {"us", "ussr"} else 0.0
        history_turn_ar[idx, 0] = safe_div(float(item.get("turn") or 0.0), 10.0)
        history_turn_ar[idx, 1] = safe_div(float(item.get("action_round") or 0.0), 7.0)
        history_vp_defcon[idx, 0] = safe_div(float(item.get("vp") or 0.0), 20.0)
        history_vp_defcon[idx, 1] = safe_div(float(item.get("defcon") or 0.0), 5.0)
        history_mask[idx] = 1.0
    return {
        "history_actions": history_actions,
        "history_sides": history_sides,
        "history_turn_ar": history_turn_ar,
        "history_vp_defcon": history_vp_defcon,
        "history_mask": history_mask,
    }


def encode_card_history(
    card_history: list[dict[str, Any]],
    spec: FeatureSpec,
    length: int = CARD_HISTORY_LENGTH,
) -> dict[str, np.ndarray]:
    card_index = {card_id: idx for idx, card_id in enumerate(spec.card_ids)}
    features = np.zeros((length, CARD_HISTORY_FEATURES), dtype=np.float32)
    mask = np.zeros((length,), dtype=np.float32)
    recent = list(card_history or [])[-length:]
    offset = length - len(recent)
    for idx, item in enumerate(recent, start=offset):
        card_id = str(item.get("card") or "")
        card = spec.card_meta.get(card_id, {})
        owner = str(card.get("player") or card.get("side") or "")
        side = str(item.get("side") or "")
        modes = set(item.get("modes") or [])
        phase = str(item.get("phase") or "").lower()
        vp_before = float(item.get("vp_before") or 0.0)
        vp_after = float(item.get("vp_after", vp_before) or 0.0)
        defcon_before = float(item.get("defcon_before") or 0.0)
        defcon_after = float(item.get("defcon_after", defcon_before) or 0.0)
        card_pos = card_index.get(card_id, -1)

        features[idx, 0] = side_sign(side) if side in {"us", "ussr"} else 0.0
        features[idx, 1] = safe_div(float(item.get("turn") or 0.0), 10.0)
        features[idx, 2] = safe_div(float(item.get("action_round") or 0.0), 7.0)
        features[idx, 3] = safe_div(vp_before, 20.0)
        features[idx, 4] = safe_div(defcon_before, 5.0)
        features[idx, 5] = safe_div(vp_after - vp_before, 20.0)
        features[idx, 6] = safe_div(defcon_after - defcon_before, 5.0)
        features[idx, 7] = 1.0 if card_pos >= 0 else 0.0
        features[idx, 8] = safe_div(card_pos + 1, spec.card_count) if card_pos >= 0 else 0.0
        features[idx, 9] = safe_div(float(card.get("ops", 0)), 4.0)
        features[idx, 10] = 1.0 if owner == "us" else 0.0
        features[idx, 11] = 1.0 if owner == "ussr" else 0.0
        features[idx, 12] = 1.0 if owner == "both" else 0.0
        features[idx, 13] = 1.0 if card.get("scoring") else 0.0
        features[idx, 14] = side_value(owner, side) if side in {"us", "ussr"} else 0.0
        features[idx, 15] = 1.0 if owner in {"us", "ussr"} and owner != side else 0.0
        features[idx, 16] = 1.0 if "headline" in phase or "headline" in modes else 0.0
        features[idx, 17] = 1.0 if "event" in modes else 0.0
        features[idx, 18] = 1.0 if "ops" in modes else 0.0
        features[idx, 19] = 1.0 if "space" in modes else 0.0
        features[idx, 20] = 1.0 if "before_ops" in modes else 0.0
        features[idx, 21] = 1.0 if "after_ops" in modes else 0.0
        features[idx, 22] = 1.0 if "discard" in modes else 0.0
        features[idx, 23] = 1.0 if card_id in DEFCON_RISK_EVENT_CARDS else 0.0
        features[idx, 24] = 1.0 if item.get("removed") else 0.0
        features[idx, 25] = 1.0 if item.get("completed") else 0.0
        mask[idx] = 1.0
    return {
        "card_history": features,
        "card_history_mask": mask,
    }


def encode_global(obs: dict[str, Any], side: str, spec: FeatureSpec) -> np.ndarray:
    features = np.zeros((GLOBAL_FEATURES,), dtype=np.float32)
    opponent = other(side)
    discard_count = len(obs.get("discard", []))
    removed_count = len(obs.get("removed", []))
    china_owner = obs.get("china_owner", "none")
    prompt = str(obs.get("prompt", "")).lower()

    features[0] = side_sign(side)
    features[1] = 1.0 if obs["current_player"] == side else 0.0
    features[2] = safe_div(obs["turn"], 10)
    features[3] = safe_div(obs["action_round"], 7)
    features[4] = safe_div(obs["vp"], 20)
    features[5] = safe_div(obs["defcon"], 5)
    features[6] = safe_div(obs["milops"]["us"], 5)
    features[7] = safe_div(obs["milops"]["ussr"], 5)
    features[8] = safe_div(obs["space"]["us"], 8)
    features[9] = safe_div(obs["space"]["ussr"], 8)
    features[10] = safe_div(obs["hand_count"]["us"], 10)
    features[11] = safe_div(obs["hand_count"]["ussr"], 10)
    features[12] = safe_div(obs["hand_count"][side], 10)
    features[13] = safe_div(obs["hand_count"][opponent], 10)
    features[14] = safe_div(obs["deck_count"], max(1, spec.card_count))
    features[15] = safe_div(discard_count, max(1, spec.card_count))
    features[16] = safe_div(removed_count, max(1, spec.card_count))
    features[17] = china_owner_value(china_owner, side)
    features[18] = 1.0 if obs.get("terminal") else 0.0
    features[19] = 1.0 if "headline" in prompt else 0.0
    features[20] = 1.0 if "ops" in prompt else 0.0
    features[21] = 1.0 if "coup" in prompt else 0.0
    features[22] = 1.0 if "realign" in prompt else 0.0
    features[23] = 1.0 if "place" in prompt or "influence" in prompt else 0.0
    return features


def encode_events(obs: dict[str, Any], side: str) -> np.ndarray:
    events = obs.get("events", {}) or {}
    features = np.zeros((EVENT_FEATURES,), dtype=np.float32)
    for idx, key in enumerate(EVENT_KEYS):
        features[idx] = encode_event_value(events.get(key, 0), side)
    return features


def encode_countries(
    obs: dict[str, Any],
    by_country: dict[str, dict[str, Any]],
    country_index: dict[str, int],
    spec: FeatureSpec,
) -> np.ndarray:
    side = obs["side"]
    opponent = other(side)
    features = np.zeros((spec.country_count, COUNTRY_FEATURES), dtype=np.float32)
    for country_id, idx in country_index.items():
        country = by_country[country_id]
        us = float(country["us"])
        ussr = float(country["ussr"])
        stability = max(1.0, float(country["stability"]))
        region = country.get("region", "")
        control = country.get("control", "none")
        neighbours = country.get("neighbours", []) or []
        neighbour_states = [by_country[n] for n in neighbours if n in by_country]
        self_inf = us if side == "us" else ussr
        opp_inf = ussr if side == "us" else us

        features[idx, 0] = safe_div(us, 10)
        features[idx, 1] = safe_div(ussr, 10)
        features[idx, 2] = safe_div(self_inf, 10)
        features[idx, 3] = safe_div(opp_inf, 10)
        features[idx, 4] = safe_div(us - ussr, 10)
        features[idx, 5] = safe_div(self_inf - opp_inf, 10)
        features[idx, 6] = safe_div(stability, 5)
        features[idx, 7] = 1.0 if country.get("bg") else 0.0
        features[idx, 8] = 1.0 if control == "us" else 0.0
        features[idx, 9] = 1.0 if control == "ussr" else 0.0
        features[idx, 10] = 1.0 if control == "none" else 0.0
        features[idx, 11] = 1.0 if control == side else 0.0
        features[idx, 12] = 1.0 if control == opponent else 0.0
        features[idx, 13] = safe_div(max(0.0, stability - us), 5)
        features[idx, 14] = safe_div(max(0.0, stability - ussr), 5)
        features[idx, 15] = safe_div(max(0.0, stability - self_inf), 5)
        features[idx, 16] = safe_div(max(0.0, stability - opp_inf), 5)
        for ridx, region_id in enumerate(REGIONS):
            features[idx, 17 + ridx] = 1.0 if region == region_id else 0.0
        features[idx, 24] = 1.0 if defcon_blocks(region, obs["defcon"]) else 0.0
        features[idx, 25] = 1.0 if not defcon_blocks(region, obs["defcon"]) else 0.0
        features[idx, 26] = 1.0 if us > 0 else 0.0
        features[idx, 27] = 1.0 if ussr > 0 else 0.0
        features[idx, 28] = 1.0 if self_inf > 0 else 0.0
        features[idx, 29] = 1.0 if opp_inf > 0 else 0.0
        features[idx, 30] = 1.0 if any(n.get("control") == "us" for n in neighbour_states) else 0.0
        features[idx, 31] = 1.0 if any(n.get("control") == "ussr" for n in neighbour_states) else 0.0
        features[idx, 32] = 1.0 if any(n.get("control") == side for n in neighbour_states) else 0.0
        features[idx, 33] = 1.0 if any(n.get("control") == opponent for n in neighbour_states) else 0.0
    return features


def encode_regions(obs: dict[str, Any], by_country: dict[str, dict[str, Any]], spec: FeatureSpec) -> np.ndarray:
    hand = set(obs.get("hand", []))
    discard = set(obs.get("discard", []))
    removed = set(obs.get("removed", []))
    features = np.zeros((spec.region_count, REGION_FEATURES), dtype=np.float32)
    for ridx, region in enumerate(REGIONS):
        countries = [c for c in by_country.values() if c.get("region") == region]
        total = max(1, len(countries))
        bg_total = max(1, sum(1 for c in countries if c.get("bg")))
        us_control = [c for c in countries if c.get("control") == "us"]
        ussr_control = [c for c in countries if c.get("control") == "ussr"]
        us_bg = [c for c in us_control if c.get("bg")]
        ussr_bg = [c for c in ussr_control if c.get("bg")]
        scoring_card = scoring_card_for_region(region)

        features[ridx, 0] = safe_div(len(countries), 20)
        features[ridx, 1] = safe_div(bg_total, 10)
        features[ridx, 2] = safe_div(len(us_control), total)
        features[ridx, 3] = safe_div(len(ussr_control), total)
        features[ridx, 4] = safe_div(len(us_bg), bg_total)
        features[ridx, 5] = safe_div(len(ussr_bg), bg_total)
        features[ridx, 6] = 1.0 if us_control else 0.0
        features[ridx, 7] = 1.0 if ussr_control else 0.0
        features[ridx, 8] = 1.0 if len(us_control) > len(ussr_control) else 0.0
        features[ridx, 9] = 1.0 if len(ussr_control) > len(us_control) else 0.0
        features[ridx, 10] = 1.0 if len(us_bg) > len(ussr_bg) else 0.0
        features[ridx, 11] = 1.0 if len(ussr_bg) > len(us_bg) else 0.0
        features[ridx, 12] = 1.0 if len(us_control) == len(countries) and countries else 0.0
        features[ridx, 13] = 1.0 if len(ussr_control) == len(countries) and countries else 0.0
        features[ridx, 14] = 1.0 if features[ridx, 6] and features[ridx, 8] and features[ridx, 10] else 0.0
        features[ridx, 15] = 1.0 if features[ridx, 7] and features[ridx, 9] and features[ridx, 11] else 0.0
        features[ridx, 16] = 1.0 if scoring_card in hand else 0.0
        features[ridx, 17] = 1.0 if scoring_card in discard else 0.0
        features[ridx, 18] = 1.0 if scoring_card in removed else 0.0
        features[ridx, 19] = 1.0 if scoring_card else 0.0
    return features


def encode_cards(
    obs: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    card_index: dict[str, int],
    spec: FeatureSpec,
) -> np.ndarray:
    side = obs["side"]
    hand = set(obs.get("hand", []))
    discard = set(obs.get("discard", []))
    removed = set(obs.get("removed", []))
    legal_values = {str(action.get("value", "")) for action in legal_actions}
    prompt = str(obs.get("prompt", "")).lower()
    features = np.zeros((spec.card_count, CARD_FEATURES), dtype=np.float32)
    for card_id, idx in card_index.items():
        card = spec.card_meta[card_id]
        player = card.get("player") or card.get("side")
        era = card.get("era", "special")
        ops = float(card.get("ops", 0))
        in_hand = card_id in hand

        features[idx, 0] = safe_div(idx + 1, spec.card_count)
        features[idx, 1] = 1.0 if player == "us" else 0.0
        features[idx, 2] = 1.0 if player == "ussr" else 0.0
        features[idx, 3] = 1.0 if player == "both" else 0.0
        features[idx, 4] = 1.0 if card.get("scoring") else 0.0
        features[idx, 5] = safe_div(ops, 4)
        features[idx, 6] = 1.0 if era == "early" else 0.0
        features[idx, 7] = 1.0 if era == "mid" else 0.0
        features[idx, 8] = 1.0 if era == "late" else 0.0
        features[idx, 9] = 1.0 if era == "special" else 0.0
        features[idx, 10] = 1.0 if card.get("recurring") else 0.0
        features[idx, 11] = 0.0 if card.get("recurring") else 1.0
        features[idx, 12] = 1.0 if card_id == "china" else 0.0
        features[idx, 13] = 1.0 if in_hand else 0.0
        features[idx, 14] = 1.0 if card_id in discard else 0.0
        features[idx, 15] = 1.0 if card_id in removed else 0.0
        features[idx, 16] = 1.0 if card_id in legal_values else 0.0
        features[idx, 17] = 1.0 if in_hand and player not in (side, "both") and not card.get("scoring") else 0.0
        features[idx, 18] = 1.0 if in_hand and player == side else 0.0
        features[idx, 19] = 1.0 if in_hand and card.get("scoring") else 0.0
        features[idx, 20] = 1.0 if in_hand and "space" in prompt else 0.0
        features[idx, 21] = 1.0 if card_id not in hand and card_id not in discard and card_id not in removed else 0.0
        features[idx, 22] = side_value(player, side)
        features[idx, 23] = 1.0 if in_hand and ops >= 3 else 0.0
        features[idx, 24] = 1.0 if card_id in DEFCON_LOWERING_EVENT_CARDS else 0.0
        features[idx, 25] = 1.0 if card_id in DEFCON_FORCED_COUP_EVENT_CARDS else 0.0
        features[idx, 26] = 1.0 if card_id in DEFCON_RANDOM_EVENT_TRIGGER_CARDS else 0.0
        features[idx, 27] = 1.0 if in_hand and obs["defcon"] <= 2 and card_id in DEFCON_RISK_EVENT_CARDS else 0.0
        features[idx, 28] = 1.0 if side == "us" and in_hand else 0.0
        features[idx, 29] = 1.0 if side == "ussr" and in_hand else 0.0
    return features


def encode_action(
    action: dict[str, Any],
    obs: dict[str, Any],
    card_index: dict[str, int],
    country_index: dict[str, int],
    by_country: dict[str, dict[str, Any]],
    spec: FeatureSpec,
) -> np.ndarray:
    features = np.zeros((ACTION_FEATURES,), dtype=np.float32)
    action_type = action.get("type", "")
    decision = action.get("decision", "")
    value = str(action.get("value", ""))
    prompt = str(action.get("prompt") or obs.get("prompt", "")).lower()
    card_id = action.get("card") if action.get("card") in card_index else value if value in card_index else None
    country_id = action.get("country") if action.get("country") in country_index else value if value in country_index else None

    features[0] = 1.0 if action_type == "saito_choice" else 0.0
    features[1] = 1.0 if action_type == "saito_dom" else 0.0
    features[2] = 1.0 if decision == "list" else 0.0
    features[3] = 1.0 if decision == "option" else 0.0
    features[4] = 1.0 if decision == "country_click" else 0.0
    features[5] = 1.0 if decision == "country_mouseup" else 0.0
    features[6] = 1.0 if card_id else 0.0
    features[7] = 1.0 if country_id else 0.0
    if card_id:
        card = spec.card_meta[card_id]
        features[8] = safe_div(card_index[card_id] + 1, spec.card_count)
        features[9] = safe_div(float(card.get("ops", 0)), 4)
        features[10] = side_value(card.get("player") or card.get("side"), obs["side"])
        features[11] = 1.0 if card.get("scoring") else 0.0
    if country_id:
        country = by_country[country_id]
        features[12] = safe_div(country_index[country_id] + 1, spec.country_count)
        features[13] = 1.0 if country.get("bg") else 0.0
        features[14] = safe_div(float(country.get("stability", 0)), 5)
        features[15] = safe_div(float(country.get("us", 0)), 10)
        features[16] = safe_div(float(country.get("ussr", 0)), 10)
        for ridx, region in enumerate(REGIONS):
            features[17 + ridx] = 1.0 if country.get("region") == region else 0.0
    features[24] = 1.0 if value == "place" or "place" in prompt or "influence" in prompt else 0.0
    features[25] = 1.0 if value == "coup" or "coup" in prompt else 0.0
    features[26] = 1.0 if value == "realign" or "realign" in prompt else 0.0
    features[27] = 1.0 if value == "space" or "space" in prompt else 0.0
    features[28] = 1.0 if "headline" in prompt else 0.0
    features[29] = 1.0 if "discard" in prompt else 0.0
    features[30] = 1.0 if "scoring" in prompt or (card_id and spec.card_meta[card_id].get("scoring")) else 0.0
    features[31] = 1.0 if "event" in prompt or value == "event" else 0.0
    features[32] = 1.0 if "ops" in prompt or value in {"before_ops", "ops"} else 0.0
    features[33] = 1.0 if value in {"before_ops", "event", "place", "coup", "realign", "space"} else 0.0
    features[34] = 1.0 if value.startswith("cancel") else 0.0
    features[35] = 1.0 if "defcon" in prompt else 0.0
    features[36] = 1.0 if "confirm" in prompt or "sure" in prompt else 0.0
    features[37] = 1.0 if "choose" in prompt or "pick" in prompt else 0.0
    features[38] = 1.0 if "opponent" in prompt else 0.0
    features[39] = 1.0
    features[HEURISTIC_PRIOR_FEATURE] = float(np.tanh(score_action({**action, "prompt": prompt}, obs=obs) / 30.0))
    if card_id:
        features[41] = 1.0 if card_id in DEFCON_LOWERING_EVENT_CARDS else 0.0
        features[42] = 1.0 if card_id in DEFCON_FORCED_COUP_EVENT_CARDS else 0.0
        features[43] = 1.0 if card_id in DEFCON_RANDOM_EVENT_TRIGGER_CARDS else 0.0
        features[44] = 1.0 if obs["defcon"] <= 2 and card_id in DEFCON_RISK_EVENT_CARDS else 0.0
    return features


def scoring_card_for_region(region: str) -> str:
    return {
        "europe": "europe",
        "asia": "asia",
        "mideast": "mideast",
        "africa": "africa",
        "camerica": "camerica",
        "samerica": "samerica",
        "seasia": "seasia",
    }.get(region, "")


def defcon_blocks(region: str, defcon: int | float) -> bool:
    if region == "europe":
        return defcon < 5
    if region in ("asia", "seasia"):
        return defcon < 4
    if region == "mideast":
        return defcon < 3
    return False


def other(side: str) -> str:
    return "ussr" if side == "us" else "us"


def side_sign(side: str) -> float:
    return 1.0 if side == "us" else -1.0


def side_value(side: str | None, perspective: str) -> float:
    if side == perspective:
        return 1.0
    if side in ("us", "ussr"):
        return -1.0
    return 0.0


def china_owner_value(owner: str, perspective: str) -> float:
    if owner == "none":
        return 0.0
    return 1.0 if owner == perspective else -1.0


def encode_event_value(value: Any, perspective: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(np.clip(value, -5, 5)) / 5.0
    if isinstance(value, str):
        if value == perspective:
            return 1.0
        if value in ("us", "ussr"):
            return -1.0
        return 1.0 if value else 0.0
    return 0.0


def safe_div(value: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float(value) / float(denom)
