from __future__ import annotations

import random
from typing import Any


US_OPENING_PROFILES = (
    {"westgermany": 3, "france": 3, "italy": 2, "iran": 2},
    {"westgermany": 4, "france": 2, "italy": 2, "iran": 2},
    {"westgermany": 4, "france": 0, "italy": 4, "iran": 2},
)

USSR_OPENING_PROFILES = (
    {"eastgermany": 4, "poland": 4, "austria": 1},
    {"eastgermany": 4, "poland": 5},
)

DEFCON_LOWERING_EVENT_CARDS = frozenset({"duckandcover", "kal007", "wwby"})
DEFCON_FORCED_COUP_EVENT_CARDS = frozenset({"ortega"})
DEFCON_RANDOM_EVENT_TRIGGER_CARDS = frozenset({"fiveyearplan"})
DEFCON_RISK_EVENT_CARDS = (
    DEFCON_LOWERING_EVENT_CARDS
    | DEFCON_FORCED_COUP_EVENT_CARDS
    | DEFCON_RANDOM_EVENT_TRIGGER_CARDS
    | frozenset({"missileenvy"})
)


def choose_heuristic_action(legal_actions: list[dict[str, Any]], rng: random.Random | None = None, obs: dict[str, Any] | None = None) -> int:
    if not legal_actions:
        raise ValueError("no legal actions")
    rng = rng or random.Random()
    scores = [score_action(action, obs=obs) for action in legal_actions]
    best_score = max(scores)
    candidates = [idx for idx, score in enumerate(scores) if score == best_score]
    return rng.choice(candidates)


def score_action(action: dict[str, Any], obs: dict[str, Any] | None = None) -> float:
    value = str(action.get("value", "")).lower()
    label = str(action.get("label", value)).lower()
    prompt = str(action.get("prompt", "")).lower()
    decision = str(action.get("decision", "")).lower()

    score = 0.0
    if decision == "list":
        score += 5
    if decision == "option":
        score += 2
    if "headline" in prompt:
        score += headline_card_score(value, label)
    elif "pick a card" in prompt:
        score += play_card_score(value, label)

    option_scores = {
        "event": 25,
        "score": 25,
        "ops": 15,
        "place": 16,
        "coup": 18,
        "realign": 4,
        "space": 8,
        "before_ops": 12,
        "after_ops": 14,
        "playevent": 20,
    }
    score += option_scores.get(value, 0)

    if value.startswith("cancel") or value in {"nope", "skip", "skipturn"}:
        score -= 30
    if "discard" in prompt and "scoring" not in label:
        score += 2
    score += defcon_risk_score(action, obs, value, label, prompt, decision)
    if decision in {"country_click", "country_mouseup"}:
        score += country_score(value, prompt)
        score += opening_setup_score(value, prompt, obs)
    return score


def defcon_risk_score(
    action: dict[str, Any],
    obs: dict[str, Any] | None,
    value: str,
    label: str,
    prompt: str,
    decision: str,
) -> float:
    if not obs:
        return 0.0
    try:
        defcon = int(obs.get("defcon") or 5)
    except (TypeError, ValueError):
        defcon = 5
    card_id = risk_card_id(action, value, label, prompt)
    if defcon > 2 or card_id not in DEFCON_RISK_EVENT_CARDS:
        return 0.0
    if value == "space":
        return 35.0
    if decision == "list" and "headline" in prompt and card_id in DEFCON_LOWERING_EVENT_CARDS:
        return -60.0
    if decision == "list" and "pick a card" in prompt:
        return -18.0
    if value in {"event", "ops", "before_ops", "after_ops", "coup"}:
        return -70.0
    return 0.0


def risk_card_id(action: dict[str, Any], value: str, label: str, prompt: str) -> str:
    for candidate in (action.get("card"), value, label, prompt):
        normalized = normalize_card_text(candidate)
        if normalized in DEFCON_RISK_EVENT_CARDS:
            return normalized
    if "duckandcover" in normalize_card_text(prompt) or "duckandcover" in normalize_card_text(label):
        return "duckandcover"
    if "kal007" in normalize_card_text(prompt) or "kal007" in normalize_card_text(label) or "sovietsshootdownkal007" in normalize_card_text(prompt + label):
        return "kal007"
    if "wewillburyyou" in normalize_card_text(prompt + label):
        return "wwby"
    if "ortegaelectedinnicaragua" in normalize_card_text(prompt + label):
        return "ortega"
    if "fiveyearplan" in normalize_card_text(prompt + label):
        return "fiveyearplan"
    if "missileenvy" in normalize_card_text(prompt + label):
        return "missileenvy"
    return ""


def normalize_card_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def headline_card_score(value: str, label: str) -> float:
    score = play_card_score(value, label)
    if value == "defectors":
        score += 18
    if "scoring" in label:
        score -= 8
    return score


def play_card_score(value: str, label: str) -> float:
    score = 0.0
    if "scoring" in label:
        score += 30
    score += {
        "redscare": 20,
        "containment": 18,
        "brezhnev": 18,
        "decolonization": 16,
        "destalinization": 16,
        "duckandcover": 14,
        "nucleartestban": 14,
        "marshall": 14,
        "warsawpact": 14,
        "usjapan": 12,
        "defectors": 10,
        "unintervention": 8,
    }.get(value, 0)
    return score


def country_score(value: str, prompt: str) -> float:
    battlegrounds = {
        "france",
        "italy",
        "westgermany",
        "eastgermany",
        "poland",
        "iran",
        "iraq",
        "israel",
        "egypt",
        "libya",
        "saudiarabia",
        "pakistan",
        "india",
        "thailand",
        "japan",
        "northkorea",
        "southkorea",
        "panama",
        "mexico",
        "cuba",
        "venezuela",
        "brazil",
        "chile",
        "argentina",
        "angola",
        "zaire",
        "nigeria",
        "southafrica",
        "algeria",
    }
    score = 5.0
    if value in battlegrounds:
        score += 12
    if "coup" in prompt and value in {"iran", "panama", "thailand", "zaire", "angola"}:
        score += 8
    if "place" in prompt and value in {"france", "thailand", "pakistan", "iran", "westgermany"}:
        score += 6
    return score


def opening_setup_score(value: str, prompt: str, obs: dict[str, Any] | None) -> float:
    if not obs:
        return 0.0
    phase = str(obs.get("phase", "")).lower()
    if not phase.startswith("setup_") and "initial placement" not in prompt and "optional +2" not in prompt:
        return 0.0

    countries = {country.get("id"): country for country in obs.get("countries", [])}
    if phase == "setup_ussr" or "ussr initial placement" in prompt:
        return profile_setup_score(value, countries, USSR_OPENING_PROFILES, "ussr")
    if phase in {"setup_us", "setup_us_bonus"} or "us initial placement" in prompt or "us optional +2" in prompt:
        if (phase == "setup_us_bonus" or "us optional +2" in prompt) and value == "iran" and int(countries.get("iran", {}).get("us", 0) or 0) < 2:
            return 90.0
        return profile_setup_score(value, countries, US_OPENING_PROFILES, "us")
    return 0.0


def profile_setup_score(
    value: str,
    countries: dict[str, dict[str, Any]],
    profiles: tuple[dict[str, int], ...],
    side: str,
) -> float:
    if value not in {country for profile in profiles for country in profile}:
        return -18.0
    country = countries.get(value, {})
    current = int(country.get(side, 0) or 0)
    wanted_scores = []
    invalid_scores = []
    for profile in profiles:
        overshoot = sum(
            max(0, int(countries.get(country_id, {}).get(side, 0) or 0) - target_inf)
            for country_id, target_inf in profile.items()
        )
        if overshoot > 0:
            invalid_scores.append(-20.0 - overshoot * 10.0)
            continue
        target = profile.get(value, 0)
        deficit = target - current
        if deficit <= 0:
            wanted_scores.append(-12.0)
            continue
        remaining = sum(max(0, target_inf - int(countries.get(country_id, {}).get(side, 0) or 0)) for country_id, target_inf in profile.items())
        wanted_scores.append(35.0 + deficit * 10.0 + remaining)
    return max(wanted_scores or invalid_scores)
