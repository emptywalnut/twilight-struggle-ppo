from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TAG_RE = re.compile(r"<[^>]+>")


def load_record(path: Path, record: int = -1) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no JSONL records in {path}")
    return records[record]


def format_record(record: dict[str, Any]) -> str:
    card_names = card_name_map(record)
    lines: list[str] = []
    run_id = record.get("episode_index", "?")
    seed = record.get("seed", "?")
    preset = record.get("start", {}).get("preset") or record.get("saito", {}).get("preset", {})
    preset_id = preset.get("id", "unknown")
    lines.append(f"Run {run_id} starts: seed={seed}, ruleset={preset_id}")

    start = record.get("start") or {}
    if start.get("hands"):
        for side in ("us", "ussr"):
            hand = list(start["hands"].get(side, []))
            lines.append(f"  {side.upper()} get {cards_text(hand, card_names)}; have {len(hand)} in hand")
    for line in initial_influence_lines(start):
        lines.append(f"  {line}")

    actions = record.get("actions", [])
    if not actions:
        append_result(lines, record)
        return "\n".join(lines)

    current_turn: int | None = None
    printed_headline: set[int] = set()
    for group in group_actions(actions):
        turn = int(group[0].get("turn") or group[0].get("before", {}).get("turn") or 0)
        if turn != current_turn:
            current_turn = turn
            snap = group[0].get("debug_before") or {}
            state = state_text(snap)
            lines.append("")
            lines.append(f"Turn {turn} starts{state}")

        phase = str(group[0].get("before", {}).get("phase") or "")
        if turn not in printed_headline and not phase.startswith("setup_"):
            headline = headline_text(actions, turn, card_names)
            lines.append(f"  Headline: {headline}")
            printed_headline.add(turn)

        lines.append("  " + action_round_text(group, card_names))
        lines.extend("    " + line for line in detail_lines(group))
        for deal_line in deal_lines(group, card_names):
            lines.append("  " + deal_line)

    append_result(lines, record)
    return "\n".join(lines)


def group_actions(actions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    last_key: tuple[Any, Any, Any, Any] | None = None
    for action in actions:
        phase = action.get("before", {}).get("phase")
        key = (action.get("turn"), action.get("action_round"), action.get("side"), phase)
        if key != last_key:
            groups.append([])
            last_key = key
        groups[-1].append(action)
    return groups


def headline_text(actions: list[dict[str, Any]], turn: int, card_names: dict[str, str]) -> str:
    headline_cards: dict[str, str] = {}
    for item in actions:
        if item.get("turn") != turn:
            continue
        before = item.get("before") or {}
        after = item.get("after") or {}
        action = item.get("action") or {}
        blob = " ".join(
            str(value)
            for value in [
                before.get("prompt"),
                after.get("prompt"),
                action.get("prompt"),
            ]
            if value
        ).lower()
        if "headline" not in blob:
            continue
        if action.get("decision") == "list" and action.get("value") in card_names:
            headline_cards[item.get("side", "?")] = card_names[action["value"]]
    if headline_cards:
        ussr = headline_cards.get("ussr", "?")
        us = headline_cards.get("us", "?")
        return f"USSR {ussr}; US {us}"
    return "not present in this bridge log"


def action_round_text(group: list[dict[str, Any]], card_names: dict[str, str]) -> str:
    first = group[0]
    side = str(first.get("side", "?")).upper()
    turn = first.get("turn", "?")
    raw_ar = int(first.get("action_round") or 0)
    turn_ar = normalize_action_round(int(turn or 0), raw_ar)
    phase = str(first.get("before", {}).get("phase") or "")
    if phase == "setup_us_bonus":
        actor_label = f"{side} setup bonus"
        display_ar = raw_ar
    elif phase.startswith("setup_"):
        actor_label = f"{side} setup"
        display_ar = raw_ar
    elif phase.startswith("headline"):
        actor_label = f"{side} headline"
        display_ar = 0
    else:
        side_ar = turn_ar + 1
        display_ar = side_ar
        active_side = str(first.get("before", {}).get("current_player") or first.get("side", "?")).upper()
        actor_label = f"{active_side} AR{side_ar}"
        if side in {"US", "USSR"} and side != active_side:
            actor_label = f"{side} response during {active_side} AR{side_ar}"
    parts: list[str] = []
    for item in group:
        action = item.get("action", {})
        decision = action.get("decision", "")
        value = action.get("value", "")
        label = clean(action.get("label") or value)
        if decision == "list" and value in card_names:
            text = card_names[value]
        elif decision == "country_click":
            text = label
        elif decision == "option":
            text = label
        else:
            text = label or clean(decision)
        if text:
            parts.append(text)

    after = group[-1].get("debug_after") or {}
    hand_count = len(after.get("hands", {}).get(first.get("side"), [])) if after.get("hands") else "?"
    detail = " -> ".join(parts) if parts else "(no action detail)"
    state = state_text(after)
    return f"{actor_label} (T{turn}.{display_ar:02d}): {detail}; have {hand_count} in hand{state}"


def normalize_action_round(turn: int, raw_ar: int) -> int:
    """Convert old cumulative bridge AR counters into per-turn Saito counters."""
    if turn <= 1:
        return raw_ar
    previous = 0
    for ts_turn in range(1, turn):
        previous += 6 if ts_turn <= 3 else 7
    normalized = raw_ar - previous
    rounds_this_turn = 6 if turn <= 3 else 7
    if 0 <= normalized < rounds_this_turn:
        return normalized
    return raw_ar


def deal_lines(group: list[dict[str, Any]], card_names: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for item in group:
        before = item.get("debug_before") or {}
        after = item.get("debug_after") or {}
        before_hands = before.get("hands") or {}
        after_hands = after.get("hands") or {}
        if not before_hands or not after_hands:
            continue
        for side in ("us", "ussr"):
            gained = list_delta(after_hands.get(side, []), before_hands.get(side, []))
            if gained:
                have = len(after_hands.get(side, []))
                lines.append(f"{side.upper()} get {cards_text(gained, card_names)}; have {have} in hand")
    return lines


def detail_lines(group: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in group:
        state_delta = item.get("state_delta") or {}
        vp_delta = state_delta.get("vp")
        if vp_delta:
            before = vp_delta.get("before")
            after = vp_delta.get("after")
            try:
                change = int(after) - int(before)
                sign = "+" if change > 0 else ""
                lines.append(f"VP: {before} -> {after} ({sign}{change})")
            except (TypeError, ValueError):
                lines.append(f"VP: {before} -> {after}")
        defcon_delta = state_delta.get("defcon")
        if defcon_delta:
            lines.append(f"DEFCON: {defcon_delta.get('before')} -> {defcon_delta.get('after')}")
        for country in state_delta.get("countries") or []:
            control = ""
            if country.get("control_before") != country.get("control_after"):
                control = f", control {country.get('control_before')} -> {country.get('control_after')}"
            lines.append(
                f"Influence: {country.get('name')} "
                f"US {country.get('us_before')} -> {country.get('us_after')}, "
                f"USSR {country.get('ussr_before')} -> {country.get('ussr_after')}{control}"
            )
        for entry in item.get("saito_log_delta") or []:
            text = clean(entry)
            if text:
                lines.append(f"Log: {text}")
        if item.get("bridge_error"):
            lines.append(f"Bridge error: {clean(item['bridge_error'])}")
    return lines


def initial_influence_lines(start: dict[str, Any]) -> list[str]:
    countries = start.get("countries") or []
    if not countries:
        return []
    us = []
    ussr = []
    for country in countries:
        name = country.get("name") or country.get("id")
        us_inf = int(country.get("us") or 0)
        ussr_inf = int(country.get("ussr") or 0)
        if us_inf:
            us.append(f"{name} {us_inf}")
        if ussr_inf:
            ussr.append(f"{name} {ussr_inf}")
    lines = []
    if us:
        lines.append(f"Initial US influence: {', '.join(us)}")
    if ussr:
        lines.append(f"Initial USSR influence: {', '.join(ussr)}")
    return lines


def list_delta(after: list[str], before: list[str]) -> list[str]:
    before_counts = Counter(before)
    gained: list[str] = []
    for item in after:
        if before_counts[item] > 0:
            before_counts[item] -= 1
        else:
            gained.append(item)
    return gained


def append_result(lines: list[str], record: dict[str, Any]) -> None:
    saito = record.get("saito", {})
    winner = saito.get("winner")
    if winner:
        lines.append("")
        lines.append(
            f"Game over: winner={winner.upper()}, reason={saito.get('terminal_reason')}, "
            f"VP={saito.get('vp')}, DEFCON={saito.get('defcon')}, steps={record.get('steps')}"
        )
    else:
        lines.append("")
        lines.append(
            f"Log ends: kind={record.get('kind')}, turn={saito.get('turn')}, "
            f"action_round={saito.get('action_round')}, VP={saito.get('vp')}, "
            f"DEFCON={saito.get('defcon')}, steps={record.get('steps')}"
        )


def card_name_map(record: dict[str, Any]) -> dict[str, str]:
    cards = record.get("cards") or []
    return {card["id"]: card.get("name", card["id"]) for card in cards if "id" in card}


def cards_text(cards: list[str], card_names: dict[str, str]) -> str:
    if not cards:
        return "nothing"
    return ", ".join(card_names.get(card, card) for card in cards)


def state_text(snapshot: dict[str, Any]) -> str:
    if not snapshot:
        return ""
    return f" [VP={snapshot.get('vp')}, DEFCON={snapshot.get('defcon')}, deck={len(snapshot.get('deck', []))}]"


def clean(value: Any) -> str:
    return TAG_RE.sub("", str(value or "")).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format a Twilight Struggle JSONL game log.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--record", type=int, default=-1, help="JSONL record index, default last")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = format_record(load_record(args.log, args.record))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
