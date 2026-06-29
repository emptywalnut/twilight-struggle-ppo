#!/usr/bin/env python3
"""Audit Saito Twilight gameplay stages against the headless bridge.

This is intentionally conservative: commands that the bridge pops as
"headless noops" are treated as bypassed, not covered, when Saito uses them
for real lifecycle work.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAITO = ROOT / "third_party/saito-lite-rust-materialized/mods/twilight/twilight.js"
BRIDGE = ROOT / "bridge/saito_bridge.js"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def saito_handlers(source: str) -> list[str]:
    pattern = re.compile(r"mv\[0\]\s*(?:===|==)\s*[\"']([^\"']+)")
    return sorted(set(pattern.findall(source)))


def init_queue(source: str) -> list[str]:
    match = re.search(r"if \(!this\.game\.state\) \{(?P<body>.*?)if \(this\.is_testing", source, re.S)
    if not match:
        return []
    queue = []
    for item in re.findall(r"this\.game\.queue\.push\([\"']([^\"']+)", match.group("body")):
        queue.append(item.split("\\t", 1)[0].split("\t", 1)[0])
    return queue


def bridge_noops(source: str) -> list[str]:
    match = re.search(r"const headlessNoops = new Set\(\[(?P<body>.*?)\]\);", source, re.S)
    if not match:
        return []
    return sorted(set(re.findall(r"[\"']([^\"']+)[\"']", match.group("body"))))


def bridge_phases(source: str) -> list[str]:
    return sorted(set(re.findall(r"this\.phase\s*=\s*[\"']([^\"']+)[\"']", source)))


def status_line(status: str) -> str:
    return {
        "covered": "COVERED",
        "partial": "PARTIAL",
        "gap": "GAP",
    }[status]


STAGES = [
    {
        "stage": "Startup deck construction",
        "saito": "initializeGame pushes DECK/XOR/ENCRYPT/SHUFFLE/DEAL/READY, then placement and round.",
        "bridge": "Bridge consumes Saito startup queue commands and implements the headless deck transport for DECK/SHUFFLE/DEAL.",
        "status": "partial",
        "gap": "Network encryption/XOR semantics are intentionally collapsed to local deterministic deck transport.",
    },
    {
        "stage": "Initial placement",
        "saito": "placement(USSR), placement(US), placement_bonus(US +2) through playerPlaceInitialInfluence/playerPlaceBonusInfluence.",
        "bridge": "Saito placement queue commands are converted into setup_ussr/setup_us/setup_us_bonus legal actions, then popped only after all influence is placed.",
        "status": "partial",
        "gap": "The setup stage is queue-aligned, but still uses bridge-native setup actions instead of Saito's browser placement handlers.",
    },
    {
        "stage": "Round start and end-round cleanup",
        "saito": "round settles outstanding VP, resets round/event flags, handles NORAD and extra-turn bonuses, calls endRound, builds headline/play/turn/deal/reshuffle queues.",
        "bridge": "Bridge now lets Saito's round handler run and processes the resulting headline/play/turn/deal/reshuffle queue.",
        "status": "partial",
        "gap": "Round lifecycle is no longer bypassed, but all round-triggered prompt branches still need rollout coverage.",
    },
    {
        "stage": "Headline selection",
        "saito": "headline supports simultaneous blind pick and Man in Earth Orbit headline-peeking sequence.",
        "bridge": "headline_ussr/headline_us pick cards sequentially and queue each selected card as an event.",
        "status": "partial",
        "gap": "Normal card choice exists, but simultaneous-pick protocol and headline-peeking are not wired.",
    },
    {
        "stage": "Headline resolution",
        "saito": "headline6/headline7 order by ops, handles Defectors, discards/removes through queue.",
        "bridge": "orders by ops, handles basic Defectors, calls Saito event handlers, then finalizes cards manually.",
        "status": "partial",
        "gap": "Catch-all fallback was removed; unresolved headline decisions now raise diagnostics. Simultaneous blind-pick protocol is still simplified.",
    },
    {
        "stage": "Action round card play",
        "saito": "round queues play/turn pairs; play resets turn flags, settles VP, handles NORAD, and calls playMove/playerTurn.",
        "bridge": "Bridge now drives Saito play/turn queue commands and exposes decisions from Saito playMove/playerTurn.",
        "status": "partial",
        "gap": "Core queue side effects run through Saito; per-card prompt coverage still needs exhaustive rollout tests.",
    },
    {
        "stage": "Operations: influence/coup/realign/space",
        "saito": "playerTurn/playOps expose browser choices and queue ops/coup/realign/space commands.",
        "bridge": "adapter captures list/options/DOM country handlers and feeds choices back to Saito.",
        "status": "partial",
        "gap": "Core interactions are wired, but legality filters and bounded placement rules are bridge-side and still prompt-specific.",
    },
    {
        "stage": "Scoring cards",
        "saito": "scoring cards run through playEvent/scoreRegion and queue VP/milops/end conditions.",
        "bridge": "scoring card choices invoke Saito playEvent through the event queue.",
        "status": "partial",
        "gap": "Region scoring itself uses Saito, but end-turn VP reward/logging and full round cleanup are not yet equivalent.",
    },
    {
        "stage": "Event sub-decisions",
        "saito": "many events stop on updateStatusWithOptions/updateStatusAndListCards/DOM handlers or custom queue commands.",
        "bridge": "generic adapter captures many options and country clicks; some command actor retries exist.",
        "status": "gap",
        "gap": "Not exhaustive. Current known blocker: event us summit can leave the queue unadvanced; many mid/late event custom prompts need coverage tests.",
    },
    {
        "stage": "Deck/deal/reshuffle by turn",
        "saito": "deal/reshuffle/sharehandsize/dynamic_deck_management add era cards, reshuffle discards, and preserve held-card counts.",
        "bridge": "Bridge processes Saito deal/reshuffle/sharehandsize commands with a local deck/discard transport.",
        "status": "partial",
        "gap": "Transport is local rather than encrypted/networked; dynamic community deck commands are only audited as they appear under the selected optional preset.",
    },
    {
        "stage": "Terminal/final scoring",
        "saito": "final_scoring and sendGameOverTransaction determine winner after round 10 or instant conditions.",
        "bridge": "Bridge lets Saito final_scoring run and also checks DEFCON/+/-20 VP terminal conditions after steps.",
        "status": "partial",
        "gap": "Final scoring path needs complete-game rollout verification.",
    },
    {
        "stage": "Game logs",
        "saito": "updateLog records event/action text during queue execution.",
        "bridge": "raw JSONL plus sorted text logs are written from bridge gameLog.",
        "status": "covered",
        "gap": "Logs exist, but they reflect bridge stage bypasses where lifecycle is manual.",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true", help="print a Markdown report")
    args = parser.parse_args()

    saito = read(SAITO)
    bridge = read(BRIDGE)
    handlers = saito_handlers(saito)
    noops = bridge_noops(bridge)
    phases = bridge_phases(bridge)
    startup = init_queue(saito)
    bypassed_real_commands = sorted(set(noops) & {"round", "headline", "deal", "reshuffle", "clear"})
    unhandled_custom = sorted(
        set(handlers)
        - set(noops)
        - {
            "event",
            "ops",
            "card",
            "discard",
            "space",
            "coup",
            "realign",
            "vp",
            "milops",
            "place",
            "remove",
            "defcon",
            "resolve",
            "setvar",
            "stability",
            "move",
            "war",
            "wargames",
        }
    )

    print("# Saito Bridge Stage Coverage Audit")
    print()
    print(f"- Saito handlers found: {len(handlers)}")
    print(f"- Saito startup queue: {', '.join(startup) if startup else 'not detected'}")
    print(f"- Bridge phases: {', '.join(phases)}")
    print(f"- Bridge no-op/bypassed commands: {', '.join(noops)}")
    print(f"- Lifecycle commands currently bypassed as no-ops: {', '.join(bypassed_real_commands) or 'none'}")
    print()
    print("| Stage | Status | Saito stage | Bridge stage | Gap |")
    print("| --- | --- | --- | --- | --- |")
    for item in STAGES:
        print(
            f"| {item['stage']} | {status_line(item['status'])} | "
            f"{item['saito']} | {item['bridge']} | {item['gap']} |"
        )
    print()
    print("## Saito Queue Commands Needing Explicit Audit")
    print()
    print(", ".join(unhandled_custom) if unhandled_custom else "None")
    print()
    print("## Verdict")
    print()
    if any(item["status"] == "gap" for item in STAGES):
        print("The bridge is not yet stage-complete. Do not treat PPO training as rules-faithful until the GAP rows are closed.")
        return 1
    print("No stage gaps detected by this audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
