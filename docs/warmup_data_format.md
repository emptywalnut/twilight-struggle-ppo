# Warmup Gameplay Data Format

This document defines the gameplay-log format needed for supervised warmup training before PPO. The goal is to convert expert or scripted full-game logs into action-level training samples:

```text
observation_before_action + legal_action_list -> expert_action_index
```

The safest format is JSONL. A text log is acceptable as a human companion, but direct warmup training needs stable IDs and replayable state, not only display names.

## Scope

Use the current training ruleset unless explicitly changed:

- ruleset: `optional_us_plus_2`
- Saito backend card IDs and country IDs
- optional deck enabled
- US setup bonus: `2`
- no Turn Zero / promo / variant cards beyond the current Saito optional deck

Card and country names in display text are not enough. Use IDs such as `decolonization`, `cambridge`, `mideast`, `westgermany`, `southafrica`.

The full country/card ID reference is in [`docs/warmup_ids.md`](warmup_ids.md).

## Preferred Input: Direct No-Seed Samples

For actual warmup training, prefer direct action samples. This does **not** require
a seed. Each decision carries the exact model input at that moment:

Schema file: [`docs/warmup_direct_sample_schema.json`](warmup_direct_sample_schema.json)

Pretty structured sample: [`examples/warmup/sample_direct_warmup_game.json`](../examples/warmup/sample_direct_warmup_game.json)

```json
{
  "format": "ts_warmup_game_v1",
  "game_id": "source-000001",
  "ruleset": "optional_us_plus_2",
  "partial": false,
  "source": {"kind": "human_log", "name": "optional source name"},
  "samples": [
    {
      "step": 0,
      "turn": 1,
      "action_round": 0,
      "side": "ussr",
      "phase": "setup_ussr",
      "observation": {
        "side": "ussr",
        "current_player": "ussr",
        "turn": 1,
        "action_round": 0,
        "vp": 0,
        "defcon": 5,
        "milops": {"us": 0, "ussr": 0},
        "space": {"us": 0, "ussr": 0},
        "hand_count": {"us": 8, "ussr": 9},
        "deck_count": 94,
        "china_owner": "ussr",
        "events": {},
        "hand": ["decolonization", "suezcrisis"],
        "discard": [],
        "removed": [],
        "countries": ["full bridge country objects here"]
      },
      "legal_actions": [
        {"type": "saito_dom", "decision": "country_click", "value": "poland", "label": "Poland"}
      ],
      "choice": {"type": "saito_dom", "decision": "country_click", "value": "poland"},
      "result_after": {
        "vp": 0,
        "defcon": 5,
        "state_delta": {"countries": ["influence/control changes here"]}
      }
    }
  ],
  "result": {"winner": "us", "terminal_reason": "final scoring", "vp": 7, "turn": 10, "steps": 420}
}
```

`observation` must be the full pre-action bridge observation, not just a compact
text line. `legal_actions` must be the exact choices available to the player at
that decision. `choice` or `expert_action_index` identifies the expert move.
`result_after` is used for audit and later weighting; behavior cloning trains
from `observation + legal_actions -> choice`.

This direct format is what `python -m struggle_ai.warmup_bc` uses first. Seeded
bridge replay is only a legacy fallback for records that do not include direct
observations.

For the current PPO architecture track, run warmup with the same model shape as
the transformer-history trainer:

```bash
python -m struggle_ai.warmup_bc \
  --input path/to/direct_warmup_games.jsonl \
  --output-dir runs/warmup_bc/my_run \
  --multi-agent \
  --model-arch transformer_history \
  --history-layers 2 \
  --history-attention-heads 4 \
  --history-dropout 0.05 \
  --graph-layers 2 \
  --graph-neighbor-hops 5 \
  --load-us-policy-from runs/ppo/selected_policies/best_us57005_ussr66001/policies/us_policy \
  --load-ussr-policy-from runs/ppo/selected_policies/best_us57005_ussr66001/policies/ussr_policy \
  --partial-warmstart
```

Use `--dry-run` first. A good dry run should report nonzero matched samples for
both `us_policy` and `ussr_policy`, with zero or explainable failures.

## Legacy Replay Input

Provide one JSONL file containing one complete game per line:

```text
warmup_games.jsonl
```

Each line must be a complete JSON object with:

```json
{
  "format": "ts_warmup_game_v1",
  "game_id": "source-000001",
  "ruleset": "optional_us_plus_2",
  "source": {
    "kind": "human_log",
    "name": "optional free-form source name"
  },
  "start": {
    "seed": 12345,
    "deck": ["duckandcover", "asia", "..."],
    "hands": {
      "us": ["marshall", "containment", "..."],
      "ussr": ["decolonization", "fidel", "..."]
    },
    "initial_influence": {
      "us": {"canada": 2, "uk": 5, "israel": 1, "iran": 1},
      "ussr": {"eastgermany": 3, "finland": 1, "syria": 1}
    }
  },
  "actions": [],
  "result": {
    "winner": "us",
    "terminal_reason": "vp_threshold",
    "vp": 20,
    "turn": 5,
    "steps": 221
  }
}
```

`seed` is useful but not sufficient by itself for external logs. If the game was not generated by our exact bridge build, include `deck`, `hands`, and dice/random outcomes where relevant.

## Required Top-Level Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `format` | yes | Must be `ts_warmup_game_v1`. |
| `game_id` | yes | Stable unique ID. Used in error reports. |
| `ruleset` | yes | Usually `optional_us_plus_2`. |
| `start.hands.us` | yes | US initial hand as Saito card IDs. |
| `start.hands.ussr` | yes | USSR initial hand as Saito card IDs. |
| `start.deck` | strongly recommended | Ordered remaining draw deck after initial deal. Required if no exact bridge seed. |
| `start.initial_influence` | strongly recommended | Starting influence if not standard for the ruleset. |
| `actions` | yes | Ordered action decisions, one entry per player choice. |
| `result` | yes | Winner and terminal reason for filtering/evaluation. |

## Action Entry Format

Each expert choice should be one action object. The loader will replay the game in the Saito bridge and match each entry against the legal actions available at that point.

```json
{
  "step": 37,
  "turn": 2,
  "action_round": 3,
  "side": "us",
  "phase": "action",
  "prompt_class": "ops_mode",
  "choice": {
    "type": "saito_choice",
    "decision": "option",
    "value": "coup",
    "label": "coup"
  },
  "context": {
    "card": "duckandcover",
    "mode": "ops"
  }
}
```

### Required Action Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `step` | yes | Zero-based decision index within this game. |
| `side` | yes | `us` or `ussr`, the side making the decision. |
| `turn` | yes | Twilight Struggle turn number. |
| `action_round` | yes | Raw bridge/Saito action-round counter if known. |
| `phase` | yes | One of `setup_ussr`, `setup_us`, `setup_us_bonus`, `headline`, `headline_resolve`, `action`, `event`, `scoring`, or `response`. |
| `choice.type` | yes | Usually `saito_choice`, `saito_dom`, or `setup`. |
| `choice.decision` | yes | Saito decision kind, e.g. `list`, `option`, `country_click`, `country_mouseup`. |
| `choice.value` | yes | Stable selected value: card ID, country ID, option ID, or DOM value. |

### Optional But Useful Action Fields

| Field | Meaning |
| --- | --- |
| `choice.label` | Human-readable display text. Ignored for primary matching. |
| `choice.card` | Card currently being resolved, if the selected value is not itself the card. |
| `choice.country` | Country target, if applicable. |
| `choice.region` | Region target, if applicable. |
| `context.card` | Card being played or resolved. |
| `context.mode` | `event`, `ops`, `space`, `coup`, `realign`, `place`, `discard`, etc. |
| `before` | Optional compact state before the action. Used for validation. |
| `after` | Optional compact state after the action. Used for validation. |
| `random` | Dice roll or sampled random result consumed by this action. |

## Random Outcomes

If the source log is not generated by our bridge with a reproducible seed, include random outcomes. Without them, replay can diverge and later action labels may no longer be legal.

Use this shape:

```json
{
  "step": 52,
  "side": "ussr",
  "choice": {
    "type": "saito_choice",
    "decision": "option",
    "value": "event"
  },
  "context": {"card": "arabisraeli"},
  "random": {
    "kind": "war_roll",
    "roll": 6,
    "adjusted": 6,
    "success": true
  }
}
```

Common random kinds:

- `coup_roll`
- `realignment_roll`
- `war_roll`
- `space_roll`
- `olympic_roll`
- `deck_shuffle`

For two-sided rolls:

```json
"random": {
  "kind": "realignment_roll",
  "us_roll": 3,
  "ussr_roll": 5,
  "us_adjusted": 4,
  "ussr_adjusted": 6
}
```

## Legal Action Matching

During import, each action is matched against the bridge legal action list in this order:

1. Exact match on `choice.type`, `choice.decision`, and `choice.value`.
2. If ambiguous, also match `choice.card`, `choice.country`, `choice.region`.
3. If still ambiguous, compare `prompt_class` and `context.mode`.
4. If no unique legal action matches, reject the game and report `game_id`, `step`, and the current legal actions.

Do not rely on labels like `"Middle-East Scoring"` or `"South Africa"` alone. Labels can change formatting; IDs should not.

## Text Log Companion Format

If you want to provide text logs like the current `.sorted.txt`, use the following strict format. This text is for inspection and fallback parsing; JSONL remains the training source of truth.

```text
Run source-000001 starts: seed=12345, ruleset=optional_us_plus_2
  US get marshall|Marshall Plan, containment|Containment, usjapan|US/Japan Defense Pact; have 8 in hand
  USSR get decolonization|Decolonization, fidel|Fidel, asia|Asia Scoring, china|China; have 9 in hand
  Initial US influence: canada=2, uk=5, israel=1, iran=1, australia=4, philippines=1, japan=1, southkorea=1, panama=1, southafrica=1
  Initial USSR influence: eastgermany=3, finland=1, syria=1, iraq=1, northkorea=3
  Deck: duckandcover, nasser, romanianab, ...

Turn 1 starts
  USSR setup (T1.00): poland|Poland -> poland|Poland -> eastgermany|East Germany
  US setup (T1.00): westgermany|West Germany -> italy|Italy -> france|France
  US setup bonus (T1.00): iran|Iran -> italy|Italy
  Headline: USSR decolonization|Decolonization; US containment|Containment
  USSR headline (T1.00): decolonization|Decolonization
  US headline (T1.00): containment|Containment
  USSR AR1 (T1.01): fidel|Fidel -> event
  US AR1 (T1.01): usjapan|US/Japan Defense Pact -> ops -> coup -> thailand|Thailand
    Roll: coup us die=5 ops=4 target=thailand
```

Rules for text logs:

- Always write IDs before display names: `id|Display Name`.
- Use `->` only to separate atomic decisions.
- Include one action group per line, in exact decision order.
- Include setup and setup bonus decisions.
- Include headline card choices before headline resolution.
- Include response decisions, e.g. Olympic boycott/participate, Blockade discard, UN Intervention card choice.
- Include dice rolls and random outcomes on indented `Roll:` lines.
- Include new hands/deals at turn boundaries if the deck is not fully supplied.

Text-only logs missing IDs, deck order, dice rolls, or response decisions are not directly warmup-trainable. They can still be used for human review.

## Minimal Valid JSONL Example

This is the smallest useful direct-training example. It assumes deterministic replay from `seed` works with our exact bridge version.

```json
{
  "format": "ts_warmup_game_v1",
  "game_id": "demo-0001",
  "ruleset": "optional_us_plus_2",
  "start": {
    "seed": 1,
    "hands": {
      "us": ["arabisraeli", "formosan", "degaulle", "unintervention", "usjapan", "koreanwar", "nato", "nasser"],
      "ussr": ["decolonization", "suezcrisis", "naziscientist", "fidel", "fiveyearplan", "asia", "socgov", "norad", "china"]
    }
  },
  "actions": [
    {
      "step": 0,
      "turn": 1,
      "action_round": 0,
      "side": "ussr",
      "phase": "setup_ussr",
      "choice": {"type": "setup", "decision": "country", "value": "poland"}
    },
    {
      "step": 1,
      "turn": 1,
      "action_round": 0,
      "side": "ussr",
      "phase": "setup_ussr",
      "choice": {"type": "setup", "decision": "country", "value": "poland"}
    },
    {
      "step": 2,
      "turn": 1,
      "action_round": 0,
      "side": "us",
      "phase": "headline",
      "choice": {"type": "saito_choice", "decision": "list", "value": "arabisraeli"}
    }
  ],
  "result": {
    "winner": "ussr",
    "terminal_reason": "nuclear_war",
    "vp": -2,
    "turn": 2,
    "steps": 95
  }
}
```

## Best Format From Our Existing Logs

If the data comes from our current system, provide the raw `games-*.jsonl`, not only `games-*.sorted.txt`.

The current raw logs already include:

- `seed`
- `start.hands`
- action order
- selected `action`
- `before` and `after` context
- optional `debug_before` and `debug_after`
- state deltas
- Saito log deltas
- result metadata

That is enough to build a converter with minimal loss. The sorted text is still useful for manual inspection.

## Quality Requirements

A game should be rejected from warmup if any of these occur:

- missing setup actions;
- missing headline choices;
- missing response decisions;
- action side does not match bridge side-to-act;
- selected action is not legal during replay;
- replayed final result differs from supplied `result`;
- card appears in hand/deck/discard/removed in an impossible zone;
- same non-recurring removed card is played twice;
- unsupported ruleset or unknown card/country ID.

## Training Use

The importer will turn accepted games into behavior-cloning samples:

```json
{
  "game_id": "source-000001",
  "step": 37,
  "side": "us",
  "policy_id": "us_policy",
  "observation": "<encoded by struggle_ai.features.encode_observation>",
  "legal_actions": ["..."],
  "expert_action_index": 12,
  "weight": 1.0
}
```

Recommended sample weights:

- `1.0` for normal action decisions.
- `0.2-0.5` for setup if setup is scripted separately.
- `1.0-2.0` for headline, scoring-card, DEFCON-sensitive coup, Blockade/UN/Olympic response, and late-game VP-critical decisions.

Do not include illegal or hindsight-corrected actions as expert labels. If a source move is known to be a mistake but still part of a real game, keep it with lower weight or mark it:

```json
"weight": 0.25,
"tags": ["human_mistake", "keep_for_distribution"]
```

## Delivery Layout

Use this directory shape:

```text
warmup_data/
  manifest.json
  games/
    part-00000.jsonl
    part-00001.jsonl
  text/
    source-000001.sorted.txt
    source-000002.sorted.txt
```

`manifest.json`:

```json
{
  "format": "ts_warmup_manifest_v1",
  "ruleset": "optional_us_plus_2",
  "created_at": "2026-06-03",
  "game_files": ["games/part-00000.jsonl", "games/part-00001.jsonl"],
  "text_log_dir": "text",
  "notes": "free-form"
}
```

The JSONL files are required for direct training. The text directory is optional but recommended for audits.
