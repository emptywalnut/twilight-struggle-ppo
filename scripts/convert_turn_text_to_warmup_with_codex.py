#!/usr/bin/env python3
"""
Convert scraped replay text logs into warmup JSONL using Codex.

Each replay file is translated independently with the configured Codex model,
then the outputs are collected into warmup-data artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate scraped replay turn-text files into warmup JSONL using "
            "Codex with model gpt-5.3-codex-spark."
        )
    )
    parser.add_argument("--workspace", type=Path, default=Path("/mnt2/users/kaile/hantao/game"))
    parser.add_argument("--turn-text-dir", type=Path, default=None)
    parser.add_argument("--hands-dir", type=Path, default=None)
    parser.add_argument("--ids-md", type=Path, default=None)
    parser.add_argument("--format-md", type=Path, default=None)
    parser.add_argument("--start", type=int, default=26)
    parser.add_argument("--end", type=int, default=165)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--model", type=str, default="gpt-5.3-codex-spark")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--part-prefix", type=str, default="part")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ids", type=str, default="")
    return parser.parse_args()


def normalize(name: str) -> str:
    text = name.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def alias_variants(name: str) -> set[str]:
    base = normalize(name)
    compact = base.replace(" ", "")
    dashed = base.replace(" ", "_")
    return {base, compact, dashed, base.replace(" ", "-")}


def parse_warmup_ids(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    card_map: dict[str, str] = {}
    country_map: dict[str, str] = {}
    section: str | None = None

    for line in text:
        l = line.strip()
        if "## Country IDs" in l:
            section = "country"
            continue
        if "## Card IDs" in l:
            section = "card"
            continue
        if not l.startswith("|") or l.startswith("| ID |"):
            continue
        if re.match(r"\|\s*-+\s*\|", l):
            continue
        cols = [c.strip() for c in l.strip("|").split("|")]
        if len(cols) < 2:
            continue

        item_id = cols[0].strip().strip("`")
        item_name = cols[1].strip()
        if not item_id or not item_name or item_id in {"Name", "---", "ID"}:
            continue

        if section == "country":
            country_map[item_id] = item_name
        elif section == "card":
            card_map[item_id] = item_name

    return {
        "country_id_by_name": {alias: cid for cid, nm in country_map.items() for alias in alias_variants(nm)},
        "country_id_by_name_no_space": {nm.lower().replace(" ", ""): cid for cid, nm in country_map.items()},
        "card_id_by_name": {alias: cid for cid, nm in card_map.items() for alias in alias_variants(nm)},
        "card_id_by_name_no_space": {nm.lower().replace(" ", ""): cid for cid, nm in card_map.items()},
        "country_table": country_map,
        "card_table": card_map,
    }


def map_name_to_id(value: str, map_alias: dict[str, str], map_nospace: dict[str, str]) -> str | None:
    if not value:
        return None
    key1 = normalize(value)
    key2 = key1.replace(" ", "")
    return map_alias.get(key1) or map_alias.get(key2) or map_nospace.get(key1.replace(" ", ""))


def load_turn_text(text_dir: Path, replay_id: str) -> str:
    file = text_dir / f"replay_{replay_id}.txt"
    if not file.exists():
        raise FileNotFoundError(f"missing turn text: {file}")
    return file.read_text(encoding="utf-8", errors="replace")


def load_turn_hands(hands_dir: Path, replay_id: str, maps: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    hand_file = hands_dir / f"replay_{replay_id}.json"
    if not hand_file.exists():
        return {"us": [], "ussr": []}

    raw = json.loads(hand_file.read_text(encoding="utf-8"))
    start = raw.get("1", {}) if isinstance(raw, dict) else {}

    card_map = {
        **maps["card_id_by_name"],
        **maps["card_id_by_name_no_space"],
    }

    out: dict[str, list[str]] = {"us": [], "ussr": []}
    for side in ("us", "ussr"):
        for name in start.get(side, []):
            cid = map_name_to_id(name, maps["card_id_by_name"], maps["card_id_by_name_no_space"])
            if cid:
                out[side].append(cid)
                continue
            out[side].append(name)
    return out


def build_prompt(
    replay_id: str,
    turn_text: str,
    start_hands: dict[str, list[str]],
    cards_map: dict[str, str],
    countries_map: dict[str, str],
    format_doc: str,
) -> str:
    sample_actions = """{
  \"step\": 0,
  \"turn\": 1,
  \"action_round\": 0,
  \"side\": \"ussr\",
  \"phase\": \"setup_ussr\",
  \"prompt_class\": \"setup\",
  \"choice\": {\"type\": \"setup\", \"decision\": \"country\", \"value\": \"us\"},
  \"context\": {\"card\": \"duckandcover\", \"mode\": \"setup\"}
}"""

    return f"""Convert this replay log into one strict JSON object for warmup training.

Rules are defined by: /mnt2/users/kaile/hantao/game/struggle/docs/warmup_data_format.md
Use only stable IDs from warmup_ids and output must be directly consumable by training code expecting
format='ts_warmup_game_v1'.

You must emit **one JSON object only**, no markdown fences.

Use this action schema:
{sample_actions}

Allowed side values: us, ussr.
Allowed phases: setup_ussr, setup_us, setup_us_bonus, headline, headline_resolve, action, event, scoring, response.
At least one action is required if parseable.

Use these stable ID maps:

Cards:
{json.dumps(cards_map, indent=2)}

Countries:
{json.dumps(countries_map, indent=2)}

Use this pre-filled start hands (turn 1 cards for each side) from scrape/hands:
{json.dumps(start_hands, indent=2)}

Replay ID:
{replay_id}

Replay text:
{turn_text}

For unmapped names, keep the label in choice.label or action context and omit hardcoding a guessed ID.
Set a top-level quality object with any warning/reject flags if there are ambiguous mappings.

Return JSON with top-level keys:
- format
- game_id
- ruleset
- source
- start
- actions
- result
- quality (object with status "ok|reject|partial" and notes list)

Minimal requirements:
- format must be "ts_warmup_game_v1"
- game_id must be the replay id
- start.hands must contain us and ussr arrays
- actions must be a list of objects

If any fields are impossible, set null and include in quality.notes.
"""


def call_codex_once(prompt: str, model: str, workspace: Path, timeout: int = 120) -> str:
    with tempfile.TemporaryDirectory(prefix="codex-warmup-") as tmpdir:
        out_file = Path(tmpdir) / "last.txt"
        cmd = [
            "codex",
            "exec",
            "--model",
            model,
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            "-C",
            str(workspace),
            "--output-last-message",
            str(out_file),
        ]

        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "codex exec failed")

        if not out_file.exists():
            raise RuntimeError("codex output file was not written")

        return out_file.read_text(encoding="utf-8", errors="replace").strip()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to recover if wrapper text exists around JSON.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


@dataclass
class ConversionResult:
    replay_id: str
    game: dict[str, Any] | None
    text_out: str
    ok: bool
    status: str
    error: str | None


def convert_replay(
    replay_id: str,
    turn_text_dir: Path,
    hands_dir: Path,
    ids_map: dict[str, dict[str, str]],
    format_doc: str,
    model: str,
    workspace: Path,
    max_retries: int = 2,
    retry_backoff: float = 1.5,
    timeout: int = 120,
    dry_run: bool = False,
) -> ConversionResult:
    try:
        raw = load_turn_text(turn_text_dir, replay_id)
        start_hands = load_turn_hands(hands_dir, replay_id, ids_map)

        if dry_run:
            return ConversionResult(
                replay_id=replay_id,
                game=None,
                text_out=raw,
                ok=True,
                status="dry_run",
                error=None,
            )

        prompt = build_prompt(
            replay_id,
            raw,
            start_hands,
            ids_map["card_table"],
            ids_map["country_table"],
            format_doc,
        )

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                codex_output = call_codex_once(prompt, model=model, workspace=workspace, timeout=timeout)
                payload = extract_json(codex_output)
                if not isinstance(payload, dict):
                    raise ValueError("codex payload is not JSON object")
                payload.setdefault("format", "ts_warmup_game_v1")
                payload.setdefault("game_id", replay_id)
                payload.setdefault("ruleset", "optional_us_plus_2")
                payload.setdefault("source", {"kind": "turn_text", "name": f"replay_{replay_id}.txt"})
                payload.setdefault("start", {})
                payload["start"].setdefault("hands", start_hands)
                if "actions" not in payload or not isinstance(payload["actions"], list):
                    payload["actions"] = []
                payload.setdefault("quality", {"status": "ok", "notes": []})
                if "result" not in payload or not isinstance(payload["result"], dict):
                    payload["result"] = {"winner": None, "terminal_reason": None, "vp": None, "turn": None, "steps": None}
                return ConversionResult(
                    replay_id=replay_id,
                    game=payload,
                    text_out=raw,
                    ok=True,
                    status=payload.get("quality", {}).get("status", "ok"),
                    error=None,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < max_retries:
                    time.sleep(retry_backoff * (attempt + 1))
                    continue
                raise RuntimeError(f"failed after {max_retries + 1} attempts: {exc}") from exc

        raise RuntimeError(str(last_err) if last_err else "conversion failed")

    except Exception as exc:
        return ConversionResult(
            replay_id=replay_id,
            game=None,
            text_out="",
            ok=False,
            status="failed",
            error=str(exc),
        )


def main() -> None:
    args = parse_args()

    workspace = args.workspace
    turn_text_dir = args.turn_text_dir or (workspace / "scrape" / "turn_text")
    hands_dir = args.hands_dir or (workspace / "scrape" / "hands")
    ids_md = args.ids_md or (workspace / "struggle" / "docs" / "warmup_ids.md")
    format_md = args.format_md or (workspace / "struggle" / "docs" / "warmup_data_format.md")
    output_dir = args.output_dir or (workspace / "struggle" / "warmup_data")

    ids = parse_warmup_ids(ids_md)
    format_doc = format_md.read_text(encoding="utf-8", errors="ignore")

    if args.ids:
        ids_to_run = [v.zfill(3) for v in args.ids.split(",") if v.strip()]
    else:
        ids_to_run = [str(i).zfill(3) for i in range(args.start, args.end + 1)]

    game_out_dir = output_dir / "games"
    text_out_dir = output_dir / "text"

    if not args.dry_run:
        game_out_dir.mkdir(parents=True, exist_ok=True)
        text_out_dir.mkdir(parents=True, exist_ok=True)

    to_convert = [rid for rid in ids_to_run if (turn_text_dir / f"replay_{rid}.txt").exists()]
    skipped = [rid for rid in ids_to_run if rid not in to_convert]

    print(f"Converting {len(to_convert)} replays (skipped {len(skipped)} missing).")

    results: list[ConversionResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        fut_map = {
            pool.submit(
                convert_replay,
                replay_id=rid,
                turn_text_dir=turn_text_dir,
                hands_dir=hands_dir,
                ids_map=ids,
                format_doc=format_doc,
                model=args.model,
                workspace=workspace,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                timeout=args.timeout,
                dry_run=args.dry_run,
            ): rid
            for rid in to_convert
        }
        for future in concurrent.futures.as_completed(fut_map):
            rid = fut_map[future]
            res = future.result()
            results.append(res)
            print(f"{rid}: {'OK' if res.ok else 'FAIL'} [{res.status}]" + (f" {res.error}" if res.error else ""))

    ok_results = [r for r in results if r.ok and r.game is not None]
    fail_results = [r for r in results if not r.ok]

    if args.dry_run:
        return

    games_file = game_out_dir / f"{args.part_prefix}-00000.jsonl"
    with games_file.open("w", encoding="utf-8") as fh:
        for res in sorted(ok_results, key=lambda x: x.replay_id):
            fh.write(json.dumps(res.game, ensure_ascii=False))
            fh.write("\n")

            text_path = text_out_dir / f"source-{res.replay_id}.sorted.txt"
            text_path.write_text(res.text_out, encoding="utf-8")

    manifest = {
        "format": "ts_warmup_manifest_v1",
        "ruleset": "optional_us_plus_2",
        "created_at": datetime.now().date().isoformat(),
        "game_files": [str(Path("games") / games_file.name)],
        "text_log_dir": "text",
        "notes": "Generated via codex gpt-5.3-codex-spark from scrape/turn_text and scrape/hands",
        "stats": {
            "attempted": len(results),
            "accepted": len(ok_results),
            "failed": len(fail_results),
            "skipped": len(skipped),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if fail_results:
        failures = [
            {"replay": r.replay_id, "status": r.status, "error": r.error}
            for r in fail_results
        ]
        (output_dir / "conversion_failures.jsonl").write_text(
            "\n".join(json.dumps(r) for r in failures) + "\n",
            encoding="utf-8",
        )

    print(f"Done. Output: {games_file}")
    print(f"Failures: {len(fail_results)}. Check conversion_failures.jsonl")


if __name__ == "__main__":
    os.chdir(os.environ.get("PWD", str(Path(__file__).resolve().parents[2])))
    main()
