#!/usr/bin/env python3
"""
Translate Twilight Struggle replay logs in scrape/turn_text into warmup JSONL
using Codex with gpt-5.3-codex-spark.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
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
            "Convert scraped replay turn-text logs into training-ready warmup data "
            "using codex exec + gpt-5.3-codex-spark."
        )
    )
    parser.add_argument("--workspace", type=Path, default=Path("/mnt2/users/kaile/hantao/game"))
    parser.add_argument("--turn-text-dir", type=Path, default=None, help="Directory containing scrape/turn_text")
    parser.add_argument("--hands-dir", type=Path, default=None, help="Directory containing scrape/hands")
    parser.add_argument("--ids-md", type=Path, default=None, help="Path to warmup_ids.md")
    parser.add_argument("--format-md", type=Path, default=None, help="Path to warmup_data_format.md")
    parser.add_argument("--ids-file", type=Path, default=None, help="CSV/plain file listing IDs")
    parser.add_argument("--ids", type=str, default="", help="Comma list of replay IDs to force")
    parser.add_argument("--start", type=int, default=26)
    parser.add_argument("--end", type=int, default=165)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--model", type=str, default="gpt-5.3-codex-spark")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for warmup_data output")
    parser.add_argument("--part-prefix", type=str, default="part")
    parser.add_argument("--part-size", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--resume", action="store_true", help="Skip IDs already in existing part files")
    parser.add_argument("--dry-run", action="store_true", help="Load+package prompts only, no codex calls")
    parser.add_argument("--verbose", action="store_true")
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
    dashed = base.replace(" ", "-")
    return {base, compact, dashed, base.replace(" ", "_")}


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
        item_id = cols[0].strip("`")
        item_name = cols[1].strip("`")
        if not item_id or not item_name or item_id in {"ID", "---", "Name"}:
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


def map_name_to_id(value: str, alias_map: dict[str, str], compact_map: dict[str, str]) -> str | None:
    if not value:
        return None
    key1 = normalize(value)
    key2 = key1.replace(" ", "")
    return alias_map.get(key1) or alias_map.get(key2) or compact_map.get(key2)


def clean_card_name(name: str) -> str:
    return name.replace("*", "").replace("\"", "").strip()


def load_turn_text(text_dir: Path, replay_id: str) -> str:
    path = text_dir / f"replay_{replay_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing turn text file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def load_turn_hands(hands_dir: Path, replay_id: str, maps: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    hand_file = hands_dir / f"replay_{replay_id}.json"
    if not hand_file.exists():
        return {"us": [], "ussr": []}
    raw = json.loads(hand_file.read_text(encoding="utf-8"))
    turn_one = raw.get("1", {}) if isinstance(raw, dict) else {}
    out: dict[str, list[str]] = {"us": [], "ussr": []}
    for side in ("us", "ussr"):
        for nm in turn_one.get(side, []):
            cid = map_name_to_id(clean_card_name(nm), maps["card_id_by_name"], maps["card_id_by_name_no_space"])
            out[side].append(cid if cid else clean_card_name(nm))
    return out


def discover_ids(turn_text_dir: Path, args: argparse.Namespace) -> list[str]:
    ids: list[str] = []

    if args.ids:
        ids = [s.strip().zfill(3) for s in args.ids.split(",") if s.strip()]
        return ids

    if args.ids_file:
        with args.ids_file.open("r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(128)
            fh.seek(0)
            if head.lstrip().startswith("replay,"):
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    if row[0].lower() in {"replay", "id", "game"}:
                        continue
                    ids.append(row[0].zfill(3))
            else:
                for line in fh:
                    v = line.strip()
                    if v:
                        ids.append(v.split(",")[0].strip().zfill(3))
    else:
        ids = []
        for path in sorted(turn_text_dir.glob("replay_*.txt")):
            m = re.search(r"replay_(\d+)\.txt$", path.name)
            if m:
                ids.append(m.group(1))
        if not ids:
            # fallback to numeric range in case file discovery fails unexpectedly.
            ids = [str(i).zfill(3) for i in range(args.start, args.end + 1)]
    return sorted({rid for rid in ids if rid and rid.strip()})


def existing_converted_ids(output_dir: Path, part_prefix: str) -> set[str]:
    seen: set[str] = set()
    game_dir = output_dir / "games"
    if not game_dir.exists():
        return seen
    for path in sorted(game_dir.glob(f"{part_prefix}-*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    gid = payload.get("game_id")
                    if isinstance(gid, str):
                        seen.add(gid.zfill(3))
                except json.JSONDecodeError:
                    continue
    return seen


def next_part_indices(output_dir: Path, part_prefix: str) -> int:
    game_dir = output_dir / "games"
    if not game_dir.exists():
        return 0
    idx = []
    for p in game_dir.glob(f"{part_prefix}-*.jsonl"):
        m = re.search(rf"{re.escape(part_prefix)}-(\\d+)\\.jsonl$", p.name)
        if m:
            idx.append(int(m.group(1)))
    return (max(idx) + 1) if idx else 0


def count_existing_games(output_dir: Path, part_prefix: str) -> int:
    game_dir = output_dir / "games"
    if not game_dir.exists():
        return 0
    total = 0
    for path in game_dir.glob(f"{part_prefix}-*.jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            total += sum(1 for _ in fh)
    return total


def build_prompt(replay_id: str, raw_text: str, start_hands: dict[str, list[str]], maps: dict[str, dict[str, str]], format_doc: str) -> str:
    action_example = r'''{
  "step": 0,
  "turn": 1,
  "action_round": 0,
  "side": "ussr",
  "phase": "setup_ussr",
  "prompt_class": "setup",
  "choice": {"type": "setup", "decision": "country", "value": "poland"}
}'''

    return f"""Convert this full replay text into ONE strict JSON object for warmup training.

Output must satisfy docs/warmup_data_format.md and use only stable ids from warmup_ids.md.
Do not use markdown fences.

Required top fields:
- format: "ts_warmup_game_v1"
- game_id: {replay_id}
- ruleset: "optional_us_plus_2"
- source: {{"kind": "turn_text", "name": "replay_XXX.txt"}}
- start: object with "seed" and "hands" (must contain arrays us/ussr)
- actions: list of action objects
- result: object with winner/terminal_reason/vp/turn/steps
- quality: object with keys status and notes

Use this action shape:
{action_example}

Allowed side: us, ussr
Allowed phase: setup_ussr, setup_us, setup_us_bonus, headline, headline_resolve, action, event, scoring, response
Use these maps for stable IDs:

Cards:
{json.dumps(maps["card_table"], indent=2)}

Countries:
{json.dumps(maps["country_table"], indent=2)}

Pre-filled start hands (turn-1):
{json.dumps(start_hands, indent=2)}

Start with this raw format documentation snippet:
{format_doc[:3000]}

Replay text:
{raw_text}

Rules:
- Every legal decision from the log should become one action in order.
- If a label cannot be confidently mapped, set choice.value to null and add note in quality.notes.
- Fill result.winner/terminal_reason/vp/turn/steps with observed values if available, otherwise null.
- Return exactly one JSON object only.
"""


def call_codex(prompt: str, model: str, workspace: Path, timeout: int) -> str:
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
            raise RuntimeError((result.stderr or result.stdout or "").strip() or "codex exec failed")
        if not out_file.exists():
            raise RuntimeError("codex output file was not written")
        return out_file.read_text(encoding="utf-8", errors="replace").strip()


def parse_json_like(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def normalize_payload(
    payload: Any,
    replay_id: str,
    start_hands: dict[str, list[str]],
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    if not isinstance(payload, dict):
        raise ValueError("payload is not object")

    payload.setdefault("format", "ts_warmup_game_v1")
    payload["game_id"] = replay_id
    payload.setdefault("ruleset", "optional_us_plus_2")
    payload.setdefault("source", {"kind": "turn_text", "name": f"replay_{replay_id}.txt"})
    start = payload.get("start") if isinstance(payload.get("start"), dict) else {}
    start.setdefault("hands", {})
    start["hands"].setdefault("us", start_hands.get("us", []))
    start["hands"].setdefault("ussr", start_hands.get("ussr", []))
    payload["start"] = start

    actions = payload.get("actions")
    if not isinstance(actions, list):
        issues.append("actions missing or not list")
        payload["actions"] = []

    result = payload.get("result")
    if not isinstance(result, dict):
        result = {"winner": None, "terminal_reason": None, "vp": None, "turn": None, "steps": None}
        issues.append("result missing or invalid; set to nulls")
    payload["result"] = result

    quality = payload.get("quality")
    if not isinstance(quality, dict):
        quality = {"status": "partial" if issues else "ok", "notes": issues}
    else:
        if not quality.get("notes"):
            quality["notes"] = []
        if not isinstance(quality["notes"], list):
            quality["notes"] = [str(quality["notes"])]
        quality["notes"].extend(issues)
    if issues:
        quality["status"] = "partial" if quality.get("status") not in {"reject"} else "reject"
    quality.setdefault("status", "ok")
    payload["quality"] = quality
    return payload, issues


@dataclass
class Conversion:
    replay_id: str
    ok: bool
    status: str
    error: str | None
    payload: dict[str, Any] | None = None
    raw_prompt: str | None = None


def convert_replay(
    replay_id: str,
    turn_text_dir: Path,
    hands_dir: Path,
    ids_map: dict[str, dict[str, str]],
    format_doc: str,
    args: argparse.Namespace,
) -> Conversion:
    try:
        turn_text = load_turn_text(turn_text_dir, replay_id)
        start_hands = load_turn_hands(hands_dir, replay_id, ids_map)
        prompt = build_prompt(replay_id, turn_text, start_hands, ids_map, format_doc)

        if args.dry_run:
            return Conversion(replay_id=replay_id, ok=True, status="dry_run", error=None, payload={}, raw_prompt=prompt)

        for attempt in range(args.max_retries + 1):
            try:
                output = call_codex(prompt, model=args.model, workspace=args.workspace, timeout=args.timeout)
                parsed = parse_json_like(output)
                payload, _notes = normalize_payload(parsed, replay_id, start_hands)
                status = payload.get("quality", {}).get("status", "ok")
                return Conversion(replay_id, True, status, None, payload, raw_prompt=None)
            except Exception as exc:  # noqa: BLE001
                if attempt >= args.max_retries:
                    raise
                time.sleep(args.retry_backoff * (attempt + 1))
        raise RuntimeError("unreachable")
    except Exception as exc:  # noqa: BLE001
        return Conversion(replay_id, False, "failed", str(exc), None, None)


def write_manifest(
    output_dir: Path,
    games_written: int,
    attempted: int,
    failed: int,
    skipped: int,
    game_files: list[str],
) -> None:
    manifest = {
        "format": "ts_warmup_manifest_v1",
        "ruleset": "optional_us_plus_2",
        "created_at": datetime.now().date().isoformat(),
        "game_files": game_files,
        "text_log_dir": "text",
        "notes": "Converted from scrape/turn_text using gpt-5.3-codex-spark in codex batch script.",
        "stats": {
            "attempted": attempted,
            "accepted": games_written,
            "failed": failed,
            "skipped": skipped,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    workspace = args.workspace
    turn_text_dir = args.turn_text_dir or (workspace / "scrape" / "turn_text")
    hands_dir = args.hands_dir or (workspace / "scrape" / "hands")
    ids_md = args.ids_md or (workspace / "struggle" / "docs" / "warmup_ids.md")
    format_md = args.format_md or (workspace / "struggle" / "docs" / "warmup_data_format.md")
    output_dir = args.output_dir or (workspace / "struggle" / "warmup_data")

    ids_map = parse_warmup_ids(ids_md)
    format_doc = format_md.read_text(encoding="utf-8", errors="ignore")

    all_ids = discover_ids(turn_text_dir, args)
    to_convert = [rid for rid in all_ids if (turn_text_dir / f"replay_{rid}.txt").exists()]
    missing = [rid for rid in all_ids if rid not in to_convert]

    if args.resume:
        done = existing_converted_ids(output_dir, args.part_prefix)
        to_convert = [rid for rid in to_convert if rid not in done]
    else:
        done = set()

    print(f"found {len(all_ids)} ids, to convert {len(to_convert)} (missing files {len(missing)}, existing done {len(done)}).")

    output_dir.mkdir(parents=True, exist_ok=True)
    games_dir = output_dir / "games"
    text_dir = output_dir / "text"
    if not args.dry_run:
        games_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

    conversions: list[Conversion] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {
            pool.submit(
                convert_replay,
                replay_id=rid,
                turn_text_dir=turn_text_dir,
                hands_dir=hands_dir,
                ids_map=ids_map,
                format_doc=format_doc,
                args=args,
            ): rid
            for rid in to_convert
        }
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            conversions.append(res)
            if args.verbose:
                if res.error:
                    print(f"{res.replay_id}: {'OK' if res.ok else 'FAIL'} {res.status} {res.error}")
                else:
                    print(f"{res.replay_id}: {'OK' if res.ok else 'FAIL'} {res.status}")

    ok = [r for r in conversions if r.ok and r.payload]
    failed = [r for r in conversions if not r.ok or r.payload is None]

    if args.dry_run:
        print(f"Dry run complete. Would run {len(to_convert)} replays.")
        return

    ok_sorted = sorted(ok, key=lambda x: x.replay_id)
    if ok_sorted:
        start_part = next_part_indices(games_dir, args.part_prefix)
        part_files: list[str] = []
        cursor = 0
        part = start_part
        while cursor < len(ok_sorted):
            chunk = ok_sorted[cursor : cursor + args.part_size]
            path = games_dir / f"{args.part_prefix}-{part:05d}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for r in chunk:
                    if not r.payload:
                        continue
                    fh.write(json.dumps(r.payload, ensure_ascii=False))
                    fh.write("\n")
                    text_path = text_dir / f"source-{r.replay_id}.sorted.txt"
                    try:
                        text_path.write_text(load_turn_text(turn_text_dir, r.replay_id), encoding="utf-8")
                    except Exception:
                        pass
            part_files.append(str(Path("games") / path.name))
            part += 1
            cursor += args.part_size
    else:
        part_files = []

    if output_dir.exists():
        # keep manifest current to include all files from this output_dir
        existing = sorted(games_dir.glob(f"{args.part_prefix}-*.jsonl")) if games_dir.exists() else []
        manifest_files = [str(Path("games") / p.name) for p in existing]
        total_accepted = count_existing_games(output_dir, args.part_prefix)
        write_manifest(
            output_dir,
            games_written=total_accepted,
            attempted=len(conversions) + done.__len__() if args.resume else len(conversions),
            failed=len(failed),
            skipped=len(missing) + len(done),
            game_files=manifest_files,
        )

    if failed:
        fail_rows = [{"replay": r.replay_id, "status": r.status, "error": r.error} for r in failed]
        (output_dir / "conversion_failures.jsonl").write_text(
            "\n".join(json.dumps(r) for r in fail_rows) + ("\n" if fail_rows else ""),
            encoding="utf-8",
        )
    else:
        fail_file = output_dir / "conversion_failures.jsonl"
        if fail_file.exists():
            fail_file.unlink()

    print(
        f"Done. accepted={len(ok_sorted)} failed={len(failed)} skipped={len(missing)} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    os.chdir(os.environ.get("PWD", str(Path(__file__).resolve().parents[2])))
    main()
