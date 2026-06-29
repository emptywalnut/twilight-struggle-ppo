from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize side-specific Twilight Struggle Elo leaderboards.")
    parser.add_argument("path", type=Path, help="Run directory or elo_ratings.json path.")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def elo_path(path: Path) -> Path:
    if path.is_dir():
        return path / "elo_ratings.json"
    return path


def player_kind(name: str) -> str:
    prefix = name.split(":", 1)[0]
    if prefix in {"initial", "random", "final"}:
        return prefix
    if prefix.startswith("steps-"):
        return "checkpoint"
    return "other"


def summarize_board(board: dict[str, Any], top: int) -> list[dict[str, Any]]:
    ratings = board.get("ratings") or {}
    games = board.get("games") or {}
    rows = [
        {
            "player": player,
            "kind": player_kind(player),
            "rating": round(float(rating), 2),
            "games": int(games.get(player, 0)),
        }
        for player, rating in ratings.items()
    ]
    rows.sort(key=lambda row: (float(row["rating"]), int(row["games"])), reverse=True)
    return rows[:top]


def main() -> None:
    args = parse_args()
    path = elo_path(args.path)
    if not path.exists():
        raise SystemExit(f"Elo file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    leaderboards = state.get("leaderboards") or {}
    payload = {
        "elo_path": str(path),
        "k_factor": state.get("k_factor"),
        "match_count": len(state.get("matches") or []),
        "leaderboards": {
            side: summarize_board(leaderboards.get(side) or {}, args.top)
            for side in ("us", "ussr")
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
