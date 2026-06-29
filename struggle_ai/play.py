from __future__ import annotations

import argparse
import random

from struggle_ai.bridge_client import TwilightBridgeClient
from struggle_ai.policies import choose_heuristic_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Twilight Struggle CLI against a bot.")
    parser.add_argument("--human", choices=["us", "ussr"], default="us")
    parser.add_argument("--bot", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(args.seed)
        while not obs["terminal"]:
            side = obs["current_player"]
            legal = obs["legal_actions"]
            print()
            print(bridge.render_text(side))
            if side == args.human:
                for idx, action in enumerate(legal):
                    print(f"{idx:3d}: {format_action(action)}")
                raw = input(f"{side.upper()} action index> ").strip()
                action_idx = int(raw)
            else:
                if args.bot == "random":
                    action_idx = rng.randrange(len(legal))
                else:
                    action_idx = choose_heuristic_action(legal, rng)
                print(f"{side.upper()} bot selects {action_idx}: {format_action(legal[action_idx])}")
            result = bridge.step(legal[action_idx])
            obs = result["observation"]
        print()
        print(bridge.render_text(args.human))
        print(f"Game over: winner={obs['winner']}")


def format_action(action: dict) -> str:
    detail = [action["type"], action.get("card", "")]
    if "country" in action:
        detail.append(action["country"])
    if "region" in action:
        detail.append(action["region"])
    return " ".join(part for part in detail if part)


if __name__ == "__main__":
    main()
