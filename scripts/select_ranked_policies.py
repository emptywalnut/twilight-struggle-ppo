from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rank_ppo_checkpoints import checkpoint_records, default_train_log, rank_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package best side-specific policies from a PPO run's checkpoint eval records."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-nuke-rate", type=float, default=0.60)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def require_policy(checkpoint: Path, policy_id: str) -> Path:
    policy_dir = checkpoint / "policies" / policy_id
    if not policy_dir.exists():
        raise FileNotFoundError(f"missing policy {policy_id} in checkpoint {checkpoint}")
    return policy_dir


def select_best(records: list[dict[str, Any]], side: str, max_nuke_rate: float) -> dict[str, Any]:
    ranked = rank_side(records, side, max_nuke_rate)
    if not ranked:
        raise RuntimeError(f"no ranked checkpoint records for {side}")
    return ranked[0]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force to replace: {out_dir}")

    train_log = default_train_log(run_dir)
    records = checkpoint_records(train_log)
    if not records:
        raise SystemExit(f"no checkpoint_eval_done/final_eval_done records in {train_log}")

    best = {
        "us": select_best(records, "us", args.max_nuke_rate),
        "ussr": select_best(records, "ussr", args.max_nuke_rate),
    }
    temp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "policies").mkdir()
    (temp_dir / "full_checkpoints").mkdir()

    manifest: dict[str, Any] = {
        "source_run_dir": str(run_dir),
        "max_nuke_rate": args.max_nuke_rate,
        "selected": best,
        "policies": {},
        "full_checkpoints": {},
    }

    for side, policy_id in (("us", "us_policy"), ("ussr", "ussr_policy")):
        checkpoint = Path(str(best[side]["checkpoint"])).resolve()
        policy_dir = require_policy(checkpoint, policy_id)
        packaged_policy = temp_dir / "policies" / policy_id
        packaged_checkpoint = temp_dir / "full_checkpoints" / f"{side}_source_{checkpoint.name}"
        copy_tree(policy_dir, packaged_policy)
        copy_tree(checkpoint, packaged_checkpoint)
        manifest["policies"][policy_id] = str(out_dir / "policies" / policy_id)
        manifest["full_checkpoints"][side] = str(out_dir / "full_checkpoints" / f"{side}_source_{checkpoint.name}")

    (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temp_dir / "README.md").write_text(
        "# Selected Twilight Struggle PPO Policies\n\n"
        f"Source run: `{run_dir}`\n\n"
        f"- US policy: `{manifest['policies']['us_policy']}`\n"
        f"- USSR policy: `{manifest['policies']['ussr_policy']}`\n\n"
        "Selection is based on `scripts/rank_ppo_checkpoints.py` side-specific eval scores.\n",
        encoding="utf-8",
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    temp_dir.rename(out_dir)
    print(json.dumps({"out_dir": str(out_dir), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
