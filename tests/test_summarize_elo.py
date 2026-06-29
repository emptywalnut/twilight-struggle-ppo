import json
import subprocess
import sys


def test_summarize_elo_splits_side_leaderboards(tmp_path):
    path = tmp_path / "elo_ratings.json"
    path.write_text(
        json.dumps(
            {
                "k_factor": 32.0,
                "matches": [{"side": "us"}, {"side": "ussr"}],
                "leaderboards": {
                    "us": {
                        "ratings": {
                            "steps-1:us_policy": 1510.0,
                            "initial:ussr_policy": 1490.0,
                        },
                        "games": {
                            "steps-1:us_policy": 10,
                            "initial:ussr_policy": 10,
                        },
                    },
                    "ussr": {
                        "ratings": {
                            "random:us_policy": 1480.0,
                            "final:ussr_policy": 1520.0,
                        },
                        "games": {
                            "random:us_policy": 10,
                            "final:ussr_policy": 10,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/summarize_elo.py", str(path)],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["match_count"] == 2
    assert payload["leaderboards"]["us"][0]["player"] == "steps-1:us_policy"
    assert payload["leaderboards"]["us"][0]["kind"] == "checkpoint"
    assert payload["leaderboards"]["ussr"][0]["player"] == "final:ussr_policy"
