from struggle_ai.train_rllib import (
    HEURISTIC_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
    league_fixed_policy_id,
    league_ids_by_side_and_pool,
    league_policy_sources,
    league_train_policy_ids,
    normalize_sampling,
)
from struggle_ai.league import train_command


def test_league_policy_sources_from_manifest():
    manifest = {
        "seed_policies": {"us": "seed-us", "ussr": "seed-ussr"},
        "best_pool": {"us": [{"checkpoint": "best-us"}], "ussr": [{"checkpoint": "best-ussr"}]},
        "snapshots": {"us": [{"checkpoint": "hist-us"}], "ussr": [{"checkpoint": "hist-ussr"}]},
        "exploiters": {"us": {"checkpoint": "exp-us"}, "ussr": "exp-ussr"},
    }
    sources = league_policy_sources(manifest)
    assert sources[league_fixed_policy_id("us", "current")]["source"] == "seed-us"
    assert sources[league_fixed_policy_id("ussr", "best", 0)]["source"] == "best-ussr"
    assert sources[league_fixed_policy_id("us", "history", 0)]["source"] == "hist-us"
    assert sources[league_fixed_policy_id("ussr", "exploiter")]["source"] == "exp-ussr"

    pools = league_ids_by_side_and_pool(sources)
    assert league_fixed_policy_id("us", "current") in pools["us"]["current"]
    assert league_fixed_policy_id("ussr", "history", 0) in pools["ussr"]["history"]


def test_league_train_policy_ids_and_sampling_normalization():
    assert league_train_policy_ids("main", "both") == ["us_policy", "ussr_policy"]
    assert league_train_policy_ids("exploiter_us", "both") == ["us_policy"]
    assert league_train_policy_ids("main", "ussr") == ["ussr_policy"]

    sampling = normalize_sampling({"current_main": 2, "heuristic": 1})
    assert sampling["current_main"] == 2 / 3
    assert sampling["heuristic"] == 1 / 3
    assert sampling["random_legal"] == 0.0


def test_baseline_policy_ids_are_distinct():
    assert RANDOM_LEGAL_POLICY_ID != HEURISTIC_POLICY_ID


def test_league_transformer_launch_uses_current_transfer_settings(tmp_path):
    class Args:
        python = "python"
        league_dir = tmp_path
        role = "main"
        train_side = "both"
        log_games_every = 25
        episodes = 300
        checkpoint_eval_every_episodes = 100
        eval_games = 10
        eval_min_games_per_side = 10
        eval_history_opponents = 3
        eval_max_episode_steps = 1200
        gpu = 1.0
        num_env_runners = 0
        train_batch_size = 1000
        minibatch_size = 256
        rollout_fragment_length = "256"
        batch_mode = "complete_episodes"
        sample_timeout_s = 300.0
        branch = "hard_filter"
        model_arch = "transformer_history"
        hidden = 256
        history_layers = 2
        history_attention_heads = 4
        history_dropout = 0.05
        graph_layers = 2
        graph_neighbor_hops = 5
        heuristic_prior_scale = 2.0
        setup_heuristic_prior_scale = 0.0
        policy_temperature = 1.0
        entropy_coeff = 0.0
        max_episode_steps = 1200
        extra = []

    manifest = {"seed_policies": {"us": "seed-us", "ussr": "seed-ussr"}}
    cmd = train_command(Args, manifest)

    def has_flag(flag: str, value: str | None = None) -> bool:
        if flag not in cmd:
            return False
        if value is None:
            return True
        idx = cmd.index(flag)
        return idx + 1 < len(cmd) and cmd[idx + 1] == value

    assert has_flag("--multi-agent")
    assert has_flag("--model-arch", "transformer_history")
    assert has_flag("--partial-warmstart")
    assert has_flag("--history-layers", "2")
    assert has_flag("--history-attention-heads", "4")
    assert has_flag("--history-dropout", "0.05")
    assert has_flag("--graph-layers", "2")
    assert has_flag("--graph-neighbor-hops", "5")
    assert has_flag("--terminal-reward-scale", "2.0")
    assert has_flag("--nuke-death-penalty", "1.0")
    assert has_flag("--defcon-risk-pick-penalty", "0.05")
    assert has_flag("--defcon-risk-commit-penalty", "0.15")
    assert has_flag("--max-episode-step-penalty", "0.25")
    assert has_flag("--reward-shaping-final-scale", "0.9")
    assert has_flag("--checkpoint-eval-every-episodes", "100")
    assert has_flag("--eval-games", "10")
    assert has_flag("--eval-min-games-per-side", "10")
    assert has_flag("--eval-history-opponents", "3")
    assert has_flag("--eval-max-episode-steps", "1200")
