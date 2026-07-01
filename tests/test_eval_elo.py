from struggle_ai.train_rllib import (
    SHARED_POLICY_ID,
    EvalPolicySwitcher,
    US_POLICY_ID,
    USSR_POLICY_ID,
    eval_games_per_side,
    eval_benchmark_policy_id,
    metric_label,
    new_elo_state,
    parse_policies_to_train,
    update_elo_match,
)


def test_eval_games_per_side_enforces_minimum():
    assert eval_games_per_side(5, 10) == 10
    assert eval_games_per_side(12, 10) == 12


def test_metric_label_sanitizes_checkpoint_names():
    assert metric_label("steps-0000010000-episodes-00000100") == "steps-0000010000-episodes-00000100"
    assert metric_label("foo/bar:baz") == "foo_bar_baz"


def test_update_elo_match_updates_both_side_board_entries():
    state = new_elo_state(k_factor=32.0)

    player_rating, opponent_rating = update_elo_match(
        state,
        "us",
        "checkpoint-1:us_policy",
        "initial:ussr_policy",
        1.0,
        k_factor=32.0,
        meta={"terminal_reason": "final_scoring"},
    )

    assert player_rating > 1500.0
    assert opponent_rating < 1500.0
    assert state["leaderboards"]["us"]["games"]["checkpoint-1:us_policy"] == 1
    assert state["leaderboards"]["us"]["games"]["initial:ussr_policy"] == 1
    assert state["leaderboards"]["ussr"]["ratings"] == {}
    assert state["matches"][0]["terminal_reason"] == "final_scoring"


def test_update_elo_match_supports_unified_bot_board():
    state = new_elo_state(k_factor=32.0)

    player_rating, opponent_rating = update_elo_match(
        state,
        "bot",
        f"checkpoint-1:{SHARED_POLICY_ID}",
        f"initial:{SHARED_POLICY_ID}",
        0.75,
        k_factor=32.0,
        meta={"games": 20, "aggregate": True},
    )

    assert player_rating > 1500.0
    assert opponent_rating < 1500.0
    assert state["leaderboards"]["bot"]["games"][f"checkpoint-1:{SHARED_POLICY_ID}"] == 1
    assert state["leaderboards"]["us"]["ratings"] == {}
    assert state["matches"][0]["aggregate"] is True


def test_parse_policies_to_train_unified_collapses_side_aliases():
    assert parse_policies_to_train(None, multi_agent=True, policy_sharing="unified") == [SHARED_POLICY_ID]
    assert parse_policies_to_train("us,ussr", multi_agent=True, policy_sharing="unified") == [SHARED_POLICY_ID]


def test_eval_benchmark_policy_id_prefers_split_pair_over_shared_policy():
    class Switcher:
        weights = {
            "split_best": {
                US_POLICY_ID: {"w": 1},
                USSR_POLICY_ID: {"w": 2},
                SHARED_POLICY_ID: {"w": 3},
            }
        }

    side_policy_ids = {"us": SHARED_POLICY_ID, "ussr": SHARED_POLICY_ID}

    assert eval_benchmark_policy_id("split_best", "us", side_policy_ids, Switcher()) == US_POLICY_ID
    assert eval_benchmark_policy_id("split_best", "ussr", side_policy_ids, Switcher()) == USSR_POLICY_ID


def test_eval_policy_switcher_caches_weights_per_policy_id():
    class Policy:
        def __init__(self):
            self.set_calls = 0

        def set_weights(self, _weights):
            self.set_calls += 1

    class Algo:
        def __init__(self):
            self.policies = {"us_policy": Policy(), "ussr_policy": Policy()}

        def get_policy(self, policy_id):
            return self.policies[policy_id]

        def compute_single_action(self, _obs, policy_id=None, explore=False):
            return 0

    algo = Algo()
    switcher = EvalPolicySwitcher(
        algo,
        current_weights={"us_policy": {"w": 1}, "ussr_policy": {"w": 2}},
        benchmark_weights={"initial": {"us_policy": {"w": 3}, "ussr_policy": {"w": 4}}},
    )
    obs = {"action_mask": [1]}

    for _ in range(3):
        switcher.action("us_policy", "current", obs)
        switcher.action("ussr_policy", "initial", obs)

    assert algo.policies["us_policy"].set_calls == 1
    assert algo.policies["ussr_policy"].set_calls == 1

    switcher.action("us_policy", "initial", obs)
    assert algo.policies["us_policy"].set_calls == 2
