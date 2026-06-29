from struggle_ai.train_rllib import EvalPolicySwitcher, eval_games_per_side, metric_label, new_elo_state, update_elo_match


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
