import numpy as np
import pytest

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.features import ACTION_FEATURES, EVENT_FEATURES, HISTORY_LENGTH, MAX_ACTIONS, REGION_FEATURES


def test_env_reset_shapes_and_mask():
    env = TwilightStruggleEnv({"seed": 11})
    try:
        obs, info = env.reset(seed=11)
        assert info["side"] == "ussr"
        assert obs["action_mask"].shape == (MAX_ACTIONS,)
        assert obs["action_features"].shape[0] == MAX_ACTIONS
        assert obs["events"].shape == (EVENT_FEATURES,)
        assert obs["regions"].shape == (7, REGION_FEATURES)
        assert obs["us_hand"].shape == (env.feature_spec.card_count,)
        assert obs["ussr_hand"].shape == (env.feature_spec.card_count,)
        assert np.sum(obs["us_hand"]) == 0
        assert np.sum(obs["ussr_hand"]) == len(env.last_obs["hand"])
        assert obs["country_adjacency"].shape == (env.feature_spec.country_count, env.feature_spec.country_count)
        assert obs["history_actions"].shape == (HISTORY_LENGTH, ACTION_FEATURES)
        assert obs["history_sides"].shape == (HISTORY_LENGTH,)
        assert obs["history_turn_ar"].shape == (HISTORY_LENGTH, 2)
        assert obs["history_vp_defcon"].shape == (HISTORY_LENGTH, 2)
        assert obs["history_mask"].shape == (HISTORY_LENGTH,)
        assert np.sum(obs["history_mask"]) == 0
        assert np.sum(obs["action_mask"]) > 0
        assert np.sum(obs["country_adjacency"]) > 0
    finally:
        env.close()


def test_env_rejects_invalid_action_index():
    env = TwilightStruggleEnv({"seed": 11})
    try:
        obs, _ = env.reset(seed=11)
        invalid = int(np.sum(obs["action_mask"])) + 10
        with pytest.raises(ValueError):
            env.step(invalid)
    finally:
        env.close()


def test_env_can_step_first_legal_action():
    env = TwilightStruggleEnv({"seed": 12})
    try:
        obs, _ = env.reset(seed=12)
        action = int(np.argmax(obs["action_mask"]))
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs["action_mask"].shape == (MAX_ACTIONS,)
        assert next_obs["history_actions"].shape == (HISTORY_LENGTH, ACTION_FEATURES)
        assert np.sum(next_obs["history_mask"]) == 1
        assert isinstance(reward, float)
        assert not truncated
        assert "vp" in info
    finally:
        env.close()


def test_hard_filter_detects_defcon_two_battleground_coup():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        obs, _ = env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "Select country for coup",
            "events": {},
            "countries": [{"id": "iran", "bg": True}],
            "terminal": False,
        }
        actions = [
            {"type": "saito_dom", "decision": "country_mouseup", "value": "iran", "prompt": "Select country for coup"},
            {"type": "saito_choice", "decision": "option", "value": "skip", "prompt": "Select country for coup"},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_battleground_coup"
    finally:
        env.close()


def test_hard_filter_detects_defcon_two_coup_mode_before_target_selection():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "prompt": "US plays 3 OPS:",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "coup", "prompt": "US plays 3 OPS:"},
            {"type": "saito_choice", "decision": "option", "value": "place", "prompt": "US plays 3 OPS:"},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_coup_mode"
    finally:
        env.close()


def test_hard_filter_detects_cuban_missile_crisis_coup_at_high_defcon():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 4,
            "side": "us",
            "prompt": "US plays 3 OPS:",
            "events": {"cubanmissilecrisis": 1},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "coup", "prompt": "US plays 3 OPS:"},
            {"type": "saito_choice", "decision": "option", "value": "realign", "prompt": "US plays 3 OPS:"},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "cuban_missile_crisis_coup_mode"
    finally:
        env.close()


def test_hard_filter_detects_olympic_boycott_at_defcon_two():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": 'US hosts the <span class="showcard" id="olympic">Olympic Games</span>:',
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "boycott", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "participate", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_olympic_boycott"
    finally:
        env.close()


def test_hard_filter_detects_headline_duck_and_cover_how_i_learned_combo():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        duck_obs = {
            **env.last_obs,
            "defcon": 3,
            "side": "us",
            "phase": "headline_us",
            "prompt": "US pick headline card",
            "headline": {"ussr": "howilearned"},
            "events": {},
            "terminal": False,
        }
        duck_actions = [
            {"type": "saito_choice", "decision": "list", "value": "duckandcover", "prompt": duck_obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "containment", "prompt": duck_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(duck_obs, duck_actions)
        assert filtered == [duck_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "headline_defcon_3_duckandcover_with_howilearned"

        hil_obs = {
            **env.last_obs,
            "defcon": 3,
            "side": "ussr",
            "phase": "headline_ussr",
            "prompt": "USSR pick headline card",
            "headline": {"us": "duckandcover"},
            "events": {},
            "terminal": False,
        }
        hil_actions = [
            {"type": "saito_choice", "decision": "list", "value": "howilearned", "prompt": hil_obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "containment", "prompt": hil_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(hil_obs, hil_actions)
        assert filtered == [hil_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "headline_defcon_3_howilearned_with_defcon_lowering_card"
    finally:
        env.close()


def test_hard_filter_detects_defcon_two_defcon_lowering_headline():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "phase": "headline_us",
            "prompt": "US pick headline card",
            "headline": {},
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "list", "value": "duckandcover", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "containment", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "headline_defcon_2_duckandcover"
    finally:
        env.close()


def test_hard_filter_blocks_defcon_two_action_round_risk_card_selection_only():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "phase": "action",
            "prompt": "US pick a card: ",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "list", "value": "wwby", "label": "We Will Bury You", "prompt": obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "marshall", "label": "Marshall Plan", "prompt": obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_pick_card_wwby"

        discard_obs = {**obs, "prompt": "Choose a card to discard:"}
        discard_actions = [
            {"type": "saito_choice", "decision": "list", "value": "fiveyearplan", "label": "Five Year Plan", "prompt": discard_obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "marshall", "label": "Marshall Plan", "prompt": discard_obs["prompt"]},
        ]
        assert env.filtered_legal_actions(discard_obs, discard_actions) == discard_actions
    finally:
        env.close()


def test_hard_filter_blocks_immediate_play_defcon_risk_card_selection():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "phase": "action",
            "prompt": '<span class="showcard" id="starwars">Star Wars</span>: Choose card to play immediately:',
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "list", "value": "duckandcover", "label": "Duck and Cover", "prompt": obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "marshall", "label": "Marshall Plan", "prompt": obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_immediate_card_duckandcover"
    finally:
        env.close()


def test_defcon_risk_action_reward_distinguishes_pick_and_commit():
    env = TwilightStruggleEnv(
        {
            "seed": 13,
            "defcon_risk_pick_penalty": 0.04,
            "defcon_risk_commit_penalty": 0.12,
            "reward_shaping_scale": 0.5,
        }
    )
    try:
        env.reset(seed=13)
        pick_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "phase": "action",
            "prompt": "US pick a card: ",
            "events": {},
            "terminal": False,
        }
        pick_action = {"type": "saito_choice", "decision": "list", "value": "wwby", "label": "We Will Bury You", "prompt": pick_obs["prompt"]}
        reward, reason = env.defcon_risk_action_reward(pick_obs, pick_action)
        assert reward == pytest.approx(-0.02)
        assert reason == "defcon_2_pick_card_wwby"

        commit_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "phase": "action",
            "prompt": "Playing opponent card:",
            "event_name": "Duck and Cover",
            "events": {},
            "terminal": False,
        }
        commit_action = {"type": "saito_choice", "decision": "option", "value": "before_ops", "prompt": commit_obs["prompt"]}
        reward, reason = env.defcon_risk_action_reward(commit_obs, commit_action)
        assert reward == pytest.approx(-0.06)
        assert reason == "defcon_2_duckandcover_event"
    finally:
        env.close()


def test_hard_filter_detects_headline_duck_and_cover_how_i_learned_combo_at_high_defcon():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        duck_obs = {
            **env.last_obs,
            "defcon": 5,
            "side": "us",
            "phase": "headline_us",
            "prompt": "US pick headline card",
            "headline": {"ussr": "howilearned"},
            "events": {},
            "terminal": False,
        }
        duck_actions = [
            {"type": "saito_choice", "decision": "list", "value": "duckandcover", "prompt": duck_obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "containment", "prompt": duck_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(duck_obs, duck_actions)
        assert filtered == [duck_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "headline_defcon_5_duckandcover_with_howilearned"

        set_defcon_obs = {
            **env.last_obs,
            "defcon": 5,
            "side": "ussr",
            "phase": "action",
            "prompt": "Set DEFCON to:",
            "headline": {"us": "duckandcover", "ussr": "howilearned"},
            "events": {},
            "terminal": False,
        }
        set_defcon_actions = [
            {"type": "saito_choice", "decision": "option", "value": "2", "prompt": set_defcon_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "5", "prompt": set_defcon_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(set_defcon_obs, set_defcon_actions)
        assert filtered == [set_defcon_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "howilearned_set_two_before_defcon_lowering_headline"
    finally:
        env.close()


def test_hard_filter_detects_five_year_plan_random_event_trigger_at_defcon_two():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "USSR playing <span>Five Year Plan</span>",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "ops", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "space", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_fiveyearplan_random_event_trigger"
    finally:
        env.close()


def test_hard_filter_detects_defcon_lowering_event_mode():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "prompt": "US playing <span>We Will Bury You</span>",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "ops", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "space", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_wwby_event"
    finally:
        env.close()


def test_hard_filter_allows_defcon_risk_card_selection_before_mode_choice():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "USSR pick a card: ",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "list", "value": "KAL007", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "abmtreaty", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == actions

        mode_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "USSR playing <span>Soviets Shoot Down KAL-007</span>",
            "events": {},
            "terminal": False,
        }
        mode_actions = [
            {"type": "saito_choice", "decision": "option", "value": "ops", "prompt": mode_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "space", "prompt": mode_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(mode_obs, mode_actions)
        assert filtered == [mode_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_kal007_event"
    finally:
        env.close()


def test_hard_filter_detects_defcon_two_forced_coup_event_selection():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "USSR pick a card: ",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "list", "value": "ortega", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "list", "value": "abmtreaty", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == actions

        mode_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "USSR playing <span>Ortega Elected in Nicaragua</span>",
            "events": {},
            "terminal": False,
        }
        mode_actions = [
            {"type": "saito_choice", "decision": "option", "value": "event", "prompt": mode_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "space", "prompt": mode_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(mode_obs, mode_actions)
        assert filtered == [mode_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_ortega_forced_coup_event"
    finally:
        env.close()


def test_hard_filter_detects_che_forced_coup_under_cuban_missile_crisis():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        mode_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "phase": "action",
            "prompt": "USSR playing <span>Che</span>",
            "events": {"cubanmissilecrisis": 1},
            "terminal": False,
        }
        mode_actions = [
            {"type": "saito_choice", "decision": "option", "value": "event", "prompt": mode_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "space", "prompt": mode_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(mode_obs, mode_actions)
        assert filtered == [mode_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_che_forced_coup_event"
    finally:
        env.close()


def test_hard_filter_detects_defcon_lowering_opponent_event_resolution():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "prompt": "Playing opponent card:",
            "event_name": "We Will Bury You",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "before_ops", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "after_ops", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == actions
        assert env.episode_filtered_actions[-1]["filter_skipped"] is True
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_wwby_event"
    finally:
        env.close()


def test_hard_filter_does_not_filter_existing_defcon_lowering_ops_placement_mode():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "USSR plays 3 OPS:",
            "event_name": "",
            "queue_top": "ops\tussr\tduckandcover\t3",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "place", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "space", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == actions
        assert not env.episode_filtered_actions
    finally:
        env.close()


def test_hard_filter_uses_queue_top_card_for_random_event_place_mode():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "USSR plays 3 OPS:",
            "event_name": "",
            "queue_top": "ops\tussr\tfiveyearplan\t3",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "place", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "space", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_fiveyearplan_random_event_trigger"
    finally:
        env.close()


def test_hard_filter_does_not_filter_defcon_lowering_country_placement_commit():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "ussr",
            "prompt": "Place 1 influence:",
            "event_name": "",
            "queue_top": "ops\tussr\tKAL007\t4",
            "events": {},
            "terminal": False,
        }
        actions = [
            {
                "type": "saito_choice",
                "decision": "country_mouseup",
                "value": "southkorea",
                "prompt": fake_obs["prompt"],
            },
            {"type": "saito_choice", "decision": "option", "value": "cancel", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == actions
        assert not env.episode_filtered_actions
    finally:
        env.close()


def test_hard_filter_does_not_treat_other_risk_cards_as_defcon_lowering_placement_commit():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "prompt": "Place 2 influence:",
            "event_name": "",
            "queue_top": "ops\tus\tmissileenvy\t2",
            "events": {},
            "terminal": False,
        }
        actions = [
            {
                "type": "saito_choice",
                "decision": "country_mouseup",
                "value": "canada",
                "prompt": fake_obs["prompt"],
            },
            {
                "type": "saito_choice",
                "decision": "country_mouseup",
                "value": "uk",
                "prompt": fake_obs["prompt"],
            },
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == actions
        assert not env.episode_filtered_actions
    finally:
        env.close()


def test_hard_filter_detects_defcon_three_coup_before_delayed_defcon_event():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        fake_obs = {
            **env.last_obs,
            "defcon": 3,
            "side": "ussr",
            "prompt": "USSR plays 3 OPS:",
            "event_name": "Duck and Cover",
            "events": {},
            "terminal": False,
        }
        actions = [
            {"type": "saito_choice", "decision": "option", "value": "coup", "prompt": fake_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "place", "prompt": fake_obs["prompt"]},
        ]
        filtered = env.filtered_legal_actions(fake_obs, actions)
        assert filtered == [actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_3_coup_before_defcon_lowering_event"
    finally:
        env.close()


def test_hard_filter_detects_summit_lower_and_defcon_set_below_two():
    env = TwilightStruggleEnv({"seed": 13, "defcon_suicide_mode": "hard_filter"})
    try:
        env.reset(seed=13)
        summit_obs = {
            **env.last_obs,
            "defcon": 2,
            "side": "us",
            "prompt": "You win the Summit:",
            "events": {},
            "terminal": False,
        }
        summit_actions = [
            {"type": "saito_choice", "decision": "option", "value": "lower", "prompt": summit_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "same", "prompt": summit_obs["prompt"]},
        ]
        assert env.filtered_legal_actions(summit_obs, summit_actions) == [summit_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_2_summit_lower"

        how_i_learned_obs = {
            **env.last_obs,
            "defcon": 4,
            "side": "ussr",
            "prompt": "Set DEFCON to:",
            "events": {},
            "terminal": False,
        }
        how_i_learned_actions = [
            {"type": "saito_choice", "decision": "option", "value": "1", "prompt": how_i_learned_obs["prompt"]},
            {"type": "saito_choice", "decision": "option", "value": "3", "prompt": how_i_learned_obs["prompt"]},
        ]
        assert env.filtered_legal_actions(how_i_learned_obs, how_i_learned_actions) == [how_i_learned_actions[1]]
        assert env.episode_filtered_actions[-1]["reason"] == "defcon_set_below_two"
    finally:
        env.close()


def test_vp_delta_reward_clamps_malformed_terminal_vp():
    env = TwilightStruggleEnv({"turn_vp_reward_scale": 0.01, "reward_shaping_scale": 1.0})
    try:
        rewards = env.vp_delta_rewards({"vp": 0}, {"vp": -10000})
        assert rewards["us"] == pytest.approx(-0.01)
        assert rewards["ussr"] == pytest.approx(0.01)

        rewards = env.vp_delta_rewards({"vp": 0}, {"vp": 10000})
        assert rewards["us"] == pytest.approx(0.01)
        assert rewards["ussr"] == pytest.approx(-0.01)
    finally:
        env.close()
