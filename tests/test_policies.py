from struggle_ai.policies import score_action


def test_heuristic_prior_prefers_spacing_defcon_risk_card_at_defcon_two():
    obs = {"defcon": 2}
    prompt = "USSR playing <span>Duck and Cover</span>"
    unsafe = {"type": "saito_choice", "decision": "option", "value": "ops", "prompt": prompt}
    space = {"type": "saito_choice", "decision": "option", "value": "space", "prompt": prompt}

    assert score_action(space, obs=obs) > score_action(unsafe, obs=obs)
    assert score_action(unsafe, obs=obs) < 0


def test_heuristic_prior_penalizes_defcon_two_defcon_lowering_headline():
    obs = {"defcon": 2}
    duck = {"type": "saito_choice", "decision": "list", "value": "duckandcover", "prompt": "US pick headline card"}
    containment = {"type": "saito_choice", "decision": "list", "value": "containment", "prompt": "US pick headline card"}

    assert score_action(containment, obs=obs) > score_action(duck, obs=obs)
