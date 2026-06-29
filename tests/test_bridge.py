import json
import subprocess

from struggle_ai.bridge_client import TwilightBridgeClient


def finish_setup(bridge: TwilightBridgeClient, obs: dict) -> dict:
    while obs["phase"].startswith("setup_"):
        obs = bridge.step(obs["legal_actions"][0])["observation"]
    return obs


def test_preset_and_card_scope():
    with TwilightBridgeClient() as bridge:
        preset = bridge.preset()
        assert preset["id"] == "optional_us_plus_2"
        assert preset["deck"] == "optional"
        assert preset["usbonus"] == 2
        assert preset["backend"] == "saito"
        card_ids = {card["id"] for card in bridge.cards()}
        for card_id in ["muslimrevolution", "cambridge", "defectors", "che", "tehran", "iraniraq", "yuri", "awacs"]:
            assert card_id in card_ids
        for card_id in ["berlinagreement", "pinochet", "perestroika", "gouzenkoaffair", "communistrevolution"]:
            assert card_id not in card_ids


def test_hidden_hand_observation_and_legal_actions():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(3)
        assert obs["side"] == obs["current_player"] == "ussr"
        assert obs["phase"] == "setup_ussr"
        assert obs["hand_count"]["us"] == 8
        assert obs["hand_count"]["ussr"] == 9
        assert "china" in obs["hand"]
        us_obs = bridge.observe("us")
        assert "hand" in us_obs
        assert us_obs["hand"] != obs["hand"]
        assert us_obs["legal_actions"] == []
        assert len(obs["legal_actions"]) > 0


def test_initial_placement_is_wired_before_headline():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(1)
        assert obs["phase"] == "setup_ussr"
        assert "Eastern Europe" in obs["prompt"]
        for _ in range(6):
            obs = bridge.step(next(action for action in obs["legal_actions"] if action["value"] == "poland"))["observation"]
        assert obs["phase"] == "setup_us"
        assert "Western Europe" in obs["prompt"]
        for _ in range(7):
            obs = bridge.step(next(action for action in obs["legal_actions"] if action["value"] == "westgermany"))["observation"]
        assert obs["phase"] == "setup_us_bonus"
        assert "+2" in obs["prompt"]
        for _ in range(2):
            obs = bridge.step(next(action for action in obs["legal_actions"] if action["value"] == "iran"))["observation"]
        assert obs["phase"] == "headline_ussr"
        countries = {country["id"]: country for country in obs["countries"]}
        assert countries["poland"]["ussr"] == 6
        assert countries["westgermany"]["us"] == 7
        assert countries["iran"]["us"] == 3


def test_headline_phase_is_wired_before_ar1():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        assert obs["phase"] == "headline_ussr"
        assert "headline" in obs["prompt"].lower()
        assert obs["legal_actions"]
        ussr_headline = obs["legal_actions"][0]["value"]

        obs = bridge.step(obs["legal_actions"][0])["observation"]
        assert obs["phase"] == "headline_us"
        assert obs["current_player"] == "us"
        assert "headline" in obs["prompt"].lower()
        us_headline = obs["legal_actions"][0]["value"]

        for _ in range(40):
            if obs["phase"] == "action" and obs["action_round"] == 0 and obs["current_player"] == "ussr":
                break
            assert obs["legal_actions"]
            obs = bridge.step(obs["legal_actions"][0])["observation"]

        log = bridge.log()
        assert log["headline"]["ussr"] == ussr_headline
        assert log["headline"]["us"] == us_headline
        assert any("headlines" in entry for entry in log["log"])
        assert obs["phase"] == "action"
        assert obs["action_round"] == 0


def test_initial_deal_contains_only_early_war_cards():
    with TwilightBridgeClient() as bridge:
        cards = {card["id"]: card for card in bridge.cards()}
        for seed in range(1, 21):
            ussr_obs = bridge.reset(seed)
            us_obs = bridge.observe("us")
            initial_hands = ussr_obs["hand"] + us_obs["hand"]
            assert initial_hands
            assert all(cards[card_id]["era"] == "early" or card_id == "china" for card_id in initial_hands)


def test_headline_submit_status_does_not_advance_action_round():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        for value in ["decolonization", "arabisraeli"]:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]

        assert "Africa or Southeast Asia" in obs["prompt"]
        first = "bulgaria"
        first = "burma"
        action = next(action for action in obs["legal_actions"] if action["value"] == first)
        obs = bridge.step(action)["observation"]
        assert first not in {action["value"] for action in obs["legal_actions"]}

        for value in ["laos", "thailand", "vietnam"]:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]

        assert obs["phase"] == "action"
        assert obs["action_round"] == 0
        assert obs["current_player"] == "ussr"
        assert obs["legal_actions"]


def test_random_legal_self_play_reaches_terminal_or_progresses():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(5)
        for _ in range(300):
            if obs["terminal"]:
                break
            legal = obs["legal_actions"]
            assert legal
            result = bridge.step(legal[0])
            obs = result["observation"]
        assert obs["turn"] >= 1


def test_ops_influence_placement_consumes_ops():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        for value in ["decolonization", "arabisraeli", "burma", "laos", "thailand", "vietnam",
                      "suezcrisis", "event", "formosan", "event",
                      "naziscientist", "event", "degaulle", "ops", "before_ops", "place"]:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]

        assert obs["prompt"] == "Place 3 influence"
        assert all(action["decision"] == "country_mouseup" for action in obs["legal_actions"])
        action = next(action for action in obs["legal_actions"] if action["value"] == "iran")
        obs = bridge.step(action)["observation"]
        assert obs["prompt"] != "Place 3 influence"


def test_one_per_country_event_placements_are_not_reoffered():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        for value in ["decolonization", "arabisraeli"]:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]

        assert "Africa or Southeast Asia" in obs["prompt"]
        first = "sudan"
        action = next(action for action in obs["legal_actions"] if action["value"] == first)
        obs = bridge.step(action)["observation"]
        assert first not in {action["value"] for action in obs["legal_actions"]}


def test_headless_submit_status_advances_action_round():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        for _ in range(300):
            if obs["terminal"] or obs["action_round"] > 2:
                break
            obs = bridge.step(obs["legal_actions"][0])["observation"]

        assert "submitting moves" not in obs["prompt"].lower()
        assert obs["action_round"] > 0 or obs["terminal"]


def test_ops_menu_hides_targetless_coup_and_realign_options():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.phase = 'action';
env.currentPlayer = 'us';
env.twilight.game.state.defcon = 5;
for (const country of Object.values(env.twilight.countries)) {
  country.us = 0;
  country.ussr = 0;
}
env.adapter.setOptions(
  'US plays 3 OPS:',
  '<ul><li class="option" id="place">place influence</li><li class="option" id="coup">launch coup</li><li class="option" id="realign">realign country</li></ul>',
  () => {}
);
console.log(JSON.stringify(env.adapter.pending.choices.map((choice) => choice.value)));
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == ["place"]


def test_targetless_junta_free_ops_exposes_narrow_skip_and_clears_limits():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.phase = 'action';
env.currentPlayer = 'us';
env.twilight.game.player = 2;
env.twilight.game.state.events.junta = 1;
env.twilight.game.state.limit_placement = 1;
env.twilight.game.state.limit_milops = 1;
env.twilight.game.state.limit_region = ['europe', 'africa', 'mideast', 'asia', 'seasia'];
env.twilight.game.queue = ['ops\tus\tjunta\t2'];
for (const country of Object.values(env.twilight.countries)) {
  country.us = 0;
  country.ussr = 0;
}
env.adapter.setOptions(
  'US plays 2 OPS:',
  '<ul><li class="option" id="coup">launch coup</li><li class="option" id="realign">realign country</li></ul>',
  () => {}
);
const choices = env.adapter.pending.choices.map((choice) => choice.value);
env.stepAsync(env.adapter.pending.choices[0]).then(() => {
  console.log(JSON.stringify({
    choices,
    nextChoices: (env.adapter.pending && env.adapter.pending.choices || []).map((choice) => choice.value),
    limitRegion: env.twilight.game.state.limit_region,
    limitPlacement: env.twilight.game.state.limit_placement,
    junta: env.twilight.game.state.events.junta,
    logTail: env.log.slice(-3),
  }));
}).catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["choices"] == ["__skip_targetless_junta_ops__"]
    assert payload["nextChoices"]
    assert payload["limitRegion"] == []
    assert payload["limitPlacement"] == 0
    assert payload["junta"] == 0
    assert any("free OPS skipped" in entry for entry in payload["logTail"])


def test_targetless_non_junta_ops_exposes_generic_skip():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.phase = 'action';
env.currentPlayer = 'ussr';
env.twilight.game.player = 1;
env.twilight.game.queue = ['ops\tussr\tnuclearsubs\t2'];
env.isLegalCoupTarget = () => false;
env.isLegalRealignTarget = () => false;
env.adapter.setOptions(
  'USSR plays 2 OPS:',
  '<ul><li class="option" id="coup">launch coup</li><li class="option" id="realign">realign country</li></ul>',
  () => {}
);
console.log(JSON.stringify(env.adapter.pending.choices.map((choice) => [choice.decision, choice.value])));
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == [["skip_targetless_ops", "__skip_targetless_ops__"]]


def test_targetless_non_junta_ops_skip_resolves_ops_marker():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
(async () => {
  const env = new SaitoTwilightKernel();
  env.reset(1);
  env.phase = 'action';
  env.currentPlayer = 'ussr';
  env.twilight.game.player = 1;
  env.twilight.game.queue = ['ops\tussr\tnuclearsubs\t2'];
  env.isLegalCoupTarget = () => false;
  env.isLegalRealignTarget = () => false;
  env.adapter.setOptions(
    'USSR plays 2 OPS:',
    '<ul><li class="option" id="coup">launch coup</li><li class="option" id="realign">realign country</li></ul>',
    () => {}
  );
  await env.stepAsync({type: 'saito_bridge', value: '__skip_targetless_ops__'});
  console.log(JSON.stringify({
    queue: env.twilight.game.queue,
    logTail: env.log.slice(-2),
  }));
})().catch((err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert any("no legal OPS target mode" in entry for entry in payload["logTail"])
    assert payload["queue"]
