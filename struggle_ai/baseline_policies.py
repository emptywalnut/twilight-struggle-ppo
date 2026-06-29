from __future__ import annotations

import numpy as np

try:
    from ray.rllib.policy.policy import Policy
except ImportError:
    Policy = object  # type: ignore[assignment,misc]


class RandomLegalPolicy(Policy):
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        rng = np.random.default_rng()
        actions = []
        for obs in obs_batch:
            mask = np.asarray(obs.get("action_mask", []), dtype=np.float32)
            legal = np.flatnonzero(mask > 0.5)
            actions.append(int(rng.choice(legal)) if len(legal) else 0)
        return np.asarray(actions, dtype=np.int64), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        return None


class HeuristicPolicy(RandomLegalPolicy):
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        actions = []
        for obs in obs_batch:
            mask = np.asarray(obs.get("action_mask", []), dtype=np.float32)
            legal = np.flatnonzero(mask > 0.5)
            if len(legal) == 0:
                actions.append(0)
                continue
            priors = np.asarray(obs.get("action_features", []), dtype=np.float32)
            if priors.ndim == 2 and priors.shape[1] > 0:
                scores = priors[legal, -1]
                actions.append(int(legal[int(np.argmax(scores))]))
            else:
                actions.append(int(legal[0]))
        return np.asarray(actions, dtype=np.int64), [], {}
