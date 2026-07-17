from __future__ import annotations

import numpy as np

try:
    from gymnasium.spaces.utils import flatdim
except ImportError:
    flatdim = None  # type: ignore[assignment]

try:
    from ray.rllib.policy.policy import Policy
except ImportError:
    Policy = object  # type: ignore[assignment,misc]


class RandomLegalPolicy(Policy):
    def __init__(self, observation_space=None, action_space=None, config=None, **kwargs):
        try:
            super().__init__(observation_space, action_space, config or {}, **kwargs)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self._action_mask_slice: slice | None = None
        self._action_features_slice: slice | None = None
        self._action_features_shape: tuple[int, ...] | None = None
        self._configure_flattened_slices(observation_space)

    def _configure_flattened_slices(self, observation_space) -> None:
        if flatdim is None or observation_space is None:
            return
        original = getattr(observation_space, "original_space", observation_space)
        spaces = getattr(original, "spaces", None)
        if not spaces:
            return
        offset = 0
        for key, space in spaces.items():
            width = int(flatdim(space))
            if key == "action_mask":
                self._action_mask_slice = slice(offset, offset + width)
            elif key == "action_features":
                self._action_features_slice = slice(offset, offset + width)
                self._action_features_shape = tuple(getattr(space, "shape", ()) or ())
            offset += width

    def _iter_obs(self, obs_batch):
        if isinstance(obs_batch, dict):
            first = next(iter(obs_batch.values()), [])
            batch_size = len(first)
            for index in range(batch_size):
                yield {key: value[index] for key, value in obs_batch.items()}
            return
        for obs in obs_batch:
            yield obs

    def _mask_and_features(self, obs) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(obs, dict):
            return (
                np.asarray(obs.get("action_mask", []), dtype=np.float32),
                np.asarray(obs.get("action_features", []), dtype=np.float32),
            )
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self._action_mask_slice is None:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
        mask = flat[self._action_mask_slice]
        features = np.asarray([], dtype=np.float32)
        if self._action_features_slice is not None:
            features = flat[self._action_features_slice]
            if self._action_features_shape:
                features = features.reshape(self._action_features_shape)
        return mask, features

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        rng = np.random.default_rng()
        actions = []
        for obs in self._iter_obs(obs_batch):
            mask, _features = self._mask_and_features(obs)
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
        for obs in self._iter_obs(obs_batch):
            mask, priors = self._mask_and_features(obs)
            legal = np.flatnonzero(mask > 0.5)
            if len(legal) == 0:
                actions.append(0)
                continue
            if priors.ndim == 2 and priors.shape[1] > 0:
                scores = priors[legal, -1]
                actions.append(int(legal[int(np.argmax(scores))]))
            else:
                actions.append(int(legal[0]))
        return np.asarray(actions, dtype=np.int64), [], {}
