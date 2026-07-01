from __future__ import annotations

import math

import numpy as np

from struggle_ai.features import HEURISTIC_PRIOR_FEATURE


def register_masked_model() -> None:
    try:
        from ray.rllib.models import ModelCatalog
    except ImportError as exc:
        raise RuntimeError("RLlib is not installed") from exc

    ModelCatalog.register_custom_model("twilight_masked_model", TorchMaskedActions)


try:
    import torch
    import torch.nn as nn
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.torch_utils import FLOAT_MIN
except ImportError:
    torch = None
    nn = None
    TorchModelV2 = None
    FLOAT_MIN = -3.4e38


if TorchModelV2 is not None:

    def mlp(input_size: int, hidden: int, output_size: int, layers: int = 2) -> nn.Sequential:
        modules: list[nn.Module] = []
        last = input_size
        for _ in range(max(0, layers - 1)):
            modules.extend([nn.Linear(last, hidden), nn.ReLU()])
            last = hidden
        modules.append(nn.Linear(last, output_size))
        modules.append(nn.ReLU())
        return nn.Sequential(*modules)


    class TorchMaskedActions(TorchModelV2, nn.Module):
        """Structured board/card/action encoder with legal-action masking.

        This keeps the RLlib action space as padded discrete candidates, but the
        logits are produced by scoring each legal action against an encoded game
        state instead of using one flat output head.
        """

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            nn.Module.__init__(self)
            self._value_out = None

            original = getattr(obs_space, "original_space", obs_space)
            cfg = model_config.get("custom_model_config", {})
            hidden = int(cfg.get("hidden", 256))
            graph_layers = int(cfg.get("graph_layers", 2))
            self.graph_neighbor_hops = max(1, int(cfg.get("graph_neighbor_hops", 2)))
            self.heuristic_prior_scale = float(cfg.get("heuristic_prior_scale", 2.0))
            self.setup_heuristic_prior_scale = float(cfg.get("setup_heuristic_prior_scale", 0.0))
            self.policy_temperature = max(0.05, float(cfg.get("policy_temperature", 1.0)))
            self.model_arch = str(cfg.get("model_arch", "feedforward"))
            self.use_transformer_history = self.model_arch == "transformer_history"

            global_size = int(np.prod(original["global"].shape))
            event_size = int(np.prod(original["events"].shape))
            country_size = int(original["countries"].shape[-1])
            region_size = int(original["regions"].shape[-1])
            card_size = int(original["cards"].shape[-1])
            action_size = int(original["action_features"].shape[-1])

            self.global_encoder = mlp(global_size + event_size, hidden, hidden)
            self.country_in = mlp(country_size, hidden, hidden)
            self.country_graph = nn.ModuleList(
                nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.LayerNorm(hidden))
                for _ in range(graph_layers)
            )
            self.region_encoder = mlp(region_size, hidden, hidden)
            self.card_encoder = mlp(card_size, hidden, hidden)
            self.action_encoder = mlp(action_size, hidden, hidden)
            if self.use_transformer_history:
                self.history_side_encoder = nn.Linear(1, hidden)
                self.history_turn_ar_encoder = nn.Linear(2, hidden)
                self.history_vp_defcon_encoder = nn.Linear(2, hidden)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=int(cfg.get("history_attention_heads", 4)),
                    dim_feedforward=hidden * 4,
                    dropout=float(cfg.get("history_dropout", 0.1)),
                    batch_first=True,
                    activation="gelu",
                )
                self.history_transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=int(cfg.get("history_layers", 2)),
                )
                self.card_history_encoder = mlp(int(original["card_history"].shape[-1]), hidden, hidden)
                card_encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=int(cfg.get("history_attention_heads", 4)),
                    dim_feedforward=hidden * 4,
                    dropout=float(cfg.get("history_dropout", 0.1)),
                    batch_first=True,
                    activation="gelu",
                )
                self.card_history_transformer = nn.TransformerEncoder(
                    card_encoder_layer,
                    num_layers=int(cfg.get("card_history_layers", cfg.get("history_layers", 2))),
                )
            self.state_encoder = nn.Sequential(
                nn.Linear(hidden * (6 if self.use_transformer_history else 4), hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.policy_query = nn.Linear(hidden, hidden)
            self.action_bias = nn.Linear(hidden, 1)
            self.value = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs"]
            state_embedding = self.encode_state(obs)
            action_embedding = self.action_encoder(obs["action_features"].float())
            query = self.policy_query(state_embedding).unsqueeze(1)
            logits = (action_embedding * query).sum(dim=-1) / math.sqrt(action_embedding.shape[-1])
            logits = logits + self.action_bias(action_embedding).squeeze(-1)
            heuristic_prior = obs["action_features"].float()[..., HEURISTIC_PRIOR_FEATURE]
            logits = logits + self.heuristic_prior_scale * heuristic_prior
            if self.setup_heuristic_prior_scale:
                global_features = obs["global"].float()
                is_setup = (
                    (global_features[:, 2] <= 0.101)
                    & (global_features[:, 3] <= 0.001)
                    & (global_features[:, 23] > 0.5)
                ).float().unsqueeze(1)
                logits = logits + self.setup_heuristic_prior_scale * is_setup * heuristic_prior
            logits = logits / self.policy_temperature
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            mask = obs["action_mask"].float()
            inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
            self._value_out = torch.nan_to_num(self.value(state_embedding).squeeze(1), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            return logits + inf_mask, state

        def encode_state(self, obs):
            global_embedding = self.global_encoder(
                torch.cat(
                    [
                        obs["global"].float().flatten(start_dim=1),
                        obs["events"].float().flatten(start_dim=1),
                    ],
                    dim=1,
                )
            )
            country_embedding = self.encode_countries(
                obs["countries"].float(),
                obs["country_adjacency"].float(),
            )
            region_embedding = self.region_encoder(obs["regions"].float()).mean(dim=1)
            card_embedding = self.card_encoder(obs["cards"].float()).mean(dim=1)
            embeddings = [global_embedding, country_embedding, region_embedding, card_embedding]
            if self.use_transformer_history:
                embeddings.append(self.encode_history(obs))
                embeddings.append(self.encode_card_history(obs))
            return self.state_encoder(
                torch.cat(embeddings, dim=1)
            )

        def encode_history(self, obs):
            action_embedding = self.action_encoder(obs["history_actions"].float())
            side_embedding = self.history_side_encoder(obs["history_sides"].float().unsqueeze(-1))
            turn_embedding = self.history_turn_ar_encoder(obs["history_turn_ar"].float())
            vp_defcon_embedding = self.history_vp_defcon_encoder(obs["history_vp_defcon"].float())
            h = action_embedding + side_embedding + turn_embedding + vp_defcon_embedding
            mask = obs["history_mask"].float()
            key_padding_mask = mask <= 0.5
            if key_padding_mask.ndim == 2:
                all_empty = key_padding_mask.all(dim=1)
                if all_empty.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_empty, 0] = False
            encoded = self.history_transformer(h, src_key_padding_mask=key_padding_mask)
            weights = mask.unsqueeze(-1)
            return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        def encode_card_history(self, obs):
            h = self.card_history_encoder(obs["card_history"].float())
            mask = obs["card_history_mask"].float()
            key_padding_mask = mask <= 0.5
            if key_padding_mask.ndim == 2:
                all_empty = key_padding_mask.all(dim=1)
                if all_empty.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_empty, 0] = False
            encoded = self.card_history_transformer(h, src_key_padding_mask=key_padding_mask)
            weights = mask.unsqueeze(-1)
            return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        def encode_countries(self, countries, adjacency):
            h = self.country_in(countries)
            eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
            adj = adjacency + eye
            adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
            msg_adj = self.neighborhood_adjacency(adj)
            for layer in self.country_graph:
                msg = torch.bmm(msg_adj, h)
                h = h + layer(torch.cat([h, msg], dim=-1))
            return h.mean(dim=1)

        def neighborhood_adjacency(self, adj):
            if self.graph_neighbor_hops <= 1:
                return adj
            neighborhood = adj
            frontier = adj
            for _ in range(1, self.graph_neighbor_hops):
                frontier = torch.bmm(frontier, adj)
                neighborhood = neighborhood + frontier
            return neighborhood / neighborhood.sum(dim=-1, keepdim=True).clamp_min(1.0)

        def value_function(self):
            return self._value_out
else:

    class TorchMaskedActions:  # type: ignore[no-redef]
        pass
