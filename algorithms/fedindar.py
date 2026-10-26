"""FedIN-EDAR: local InstanceNorm plus EMD and temporal adaptive FedProx."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from models.unet import instance_norm_state_keys, output_conv_state_keys
from .r3edar import (
    OUTPUT_REGIONS,
    aggregate_r3edar_layers,
    compute_global_proportions,
    compute_regional_weights,
)

PROFILE_BINS = 10
REGIONS = ("wt", "tc", "et")
_EPS = 1e-12

TUMOR_BURDEN_BIN_EDGES = np.asarray(
    [
        0.0,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        1.0,
    ],
    dtype=np.float64,
)


def profile_metric_key(region: str, bin_index: int) -> str:
    return f"tumor-profile-{region}-{bin_index}"


def _emd(
    histogram_a: np.ndarray,
    histogram_b: np.ndarray,
) -> float:
    """Calculate 1-D Earth Mover's Distance between normalized histograms."""

    if (
        histogram_a.shape != (PROFILE_BINS,)
        or histogram_b.shape != (PROFILE_BINS,)
    ):
        raise ValueError(
            f"Expected {PROFILE_BINS}-bin tumour-burden histograms"
        )

    return float(
        np.mean(
            np.abs(
                np.cumsum(histogram_a) - np.cumsum(histogram_b)
            )
        )
    )


class FedINDARStrategy(FedAvg):
    """FedIN-EDAR with EMD-based and temporally adaptive proximal penalties."""

    def __init__(
        self,
        *,
        optimization_rounds: int,
        base_mu: float = 0.01,
        alpha: float = 2.0,
        temporal_beta: float = 0.5,
        min_mu: float = 0.001,
        max_mu: float = 0.1,
        redar_kappa: float = 0.1,
        redar_gamma: float = 1.0,
        fedr3_warmup_round: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if (
            optimization_rounds < 1
            or base_mu < 0
            or alpha < 0
            or temporal_beta < 0
            or min_mu < 0
            or max_mu < min_mu
        ):
            raise ValueError(
                "Invalid FedIN-EDAR penalty configuration"
            )

        self.optimization_rounds = optimization_rounds
        self.base_mu = base_mu
        self.alpha = alpha
        self.temporal_beta = temporal_beta
        self.min_mu = min_mu
        self.max_mu = max_mu

        self.local_in_keys = instance_norm_state_keys()

        self.mu_by_node: dict[int, float] = {}
        self.profile_mu_by_node: dict[int, float] = {}
        
        # FedIN-R3EDAR (Module 1): per-client profiles kept for EMD-guided selection
        self.profile_by_node: dict[int, dict[str, np.ndarray]] = {}
        self.count_by_node: dict[int, float] = {}
        self.global_profile: dict[str, np.ndarray] | None = None

        # FedIN-R3EDAR: Regional voxel counts and global class proportions
        self.regional_voxels_by_node: dict[int, dict[str, float]] = {}
        self.total_foreground_by_node: dict[int, float] = {}
        self.global_proportions: dict[str, float] = {"ncr": 1.0 / 3.0, "ed": 1.0 / 3.0, "et": 1.0 / 3.0}

        self.profile_audit: list[dict[str, Any]] = []
        self.round_audit: list[dict[str, Any]] = []

        self._global_state: dict[str, torch.Tensor] | None = None

        # FedIN-R3EDAR Aggregation parameters
        self.redar_kappa = float(redar_kappa)
        self.redar_gamma = float(redar_gamma)
        self.fedr3_warmup_round = int(fedr3_warmup_round)
        self.output_weight_key, self.output_bias_key = output_conv_state_keys()

    def _messages_with_client_config(
        self,
        base_messages: Iterable[Message],
        arrays: ArrayRecord,
        config: ConfigRecord,
        profile_only: bool,
    ) -> list[Message]:
        messages: list[Message] = []

        for base in base_messages:
            node_id = base.metadata.dst_node_id

            if (
                not profile_only
                and node_id not in self.mu_by_node
            ):
                print(
                    f"[FedIN-EDAR] Warning: node {node_id} profile is missing, "
                    f"falling back to base_mu={self.base_mu}",
                    flush=True,
                )
                self.mu_by_node[node_id] = self.base_mu

            node_config = ConfigRecord(dict(config))

            node_config["fedindar-enabled"] = True
            node_config["fedindar-profile-only"] = profile_only
            node_config["proximal_mu"] = float(
                self.mu_by_node.get(
                    node_id,
                    self.base_mu,
                )
            )

            messages.append(
                Message(
                    content=RecordDict(
                        {
                            self.arrayrecord_key: arrays,
                            self.configrecord_key: node_config,
                        }
                    ),
                    message_type=MessageType.TRAIN,
                    dst_node_id=node_id,
                )
            )

        return messages

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._global_state = arrays.to_torch_state_dict()

        messages = super().configure_train(
            server_round,
            arrays,
            config,
            grid,
        )

        return self._messages_with_client_config(
            messages,
            arrays,
            config,
            profile_only=(server_round == 1),
        )

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        if server_round == 1:
            return []

        base_messages = list(
            super().configure_evaluate(
                server_round,
                arrays,
                config,
                grid,
            )
        )

        final_test = (
            server_round == self.optimization_rounds + 1
        )

        messages: list[Message] = []

        for base in base_messages:
            node_config = ConfigRecord(dict(config))

            node_config["fedindar-enabled"] = True
            node_config["fedindar-final-test"] = final_test

            messages.append(
                Message(
                    content=RecordDict(
                        {
                            self.arrayrecord_key: arrays,
                            self.configrecord_key: node_config,
                        }
                    ),
                    message_type=MessageType.EVALUATE,
                    dst_node_id=base.metadata.dst_node_id,
                )
            )

        return messages

    def _aggregate_profile_round(
        self,
        replies: list[Message],
    ) -> tuple[ArrayRecord, MetricRecord]:

        counts = np.asarray(
            [
                float(
                    reply.content["metrics"][
                        self.weighted_by_key
                    ]
                )
                for reply in replies
            ],
            dtype=np.float64,
        )

        weights = counts / max(
            counts.sum(),
            _EPS,
        )

        profiles = [
            {
                region: np.asarray(
                    [
                        float(
                            reply.content["metrics"][
                                profile_metric_key(
                                    region,
                                    i,
                                )
                            ]
                        )
                        for i in range(PROFILE_BINS)
                    ]
                )
                for region in REGIONS
            }
            for reply in replies
        ]

        global_profile = {
            region: sum(
                weight * profile[region]
                for weight, profile in zip(
                    weights,
                    profiles,
                )
            )
            for region in REGIONS
        }

        global_profile = {
            region: values
            / max(values.sum(), _EPS)
            for region, values in global_profile.items()
        }

        # FedIN-R3EDAR (Module 1): keep global profile for EMD-guided selection
        self.global_profile = {
            region: values.copy()
            for region, values in global_profile.items()
        }

        rows = []

        for reply, count, weight, profile in zip(
            replies,
            counts,
            weights,
            profiles,
        ):
            distance = float(
                np.mean(
                    [
                        _emd(
                            profile[region],
                            global_profile[region],
                        )
                        for region in REGIONS
                    ]
                )
            )

            mu = float(
                np.clip(
                    self.base_mu
                    * np.exp(self.alpha * distance),
                    self.min_mu,
                    self.max_mu,
                )
            )

            node_id = reply.metadata.src_node_id

            self.mu_by_node[node_id] = mu
            self.profile_mu_by_node[node_id] = mu

            # FedIN-R3EDAR: record regional voxel counts from client metrics
            m = reply.content["metrics"]
            node_fg = float(m.get("total_foreground_voxels", 1.0))
            self.total_foreground_by_node[node_id] = node_fg
            self.regional_voxels_by_node[node_id] = {
                "ncr": float(m.get("voxel_count_ncr", 0.0)),
                "ed": float(m.get("voxel_count_ed", 0.0)),
                "et": float(m.get("voxel_count_et", 0.0)),
            }

            # FedIN-R3EDAR (Module 1): keep profile and sample count per node
            self.profile_by_node[node_id] = {
                region: profile[region].copy()
                for region in REGIONS
            }
            self.count_by_node[node_id] = float(count)

            rows.append(
                {
                    "node_id": str(node_id),
                    "num_examples": float(count),
                    "sample_weight": float(weight),
                    "emd_distance": distance,
                    "profile_mu": mu,
                    "regional_voxels": self.regional_voxels_by_node[node_id],
                    "total_foreground": node_fg,
                    "histograms": {
                        region: profile[region].tolist()
                        for region in REGIONS
                    },
                }
            )

        # Calculate global class proportions pi_r = sum(q_k,r) / sum(N_k) across all clients
        self.global_proportions = compute_global_proportions(
            self.regional_voxels_by_node,
            self.total_foreground_by_node,
        )

        self.profile_audit = [
            {
                "global_histograms": {
                    region: values.tolist()
                    for region, values in global_profile.items()
                },
                "global_class_proportions": self.global_proportions,
                "clients": rows,
            }
        ]

        if self._global_state is None:
            raise RuntimeError(
                "FedIN-EDAR profile round has no initial global state"
            )

        return (
            ArrayRecord(self._global_state),
            MetricRecord(
                {
                    "profile_phase": 1.0,
                    "num-examples": float(counts.sum()),
                }
            ),
        )

    def _shared_update_divergence(
        self,
        states: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:

        if self._global_state is None:
            raise RuntimeError(
                "FedIN-EDAR has no global model"
            )

        divergences = []

        for state in states:
            absolute_sum = 0.0
            elements = 0

            for name, global_value in self._global_state.items():

                if (
                    name in self.local_in_keys
                    or not torch.is_floating_point(global_value)
                ):
                    continue

                difference = (
                    state[name]
                    .detach()
                    .cpu()
                    .to(torch.float64)
                    - global_value
                    .detach()
                    .cpu()
                    .to(torch.float64)
                )

                absolute_sum += float(
                    difference.abs().sum().item()
                )

                elements += difference.numel()

            divergences.append(
                absolute_sum / max(elements, 1)
            )

        return np.asarray(
            divergences,
            dtype=np.float64,
        )

    def _next_round_penalties(
        self,
        replies: list[Message],
        divergences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        mean = float(divergences.mean())
        std = float(divergences.std())

        if std <= _EPS:
            normalized = np.zeros_like(
                divergences
            )
        else:
            normalized = np.clip(
                (divergences - mean) / std,
                -3.0,
                3.0,
            )

        sent_mu = np.asarray(
            [
                self.mu_by_node[
                    reply.metadata.src_node_id
                ]
                for reply in replies
            ]
        )

        profile_mu = np.asarray(
            [
                self.profile_mu_by_node[
                    reply.metadata.src_node_id
                ]
                for reply in replies
            ]
        )

        next_mu = np.clip(
            profile_mu
            * np.exp(
                self.temporal_beta
                * normalized
            ),
            self.min_mu,
            self.max_mu,
        )

        for reply, mu in zip(
            replies,
            next_mu,
        ):
            self.mu_by_node[
                reply.metadata.src_node_id
            ] = float(mu)

        return (
            sent_mu,
            normalized,
            next_mu,
        )

    def _aggregate_shared_layers(
        self,
        server_round: int,
        replies: list[Message],
    ) -> ArrayRecord:

        states = [
            reply.content[
                self.arrayrecord_key
            ].to_torch_state_dict()
            for reply in replies
        ]

        counts = np.asarray(
            [
                float(
                    reply.content["metrics"][
                        self.weighted_by_key
                    ]
                )
                for reply in replies
            ],
            dtype=np.float64,
        )

        weights = counts / max(
            counts.sum(),
            _EPS,
        )

        divergences = self._shared_update_divergence(
            states
        )

        (
            sent_mu,
            normalized,
            next_mu,
        ) = self._next_round_penalties(
            replies,
            divergences,
        )

        reference = (
            self._global_state
            if self._global_state is not None
            else states[0]
        )

        # --- FedIN-R3EDAR: Regional Reliability-Aware Aggregation ---
        reliability_by_node = {
            reply.metadata.src_node_id: {
                r: float(reply.content["metrics"].get(f"reliability_{r}", 1.0))
                for r in OUTPUT_REGIONS
            }
            for reply in replies
        }
        alpha_weights = compute_regional_weights(
            node_ids=[reply.metadata.src_node_id for reply in replies],
            reliability_by_node=reliability_by_node,
            regional_voxels_by_node=self.regional_voxels_by_node,
            total_foreground_by_node=self.total_foreground_by_node,
            global_proportions=self.global_proportions,
            use_reliability=(server_round > self.fedr3_warmup_round),
            kappa=self.redar_kappa,
            gamma=self.redar_gamma,
        )

        aggregated = aggregate_r3edar_layers(
            reference=reference,
            states=states,
            sample_weights=weights,
            alpha_weights=alpha_weights,
            output_keys=(self.output_weight_key, self.output_bias_key),
            local_in_keys=self.local_in_keys,
        )

        self.round_audit.append(
            {
                "flower_round": server_round,
                "optimization_round": server_round - 1,
                "node_ids": [
                    str(
                        reply.metadata.src_node_id
                    )
                    for reply in replies
                ],
                "num_examples": counts.tolist(),
                "aggregation_weights": weights.tolist(),
                "alpha_weights_ncr": alpha_weights["ncr"].tolist(),
                "alpha_weights_ed": alpha_weights["ed"].tolist(),
                "alpha_weights_et": alpha_weights["et"].tolist(),
                "profile_mu": [
                    float(
                        self.profile_mu_by_node[
                            reply.metadata.src_node_id
                        ]
                    )
                    for reply in replies
                ],
                "proximal_mu_sent": sent_mu.tolist(),
                "update_divergence": divergences.tolist(),
                "normalized_update_divergence": normalized.tolist(),
                "proximal_mu_next_round": next_mu.tolist(),
                "private_instance_norm_keys": sorted(
                    self.local_in_keys
                ),
            }
        )

        return ArrayRecord(aggregated)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[
        ArrayRecord | None,
        MetricRecord | None,
    ]:

        valid, _ = self._check_and_log_replies(
            replies,
            is_train=True,
        )

        if not valid:
            return None, None

        if server_round == 1:
            arrays, metrics = self._aggregate_profile_round(valid)

            # Keep global model for next round
            self._global_state = arrays.to_torch_state_dict()

            return arrays, metrics

        arrays = self._aggregate_shared_layers(
            server_round,
            valid,
        )

        # Update global model after aggregation
        self._global_state = arrays.to_torch_state_dict()

        return (
            arrays,
            self.train_metrics_aggr_fn(
                [
                    reply.content
                    for reply in valid
                ],
                self.weighted_by_key,
            ),
        )