"""Reproducible Sliding-Window Collaborator Selection for Flower Strategies."""

from collections.abc import Iterable
from logging import INFO
from math import ceil
from random import Random
import numpy as np
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp import Grid
from flwr.supercore import log


def select_sliding_window(
    client_ids: list[int],
    server_round: int,
    fraction: float = 0.2,
    seed: int = 42,
) -> list[int]:
    """Select reproducible non-overlapping client windows."""

    ordered = sorted(int(cid) for cid in client_ids)

    w_size = max(1, int(len(ordered) * fraction))
    n_windows = ceil(len(ordered) / w_size)

    cycle, w_idx = divmod(
        max(server_round - 1, 0),
        n_windows,
    )

    pad = (-len(ordered)) % w_size
    padded = ordered + ordered[:pad] if pad else ordered

    Random(seed + cycle).shuffle(padded)

    return padded[
        w_idx * w_size : (w_idx + 1) * w_size
    ]

# ---------------------------------------------------------------------------
# FedIN-R3EDAR (Module 1): EMD-guided collaborator selection
# ---------------------------------------------------------------------------

_SELECTOR_EPS = 1e-12


def _group_emd(
    selected: list[int],
    profiles: dict[int, dict[str, np.ndarray]],
    counts: dict[int, float],
    global_profile: dict[str, np.ndarray],
) -> float:
    """EMD between the combined profile of a group and the global profile.

    The group profile is the sample-count weighted mean of the selected
    hospitals' tumour-burden histograms (one histogram per region).
    """

    total = sum(counts[node] for node in selected)

    distances = []
    for region, target in global_profile.items():
        mixed = sum(
            counts[node] * profiles[node][region]
            for node in selected
        ) / max(total, _SELECTOR_EPS)

        distances.append(
            np.mean(np.abs(np.cumsum(mixed) - np.cumsum(target)))
        )

    return float(np.mean(distances))


def select_emd_guided(
    client_ids: list[int],
    profiles: dict[int, dict[str, np.ndarray]],
    counts: dict[int, float],
    global_profile: dict[str, np.ndarray],
    staleness: dict[int, int],
    fraction: float = 0.2,
    staleness_weight: float = 0.1,
) -> list[int]:
    """Greedy EMD-guided group selection (FedIN-R3EDAR, Module 1).

    Adds one hospital at a time, each time choosing the hospital that makes
    the group's combined profile closest to the global profile (lowest EMD).
    Hospitals that have not been selected for many rounds get a bonus, so
    no hospital is ignored for too long.

        score(k) = EMD(group + k) - staleness_weight * staleness(k) / max_staleness
    """

    ordered = sorted(int(cid) for cid in client_ids if int(cid) in profiles)
    if not ordered:
        return []

    group_size = max(1, int(len(ordered) * fraction))
    max_staleness = max(max(staleness.get(n, 0) for n in ordered), 1)

    selected: list[int] = []
    remaining = list(ordered)

    while len(selected) < group_size and remaining:
        best_node, best_score = None, None

        for node in remaining:
            emd = _group_emd(selected + [node], profiles, counts, global_profile)
            bonus = staleness.get(node, 0) / max_staleness
            score = emd - staleness_weight * bonus

            if best_score is None or score < best_score - 1e-12:
                best_node, best_score = node, score

        selected.append(best_node)
        remaining.remove(best_node)

    return selected


class SlidingWindowTrainMixin:
    """Strategy mixin for reproducible sliding-window client selection."""

    def __init__(
        self,
        *args: Any,
        collaborator_selection_fraction: float = 0.2,
        collaborator_selection_seed: int = 42,
        **kwargs: Any,
    ) -> None:

        if not 0.0 < collaborator_selection_fraction <= 1.0:
            raise ValueError(
                "collaborator_selection_fraction must be in (0, 1]"
            )

        self.fraction = collaborator_selection_fraction
        self.seed = collaborator_selection_seed
        self.collaborator_selection_audit: list[dict] = []

        super().__init__(*args, **kwargs)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:

        node_ids = list(grid.get_node_ids())

        # Keep FedINDAR global state updated
        if hasattr(self, "_global_state"):
            self._global_state = arrays.to_torch_state_dict()

        is_fedindar = hasattr(self, "_messages_with_client_config")

        # Round 1:
        # Every client must send its tumour profile.
        if is_fedindar and server_round == 1:
            selected = node_ids
        else:
            # For FedINDAR, training starts at round 2 (window 0)
            window_round = (server_round - 1) if is_fedindar else server_round

            selected = select_sliding_window(
                node_ids,
                window_round,
                self.fraction,
                self.seed,
            )

        if not selected:
            return []

        self.collaborator_selection_audit.append(
            {
                "server_round": server_round,
                "selected_node_ids": selected,
                "selected_count": len(selected),
                "available_count": len(node_ids),
            }
        )

        log(
            INFO,
            "configure_train: selected %s nodes (out of %s): %s",
            len(selected),
            len(node_ids),
            selected,
        )

        train_config = ConfigRecord(dict(config))
        train_config["server-round"] = server_round

        # FedINDAR-specific configuration
        if hasattr(self, "_messages_with_client_config"):

            base_record = RecordDict(
                {
                    self.arrayrecord_key: arrays,
                    self.configrecord_key: train_config,
                }
            )

            base_messages = list(
                self._construct_messages(
                    base_record,
                    selected,
                    MessageType.TRAIN,
                )
            )

            return self._messages_with_client_config(
                base_messages,
                arrays,
                train_config,
                profile_only=(server_round == 1),
            )

        # Fallback for normal strategies
        record = RecordDict(
            {
                self.arrayrecord_key: arrays,
                self.configrecord_key: train_config,
            }
        )

        return list(
            self._construct_messages(
                record,
                selected,
                MessageType.TRAIN,
            )
        )