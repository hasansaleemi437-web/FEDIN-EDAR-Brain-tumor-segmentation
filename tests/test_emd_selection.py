"""Quick tests for FedIN-R3EDAR Module 1 (EMD-guided collaborator selection).

Uses fake hospital profiles, so no dataset or GPU is needed.
Run with:  python -m pytest tests/test_emd_selection.py -v
"""

import numpy as np

from algorithms.collaborator_selector import (
    EMDGuidedTrainMixin,
    _group_emd,
    select_emd_guided,
)

REGIONS = ("wt", "tc", "et")
NUM_CLIENTS = 33


def _fake_federation(num_clients=NUM_CLIENTS, seed=0):
    """Random 10-bin profiles per region, sample counts and the global profile."""
    rng = np.random.default_rng(seed)

    def histogram():
        values = rng.random(10)
        return values / values.sum()

    profiles = {n: {r: histogram() for r in REGIONS} for n in range(num_clients)}
    counts = {n: float(rng.integers(5, 300)) for n in range(num_clients)}
    total = sum(counts.values())
    global_profile = {
        r: sum(counts[n] * profiles[n][r] for n in profiles) / total
        for r in REGIONS
    }
    return profiles, counts, global_profile


class _DummyBase:
    """Stands in for the Flower strategy behind the mixin."""

    def __init__(self, **kwargs):
        pass


class _EMDStrategy(EMDGuidedTrainMixin, _DummyBase):
    pass


def test_selects_twenty_percent_without_duplicates():
    profiles, counts, global_profile = _fake_federation()
    staleness = {n: 0 for n in profiles}

    selected = select_emd_guided(
        list(profiles), profiles, counts, global_profile, staleness, fraction=0.2
    )

    assert len(selected) == int(NUM_CLIENTS * 0.2)
    assert len(set(selected)) == len(selected)


def test_group_is_closer_to_global_than_random_groups():
    profiles, counts, global_profile = _fake_federation()
    staleness = {n: 0 for n in profiles}

    selected = select_emd_guided(
        list(profiles), profiles, counts, global_profile, staleness, fraction=0.2
    )
    emd_selected = _group_emd(selected, profiles, counts, global_profile)

    rng = np.random.default_rng(1)
    random_emds = [
        _group_emd(
            list(rng.choice(list(profiles), len(selected), replace=False)),
            profiles, counts, global_profile,
        )
        for _ in range(200)
    ]

    assert emd_selected < np.mean(random_emds)


def test_round_one_selects_all_hospitals_for_profiling():
    strategy = _EMDStrategy(collaborator_selection_fraction=0.2)
    node_ids = list(range(NUM_CLIENTS))

    assert strategy._select_nodes(node_ids, server_round=1, is_fedindar=True) == node_ids


def test_falls_back_to_sliding_window_without_profiles():
    strategy = _EMDStrategy(collaborator_selection_fraction=0.2)

    selected = strategy._select_nodes(list(range(NUM_CLIENTS)), server_round=2, is_fedindar=True)

    assert len(selected) == int(NUM_CLIENTS * 0.2)


def test_staleness_lets_every_hospital_participate():
    profiles, counts, global_profile = _fake_federation()
    strategy = _EMDStrategy(collaborator_selection_fraction=0.2, emd_staleness_weight=0.1)
    strategy.profile_by_node = profiles
    strategy.count_by_node = counts
    strategy.global_profile = global_profile

    seen = set()
    for server_round in range(2, 22):  # 20 training rounds
        seen.update(strategy._select_nodes(list(profiles), server_round, is_fedindar=True))

    assert seen == set(profiles)