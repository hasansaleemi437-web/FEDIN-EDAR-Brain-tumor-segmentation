"""FedIN-R3EDAR: Core Aggregation and Regional Weighting Engine.

Handles:
1. Global class proportion calculation: pi_r = sum(q_k,r) / sum(N_k)
2. Smoothed regional support and reliability-aware weight computation:
       alpha_{k,r} = q~_{k,r} * (rho_{k,r} + eps)^gamma / sum_j(...)
3. Model parameter aggregation:
   - Private InstanceNorm: preserved locally (not aggregated)
   - Final output layer (weight & bias): Channel 0 via FedAvg, Channels 1,2,3 via alpha_{k,r}
   - Shared backbone layers: sample-weighted FedAvg
"""

from __future__ import annotations

import numpy as np
import torch

OUTPUT_REGIONS = ("ncr", "ed", "et")
CHANNEL_MAP = {"ncr": 1, "ed": 2, "et": 3}
_EPS = 1e-12


def compute_global_proportions(
    regional_voxels_by_node: dict[int, dict[str, float]],
    total_foreground_by_node: dict[int, float],
) -> dict[str, float]:
    """Calculate global class proportions pi_r = sum(q_k,r) / sum(N_k)."""
    total_fg = sum(total_foreground_by_node.values())
    if total_fg <= 0:
        return {r: 1.0 / 3.0 for r in OUTPUT_REGIONS}

    return {
        r: sum(regional_voxels_by_node.get(nid, {}).get(r, 0.0) for nid in regional_voxels_by_node) / total_fg
        for r in OUTPUT_REGIONS
    }


def compute_regional_weights(
    node_ids: list[int],
    reliability_by_node: dict[int, dict[str, float]],
    regional_voxels_by_node: dict[int, dict[str, float]],
    total_foreground_by_node: dict[int, float],
    global_proportions: dict[str, float],
    use_reliability: bool = True,
    kappa: float = 0.1,
    gamma: float = 1.0,
    eps: float = _EPS,
) -> dict[str, np.ndarray]:
    """Compute normalized region-wise aggregation weights alpha_{k,r} for NCR, ED, and ET.

    Formula:
        q~_{k,r} = q_{k,r} + kappa * pi_r * N_k
        score_{k,r} = q~_{k,r} * (rho_{k,r} + eps)^gamma
        alpha_{k,r} = score_{k,r} / sum_j(score_{j,r})
    """
    alpha_weights: dict[str, np.ndarray] = {}

    for r in OUTPUT_REGIONS:
        pi_r = global_proportions.get(r, 1.0 / 3.0)
        raw_scores = []

        for node_id in node_ids:
            q_kr = regional_voxels_by_node.get(node_id, {}).get(r, 0.0)
            n_k = total_foreground_by_node.get(node_id, 1.0)
            q_tilde = q_kr + kappa * pi_r * n_k

            rho = reliability_by_node.get(node_id, {}).get(r, 1.0) if use_reliability else 1.0
            score = q_tilde * ((rho + eps) ** gamma)
            raw_scores.append(max(score, eps))

        scores_arr = np.asarray(raw_scores, dtype=np.float64)
        total = scores_arr.sum()
        alpha_weights[r] = scores_arr / (total if total > 0 else 1.0)

    return alpha_weights


def aggregate_r3edar_layers(
    reference: dict[str, torch.Tensor],
    states: list[dict[str, torch.Tensor]],
    sample_weights: np.ndarray | list[float],
    alpha_weights: dict[str, np.ndarray],
    output_keys: tuple[str, str | None],
    local_in_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Perform FedIN-R3EDAR model parameter aggregation.

    1. Private InstanceNorm: untouched (stays local)
    2. Final output layer (weight & bias):
       - Channel 0 (Background): sample-size weighted FedAvg
       - Channels 1, 2, 3 (NCR, ED, ET): region-wise alpha_{k,r} weighted
    3. Shared backbone layers: sample-size weighted FedAvg
    """
    output_weight_key, output_bias_key = output_keys
    output_keys_set = {output_weight_key}
    if output_bias_key is not None:
        output_keys_set.add(output_bias_key)

    aggregated: dict[str, torch.Tensor] = {}

    for name, ref_val in reference.items():
        # 1. Private InstanceNorm or non-floating point buffers
        if name in local_in_keys or not torch.is_floating_point(ref_val):
            aggregated[name] = ref_val.detach().cpu().clone()

        # 2. Final Output Layer Convolution
        elif name in output_keys_set:
            accumulated = torch.zeros_like(ref_val, dtype=torch.float64)

            # Channel 0 (Background): FedAvg
            for state, w in zip(states, sample_weights):
                accumulated[0] += float(w) * state[name][0].detach().cpu().to(torch.float64)

            # Channels 1, 2, 3 (NCR, ED, ET): Region-wise alpha
            for r in OUTPUT_REGIONS:
                ch = CHANNEL_MAP[r]
                for state, a in zip(states, alpha_weights[r]):
                    accumulated[ch] += float(a) * state[name][ch].detach().cpu().to(torch.float64)

            aggregated[name] = accumulated.to(dtype=ref_val.dtype)

        # 3. All other shared backbone layers: Standard FedAvg
        else:
            val = sum(
                float(w) * state[name].detach().cpu().to(torch.float64)
                for state, w in zip(states, sample_weights)
            )
            aggregated[name] = val.to(dtype=ref_val.dtype)

    return aggregated
