"""FedR3: Region-Wise Annotation Reliability Estimation.

Estimates hospital label reliability rho_{k,r} in [0, 1] separately for
NCR/NET, ED, and ET using PyTorch built-in Categorical entropy and Dice disagreement:

    rho_{k,r} = EMA[ exp(-a * D_{k,r} - b * H_{k,r}) ]
"""

from __future__ import annotations

import numpy as np
import torch

OUTPUT_REGIONS = ("ncr", "ed", "et")
CLASS_TO_REGION = {1: "ncr", 2: "ed", 3: "et"}


def estimate_local_reliability(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 2,
    a: float = 1.0,
    b: float = 1.0,
) -> dict[str, float]:
    """Estimate region-wise annotation reliability rho_{k,r} in [0, 1] using PyTorch built-ins.

    For region r:
        D_{k,r} = 1 - Dice(pred_r, label_r) is label disagreement
        H_{k,r} = normalized Shannon entropy of class probabilities
        rho_{k,r} = exp(-a * D_{k,r} - b * H_{k,r})
    """
    model.eval()
    entropy_acc = {r: 0.0 for r in OUTPUT_REGIONS}
    disagreement_acc = {r: 0.0 for r in OUTPUT_REGIONS}
    region_count = {r: 0 for r in OUTPUT_REGIONS}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            if labels.ndim == 5:
                labels = labels[:, 0]

            logits = model(images)
            probs = torch.softmax(logits, dim=1)  # [B, 4, D, H, W]
            preds = torch.argmax(probs, dim=1)

            # PyTorch built-in: Shannon entropy per voxel
            # Shape [B, 4, D, H, W] -> permute to [B, D, H, W, 4] for Categorical
            dist = torch.distributions.Categorical(probs=probs.permute(0, 2, 3, 4, 1))
            norm_entropy = dist.entropy() / np.log(4.0)  # normalized to [0, 1]

            for class_idx, name in CLASS_TO_REGION.items():
                target_mask = (labels == class_idx)
                pred_mask = (preds == class_idx)
                union_mask = target_mask | pred_mask

                if union_mask.sum() > 0:
                    h_val = float(norm_entropy[union_mask].mean().item())
                    intersection = float((target_mask & pred_mask).sum().item())
                    total = float(target_mask.sum().item() + pred_mask.sum().item())
                    dice = (2.0 * intersection) / max(total, 1.0)
                    d_val = 1.0 - dice

                    entropy_acc[name] += h_val
                    disagreement_acc[name] += d_val
                    region_count[name] += 1

    scores = {}
    for r in OUTPUT_REGIONS:
        if region_count[r] > 0:
            avg_d = disagreement_acc[r] / region_count[r]
            avg_h = entropy_acc[r] / region_count[r]
            raw_rho = float(np.exp(-a * avg_d - b * avg_h))
            scores[r] = float(np.clip(raw_rho, 0.0, 1.0))
        else:
            scores[r] = 1.0  # neutral reliability if region is not present in evaluated batches

    return scores


def compute_smoothed_reliability(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    client_state: dict,
    ema_alpha: float = 0.3,
    a: float = 1.0,
    b: float = 1.0,
) -> dict[str, float]:
    """Estimate reliability and update running EMA in client state.

    Returns:
        dict with keys 'reliability_ncr', 'reliability_ed', 'reliability_et'.
    """
    raw_scores = estimate_local_reliability(
        model=model,
        dataloader=dataloader,
        device=device,
        a=a,
        b=b,
    )

    result = {}
    for r in OUTPUT_REGIONS:
        prev_ema = float(client_state.get(f"ema_rel_{r}", 1.0))
        updated_ema = (1.0 - ema_alpha) * prev_ema + ema_alpha * raw_scores[r]
        client_state[f"ema_rel_{r}"] = float(updated_ema)
        result[f"reliability_{r}"] = float(updated_ema)

    return result
