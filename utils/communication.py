"""FeTS 2022 Communication Payload & Cost Measurement Utility."""

from pathlib import Path
import json
import pandas as pd
import torch


def measure_model_payload(model: torch.nn.Module) -> dict[str, int]:
    """Client helper: calculates exact tensor bytes sent over the network."""
    total_bytes = sum(
        param.numel() * param.element_size() 
        for param in model.state_dict().values()
    )
    return {
        "comm_bytes_uplink": int(total_bytes),
        "comm_bytes_downlink": int(total_bytes),
        "comm_bytes_total": int(total_bytes * 2),
    }


def audit_communication_cost(history_csv_path: Path, num_rounds: int, context=None) -> dict:
    """Server helper: computes FeTS Table 3 normalized ratio and prints audit."""
    total_clients = int(context.run_config.get("num-clients", 33)) if context else 33
    algorithm = str(context.run_config.get("algorithm", "fl")).upper() if context else "FL"

    # FedIN-R3EDAR (Module 1): show which collaborator selector was used
    selector = str(context.run_config.get("collaborator-selector", "fixed")).lower() if context else "fixed"
    selector_label = {"emd": "EMD-Guided Selection", "sliding": "Sliding Selection"}.get(selector, "All Clients")
    history_csv = Path(history_csv_path)
    if not history_csv.exists() or history_csv.stat().st_size == 0:
        return {}

    try:
        df = pd.read_csv(history_csv)
    except Exception:
        return {}

    train_rows = df[df["phase"] == "train"] if "phase" in df.columns else df

    # Model size in MB derived from actual recorded bytes
    if "comm_bytes_uplink" in train_rows.columns and not train_rows["comm_bytes_uplink"].isna().all():
        model_size_mb = float(train_rows["comm_bytes_uplink"].dropna().iloc[0]) / (1024 * 1024)
    else:
        model_size_mb = 18.374  # standard FeTS 3D U-Net fallback

    actual_client_rounds = len(train_rows)
    baseline_client_rounds = num_rounds * total_clients

    # Physical data in GB
    per_client_round_mb = 2 * model_size_mb
    actual_data_gb = (actual_client_rounds * per_client_round_mb) / 1024
    baseline_data_gb = (baseline_client_rounds * per_client_round_mb) / 1024

    # FeTS Challenge Table 3 Ratio (Normalized relative to Full FL baseline)
    fets_comm_cost_ratio = actual_client_rounds / max(baseline_client_rounds, 1)
    bandwidth_savings_pct = (1.0 - fets_comm_cost_ratio) * 100

    stats = {
        "algorithm": algorithm,
        "model_size_mb": round(model_size_mb, 2),
        "total_rounds": num_rounds,
        "actual_client_rounds": actual_client_rounds,
        "baseline_client_rounds": baseline_client_rounds,
        "actual_data_gb": round(actual_data_gb, 2),
        "baseline_data_gb": round(baseline_data_gb, 2),
        "fets_comm_cost_ratio": round(fets_comm_cost_ratio, 4),
        "bandwidth_savings_pct": round(bandwidth_savings_pct, 2),
    }

    # Print Formatted FeTS Audit
    header = f"FEDERATED LEARNING COMMUNICATION COST AUDIT ({algorithm})"
    print(f"\n{'=' * 78}\n{header:^78}\n{'=' * 78}")
    print(f" Model Payload (Single Direction)   : {stats['model_size_mb']} MB")
    print(f" Total Completed Rounds             : {num_rounds}")
    print(f" Actual Client-Rounds Transmitted   : {actual_client_rounds} ({selector_label})")
    print(f" Baseline Client-Rounds (All Nodes) : {baseline_client_rounds}")
    print(f"{'-' * 78}")
    print(f" Baseline Network Data (All Nodes)  : {stats['baseline_data_gb']} GB")
    print(f" Actual Network Data Transmitted    : {stats['actual_data_gb']} GB")
    print(f" FeTS Comm. Cost Ratio (Table 3)    : {stats['fets_comm_cost_ratio']} (Baseline = 1.0000)")
    print(f" Bandwidth Savings vs Full FL       : {stats['bandwidth_savings_pct']}% Reduction")
    print(f"{'=' * 78}\n", flush=True)

    # Save to artifacts JSON
    try:
        out_json = Path(f"artifacts/{algorithm.lower()}_communication_cost.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass

    return stats
