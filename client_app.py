"""FeTS 2022 3D MRI Brain Tumor Segmentation: Flower ClientApp."""

import gc
from pathlib import Path
import time
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from task import load_data, test as test_fn
from algorithms import get_trainer
from models import create_model
from utils.summary import append_client_history
from utils.communication import measure_model_payload

from models.unet import instance_norm_state_keys
from algorithms.reliability import compute_smoothed_reliability
from algorithms.fedindar import (
    PROFILE_BINS,
    REGIONS,
    TUMOR_BURDEN_BIN_EDGES,
    profile_metric_key,
)

# Flower ClientApp
app = ClientApp()


def get_local_instance_norm_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get this client's local InstanceNorm parameters."""
    local_keys = instance_norm_state_keys()
    state_dict = model.state_dict()

    return {
        key: state_dict[key].detach().cpu().clone()
        for key in local_keys
        if key in state_dict
    }


def restore_local_instance_norm(model: torch.nn.Module, partition_id: int, device: torch.device) -> bool:
    """Restore this client's local InstanceNorm parameters from disk if previously saved."""
    local_in_file = Path("artifacts/local_instance_norm") / f"client_{partition_id}.pt"

    if local_in_file.exists():
        try:
            local_in_state = torch.load(local_in_file, map_location="cpu")
        except Exception as e:
            print(
                f"[Client {partition_id}] Warning: Could not deserialize {local_in_file} ({e}). "
                "Starting with global InstanceNorm.",
                flush=True,
            )
            return False

        if isinstance(local_in_state, dict):
            state = model.state_dict()
            for key, value in local_in_state.items():
                if key in state and isinstance(value, torch.Tensor):
                    state[key].copy_(value.to(device))
            return True
    return False


@app.train()
def train(msg: Message, context: Context):
    """Train the 3D U-Net on local institutional MRI data."""

    # Create the model using the model registry
    model_name = context.run_config.get("model_name").lower()
    model = create_model(model_name)

    # Load the weights received from the server
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    # Select device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Identify client partition and strategy configuration
    partition_id = int(context.node_config["partition-id"])
    config = msg.content["config"]
    algorithm = str(context.run_config.get("algorithm", "fedavg")).lower()
    is_fedindar = bool(config.get("fedindar-enabled", False)) or (algorithm == "fedindar")

    # Restore this client's local InstanceNorm state if running FedINDAR
    if is_fedindar:
        restore_local_instance_norm(model, partition_id, device)

    # Check for FedINDAR initial profiling phase
    profile_only = bool(
        config.get("fedindar-profile-only", False)
    )

    if profile_only:
        _, valloader = load_data(
            partition_id,
            context,
        )

        profile_metrics = {
            profile_metric_key(region, i): 0.0
            for region in REGIONS
            for i in range(PROFILE_BINS)
        }

        # Collect tumour burden histograms across regions in a single data pass
        counts = {region: np.zeros(PROFILE_BINS, dtype=np.float64) for region in REGIONS}

        # FedIN-R3EDAR (Step 1): Voxel counts per tumor class
        q_voxels = {"ncr": 0.0, "ed": 0.0, "et": 0.0}
        total_foreground = 0.0

        for batch in valloader:
            labels = batch["label"]
            if labels.ndim == 5:
                labels = labels[:, 0]
            labels = labels.detach().cpu().numpy()

            for label in labels:
                total_voxels = max(label.size, 1)
                wt_mask = label > 0
                tc_mask = np.logical_or(label == 1, label == 3)
                et_mask = label == 3

                masks = {"wt": wt_mask, "tc": tc_mask, "et": et_mask}
                for region, mask in masks.items():
                    burden = float(mask.sum()) / total_voxels
                    counts[region] += np.histogram([burden], bins=TUMOR_BURDEN_BIN_EDGES)[0]

                # Built-in NumPy vectorized regional voxel tallies:
                # 1: NCR/NET, 2: ED, 3: ET
                ncr_count = float((label == 1).sum())
                ed_count = float((label == 2).sum())
                et_count = float((label == 3).sum())
               
                q_voxels["ncr"] += ncr_count
                q_voxels["ed"] += ed_count
                q_voxels["et"] += et_count
                total_foreground += (ncr_count + ed_count + et_count)

        for region in REGIONS:
            if counts[region].sum() > 0:
                counts[region] /= counts[region].sum()
            for i, value in enumerate(counts[region]):
                profile_metrics[profile_metric_key(region, i)] = float(value)

        # FedIN-R3EDAR: transmit regional voxel counts (q_k,r) and foreground (N_k)
        profile_metrics["voxel_count_ncr"] = q_voxels["ncr"]
        profile_metrics["voxel_count_ed"] = q_voxels["ed"]
        profile_metrics["voxel_count_et"] = q_voxels["et"]
        profile_metrics["total_foreground_voxels"] = max(total_foreground, 1.0)

        print(
            f"[Client {partition_id}] ---> "
            "FedIN-EDAR profile collected. "
            "Round 1 training skipped.",
            flush=True,
        )

        local_in_dir = Path("artifacts/local_instance_norm")
        local_in_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            get_local_instance_norm_state(model),
            local_in_dir / f"client_{partition_id}.pt",
        )

        return Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord(
                        model.state_dict()
                    ),
                    "metrics": MetricRecord(
                        {
                            **profile_metrics,
                            "num-examples": len(
                                valloader.dataset
                            ),
                            "train_loss": 0.0,
                        }
                    ),
                }
            ),
            reply_to=msg,
        )

    # Load local training data
    trainloader, _ = load_data(
        partition_id,
        context,
    )
    print(f"\n[Client {partition_id}] ---> Loaded {len(trainloader.dataset)} patients from FeTS institution partition.", flush=True)

    # Get training algorithm trainer
    trainer = get_trainer(
        algorithm,
        proximal_mu=float(config.get("proximal_mu", context.run_config.get("proximal_mu", 0.01))),
    )

    print(f"[Client {partition_id}] ---> Starting 3D U-Net training on {device}...", flush=True)

    # Train the model
    train_start = time.perf_counter()
    train_loss = trainer.train(
        model,
        trainloader,
        context.run_config["local-epochs"],
        config["lr"],
        device,
    )
    train_time_sec = time.perf_counter() - train_start

    # Save local InstanceNorm state for FedINDAR
    if is_fedindar:
        local_in_dir = Path("artifacts/local_instance_norm")
        local_in_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            get_local_instance_norm_state(model),
            local_in_dir / f"client_{partition_id}.pt",
        )

    print(f"[Client {partition_id}] ---> Training completed! Loss: {train_loss:.4f}", flush=True)

    state = dict(context.state.get("client_state", ConfigRecord({})))
    current_round = int(state.get("current_round", 1))
    current_round = int(config.get("server-round", current_round))
    append_client_history(
        context.run_id, algorithm, current_round, partition_id, "train",
        len(trainloader.dataset), float(train_loss), train_time_sec,
    )

    # FedIN-R3EDAR: Estimate regional reliability with EMA smoothing (12 bytes)
    rel_scores = {"reliability_ncr": 1.0, "reliability_ed": 1.0, "reliability_et": 1.0}
    if is_fedindar:
        rel_scores = compute_smoothed_reliability(
            model=model,
            dataloader=trainloader,
            device=device,
            client_state=state,
            ema_alpha=float(config.get("fedr3-ema-alpha", 0.3)),
        )

    context.state["client_state"] = ConfigRecord({
        "last_train_loss": float(train_loss),
        "current_round": current_round,
        "ema_rel_ncr": float(state.get("ema_rel_ncr", 1.0)),
        "ema_rel_ed": float(state.get("ema_rel_ed", 1.0)),
        "ema_rel_et": float(state.get("ema_rel_et", 1.0)),
    })

    # Return updated model and training metrics
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        **rel_scores,
        **measure_model_payload(model),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return Message(
        content=RecordDict(
            {
                "arrays": model_record,
                "metrics": MetricRecord(metrics),
                "client-info": ConfigRecord({"institution-id": partition_id}),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the 3D U-Net on local institutional MRI data."""

    # Create the model using the model registry
    model_name = context.run_config.get("model_name").lower()
    model = create_model(model_name)
    partition_id = int(context.node_config["partition-id"])

    # Select device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Load the weights received from the server
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    config = msg.content["config"] if "config" in msg.content else {}
    algorithm = str(context.run_config.get("algorithm", "fedavg")).lower()
    is_fedindar = bool(config.get("fedindar-enabled", False)) or (algorithm == "fedindar")

    # Restore local InstanceNorm weights if running FedINDAR
    if is_fedindar:
        restore_local_instance_norm(model, partition_id, device)

    # Load local validation data
    _, valloader = load_data(partition_id, context)
    print(f"\n[Client {partition_id}] ---> Evaluating on {len(valloader.dataset)} local validation patients...", flush=True)

    # Evaluate the model
    eval_start = time.perf_counter()
    eval_metrics = test_fn(
        model,
        valloader,
        device,
    )
    eval_time_sec = time.perf_counter() - eval_start
    print(f"[Client {partition_id}] ---> Evaluation completed! Loss: {eval_metrics.get('eval_loss', 0.0):.4f} | Dice WT: {eval_metrics.get('dice_wt', 0.0):.4f}", flush=True)

    state = context.state.get("client_state", ConfigRecord({}))
    train_loss = float(state.get("last_train_loss", 0.0))
    current_round = int(config.get("server-round", state.get("current_round", 1)))

    append_client_history(
        context.run_id, algorithm, current_round, partition_id, "evaluate",
        len(valloader.dataset), float(eval_metrics.get("eval_loss", 0.0)),
        eval_time_sec, eval_metrics,
    )

    context.state["client_state"] = ConfigRecord({
        "last_train_loss": train_loss,
        "current_round": current_round + 1,
    })

    # Return evaluation metrics
    metrics = {
        **eval_metrics,
        "num-examples": len(valloader.dataset),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(metrics),
            }
        ),
        reply_to=msg,
    )
