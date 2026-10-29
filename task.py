"""FeTS 2022 3D MRI Brain Tumor Segmentation: Task and Evaluation Module."""

import csv
import gc
import tomllib
from pathlib import Path
import torch
import torch.nn as nn
from flwr.app import Context
from datasets_loaders import create_dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from utils.metrics import fets_region_metrics
from label_noise import apply_label_noise
_LOADER = None
_LOADER_KEY = None



def get_loader(context=None):
    global _LOADER, _LOADER_KEY
    
    if context is not None:
        cfg = context.run_config
    else:
        cfg = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]

    # Resolve relative paths against the repository root (use workspace CWD first)
    project_root = Path.cwd().resolve() if (Path.cwd() / "data").exists() else Path(__file__).parent.resolve()
    
    data_root = Path(cfg.get("data-root", "data/MICCAI_FeTS2022_TrainingData"))
    if not data_root.is_absolute():
        data_root = project_root / data_root

    # Support all variations: partition-csv, partition_csv, partitioning-csv, partitioning_csv, partitioning
    raw_p = None
    if context is not None:
        raw_p = (
            context.run_config.get("partition-csv")
            or context.run_config.get("partition_csv")
            or context.run_config.get("partitioning-csv")
            or context.run_config.get("partitioning_csv")
            or context.run_config.get("partitioning")
        )
    if not raw_p:
        raw_p = (
            cfg.get("partition-csv")
            or cfg.get("partition_csv")
            or cfg.get("partitioning-csv")
            or cfg.get("partitioning_csv")
            or cfg.get("partitioning")
            or "data/MICCAI_FeTS2022_TrainingData/partitioning_1.csv"
        )

    raw_p_str = str(raw_p).strip()
    if raw_p_str in ("2", "partitioning_2", "partitioning_2.csv") or "partitioning_2" in raw_p_str:
        candidate = data_root / "partitioning_2.csv"
        partition_csv = candidate if candidate.exists() else Path(raw_p_str)
    elif raw_p_str in ("1", "partitioning_1", "partitioning_1.csv"):
        candidate = data_root / "partitioning_1.csv"
        partition_csv = candidate if candidate.exists() else Path(raw_p_str)
    else:
        partition_csv = Path(raw_p_str)

    # If num-clients is > 23 and partition_csv is still partitioning_1, auto-select partitioning_2
    num_clients = int(cfg.get("num-clients", cfg.get("num_clients", 0)))
    if context is not None:
        num_clients = int(context.run_config.get("num-clients", context.run_config.get("num_clients", num_clients)))
    
    if num_clients > 23 and "partitioning_1" in partition_csv.name:
        candidate_2 = data_root / "partitioning_2.csv"
        if candidate_2.exists():
            partition_csv = candidate_2

    if not partition_csv.is_absolute():
        if (project_root / partition_csv).exists():
            partition_csv = project_root / partition_csv
        elif (data_root / partition_csv.name).exists():
            partition_csv = data_root / partition_csv.name
        else:
            partition_csv = project_root / partition_csv

    global_test_fraction = float(cfg.get("global-test-fraction", 0.15))
    seed = int(cfg.get("seed", 42))

    cache_key = (str(partition_csv.resolve()), str(data_root.resolve()), global_test_fraction, seed)
    if _LOADER is not None and _LOADER_KEY == cache_key:
        return _LOADER

    _LOADER = create_dataset(
        "fets2022",
        root_dir=data_root,
        partition_csv=partition_csv,
        global_test_fraction=global_test_fraction,
        seed=seed,
    )
    _LOADER_KEY = cache_key
    return _LOADER

def load_data(partition_id: int, context):
    loader = get_loader(context)
    batch_size = int(context.run_config.get("batch-size", 1))
    out = loader.load_partition(partition_id=partition_id, batch_size=batch_size)
    return apply_label_noise(out, partition_id, context.run_config)

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    # Reuse the loader built with the run config (data-root / partition-csv overrides
    loader = _LOADER if _LOADER is not None else get_loader()
    return loader.load_global_test(batch_size=1)

def test(net: torch.nn.Module, testloader, device: torch.device) -> dict[str, float]:
    """Evaluate 3D U-Net using sliding-window inference and return Dice/HD95 metrics."""
    net.eval()
    net.to(device)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    totals = {
        "eval_loss": 0.0,
        "dice_et": 0.0,
        "dice_tc": 0.0,
        "dice_wt": 0.0,
        "hd95_et": 0.0,
        "hd95_tc": 0.0,
        "hd95_wt": 0.0,
        "pred_et_voxels": 0.0,
        "pred_tc_voxels": 0.0,
        "pred_wt_voxels": 0.0,
        "target_et_voxels": 0.0,
        "target_tc_voxels": 0.0,
        "target_wt_voxels": 0.0,
    }
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            # 3D sliding window inference with 96x96x96 patches
            logits = sliding_window_inference(
                inputs=images,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=net,
            )
            totals["eval_loss"] += criterion(logits, labels).item()
            # Calculate Dice and Hausdorff HD95 for ET, TC, and WT
            region_scores = fets_region_metrics(logits, labels)
            for key in (
                "dice_et", "dice_tc", "dice_wt", "hd95_et", "hd95_tc", "hd95_wt",
                "pred_et_voxels", "pred_tc_voxels", "pred_wt_voxels",
                "target_et_voxels", "target_tc_voxels", "target_wt_voxels",
            ):
                if key in region_scores:
                    totals[key] += region_scores[key]

            # Immediately release large 3D tensors from memory
            del images, labels, logits, region_scores
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    count = max(len(testloader), 1)
    return {k: round(v / count, 4) for k, v in totals.items()}

def run_global_benchmark(model: torch.nn.Module, device: torch.device, algorithm: str = "fedavg") -> dict[str, float]:
    """Run one-off final benchmark on the unseen global test set and save to artifacts."""
    print(f"\n[Server] ---> Running Final Benchmark on Unseen Test Set ({algorithm.upper()})...", flush=True)

    test_dataloader = load_centralized_dataset()
    eval_metrics = test(model, test_dataloader, device)

    print(
        f"[Server] ---> Final Test Results | "
        f"Loss: {eval_metrics.get('eval_loss', 0.0):.4f} | "
        f"Dice (ET/TC/WT): {eval_metrics.get('dice_et', 0.0):.4f} / {eval_metrics.get('dice_tc', 0.0):.4f} / {eval_metrics.get('dice_wt', 0.0):.4f} | "
        f"HD95 (ET/TC/WT): {eval_metrics.get('hd95_et', 0.0):.2f} / {eval_metrics.get('hd95_tc', 0.0):.2f} / {eval_metrics.get('hd95_wt', 0.0):.2f}",
        flush=True,
    )

    # Save final benchmark to artifacts/detail_metrics_{algorithm}.csv
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_file = artifacts_dir / f"detail_metrics_{algorithm}.csv"
    file_exists = csv_file.exists()

    row = {
        "strategy": algorithm,
        "test_loss": eval_metrics.get("eval_loss", 0.0),
        "dice_et": eval_metrics.get("dice_et", 0.0),
        "dice_tc": eval_metrics.get("dice_tc", 0.0),
        "dice_wt": eval_metrics.get("dice_wt", 0.0),
        "hd95_et": eval_metrics.get("hd95_et", 0.0),
        "hd95_tc": eval_metrics.get("hd95_tc", 0.0),
        "hd95_wt": eval_metrics.get("hd95_wt", 0.0),
    }
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return eval_metrics
