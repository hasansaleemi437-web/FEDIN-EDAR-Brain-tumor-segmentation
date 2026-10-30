"""Weights & Biases (W&B) Real-Time Experiment Tracking for FeTS 2022.

This module handles live experiment tracking, registering only the active training
hyperparameters and streaming round-by-round train/evaluation metrics in real time.
"""

from __future__ import annotations

import os
import sys
from typing import Any
import pandas as pd
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def is_wandb_available() -> bool:
    """Check if the wandb library is installed."""
    return _WANDB_AVAILABLE


def init_wandb(
    context: Any,
    algorithm: str,
    model_name: str,
    num_rounds: int,
) -> bool:
    """Initialize W&B session before training starts so the run appears live in dashboard."""
    cfg = getattr(context, "run_config", {})
    if not cfg.get("use-wandb", True):
        return False

    if not _WANDB_AVAILABLE:
        print("[W&B] Notice: 'wandb' package is not installed. Tracking disabled.", flush=True)
        return False

    if wandb.run is not None:
        return True

    # 1. Project, entity, and endpoint detection
    project = (
        cfg.get("wandb-project")
        or os.environ.get("WANDB_PROJECT")
        or "Conference2"
    )
    entity = (
        cfg.get("wandb-entity")
        or os.environ.get("WANDB_ENTITY")
        or "syedhaseeburrehman77-university-of-management-technolog"
    )
    base_url = (
        cfg.get("wandb-base-url")
        or os.environ.get("WANDB_BASE_URL")
        or "https://api.forge.coreweave.com"
    )
    mode = str(cfg.get("wandb-mode", os.environ.get("WANDB_MODE", "online"))).strip()
    api_key = str(cfg.get("wandb-api-key", "") or os.environ.get("WANDB_API_KEY", "")).strip()

    # If no key found and terminal is interactive, prompt user in terminal
    if not api_key and sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        try:
            print(f"\n[W&B] No API key detected for {base_url}.")
            entered = input(f"[W&B] Enter your W&B API key from {base_url}/authorize (or press Enter to run offline): ").strip()
            if entered:
                api_key = entered
        except Exception:
            pass

    try:
        if api_key:
            wandb.login(key=api_key, host=base_url if base_url else None)
        elif base_url:
            wandb.login(host=base_url)
    except Exception:
        pass

    settings = None
    if base_url:
        clean_url = str(base_url).rstrip("/")
        if "/wandb" in clean_url and not clean_url.endswith("/api"):
            clean_url = clean_url.split("/wandb")[0]
        if "forge.coreweave.com" in clean_url and "api.forge.coreweave.com" not in clean_url:
            clean_url = clean_url.replace("forge.coreweave.com", "api.forge.coreweave.com")
        settings = wandb.Settings(base_url=clean_url)

    # 2. Extract ONLY the active hyperparameters directly from the TOML run config
    selector = str(cfg.get("collaborator-selector", "fixed")).lower()
    hyperparameters: dict[str, Any] = {
        "algorithm": algorithm.lower(),
        "model_name": cfg.get("model_name", model_name),
        "num_server_rounds": int(num_rounds),
        "collaborator_selector": selector,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Training parameters directly from TOML config
    for key in ("learning-rate", "batch-size", "local-epochs", "num-clients"):
        if key in cfg:
            hyperparameters[key.replace("-", "_")] = cfg[key]

    # Selector parameters directly from TOML config
    if selector in ("sliding", "emd"):
        for key in ("collaborator-fraction", "collaborator-selector-seed"):
            if key in cfg:
                hyperparameters[key.replace("-", "_")] = cfg[key]
    if selector == "emd" and "emd-staleness-weight" in cfg:
        hyperparameters["emd_staleness_weight"] = cfg["emd-staleness-weight"]

    # Algorithm-specific parameters directly from TOML config
    if algorithm in ("fedprox", "fedindar") and "proximal_mu" in cfg:
        hyperparameters["proximal_mu"] = cfg["proximal_mu"]

    if algorithm == "fedindar":
        for key in (
            "redar-kappa",
            "redar-gamma",
            "fedr3-reliability-warmup",
            "reliability-mu-enabled",
            "reliability-mu-lambda",
        ):
            if key in cfg:
                hyperparameters[key.replace("-", "_")] = cfg[key]
    elif algorithm == "regsimagg":
        for key in ("regsimagg-regularization-round", "regsimagg-distance-mode"):
            if key in cfg:
                hyperparameters[key.replace("-", "_")] = cfg[key]

    run_name = f"{algorithm.upper()}_{model_name}_{selector}_r{num_rounds}"

    # 3. Connect to W&B live
    try:
        init_args: dict[str, Any] = {
            "name": run_name,
            "config": hyperparameters,
            "mode": mode,
        }
        if project:
            init_args["project"] = project
        if entity:
            init_args["entity"] = entity
        if settings:
            init_args["settings"] = settings

        run = wandb.init(**init_args)
        run_url = getattr(run, "url", None) or getattr(run, "get_url", lambda: project)()
        print(f"\n{'=' * 78}")
        print(f"[W&B] ---> Connected to Weights & Biases (Live Run Active)")
        print(f"[W&B] Project      : {project}")
        print(f"[W&B] Run Name     : {run_name}")
        print(f"[W&B] Live Monitor : {run_url}")
        print(f"{'=' * 78}\n", flush=True)
        return True
    except Exception as exc:
        err_msg = str(exc).strip() or repr(exc)
        print(f"\n[W&B] Warning: Online connection to {base_url} failed ({err_msg}).", flush=True)
        print("[W&B] Note: If HTTP 401 Unauthorized occurs, the API key is not valid for this host.", flush=True)
        print(f"[W&B] Make sure your key was generated on {base_url}/authorize.", flush=True)

        # If terminal is interactive, allow user to re-enter valid key
        if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            try:
                new_key = input(f"[W&B] Please enter your valid API key (or press Enter for offline mode): ").strip()
                if new_key:
                    wandb.login(key=new_key, host=base_url, relogin=True)
                    init_args["mode"] = "online"
                    run = wandb.init(**init_args)
                    run_url = getattr(run, "url", None) or getattr(run, "get_url", lambda: project)()
                    print(f"\n[W&B] ---> Connected! Live Monitor: {run_url}\n", flush=True)
                    return True
            except Exception:
                pass

        print("[W&B] Activating OFFLINE mode: creating local 'wandb/' run folder so real-time experiment tracking continues...", flush=True)
        try:
            offline_args = {
                "name": run_name,
                "config": hyperparameters,
                "mode": "offline",
            }
            if project:
                offline_args["project"] = project
            run = wandb.init(**offline_args)
            run_dir = getattr(run, "dir", "wandb/")
            print(f"\n{'=' * 78}")
            print(f"[W&B] ---> Tracking active in OFFLINE mode!")
            print(f"[W&B] Local Run Directory : {run_dir}")
            print(f"[W&B] Real-time round metrics are being tracked locally.")
            print(f"[W&B] To sync later: wandb login --host={base_url} && wandb sync {run_dir}")
            print(f"{'=' * 78}\n", flush=True)
            return True
        except Exception as e_offline:
            print(f"[W&B] Offline tracking note: {e_offline}. Training continues without W&B.\n", flush=True)
            return False


def log_live_round_metrics(
    server_round: int,
    metrics: dict[str, Any] | None = None,
    phase: str = "train",
) -> None:
    """Stream round metrics directly to W&B in real time."""
    if not _WANDB_AVAILABLE or wandb is None or wandb.run is None or not metrics:
        return
    try:
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not pd.isna(v):
                clean_k = str(k).replace("eval_", "").replace("train_", "")
                clean[f"{phase}/{clean_k}"] = float(v)
        if clean:
            clean["round"] = int(server_round)
            wandb.log(clean, step=int(server_round))
    except Exception:
        pass


def finish_wandb(*args: Any, **kwargs: Any) -> None:
    """Finalize the W&B run cleanly."""
    if not _WANDB_AVAILABLE or wandb is None or wandb.run is None:
        return
    try:
        benchmark = kwargs.get("benchmark_metrics")
        if benchmark and isinstance(benchmark, dict):
            for k, v in benchmark.items():
                try:
                    wandb.run.summary[f"benchmark/{k}"] = float(v)
                except Exception:
                    pass
        wandb.finish()
        print("[W&B] Live run finished successfully.\n", flush=True)
    except Exception:
        pass


# Backward-compatible alias
log_experiment_to_wandb = finish_wandb
