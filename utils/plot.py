"""Plots based on the per-client history CSV.

Dice summaries are unweighted across institutions. These are client-side
validation results, not centralized server evaluation results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

STRATEGY_COLORS = {
    "fedavg": "#d62728",
    "fedprox": "#1f77b4",
    "regsimagg": "#2ca02c",
    "fedindar": "#ff7f0e",
}
STRATEGY_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "regsimagg": "RegSimAgg",
    "fedindar": "FedIN-R³EDAR",
}
STRATEGY_ORDER = list(STRATEGY_LABELS)

REGIONS = [("wt", "WT"), ("tc", "TC"), ("et", "ET")]

def _load_history(history_csv: str | Path) -> pd.DataFrame:
    """Load current or older headed client-history CSVs."""
    path = Path(history_csv)
    if not path.exists():
        raise FileNotFoundError(f"History CSV not found: {path.resolve()}")

    df = pd.read_csv(path)
    if "strategy" not in df.columns:
        raise ValueError(
            "CSV needs a header row containing a 'strategy' column."
        )

    df["strategy"] = df["strategy"].astype(str).str.lower()
    for col in ("round", "institution_id", "num_examples", "train_cases",
                "total_cases", "dice_et", "dice_tc", "dice_wt",
                "hd95_et", "hd95_tc", "hd95_wt", "aggregation_weight"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Current schema uses phase='evaluate'; older schema may have no phase.
    if "phase" not in df.columns:
        if "dice_wt" not in df.columns:
            raise ValueError("CSV has neither a 'phase' nor a 'dice_wt' column.")
        df["phase"] = np.where(df["dice_wt"].notna(), "evaluate", "train")
    else:
        df["phase"] = df["phase"].astype(str).str.lower()

    # Older history schema used train_loss/eval_loss instead of loss.
    if "loss" not in df.columns:
        df["loss"] = np.nan
        if "eval_loss" in df.columns:
            df.loc[df["phase"] == "evaluate", "loss"] = df["eval_loss"]
        if "train_loss" in df.columns:
            df.loc[df["phase"] == "train", "loss"] = df["train_loss"]

    return df


def _evaluation_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Select client evaluation rows with a usable validation metric."""
    if "dice_wt" not in df.columns:
        raise ValueError("CSV is missing the 'dice_wt' metric.")
    rows = df[(df["phase"] == "evaluate") & df["dice_wt"].notna()].copy()
    if rows.empty:
        raise ValueError("No client evaluation rows with Dice scores were found.")

    # Protect against repeated client/round rows from retries or concatenation.
    keys = [c for c in ("strategy", "round", "institution_id") if c in rows]
    return rows.drop_duplicates(keys, keep="last")


def _save_figure(fig, output_path: str | Path) -> None:
    """Save requested image and a vector PDF beside it."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        str(out),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    if out.suffix.lower() != ".pdf":
        fig.savefig(
            str(out.with_suffix(".pdf")),
            format="pdf",
            bbox_inches="tight",
        )

    plt.close(fig)
    print(f"[Plot] Saved: {out}")

def _label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy.upper())
    
def _ordered_strategies(df: pd.DataFrame) -> list[str]:
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]
    strategies += [s for s in df["strategy"].unique() if s not in strategies]
    return strategies


def _family_info(family: str) -> tuple[str, str]:
    """Return (column prefix, axis name) for 'dice' or 'hd95'."""
    if family == "dice":
        return "dice", "Dice"
    if family == "hd95":
        return "hd95", "HD95 (mm)"
    raise ValueError("family must be 'dice' or 'hd95'")


def _panel_grid(layout: str):
    """Three panels (WT, TC, ET) sharing the y-axis."""
    if layout == "column":
        fig, axes = plt.subplots(
            3, 1, figsize=(5.4, 10.0), sharex=True, sharey=True
        )
    else:
        fig, axes = plt.subplots(
            1, 3, figsize=(13.0, 4.4), sharey=True
        )
    return fig, list(np.atleast_1d(axes))


def _shared_legend(fig, axes) -> None:
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02)
        )


def _finish_panels(fig, axes, title: str, ylabel: str, layout: str) -> None:
    targets = axes[:1] if layout != "column" else axes

    for ax in targets:
        ax.set_ylabel(ylabel)

    fig.suptitle(title, y=1.0)
    _shared_legend(fig, axes)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])


def plot_convergence(
    history_csv: str | Path = "artifacts/client_history.csv",
    family: str = "dice",
    output_path: str | Path | None = None,
    layout: str = "row",
) -> None:
    """Mean client validation score per round for WT, TC and ET (± site SD)."""
    df = _evaluation_rows(_load_history(history_csv))
    prefix, name = _family_info(family)
    if output_path is None:
        output_path = ("artifacts/figure1_convergence.png" if family == "dice"
                       else "artifacts/figure1b_convergence_hd95.png")
 
    fig, axes = _panel_grid(layout)
    for ax, (key, title) in zip(axes, REGIONS):
        metric = f"{prefix}_{key}"
        if metric not in df.columns:
            ax.set_visible(False)
            continue
 
        grouped = (
            df.groupby(["strategy", "round"])[metric]
            .agg(mean="mean", std="std", sites="count")
            .reset_index()
            .sort_values("round")
        )
        for strategy, data in grouped.groupby("strategy"):
            data = data.sort_values("round")
            color = STRATEGY_COLORS.get(strategy, "#333333")
            mean = data["mean"].to_numpy()
            std = data["std"].fillna(0).to_numpy()
            low, high = mean - std, mean + std
            if family == "dice":
                low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
            else:
                low = np.clip(low, 0, None)
            ax.plot(data["round"], mean, marker="o", color=color, label=_label(strategy))
            ax.fill_between(data["round"], low, high, color=color, alpha=0.16)
 
        ax.set_title(title)
        ax.set_xlabel("Federated round")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle=":", alpha=0.6)
        if family == "dice":
            ax.set_ylim(0, 1)
 
    _finish_panels(fig, axes, "Client-side validation performance",
                   f"Mean client validation {name} (\u00b1 site SD)", layout)
    _save_figure(fig, output_path)


def _last_round_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return each strategy's own latest available evaluation round."""
    rounds = df.groupby("strategy")["round"].max().to_dict()

    rows = pd.concat(
        [
            df[(df["strategy"] == strategy) & (df["round"] == round_num)]
            for strategy, round_num in rounds.items()
        ],
        ignore_index=True,
    )

    return rows, rounds


def plot_client_distribution(
    history_csv: str | Path = "artifacts/client_history.csv",
    family: str = "dice",
    output_path: str | Path | None = None,
    layout: str = "row",
) -> None:
    """Show per-institution score distributions for WT, TC and ET."""

    df, rounds = _last_round_rows(
        _evaluation_rows(_load_history(history_csv))
    )

    prefix, name = _family_info(family)

    if output_path is None:
        output_path = (
            "artifacts/figure2_client_distribution.png"
            if family == "dice"
            else "artifacts/figure2b_client_distribution_hd95.png"
        )

    strategies = _ordered_strategies(df)
    fig, axes = _panel_grid(layout)
    rng = np.random.default_rng(42)

    for ax, (key, title) in zip(axes, REGIONS):
        metric = f"{prefix}_{key}"

        if metric not in df.columns:
            ax.set_visible(False)
            continue

        valid_strategies = []
        values = []
        labels = []

        for strategy in strategies:
            scores = df.loc[
                df["strategy"] == strategy, metric
            ].dropna().to_numpy()

            if len(scores) == 0:
                continue

            valid_strategies.append(strategy)
            values.append(scores)

            labels.append(
                f"{_label(strategy)}\n"
                f"(r={int(rounds[strategy])}, n={len(scores)})"
            )

        if not values:
            ax.set_visible(False)
            continue

        try:
            boxes = ax.boxplot(
                values,
                tick_labels=labels,
                patch_artist=True,
                showfliers=False,
            )
        except TypeError:
            boxes = ax.boxplot(
                values,
                labels=labels,
                patch_artist=True,
                showfliers=False,
            )

        for i, (strategy, scores) in enumerate(
            zip(valid_strategies, values), start=1
        ):
            boxes["boxes"][i - 1].set_facecolor(
                STRATEGY_COLORS.get(strategy, "#888888")
            )
            boxes["boxes"][i - 1].set_alpha(0.45)

            x = rng.normal(i, 0.045, size=len(scores))

            ax.scatter(
                x,
                scores,
                color="black",
                s=20,
                alpha=0.7,
                zorder=3,
            )

        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        if family == "dice":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)

    _finish_panels(
        fig,
        axes,
        "Validation score distribution across institutions",
        f"Client validation {name}",
        layout,
    )
    _save_figure(fig, output_path)

def plot_subregions(
    history_csv: str | Path = "artifacts/client_history.csv",
    output_path: str | Path = "artifacts/figure3_subregions.png",
) -> None:
    """Compare equal-site mean Dice and SEM at each method's latest round."""
    df, rounds = _last_round_rows (_evaluation_rows(_load_history(history_csv)))
    regions = [
        ("dice_wt", "WT"),
        ("dice_tc", "TC"),
        ("dice_et", "ET"),
    ]
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]
    strategies += [s for s in df["strategy"].unique() if s not in strategies]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(regions))
    width = 0.8 / max(len(strategies), 1)

    for i, strategy in enumerate(strategies):
        site_rows = df[df["strategy"] == strategy]
        means, sems = [], []
        for metric, _ in regions:
            scores = site_rows[metric].dropna()
            means.append(scores.mean() if len(scores) else np.nan)
            sems.append(scores.std(ddof=1) / np.sqrt(len(scores))
                        if len(scores) > 1 else 0.0)

        offset = (i - (len(strategies) - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=sems, capsize=3,
            color=STRATEGY_COLORS.get(strategy, "#888888"),
            label=f"{_label(strategy)} (r={int(rounds[strategy])})",
        )

    ax.set_ylabel("Mean client validation Dice (± SEM)")
    ax.set_title("Client validation Dice by tumour region")
    ax.set_xticks(x, [name for _, name in regions])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    _save_figure(fig, output_path)


def plot_heterogeneity_robustness(
    history_csv: str | Path = "artifacts/client_history.csv",
    family: str = "dice",
    output_path: str | Path | None = None,
    layout: str = "row",
) -> None:
    """Association between institution size and final score, per region."""
    df, rounds = _last_round_rows(
        _evaluation_rows(_load_history(history_csv))
    )

    prefix, name = _family_info(family)

    if output_path is None:
        output_path = (
            "artifacts/figure4_heterogeneity.png" if family == "dice"
            else "artifacts/figure4b_heterogeneity_hd95.png"
        )

    if "train_cases" in df.columns and df["train_cases"].notna().any():
        x_col, x_label = "train_cases", "Institution training cases"

    elif "total_cases" in df.columns and df["total_cases"].notna().any():
        x_col, x_label = (
            "total_cases", "Institution train + validation cases"
        )

    elif "num_examples" in df.columns and df["num_examples"].notna().any():
        x_col, x_label = (
            "num_examples", "Institution evaluated cases"
        )

    else:
        raise ValueError(
            "CSV needs train_cases, total_cases, or num_examples for this plot."
        )

    fig, axes = _panel_grid(layout)

    for ax, (key, title) in zip(axes, REGIONS):
        metric = f"{prefix}_{key}"

        if metric not in df.columns:
            ax.set_visible(False)
            continue

        for strategy, data in df.groupby("strategy"):
            data = data.dropna(subset=[x_col, metric])
            color = STRATEGY_COLORS.get(strategy, "#333333")

            ax.scatter(
                data[x_col], data[metric],
                color=color, alpha=0.75,
                label=f"{_label(strategy)} (r={int(rounds[strategy])})"
            )

            if len(data) >= 2 and data[x_col].nunique() >= 2:
                slope, intercept = np.polyfit(
                    data[x_col], data[metric], 1
                )

                xs = np.linspace(
                    data[x_col].min(), data[x_col].max(), 100
                )

                ax.plot(
                    xs, slope * xs + intercept,
                    color=color, linewidth=1.8
                )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, linestyle=":", alpha=0.6)

        if family == "dice":
            ax.set_ylim(0, 1)

    _finish_panels(
        fig, axes,
        "Association between client size and validation performance",
        f"Client validation {name}",
        layout
    )

    _save_figure(fig, output_path)

def plot_aggregation_weights(
    history_csv: str | Path = "artifacts/client_history.csv",
    output_path: str | Path = "artifacts/figure5_aggregation_weights.png",
    top_clients: int = 5,
) -> None:
    """Plot actual logged training aggregation weights; skip if unavailable."""
    df = _load_history(history_csv)
    if "aggregation_weight" not in df.columns:
        print("[Plot] No aggregation_weight column; skipping weight plot.")
        return

    rows = df[
        (df["phase"] == "train")
        & df["aggregation_weight"].notna()
        & df["round"].notna()
        & df["institution_id"].notna()
    ].copy()
    if rows.empty:
        print("[Plot] No actual training aggregation weights; skipping weight plot.")
        return

    rows = rows.drop_duplicates(
        ["strategy", "round", "institution_id"], keep="last"
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plotted = False

    for strategy, strategy_rows in rows.groupby("strategy"):
        means = strategy_rows.groupby("institution_id")["aggregation_weight"].mean()
        selected = means.nlargest(top_clients).index
        color = STRATEGY_COLORS.get(strategy, "#333333")

        for client_id in selected:
            site = strategy_rows[
                strategy_rows["institution_id"] == client_id
            ].sort_values("round")
            ax.plot(
                site["round"], site["aggregation_weight"],
                marker="o", color=color, alpha=0.75,
                label=f"{_label(strategy)} — site {int(client_id)}",
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print("[Plot] No client weights to plot.")
        return

    ax.set_xlabel("Federated round")
    ax.set_ylabel("Actual training aggregation weight")
    ax.set_title(f"Top {top_clients} client aggregation-weight trajectories")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    _save_figure(fig, output_path)


def generate_all_paper_figures(
    history_csv: str | Path = "artifacts/client_history.csv",
) -> None:
    """Generate descriptive figures from the supplied client-history CSV."""
    plot_convergence(history_csv)
    plot_client_distribution(history_csv)
    plot_subregions(history_csv)
    plot_heterogeneity_robustness(history_csv)
    plot_aggregation_weights(history_csv)


if __name__ == "__main__":
    generate_all_paper_figures()