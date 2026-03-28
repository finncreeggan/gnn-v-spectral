"""
Generate publication-ready figures for both experiments.

Structural-noise experiment (Part I):
  - 4 joint comparison plots (family x noise_type), each with 9 model lines
  - 36 per-model diagnostic plots (4 settings x 9 models)

Feature-informativeness experiment (Part II):
  - 12 joint plots (family x noise_type x noise_level), validation + test curves
  - 108 per-model plots (12 settings x 9 models)

X-axis conventions:
  - Structural-noise plots: structural_noise_frac (0.00 -> 0.45)
  - Feature-informativeness plots: feature_informativeness_frac (1.00 -> 0.00)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Visual constants ────────────────────────────────────────────────────────
MODEL_COLORS = {
    "whole_eigen_logreg": "#1f77b4",
    "whole_eigen_rf": "#aec7e8",
    "kcut_eigen_logreg": "#ff7f0e",
    "kcut_eigen_rf": "#ffbb78",
    "regularized_eigen_logreg": "#2ca02c",
    "regularized_eigen_rf": "#98df8a",
    "sgc": "#d62728",
    "gcn": "#9467bd",
    "gat": "#8c564b",
}

MODEL_LABELS = {
    "whole_eigen_logreg": "Whole Eigen LR",
    "whole_eigen_rf": "Whole Eigen RF",
    "kcut_eigen_logreg": "k-cut Eigen LR",
    "kcut_eigen_rf": "k-cut Eigen RF",
    "regularized_eigen_logreg": "Reg. Eigen LR",
    "regularized_eigen_rf": "Reg. Eigen RF",
    "sgc": "SGC",
    "gcn": "GCN",
    "gat": "GAT",
}

NOISE_TYPE_LABELS = {
    "random": "Random Edge Deletion",
    "targeted_betweenness": "Targeted Edge Deletion",
}

FAMILY_LABELS = {
    "sbm": "SBM",
    "lfr": "LFR",
}


def _style_ax(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


# ═══════════════════════════════════════════════════════════════════════════
#  PART I: Structural-noise figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_structural_noise_joint(
    condition_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate 4 joint structural-noise comparison plots.

    One plot per (family, structural_noise_type), each containing one line
    per model showing mean test ARI +/- std across the 5 base graphs.
    """
    df = pd.read_csv(condition_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    families = df["family"].unique()
    noise_types = [nt for nt in df["structural_noise_type"].unique() if nt != "clean"]

    for family in families:
        for noise_type in noise_types:
            sub = df[(df["family"] == family) & (df["structural_noise_type"] == noise_type)]
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            for model in sub["model"].unique():
                ms = sub[sub["model"] == model].sort_values("structural_noise_frac")
                x = ms["structural_noise_frac"].values
                y = ms["mean_test_ari_overall"].values
                yerr = ms["std_test_ari_overall"].values

                color = MODEL_COLORS.get(model, None)
                label = MODEL_LABELS.get(model, model)
                ax.errorbar(x, y, yerr=yerr, marker="o", label=label,
                            color=color, capsize=3, linewidth=2)

            title = (
                f"{FAMILY_LABELS.get(family, family)} — "
                f"{NOISE_TYPE_LABELS.get(noise_type, noise_type)}"
            )
            _style_ax(ax, "Structural Noise Fraction", "Test ARI", title)

            fname = f"joint_{family}_{noise_type}.pdf"
            path = output_dir / fname
            fig.savefig(path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            saved.append(path)
            logger.info("Saved joint plot: %s", path)

    return saved


def plot_structural_noise_by_model(
    graph_summary_csv: str | Path,
    condition_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate 36 per-model diagnostic plots for the structural-noise sweep.

    Each plot shows transparent graph-level ARI points, a prominent mean
    point, and a fitted trend line.
    """
    graph_df = pd.read_csv(graph_summary_csv)
    cond_df = pd.read_csv(condition_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    families = graph_df["family"].unique()
    noise_types = [nt for nt in graph_df["structural_noise_type"].unique() if nt != "clean"]

    for family in families:
        for noise_type in noise_types:
            for model in graph_df["model"].unique():
                gsub = graph_df[
                    (graph_df["family"] == family)
                    & (graph_df["structural_noise_type"] == noise_type)
                    & (graph_df["model"] == model)
                ]
                csub = cond_df[
                    (cond_df["family"] == family)
                    & (cond_df["structural_noise_type"] == noise_type)
                    & (cond_df["model"] == model)
                ]
                if gsub.empty:
                    continue

                fig, ax = plt.subplots(figsize=(8, 5))

                # graph-level points (transparent)
                ax.scatter(
                    gsub["structural_noise_frac"],
                    gsub["mean_test_ari"],
                    alpha=0.3, color=MODEL_COLORS.get(model, "gray"),
                    label="Graph-level ARI", zorder=2,
                )

                # condition-level mean
                csub_sorted = csub.sort_values("structural_noise_frac")
                ax.plot(
                    csub_sorted["structural_noise_frac"],
                    csub_sorted["mean_test_ari_overall"],
                    "o-", color=MODEL_COLORS.get(model, "gray"),
                    linewidth=2, markersize=8, label="Mean ARI", zorder=3,
                )

                # trend line
                x = csub_sorted["structural_noise_frac"].values
                y = csub_sorted["mean_test_ari_overall"].values
                if len(x) > 1:
                    z = np.polyfit(x, y, min(3, len(x) - 1))
                    p = np.poly1d(z)
                    xfit = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xfit, p(xfit), "--", color="black",
                            alpha=0.5, label="Trend", zorder=1)

                label = MODEL_LABELS.get(model, model)
                title = (
                    f"{label} — "
                    f"{FAMILY_LABELS.get(family, family)} / "
                    f"{NOISE_TYPE_LABELS.get(noise_type, noise_type)}"
                )
                _style_ax(ax, "Structural Noise Fraction", "Test ARI", title)

                fname = f"{model}_{family}_{noise_type}.pdf"
                path = output_dir / fname
                fig.savefig(path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                saved.append(path)

    logger.info("Saved %d per-model structural-noise plots to %s", len(saved), output_dir)
    return saved


# ═══════════════════════════════════════════════════════════════════════════
#  PART II: Feature-informativeness figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_informativeness_joint(
    condition_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate 12 joint feature-informativeness plots.

    One plot per (family, structural_noise_type, structural_noise_code).
    Each contains validation and test curves for all 9 models.
    """
    df = pd.read_csv(condition_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for family in df["family"].unique():
        for noise_type in df["structural_noise_type"].unique():
            for noise_code in df["structural_noise_code"].unique():
                sub = df[
                    (df["family"] == family)
                    & (df["structural_noise_type"] == noise_type)
                    & (df["structural_noise_code"].astype(str) == str(noise_code))
                ]
                if sub.empty:
                    continue

                fig, ax = plt.subplots(figsize=(12, 7))

                for model in sub["model"].unique():
                    ms = sub[sub["model"] == model].sort_values(
                        "feature_informativeness_frac", ascending=False
                    )
                    x = ms["feature_informativeness_frac"].values
                    color = MODEL_COLORS.get(model, None)
                    label = MODEL_LABELS.get(model, model)

                    # test curve (solid)
                    ax.errorbar(
                        x,
                        ms["mean_test_ari_overall"].values,
                        yerr=ms["std_test_ari_overall"].values,
                        marker="o", label=f"{label} (test)",
                        color=color, capsize=3, linewidth=2,
                    )
                    # validation curve (dashed)
                    ax.errorbar(
                        x,
                        ms["mean_validation_ari_overall"].values,
                        yerr=ms["std_validation_ari_overall"].values,
                        marker="s", label=f"{label} (val)",
                        color=color, capsize=3, linewidth=1.5,
                        linestyle="--", alpha=0.6,
                    )

                noise_frac = sub["structural_noise_frac"].iloc[0]
                title = (
                    f"{FAMILY_LABELS.get(family, family)} — "
                    f"{NOISE_TYPE_LABELS.get(noise_type, noise_type)} — "
                    f"Structural Noise = {noise_frac:.2f}"
                )
                _style_ax(
                    ax,
                    "Feature Informativeness",
                    "ARI",
                    title,
                )
                ax.invert_xaxis()  # 1.0 (informative) on left, 0.0 on right

                fname = f"joint_{family}_{noise_type}_{noise_code}.pdf"
                path = output_dir / fname
                fig.savefig(path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                saved.append(path)
                logger.info("Saved joint feature plot: %s", path)

    return saved


def plot_feature_informativeness_by_model(
    graph_summary_csv: str | Path,
    condition_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate 108 per-model feature-informativeness plots.

    Each plot corresponds to one (family, noise_type, noise_code, model)
    and shows graph-level points plus mean trend lines for validation
    and test ARI.
    """
    graph_df = pd.read_csv(graph_summary_csv)
    cond_df = pd.read_csv(condition_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for family in graph_df["family"].unique():
        for noise_type in graph_df["structural_noise_type"].unique():
            for noise_code in graph_df["structural_noise_code"].unique():
                for model in graph_df["model"].unique():
                    gsub = graph_df[
                        (graph_df["family"] == family)
                        & (graph_df["structural_noise_type"] == noise_type)
                        & (graph_df["structural_noise_code"].astype(str) == str(noise_code))
                        & (graph_df["model"] == model)
                    ]
                    csub = cond_df[
                        (cond_df["family"] == family)
                        & (cond_df["structural_noise_type"] == noise_type)
                        & (cond_df["structural_noise_code"].astype(str) == str(noise_code))
                        & (cond_df["model"] == model)
                    ]
                    if gsub.empty:
                        continue

                    fig, ax = plt.subplots(figsize=(8, 5))

                    color = MODEL_COLORS.get(model, "gray")

                    # graph-level test ARI points
                    ax.scatter(
                        gsub["feature_informativeness_frac"],
                        gsub["mean_test_ari"],
                        alpha=0.3, color=color,
                        label="Graph-level test ARI", zorder=2,
                    )

                    csub_sorted = csub.sort_values(
                        "feature_informativeness_frac", ascending=False
                    )
                    x = csub_sorted["feature_informativeness_frac"].values

                    # mean test curve
                    ax.plot(
                        x,
                        csub_sorted["mean_test_ari_overall"].values,
                        "o-", color=color, linewidth=2,
                        label="Mean test ARI", zorder=3,
                    )
                    # mean validation curve
                    ax.plot(
                        x,
                        csub_sorted["mean_validation_ari_overall"].values,
                        "s--", color=color, linewidth=1.5, alpha=0.6,
                        label="Mean val ARI", zorder=3,
                    )

                    noise_frac = csub["structural_noise_frac"].iloc[0] if not csub.empty else "?"
                    label_str = MODEL_LABELS.get(model, model)
                    title = (
                        f"{label_str} — "
                        f"{FAMILY_LABELS.get(family, family)} / "
                        f"{NOISE_TYPE_LABELS.get(noise_type, noise_type)} / "
                        f"SN={noise_frac}"
                    )
                    _style_ax(ax, "Feature Informativeness", "ARI", title)
                    ax.invert_xaxis()

                    fname = f"{model}_{family}_{noise_type}_{noise_code}.pdf"
                    path = output_dir / fname
                    fig.savefig(path, bbox_inches="tight", dpi=300)
                    plt.close(fig)
                    saved.append(path)

    logger.info(
        "Saved %d per-model feature-informativeness plots to %s",
        len(saved), output_dir,
    )
    return saved


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience: generate all figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(results_root: str | Path) -> None:
    """Generate every figure from the standard results directory layout."""
    results_root = Path(results_root)

    # ── structural noise ────────────────────────────────────────────────
    sn = results_root / "structural_noise"
    sn_cond = sn / "summary" / "structural_noise_plot_summary.csv"
    sn_graph = sn / "summary" / "graph_level_structural_noise_summary.csv"

    if sn_cond.exists():
        plot_structural_noise_joint(sn_cond, sn / "plots" / "joint")
        if sn_graph.exists():
            plot_structural_noise_by_model(sn_graph, sn_cond, sn / "plots" / "by_model")
    else:
        logger.warning("Structural-noise summary not found: %s", sn_cond)

    # ── feature informativeness ─────────────────────────────────────────
    fi = results_root / "feature_informativeness"
    fi_cond = fi / "summary" / "feature_informativeness_plot_summary.csv"
    fi_graph = fi / "summary" / "graph_level_feature_informativeness_summary.csv"

    if fi_cond.exists():
        plot_feature_informativeness_joint(fi_cond, fi / "plots" / "joint")
        if fi_graph.exists():
            plot_feature_informativeness_by_model(
                fi_graph, fi_cond, fi / "plots" / "by_model"
            )
    else:
        logger.warning(
            "Feature-informativeness summary not found: %s", fi_cond
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    plot_all(root)
