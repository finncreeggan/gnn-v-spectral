"""
Produce graph-level and condition-level summary CSVs from raw experiment
results for both the structural-noise and feature-informativeness experiments.

Summary hierarchy:
    raw results -> graph-level summary -> condition-level plotting summary

The condition-level summary averages across the 5 base graphs at each
experimental condition, which is the independent unit of uncertainty.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ─── Structural-noise summaries ─────────────────────────────────────────────

STRUCTURAL_GROUP_COLS = [
    "graph_id", "family", "base_graph_id",
    "structural_noise_type", "structural_noise_code", "structural_noise_frac",
    "model",
]

STRUCTURAL_CONDITION_COLS = [
    "family", "structural_noise_type",
    "structural_noise_code", "structural_noise_frac",
    "model",
]


def summarize_structural_noise_graph_level(
    raw_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Graph-level summary: one row per (graph_id, model).

    In the first-pass benchmark with one split per graph this is essentially
    a copy of the raw table with selected columns.  When multiple splits are
    evaluated in the future, this will average across splits.
    """
    raw = pd.read_csv(raw_csv)
    summary = (
        raw
        .groupby(STRUCTURAL_GROUP_COLS, as_index=False)
        .agg(
            mean_validation_ari=("best_validation_ari", "mean"),
            mean_test_ari=("test_ari", "mean"),
        )
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    logger.info("Saved graph-level structural-noise summary: %s", output_csv)
    return summary


def summarize_structural_noise_condition_level(
    graph_summary_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Condition-level plotting summary: averaged across 5 base graphs."""
    graph_summary = pd.read_csv(graph_summary_csv)
    condition = (
        graph_summary
        .groupby(STRUCTURAL_CONDITION_COLS, as_index=False)
        .agg(
            mean_validation_ari_overall=("mean_validation_ari", "mean"),
            std_validation_ari_overall=("mean_validation_ari", "std"),
            mean_test_ari_overall=("mean_test_ari", "mean"),
            std_test_ari_overall=("mean_test_ari", "std"),
            n_graphs=("mean_test_ari", "count"),
        )
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    condition.to_csv(output_csv, index=False)
    logger.info("Saved condition-level structural-noise summary: %s", output_csv)
    return condition


# ─── Feature-informativeness summaries ──────────────────────────────────────

FEATURE_GROUP_COLS = [
    "graph_id", "family", "base_graph_id",
    "structural_noise_type", "structural_noise_code", "structural_noise_frac",
    "feature_informativeness_code", "feature_informativeness_frac",
    "feature_noise_frac",
    "model",
]

FEATURE_CONDITION_COLS = [
    "family", "structural_noise_type",
    "structural_noise_code", "structural_noise_frac",
    "feature_informativeness_code", "feature_informativeness_frac",
    "feature_noise_frac",
    "model",
]


def summarize_feature_informativeness_graph_level(
    raw_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Graph-level summary: one row per (graph_id, feature_info_code, model)."""
    raw = pd.read_csv(raw_csv)
    summary = (
        raw
        .groupby(FEATURE_GROUP_COLS, as_index=False)
        .agg(
            mean_validation_ari=("best_validation_ari", "mean"),
            mean_test_ari=("test_ari", "mean"),
        )
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    logger.info(
        "Saved graph-level feature-informativeness summary: %s", output_csv
    )
    return summary


def summarize_feature_informativeness_condition_level(
    graph_summary_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Condition-level plotting summary: averaged across 5 base graphs."""
    graph_summary = pd.read_csv(graph_summary_csv)
    condition = (
        graph_summary
        .groupby(FEATURE_CONDITION_COLS, as_index=False)
        .agg(
            mean_validation_ari_overall=("mean_validation_ari", "mean"),
            std_validation_ari_overall=("mean_validation_ari", "std"),
            mean_test_ari_overall=("mean_test_ari", "mean"),
            std_test_ari_overall=("mean_test_ari", "std"),
            n_graphs=("mean_test_ari", "count"),
        )
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    condition.to_csv(output_csv, index=False)
    logger.info(
        "Saved condition-level feature-informativeness summary: %s", output_csv
    )
    return condition


def summarize_all(results_root: str | Path) -> None:
    """Convenience: run all four summarization steps.

    Expects the standard results directory layout:
        results_root/structural_noise/raw/structural_noise_results.csv
        results_root/feature_informativeness/raw/feature_informativeness_results.csv
    """
    results_root = Path(results_root)

    # ── structural noise ────────────────────────────────────────────────
    sn_raw = results_root / "structural_noise" / "raw" / "structural_noise_results.csv"
    sn_graph = results_root / "structural_noise" / "summary" / "graph_level_structural_noise_summary.csv"
    sn_cond = results_root / "structural_noise" / "summary" / "structural_noise_plot_summary.csv"

    if sn_raw.exists():
        summarize_structural_noise_graph_level(sn_raw, sn_graph)
        summarize_structural_noise_condition_level(sn_graph, sn_cond)
    else:
        logger.warning("Structural-noise raw results not found: %s", sn_raw)

    # ── feature informativeness ─────────────────────────────────────────
    fi_raw = results_root / "feature_informativeness" / "raw" / "feature_informativeness_results.csv"
    fi_graph = results_root / "feature_informativeness" / "summary" / "graph_level_feature_informativeness_summary.csv"
    fi_cond = results_root / "feature_informativeness" / "summary" / "feature_informativeness_plot_summary.csv"

    if fi_raw.exists():
        summarize_feature_informativeness_graph_level(fi_raw, fi_graph)
        summarize_feature_informativeness_condition_level(fi_graph, fi_cond)
    else:
        logger.warning(
            "Feature-informativeness raw results not found: %s", fi_raw
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    summarize_all(root)
