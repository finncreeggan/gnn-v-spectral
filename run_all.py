"""
Full pipeline orchestrator.

Usage:
    python run_all.py                          # run everything
    python run_all.py --experiment 1           # structural noise only
    python run_all.py --experiment 2           # feature informativeness only
    python run_all.py --summarize-only         # just summarize + plot existing results
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from data import DEFAULT_DATASET_ROOT

DATA_ROOT = Path(DEFAULT_DATASET_ROOT)
RESULTS_ROOT = Path("results")

logger = logging.getLogger(__name__)


def step_build_metadata() -> tuple[Path, Path]:
    """Step 1: Build enriched metadata tables."""
    from pipeline.build_metadata_tables import save_metadata_tables

    logger.info("Building metadata tables from %s ...", DATA_ROOT)
    return save_metadata_tables(DATA_ROOT)


def step_run_experiment_1(structural_table_path: Path) -> Path:
    """Step 2: Run structural-noise experiment."""
    from pipeline.run_structural_noise import run_structural_noise_experiment

    table = pd.read_csv(structural_table_path)
    out = RESULTS_ROOT / "structural_noise" / "raw" / "structural_noise_results.csv"

    logger.info("Running Experiment 1 (%d graph rows) ...", len(table))
    return run_structural_noise_experiment(table, out)


def step_generate_features(feature_table_path: Path) -> pd.DataFrame:
    """Step 3: Generate feature matrices for experiment 2."""
    from pipeline.generate_feature_informativeness import generate_all_features

    table = pd.read_csv(feature_table_path)
    features_root = DATA_ROOT / "features"

    logger.info("Generating features for %d rows ...", len(table))
    enriched = generate_all_features(table, features_root, dataset_root=DATA_ROOT)
    enriched.to_csv(feature_table_path, index=False)
    return enriched


def step_run_experiment_2(feature_table_path: Path) -> Path:
    """Step 4: Run feature-informativeness experiment."""
    from pipeline.run_feature_informativeness import (
        run_feature_informativeness_experiment,
    )

    table = pd.read_csv(feature_table_path)
    out = (
        RESULTS_ROOT / "feature_informativeness" / "raw"
        / "feature_informativeness_results.csv"
    )

    logger.info("Running Experiment 2 (%d graph rows) ...", len(table))
    return run_feature_informativeness_experiment(table, out)


def step_summarize() -> None:
    """Step 5: Summarize results."""
    from pipeline.summarize_results import summarize_all

    logger.info("Summarizing results ...")
    summarize_all(RESULTS_ROOT)


def step_plot() -> None:
    """Step 6: Generate plots."""
    from pipeline.plot_results import plot_all

    logger.info("Generating plots ...")
    plot_all(RESULTS_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2], default=None,
        help="Run only experiment 1 or 2 (default: both)",
    )
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="Skip experiments; just summarize and plot existing results",
    )
    args = parser.parse_args()

    if args.summarize_only:
        step_summarize()
        step_plot()
        return

    structural_path, feature_path = step_build_metadata()

    if args.experiment is None or args.experiment == 1:
        step_run_experiment_1(structural_path)

    if args.experiment is None or args.experiment == 2:
        step_generate_features(feature_path)
        step_run_experiment_2(feature_path)

    step_summarize()
    step_plot()

    logger.info("All done. Results in %s", RESULTS_ROOT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
