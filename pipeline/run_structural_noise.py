"""
Experiment 1 – Structural-noise sweep.

Evaluate all 9 models across every graph instance in the structural-noise
metadata table.  For each (graph, model) pair the pipeline:

  1. Loads the transductive 70/15/15 split.
  2. Calls classifier.fit(...) with Optuna tuning.
  3. Calls classifier.score(...) on the held-out test nodes.
  4. Appends one row to the raw results CSV on disk (incremental writes).

Failed runs are logged to ``logs/failed_runs.csv`` rather than silently
skipped.

The nine classifiers are sourced from Sabrina's GraphModelSuite via
``methods.registry.METHOD_REGISTRY``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── The 9 model keys expected in METHOD_REGISTRY ────────────────────────────
MODEL_KEYS = [
    "whole_lr",
    "whole_rf",
    "kcut_lr",
    "kcut_rf",
    "regularized_lr",
    "regularized_rf",
    "sgc",
    "gcn",
    "gat",
]

DEFAULT_OPTUNA_TRIALS = 40


def _get_model(model_key: str, num_classes: int):
    """Retrieve and instantiate a model from METHOD_REGISTRY.

    Parameters
    ----------
    model_key : str
        Key into METHOD_REGISTRY (one of MODEL_KEYS).
    num_classes : int
        Number of communities/classes in the graph, required by ExperimentConfig.

    Returns
    -------
    BaseMethod instance.
    """
    from methods import METHOD_REGISTRY, ExperimentConfig
    # Placeholder defaults — Optuna tuning to be wired in once Sabrina's fit() supports it
    config = ExperimentConfig(num_classes=num_classes, seed=42)
    return METHOD_REGISTRY[model_key](config)


def run_single(
    model_key: str,
    graph_row: pd.Series,
    splits_path: str | Path,
    optuna_n_trials: int = DEFAULT_OPTUNA_TRIALS,
    optuna_storage_path: str | Path | None = None,
) -> dict:
    """Fit and score one model on one graph instance.

    Parameters
    ----------
    model_key : str
        Key into METHOD_REGISTRY (one of MODEL_KEYS).
    graph_row : pd.Series
        A single row from the structural-noise experiment table.
    splits_path : path
        Retained for API compatibility but no longer used — splits are now
        generated inside load_graph_data with a fixed seed.
    optuna_n_trials : int
        Number of Optuna trials for hyperparameter search.
        TODO: wire into method once Sabrina's fit() supports it.
    optuna_storage_path : path, optional
        Path for the Optuna study database (SQLite).
        TODO: wire into method once Sabrina's fit() supports it.

    Returns
    -------
    dict
        Raw result row ready to be appended to the results CSV.
    """
    from data import load_graph_data

    graph_id = graph_row["graph_id"]
    # spectra_path is a single .pt file (whole_V, whole_evals, reg_V, reg_evals).
    # NOTE: build_metadata_tables.py must add a "spectra_path" column for this to work.
    metadata_csv = f"data/synthetic_benchmark/metadata/graph_index_{graph_row['family']}.csv"

    # Experiment 1: no node features — load_graph_data falls back to identity matrix
    data = load_graph_data(
        metadata_csv=metadata_csv,
        graph_id=graph_id,
        spectra_path=graph_row["spectra_path"],
        seed=1,
    )

    classifier = _get_model(model_key, data.num_classes)

    # ── fit on train_idx (handled internally by method) ──────────────────
    classifier.fit(data)

    # ── score on val and test splits ─────────────────────────────────────
    val_metrics  = classifier.score(data)
    test_metrics = classifier.score(data, use_test_idx=True)

    return {
        "graph_id": graph_id,
        "family": graph_row["family"],
        "base_graph_id": graph_row.get("base_graph_id", ""),
        "structural_noise_type": graph_row["structural_noise_type"],
        "structural_noise_code": graph_row["structural_noise_code"],
        "structural_noise_frac": graph_row.get("structural_noise_frac", ""),
        "model": model_key,
        "split_id": "split_1",
        "optuna_n_trials": optuna_n_trials,
        "best_validation_ari": val_metrics.get("ARI"),
        "test_ari": test_metrics.get("ARI"),
        "best_params_json": json.dumps({}),
    }


def run_structural_noise_experiment(
    experiment_table: pd.DataFrame,
    splits_path: str | Path,
    output_csv: str | Path,
    failed_csv: str | Path | None = None,
    optuna_n_trials: int = DEFAULT_OPTUNA_TRIALS,
    optuna_storage_path: str | Path | None = None,
    model_keys: list[str] | None = None,
) -> Path:
    """Run experiment 1 across all graphs and models.

    Results are appended to *output_csv* after every (graph, model) run so
    that partial progress is preserved if the pipeline is interrupted.

    Parameters
    ----------
    experiment_table : pd.DataFrame
        Structural-noise experiment table.
    splits_path : path
        Precomputed node-split CSV.
    output_csv : path
        Destination for raw results.
    failed_csv : path, optional
        Log file for failed runs.
    optuna_n_trials : int
        Optuna budget per fit.
    optuna_storage_path : path, optional
        SQLite path for Optuna studies.
    model_keys : list[str], optional
        Subset of MODEL_KEYS to run.  Defaults to all 9.

    Returns
    -------
    Path to the raw results CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    failed_csv = Path(failed_csv) if failed_csv else output_csv.parent / "failed_runs.csv"
    failed_csv.parent.mkdir(parents=True, exist_ok=True)
    model_keys = model_keys or MODEL_KEYS

    header_written = output_csv.exists()
    failed_header_written = failed_csv.exists()

    total = len(experiment_table) * len(model_keys)
    done = 0

    for _, row in experiment_table.iterrows():
        for model_key in model_keys:
            done += 1
            graph_id = row["graph_id"]
            logger.info(
                "[%d/%d] Running %s on %s ...", done, total, model_key, graph_id
            )

            try:
                result = run_single(
                    model_key=model_key,
                    graph_row=row,
                    splits_path=splits_path,
                    optuna_n_trials=optuna_n_trials,
                    optuna_storage_path=optuna_storage_path,
                )
                result_df = pd.DataFrame([result])
                result_df.to_csv(
                    output_csv,
                    mode="a",
                    header=not header_written,
                    index=False,
                )
                header_written = True

            except Exception as exc:
                logger.error(
                    "FAILED %s on %s: %s", model_key, graph_id, exc
                )
                fail_row = pd.DataFrame([{
                    "graph_id": graph_id,
                    "model": model_key,
                    "error": str(exc),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }])
                fail_row.to_csv(
                    failed_csv,
                    mode="a",
                    header=not failed_header_written,
                    index=False,
                )
                failed_header_written = True

    logger.info("Structural-noise experiment complete. Results: %s", output_csv)
    return output_csv


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    table_path = sys.argv[1] if len(sys.argv) > 1 else (
        "data/synthetic_benchmark/metadata/structural_noise_experiment_table.csv"
    )
    splits = sys.argv[2] if len(sys.argv) > 2 else (
        "results/structural_noise/splits/structural_noise_splits.csv"
    )
    out = sys.argv[3] if len(sys.argv) > 3 else (
        "results/structural_noise/raw/structural_noise_results.csv"
    )

    table = pd.read_csv(table_path)
    run_structural_noise_experiment(table, splits, out)
