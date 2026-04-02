"""
Experiment 2 – Feature-informativeness sweep.

Evaluate all 9 models across 60 selected graphs (3 structural-noise levels x
2 noise types x 2 families x 5 base graphs) at 11 feature-informativeness
levels each.

For each (graph, feature_informativeness, model) triple the pipeline:

  1. Loads the fixed transductive 70/15/15 split (same split reused across
     all informativeness levels for a given graph).
  2. Calls classifier.fit(...) passing precomputed spectral embeddings.
  3. Calls classifier.score(...) on the held-out test nodes.
  4. Appends one row to the raw results CSV.

Structure-only models ignore the feature_path argument and act as reference
baselines.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from data import DEFAULT_DATASET_ROOT
from pipeline.run_structural_noise import MODEL_KEYS, DEFAULT_OPTUNA_TRIALS, _get_model

logger = logging.getLogger(__name__)


def run_single_feature(
    model_key: str,
    row: pd.Series,
    optuna_n_trials: int = DEFAULT_OPTUNA_TRIALS,
    optuna_storage_path: str | Path | None = None,
) -> dict:
    """Fit and score one model on one graph at one feature-informativeness level."""
    from data import load_graph_data
    from methods.spectral.spectral_method import SpectralMethod

    graph_id = row["graph_id"]
    metadata_csv = str(
        Path(DEFAULT_DATASET_ROOT) / "metadata" / f"graph_index_{row['family']}.csv"
    )

    feature_path = row.get("feature_path")
    if feature_path is not None:
        feature_path = str(Path(DEFAULT_DATASET_ROOT) / feature_path)

    data = load_graph_data(
        metadata_csv=metadata_csv,
        graph_id=graph_id,
        features_pt=feature_path,
        seed=1,
    )

    classifier = _get_model(model_key, data.num_classes)

    # Pass precomputed embeddings for spectral methods
    if isinstance(classifier, SpectralMethod):
        embedding_map = {
            "whole": data.whole_eigenspectrum,
            "kcut": data.kcut_eigenspectrum,
            "regularized": data.regularized_eigenspectrum,
        }
        classifier.fit(data, embeddings=embedding_map[classifier.embedding_type])
    else:
        classifier.fit(data)

    val_metrics = classifier.score(data)
    test_metrics = classifier.score(data, use_test_idx=True)

    return {
        "graph_id": graph_id,
        "family": row["family"],
        "base_graph_id": row.get("base_graph_id", ""),
        "structural_noise_type": row["structural_noise_type"],
        "structural_noise_code": row["structural_noise_code"],
        "structural_noise_frac": row.get("structural_noise_frac", ""),
        "feature_informativeness_code": row["feature_informativeness_code"],
        "feature_informativeness_frac": row["feature_informativeness_frac"],
        "feature_noise_frac": row.get("feature_noise_frac", ""),
        "feature_generation_seed": row.get("feature_generation_seed", ""),
        "model": model_key,
        "split_id": "split_1",
        "optuna_n_trials": optuna_n_trials,
        "best_validation_ari": val_metrics.get("ARI"),
        "test_ari": test_metrics.get("ARI"),
        "best_params_json": json.dumps({}),
    }


def run_feature_informativeness_experiment(
    experiment_table: pd.DataFrame,
    output_csv: str | Path,
    failed_csv: str | Path | None = None,
    optuna_n_trials: int = DEFAULT_OPTUNA_TRIALS,
    optuna_storage_path: str | Path | None = None,
    model_keys: list[str] | None = None,
) -> Path:
    """Run experiment 2 across all graphs, informativeness levels, and models."""
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
            fi_code = row["feature_informativeness_code"]
            logger.info(
                "[%d/%d] Running %s on %s (fi=%s) ...",
                done, total, model_key, graph_id, fi_code,
            )

            try:
                result = run_single_feature(
                    model_key=model_key, row=row,
                    optuna_n_trials=optuna_n_trials,
                    optuna_storage_path=optuna_storage_path,
                )
                result_df = pd.DataFrame([result])
                result_df.to_csv(
                    output_csv, mode="a", header=not header_written,
                    index=False,
                )
                header_written = True

            except Exception as exc:
                logger.error(
                    "FAILED %s on %s (fi=%s): %s",
                    model_key, graph_id, fi_code, exc,
                )
                fail_row = pd.DataFrame([{
                    "graph_id": graph_id,
                    "feature_informativeness_code": fi_code,
                    "model": model_key,
                    "error": str(exc),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }])
                fail_row.to_csv(
                    failed_csv, mode="a", header=not failed_header_written,
                    index=False,
                )
                failed_header_written = True

    logger.info(
        "Feature-informativeness experiment complete. Results: %s", output_csv
    )
    return output_csv


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    table_path = sys.argv[1] if len(sys.argv) > 1 else (
        str(Path(DEFAULT_DATASET_ROOT) / "metadata"
            / "feature_informativeness_experiment_table.csv")
    )
    out = sys.argv[2] if len(sys.argv) > 2 else (
        "results/feature_informativeness/raw/feature_informativeness_results.csv"
    )

    table = pd.read_csv(table_path)
    run_feature_informativeness_experiment(table, out)
