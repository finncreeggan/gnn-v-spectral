"""
Generate synthetic 5-dimensional node-feature matrices at varying
feature-informativeness levels for the second experiment.

For each selected graph and each informativeness level alpha:

    X_i = alpha * f5(y_i) + (1 - alpha) * eps_i,    eps_i ~ N(0, I_5)

where y_i is the planted class label, f5 is a fixed 5th-degree nonlinear
mapping from class labels to R^5, and alpha is the feature-informativeness
fraction.  When alpha = 1.0 the features are maximally informative; when
alpha = 0.0 they are pure noise.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 11 feature-informativeness levels (both endpoints included) ─────────────
INFORMATIVENESS_CODES = (
    "100", "090", "080", "070", "060",
    "050", "040", "030", "020", "010", "000",
)

FEATURE_DIM = 5


def _nonlinear_5th_degree_mapping(
    labels: np.ndarray,
    n_communities: int,
) -> np.ndarray:
    """Fixed 5th-degree nonlinear mapping from class labels to R^5.

    Each community k is assigned a deterministic 5-D prototype via a
    polynomial of degree 5 applied to evenly-spaced angles, ensuring
    that prototypes are well-separated in feature space.

    Parameters
    ----------
    labels : ndarray of shape (n_nodes,)
        Integer class labels in [0, n_communities).
    n_communities : int
        Number of distinct communities.

    Returns
    -------
    ndarray of shape (n_nodes, 5)
        Deterministic feature vectors (no noise yet).
    """
    angles = np.linspace(0, 2 * np.pi, n_communities, endpoint=False)

    # Build a (n_communities, 5) prototype matrix using polynomial of degree 5
    prototypes = np.column_stack([
        np.sin(angles) ** 5,
        np.cos(angles) ** 3 * np.sin(angles) ** 2,
        np.sin(2 * angles) ** 5,
        np.cos(3 * angles) ** 5,
        np.sin(angles) ** 2 * np.cos(2 * angles) ** 3,
    ])

    return prototypes[labels]


def generate_features_for_graph(
    labels: np.ndarray,
    n_communities: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single (n_nodes, 5) feature matrix at informativeness alpha.

    Parameters
    ----------
    labels : ndarray of shape (n_nodes,)
        Planted community labels.
    n_communities : int
        Number of distinct communities.
    alpha : float
        Feature-informativeness fraction in [0, 1].
    rng : numpy Generator
        Random number generator for reproducibility.

    Returns
    -------
    ndarray of shape (n_nodes, 5)
    """
    signal = _nonlinear_5th_degree_mapping(labels, n_communities)
    noise = rng.standard_normal(signal.shape)
    return alpha * signal + (1.0 - alpha) * noise


def generate_all_features(
    feature_table: pd.DataFrame,
    output_root: str | Path,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Generate and save feature matrices for every row in the experiment table.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Feature-informativeness experiment table produced by
        :func:`build_metadata_tables.build_feature_experiment_table`.
        Must contain columns: graph_id, label_path, feature_informativeness_code,
        feature_informativeness_frac, feature_path, and num_communities.
    output_root : path
        Root directory under which feature .npy files will be saved.
    base_seed : int
        Starting seed; each (graph, informativeness) pair gets a unique seed.

    Returns
    -------
    pd.DataFrame
        The input table with an added ``feature_generation_seed`` column.
    """
    output_root = Path(output_root)
    feature_table = feature_table.copy()
    seeds: list[int] = []

    for idx, row in feature_table.iterrows():
        seed = base_seed + idx
        seeds.append(seed)
        rng = np.random.default_rng(seed)

        label_path = Path(row["label_path"])
        if not label_path.exists():
            logger.warning(
                "Label file not found: %s – skipping feature generation for %s",
                label_path,
                row["graph_id"],
            )
            continue

        labels = np.load(label_path)
        n_communities = int(row.get("num_communities", len(np.unique(labels))))

        alpha = float(row["feature_informativeness_frac"])
        features = generate_features_for_graph(labels, n_communities, alpha, rng)

        out_path = Path(row["feature_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, features)

        logger.debug(
            "Saved features: %s  (alpha=%.2f, seed=%d, shape=%s)",
            out_path, alpha, seed, features.shape,
        )

    feature_table["feature_generation_seed"] = seeds

    logger.info(
        "Generated %d feature matrices under %s", len(feature_table), output_root
    )
    return feature_table


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    table_path = sys.argv[1] if len(sys.argv) > 1 else (
        "data/synthetic_benchmark/metadata/feature_informativeness_experiment_table.csv"
    )
    table = pd.read_csv(table_path)
    output = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic_benchmark/features"
    generate_all_features(table, output)
