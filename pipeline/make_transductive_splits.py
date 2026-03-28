"""
Create fixed transductive 70/15/15 train / validation / test node splits.

Each graph gets one deterministic split saved as a CSV with columns:
    graph_id, split_id, split_role, node_id

The same split is reused across all models and (for experiment 2) across all
feature-informativeness levels so that observed performance differences can be
attributed to the experimental variable rather than to changing node partitions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DEFAULT_SPLIT_ID = "split_1"


def make_split(
    n_nodes: int,
    seed: int = 0,
    split_id: str = DEFAULT_SPLIT_ID,
) -> pd.DataFrame:
    """Generate a single 70/15/15 split for *n_nodes* nodes.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes in the graph.
    seed : int
        Random seed for reproducibility.
    split_id : str
        Identifier for this split (e.g. ``"split_1"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``split_id``, ``split_role``, ``node_id``.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_nodes)

    n_train = int(np.floor(n_nodes * TRAIN_FRAC))
    n_val = int(np.floor(n_nodes * VAL_FRAC))

    train_ids = perm[:n_train]
    val_ids = perm[n_train : n_train + n_val]
    test_ids = perm[n_train + n_val :]

    records = (
        [{"split_id": split_id, "split_role": "train", "node_id": int(nid)} for nid in train_ids]
        + [{"split_id": split_id, "split_role": "validation", "node_id": int(nid)} for nid in val_ids]
        + [{"split_id": split_id, "split_role": "test", "node_id": int(nid)} for nid in test_ids]
    )
    return pd.DataFrame(records)


def make_splits_for_table(
    experiment_table: pd.DataFrame,
    output_path: str | Path,
    base_seed: int = 0,
) -> Path:
    """Create and save splits for every unique graph in the experiment table.

    Parameters
    ----------
    experiment_table : pd.DataFrame
        Must contain ``graph_id`` and ``n_nodes`` columns.
    output_path : path
        CSV file to write.  Parent directories are created if needed.
    base_seed : int
        The seed for graph *i* (in sorted order) is ``base_seed + i``.

    Returns
    -------
    Path
        The path to the saved splits CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_graphs = (
        experiment_table[["graph_id", "n_nodes"]]
        .drop_duplicates(subset="graph_id")
        .sort_values("graph_id")
        .reset_index(drop=True)
    )

    all_splits = []
    for i, (_, row) in enumerate(unique_graphs.iterrows()):
        graph_id = row["graph_id"]
        n_nodes = int(row["n_nodes"])
        split_df = make_split(n_nodes, seed=base_seed + i)
        split_df["graph_id"] = graph_id
        all_splits.append(split_df)

    splits = pd.concat(all_splits, ignore_index=True)
    splits = splits[["graph_id", "split_id", "split_role", "node_id"]]
    splits.to_csv(output_path, index=False)

    logger.info(
        "Saved splits for %d graphs (%d rows) to %s",
        len(unique_graphs),
        len(splits),
        output_path,
    )
    return output_path


def load_split(
    splits_path: str | Path,
    graph_id: str,
    split_id: str = DEFAULT_SPLIT_ID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train / validation / test node IDs for one graph.

    Returns
    -------
    (train_ids, val_ids, test_ids)
        Each is a 1-D int array of node indices.
    """
    df = pd.read_csv(splits_path)
    mask = (df["graph_id"] == graph_id) & (df["split_id"] == split_id)
    sub = df[mask]

    train_ids = sub.loc[sub["split_role"] == "train", "node_id"].values
    val_ids = sub.loc[sub["split_role"] == "validation", "node_id"].values
    test_ids = sub.loc[sub["split_role"] == "test", "node_id"].values
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    table_path = sys.argv[1] if len(sys.argv) > 1 else (
        "data/synthetic_benchmark/metadata/structural_noise_experiment_table.csv"
    )
    out = sys.argv[2] if len(sys.argv) > 2 else (
        "results/structural_noise/splits/structural_noise_splits.csv"
    )
    table = pd.read_csv(table_path)
    make_splits_for_table(table, out)
