from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from jaxtyping import Int


def _validate_graph_and_labels(G: nx.Graph, labels: np.ndarray) -> np.ndarray:
    """
    Validate that labels align with node ids 0..n-1 and return labels as int64.
    """
    labels = np.asarray(labels, dtype=np.int64)

    n_nodes = G.number_of_nodes()
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}.")

    if len(labels) != n_nodes:
        raise ValueError(
            f"Label length {len(labels)} does not match node count {n_nodes}."
        )

    expected_nodes = list(range(n_nodes))
    actual_nodes = sorted(G.nodes())
    if actual_nodes != expected_nodes:
        raise ValueError(
            "Graph nodes must be contiguous integers 0..n-1 for label alignment."
        )

    return labels


def format_base_graph_id(index: int) -> str:
    """
    Convert an integer like 1 into 'graph001'.
    """
    if index < 1:
        raise ValueError("Base graph index must be >= 1.")
    return f"graph{index:03d}"


def format_noise_code(noise_frac: float) -> str:
    """
    Convert a noise fraction like 0.05 into '005', 0.45 into '045', etc.
    """
    if not (0.0 <= noise_frac <= 1.0):
        raise ValueError("noise_frac must be in [0, 1].")

    scaled = round(noise_frac * 100)
    return f"{scaled:03d}"


def make_graph_id(
    base_graph_id: str,
    noise_code: str,
    noise_type: str,
    family: str,
) -> str:
    """
    Build a graph identifier like:
    graph001_005_random_sbm
    """
    return f"{base_graph_id}_{noise_code}_{noise_type}_{family}"


def make_output_paths(
    dataset_root: str | Path,
    family: str,
    noise_type: str,
    graph_id: str,
) -> dict[str, Path]:
    """
    Build output paths for one graph instance.

    Expected structure:
      dataset_root/
        sbm/
          clean/edges/
          clean/labels/
          random/edges/
          random/labels/
          targeted_betweenness/edges/
          targeted_betweenness/labels/
        lfr/
          ...
        metadata/
          graph_index_sbm.csv
          graph_index_lfr.csv
    """
    dataset_root = Path(dataset_root)

    edge_dir = dataset_root / family / noise_type / "edges"
    label_dir = dataset_root / family / noise_type / "labels"
    metadata_dir = dataset_root / "metadata"

    edge_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    return {
        "edge_dir": edge_dir,
        "label_dir": label_dir,
        "metadata_dir": metadata_dir,
        "edge_path": edge_dir / f"{graph_id}.csv",
        "label_path": label_dir / f"{graph_id}_labels.npy",
        "metadata_path": metadata_dir / f"graph_index_{family}.csv",
    }


def _canonical_comm_pair(label_u: int, label_v: int) -> str:
    """
    Build a canonical undirected community-pair string.
    For example, labels 3 and 1 become '1_3'.
    """
    a, b = sorted((int(label_u), int(label_v)))
    return f"{a}_{b}"


def graph_to_edgelist_df(G: nx.Graph, labels: np.ndarray) -> pd.DataFrame:
    """
    Convert a graph into the agreed edge-list CSV format with columns:
      src, dst, same_comm, comm_pair

    Notes
    -----
    - Since the graph is undirected, edges are saved once with src < dst.
    - comm_pair is stored canonically as min_label_max_label, e.g. '1_3'.
    """
    labels = _validate_graph_and_labels(G, labels)

    rows: list[dict[str, Any]] = []
    for u, v in sorted(G.edges()):
        src, dst = sorted((int(u), int(v)))
        same_comm = int(labels[src] == labels[dst])
        comm_pair = _canonical_comm_pair(labels[src], labels[dst])

        rows.append(
            {
                "src": src,
                "dst": dst,
                "same_comm": same_comm,
                "comm_pair": comm_pair,
            }
        )

    return pd.DataFrame(rows, columns=["src", "dst", "same_comm", "comm_pair"])


def save_graph_edgelist(G: nx.Graph, labels: np.ndarray, path: str | Path) -> Path:
    """
    Save the graph edge list CSV in the agreed format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = graph_to_edgelist_df(G, labels)
    df.to_csv(path, index=False)
    return path


def save_labels(labels: np.ndarray, path: str | Path) -> Path:
    """
    Save a 1D integer label array as .npy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels, dtype=np.int64)
    np.save(path, labels)
    return path


def make_metadata_row(
    *,
    graph_id: str,
    family: str,
    base_graph_id: str,
    seed: int,
    noise_type: str,
    noise_code: str,
    noise_frac: float,
    edge_path: str | Path,
    label_path: str | Path,
    stats: dict[str, Any],
    family_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Build one metadata row for graph_index_{family}.csv.

    `stats` should come from characterize.py.
    `family_metadata` should come from the clean generator / perturbation pipeline.
    """
    row: dict[str, Any] = {
        "graph_id": graph_id,
        "family": family,
        "base_graph_id": base_graph_id,
        "seed": seed,
        "noise_type": noise_type,
        "noise_code": noise_code,
        "noise_frac": noise_frac,
        "edge_path": str(edge_path),
        "label_path": str(label_path),
    }

    row.update(stats)
    row.update(family_metadata)
    return row


def write_metadata_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """
    Write all metadata rows to a CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def load_edge_index(edge_path: str | Path) -> Int[torch.Tensor, "2 num_edges"]:
    """
    Load an edge-list CSV and return an undirected edge_index tensor.

    The CSV must have 'src' and 'dst' columns (stored once with src < dst).
    Both directions are added so the result is undirected.

    Parameters
    ----------
    edge_path : str | Path

    Returns
    -------
    Int[Tensor, "2 num_edges"]
    """
    df = pd.read_csv(edge_path)
    src = torch.tensor(df["src"].to_numpy(dtype=np.int64), dtype=torch.long)
    dst = torch.tensor(df["dst"].to_numpy(dtype=np.int64), dtype=torch.long)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


__all__ = [
    "format_base_graph_id",
    "format_noise_code",
    "make_graph_id",
    "make_output_paths",
    "graph_to_edgelist_df",
    "save_graph_edgelist",
    "save_labels",
    "load_edge_index",
    "make_metadata_row",
    "write_metadata_csv",
]