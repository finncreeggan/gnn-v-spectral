""" Data Loading"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Int, Float
from dataclasses import dataclass

from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

from data.generators.io import load_edge_index

DEFAULT_DATASET_ROOT = "data/cache/synthetic"


@dataclass
class GraphData():
    graph: BaseData
    graph_id: str

    noise_fraction: float
    num_classes: int
    labels: Int[torch.Tensor, "n_nodes"]

    whole_eigenvals: Float[torch.Tensor, "n_nodes"]
    kcut_eigenvals: Float[torch.Tensor, "n_eigenvectors"] 
    regularized_eigenvals: Float[torch.Tensor, "n_nodes"]

    whole_eigenspectrum: Float[torch.Tensor, "n_nodes n_nodes"] 
    kcut_eigenspectrum: Float[torch.Tensor, "n_nodes n_eigenvectors"]
    regularized_eigenspectrum: Float[torch.Tensor, "n_nodes n_nodes"]

    features: Float[torch.Tensor, "n_nodes feature_dim"]

    train_idx: Int[torch.Tensor, "num_train_nodes"]
    val_idx: Int[torch.Tensor, "num_valid_nodes"]
    test_idx: Int[torch.Tensor, "num_test_nodes"]


#### Dataloading ####
def load_graph_data(
    metadata_csv: str | Path,
    graph_id: str,
    *,
    features_pt: str | Path | None = None,
    seed: int = 0,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> GraphData:
    """
    Load a single graph by graph_id from a metadata CSV into a GraphData object.

    Parameters
    ----------
    metadata_csv : str | Path
        Path to graph_index_{family}.csv.
    graph_id : str
        Row identifier in the metadata CSV.
    features_pt : str | Path | None
        Path to a .pt file containing a Float[Tensor, "n_nodes feature_dim"]
        node feature matrix. If None, falls back to an n_nodes x n_nodes
        one-hot identity matrix.
    seed : int
        Passed to torch.manual_seed before generating the 70/15/15
        train/val/test split so the partition is deterministic. Default 0.
    dataset_root : str | Path
        Root used to resolve relative paths from the CSV.

    Returns
    -------
    GraphData
    """
    dataset_root = Path(dataset_root)
    row = pd.read_csv(metadata_csv).set_index("graph_id").loc[graph_id]

    edge_path  = str(dataset_root / row["edge_path"])
    label_path = str(dataset_root / row["label_path"])

    labels     = torch.from_numpy(np.load(label_path))
    num_nodes  = len(labels)
    edge_index = load_edge_index(edge_path)
    graph      = Data(edge_index=edge_index, num_nodes=num_nodes)

    from methods.spectral.embeddings import kcut_eigenspectrum # To avoid import circular loop

    spectra     = torch.load(str(dataset_root / row["spectra_path"]), weights_only=False)
    whole_V     = spectra["whole_V"]
    whole_evals = spectra["whole_evals"]

    kcut_V, kcut_evals = kcut_eigenspectrum(
        edge_index, num_nodes, all_V=whole_V, all_eigenvalues=whole_evals
    )

    if features_pt is not None:
        features_pt = Path(features_pt)
        if features_pt.suffix == ".npy":
            features = torch.from_numpy(np.load(features_pt)).float()
        else:
            features = torch.load(features_pt, weights_only=False)
    else:
        rng = torch.Generator().manual_seed(seed)
        features = torch.randn(num_nodes, 5, generator=rng)

    torch.manual_seed(seed)
    perm      = torch.randperm(num_nodes)
    train_end = int(0.7  * num_nodes)
    val_end   = int(0.85 * num_nodes)
    train_idx = perm[:train_end]
    val_idx   = perm[train_end:val_end]
    test_idx  = perm[val_end:]

    return GraphData(
        graph=graph,
        graph_id=graph_id,
        noise_fraction=float(row["noise_frac"]), #type: ignore
        num_classes=int(row["num_communities"]), #type: ignore
        labels=labels,
        whole_eigenspectrum=whole_V,
        kcut_eigenspectrum=kcut_V,
        regularized_eigenspectrum=spectra["reg_V"],
        whole_eigenvals=whole_evals,
        kcut_eigenvals=kcut_evals,
        regularized_eigenvals=spectra["reg_evals"],
        features=features,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
