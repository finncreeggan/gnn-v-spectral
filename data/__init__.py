""" Data Loading"""

import torch
from jaxtyping import Int, Float
from dataclasses import dataclass

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
import torch_geometric.transforms as T

DEFAULT_DATASET_ROOT = "data/cache/synthetic"

@dataclass
class DatasetNames():
    SBM_RAND: str = "sbm_random"
    SBM_TARG: str = "sbm_targeted"
    LFR_RAND: str = "lfr_random"
    LFR_TARG: str = "lfr_targetd"

@dataclass
class GraphData():
    graph: BaseData
    num_classes: int
    labels: Int[torch.Tensor, "n_nodes"]

    train_idx: Int[torch.Tensor, "num_train_nodes"]
    val_idx: Int[torch.Tensor, "num_valid_nodes"]
    test_idx: Int[torch.Tensor, "num_test_nodes"]

    whole_eigenspectrum: Float[torch.Tensor, "n_nodes n_nodes"]
    kcut_eigenspectrum: Float[torch.Tensor, "n_nodes n_eigenvectors"]
    regularized_eigenspectrum: Float[torch.Tensor, "n_nodes n_nodes"]

    whole_eigenvals: Float[torch.Tensor, "n_nodes"]
    kcut_eigenvals: Float[torch.Tensor, "n_eigenvectors"]
    regularized_eigenvals: Float[torch.Tensor, "n_nodes"]

    features: Float[torch.Tensor, "n_nodes feature_dim"]


#### Dataloading ####
def load_graph_data(dataset_name: str) -> GraphData:
    "IN PROGRESS"
    if dataset_name == DatasetNames.SBM_RAND:
        pass
    elif dataset_name == DatasetNames.SBM_TARG:
        pass
    else:
        pass

    return GraphData(graph=graph_data, num_classes=dataset.num_classes, labels=labels, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
