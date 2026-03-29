"""Evaluation metrics for community detection methods."""

from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Int
from sklearn.metrics import adjusted_rand_score

from data import GraphData
from methods.base import BaseMethod


def compute_ari(
    data: GraphData,
    pred_labels: Int[torch.Tensor, "n_nodes"],
) -> float:
    """
    Compute Adjusted Rand Index on data.val_idx nodes.

    Parameters
    ----------
    data : GraphData
        Provides ground-truth labels and valid_idx split.
    pred_labels : Int[torch.Tensor, "n_nodes"]
        Predicted community labels for all nodes in the graph.

    Returns
    -------
    float
        ARI in [-1, 1]; 0 = chance, 1 = perfect agreement.
    """

    true = data.labels[data.val_idx].cpu().numpy()
    pred = pred_labels[data.val_idx].cpu().numpy()
    return float(adjusted_rand_score(true, pred))



def compute_relative_ari(*, ari: float, baseline_ari: float) -> float:
    """
    Compute relative ARI = ARI at noise level x / ARI at noise level 0.

    Parameters
    ----------
    ari : float
        ARI at noise level x.
    baseline_ari : float
        ARI at noise level 0 (clean graph). Must be non-zero.

    Returns
    -------
    float
        Ratio in (-inf, 1]; 1 means no degradation relative to baseline.

    Raises
    ------
    ZeroDivisionError
        If baseline_ari is zero.
    """
    if baseline_ari == 0.0:
        raise ZeroDivisionError(
            "baseline_ari is 0; relative ARI is undefined. "
            "The method achieved chance-level performance on the clean graph."
        )
    return ari / baseline_ari


def count_parameters(model: BaseMethod) -> int:
    """
    Count the number of learnable parameters in a fitted community detection model.

    Dispatches on model type:
    - torch.nn.Module subclasses (GCN, GAT, SGC): sums all requires_grad parameters.
    - sklearn-based models (SpectralMethod with LR): counts elements in coef_ and
      intercept_ arrays if present; returns 0 for parameter-free methods (e.g. LP).

    Parameters
    ----------
    model : BaseMethod
        A fitted method instance from METHOD_REGISTRY.

    Returns
    -------
    int
        Total number of learnable scalar parameters.
    """
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # sklearn-based path: inspect fitted coefficient arrays
    n_params = 0
    for attr in ("coef_", "intercept_"):
        arr = getattr(model, attr, None)
        if arr is not None:
            n_params += int(np.asarray(arr).size)
    return n_params
