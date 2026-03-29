"""Simple Graph Convolution (SGC) for transductive community detection."""

from __future__ import annotations

from typing import Self

from data import GraphData
from methods.base import BaseMethod, ExperimentConfig


class SGC(BaseMethod):
    """
    SGC (Wu et al., 2019) for transductive node classification / community detection.

    Collapses k graph-convolution steps into a single pre-computed feature
    propagation, then fits a linear classifier. Acts as a middle-ground between
    spectral and GNN-based methods.

    Relevant config fields: hidden_dim, num_layers, lr, epochs, dropout, k_hops.

    Parameters
    ----------
    config : ExperimentConfig
    """

    def __init__(self, config: ExperimentConfig) -> None:
       pass

    def fit(
        self,
        data: GraphData,
        *,
        study_name: str | None = None,
        optuna_storage_path: str | None = None,
    ) -> Self:
        """
        Pre-propagate features for config.k_hops steps; train linear classifier
        on data.train_idx nodes for config.epochs steps.

        Parameters
        ----------
        data : GraphData
        study_name : str | None
            Optuna study name; passed through to hyperparameter search if used.
        optuna_storage_path : str | None
            Path to Optuna storage backend; passed through if used.

        Returns
        -------
        Self
        """
        return self

    def score(
        self,
        data: GraphData,
        *,
        use_test_idx: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate SGC predictions on data.val_idx (or data.test_idx) nodes.

        ARI computed via sklearn.metrics.
        relative_ARI is float("nan"); filled in at pipeline level.

        Parameters
        ----------
        data : GraphData
        use_test_idx : bool
            If True, evaluate on data.test_idx instead of data.val_idx.

        Returns
        -------
        dict[str, float]
            Keys: "ARI", "relative_ARI".
        """
        return {}
