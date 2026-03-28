"""Base types shared by all community detection methods."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Self

from data import GraphData


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Unified configuration for all 9 community detection methods.

    Fields with None defaults are method-specific; each method validates
    that its required fields are populated inside fit().

    Parameters
    ----------
    num_classes : int
        Number of ground-truth communities.
    seed : int
        Global random seed for reproducibility.
    n_eigenvectors : int | None
        Number of eigenvectors to retain; required for embedding_type="kcut".
    hidden_dim : int | None
        Hidden layer dimensionality for GNN methods.
    num_layers : int | None
        Number of message-passing layers for GNN methods.
    lr : float | None
        Learning rate for GNN training.
    epochs : int | None
        Number of training epochs for GNN methods.
    dropout : float | None
        Dropout probability for GNN methods.
    num_heads : int | None
        Number of attention heads; required for GAT.
    k_hops : int | None
        Number of propagation hops; required for SGC.
    n_estimators : int | None
        Number of trees in the random forest; required for classifier_type="rf".
    """

    num_classes: int
    seed: int

    # GNN shared
    hidden_dim: int | None = None
    num_layers: int | None = None
    lr: float | None = None
    epochs: int | None = None
    dropout: float | None = None
    # GAT-specific
    num_heads: int | None = None
    # SGC-specific
    k_hops: int | None = None
    # RF-specific
    n_estimators: int | None = None


class BaseMethod(abc.ABC):
    """
    Abstract base class for all transductive community detection methods.

    Subclasses implement fit() and score() only; __init__ stores config.

    Protocol
    --------
    fit(data)   — trains/fits using data.train_idx nodes only
    score(data) — evaluates on data.valid_idx nodes only

    score() returns a dict with three keys:
        "ARI"          : float — adjusted rand index (sklearn.metrics)
        "NMI"          : float — normalised mutual information (sklearn.metrics)
        "relative_ARI" : float — ARI / baseline_ARI at noise=0;
                                 always float("nan") here, filled at pipeline level
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def fit(
        self,
        data: GraphData,
        *,
        study_name: str | None = None,
        optuna_storage_path: str | None = None,
    ) -> Self:
        """
        Fit or train the method using data.train_idx nodes only.

        Parameters
        ----------
        data : GraphData
            Graph and associated split indices.
        study_name : str | None
            Optuna study name; passed through to hyperparameter search if used.
        optuna_storage_path : str | None
            Path to Optuna storage backend; passed through if used.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def score(
        self,
        data: GraphData,
        *,
        use_test_idx: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate predictions on data.val_idx (or data.test_idx) nodes.

        Parameters
        ----------
        data : GraphData
            Graph and associated split indices.
        use_test_idx : bool
            If True, evaluate on data.test_idx instead of data.val_idx.

        Returns
        -------
        dict[str, float]
            Keys: "ARI", "NMI", "relative_ARI".
            "relative_ARI" is float("nan"); computed at pipeline level.
        """
        raise NotImplementedError
