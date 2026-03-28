# Classifiers: logistic regression, label propagation

from __future__ import annotations

import abc
from typing import Self

import torch
from jaxtyping import Float, Int
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torch_geometric.nn import LabelPropagation

from data import GraphData


class SpectralClassifier(abc.ABC):
    """
    Common interface for spectral downstream classifiers.

    Both classifiers are transductive: fit() sees the full graph and all
    embeddings, but only uses train_idx labels as supervision. predict()
    returns class assignments for every node.

    embeddings and features are concatenated before fitting/predicting.

    When SpectralMethod.fit() is called, it should pass data.graph.x as
    the features argument.
    """

    @abc.abstractmethod
    def fit(
        self,
        data: GraphData,
        embeddings: Float[Tensor, "n_nodes d_emb"],
        features: Float[Tensor, "n_nodes d_feat"],
    ) -> Self: ...

    @abc.abstractmethod
    def predict(
        self,
        embeddings: Float[Tensor, "n_nodes d_emb"],
        features: Float[Tensor, "n_nodes d_feat"],
    ) -> Int[Tensor, "n_nodes"]: ...


class LRClassifier(SpectralClassifier):
    """
    Logistic regression fit on train_idx embeddings, predict on all nodes.

    Parameters
    ----------
    seed : int
        Random seed passed to sklearn LogisticRegression.
    """

    def __init__(self, *, seed: int) -> None:
        self._lr = LogisticRegression(max_iter=1000, random_state=seed)

    def fit(
        self,
        data: GraphData,
        embeddings: Float[Tensor, "n_nodes d_emb"],
        features: Float[Tensor, "n_nodes d_feat"],
    ) -> Self:
        X = torch.cat([embeddings, features], dim=-1)
        X_train = X[data.train_idx].numpy()
        y_train = data.labels[data.train_idx].numpy()
        self._lr.fit(X_train, y_train)
        return self

    def predict(
        self,
        embeddings: Float[Tensor, "n_nodes d_emb"],
        features: Float[Tensor, "n_nodes d_feat"],
    ) -> Int[Tensor, "n_nodes"]:
        X = torch.cat([embeddings, features], dim=-1)
        preds = self._lr.predict(X.numpy())
        return torch.from_numpy(preds)


class RFClassifier(SpectralClassifier):
    """
    Random forest fit on train_idx embeddings, predict on all nodes.

    Parameters
    ----------
    seed : int
        Random seed passed to sklearn RandomForestClassifier.
    n_estimators : int
        Number of trees in the forest.
    """

    def __init__(self, *, seed: int, n_estimators: int) -> None:
        self._rf = _RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed
        )

    def fit(
        self,
        data: GraphData,
        embeddings: Float[Tensor, "n_nodes d_emb"],
        features: Float[Tensor, "n_nodes d_feat"],
    ) -> Self:
        X = torch.cat([embeddings, features], dim=-1)
        X_train = X[data.train_idx].numpy()
        y_train = data.labels[data.train_idx].numpy()
        self._rf.fit(X_train, y_train)
        return self

    def predict(
        self,
        embeddings: Float[Tensor, "n_nodes d_emb"],
        features: Float[Tensor, "n_nodes d_feat"],
    ) -> Int[Tensor, "n_nodes"]:
        X = torch.cat([embeddings, features], dim=-1)
        preds = self._rf.predict(X.numpy())
        return torch.from_numpy(preds)


class LPClassifier(SpectralClassifier):
    """
    Label propagation seeded from train_idx ground-truth labels.

    Propagates soft label distributions over the graph edges using
    torch_geometric's LabelPropagation (closed-form, no gradient needed).
    Embeddings are ignored; structure comes from data.graph.edge_index.

    Parameters
    ----------
    num_layers : int
        Number of propagation iterations.
    alpha : float
        Residual weight (keeps original seed labels; 0 < alpha < 1).
    """

    def __init__(self, *, num_layers: int = 50, alpha: float = 0.9) -> None:
        self._lp = LabelPropagation(num_layers=num_layers, alpha=alpha)
        self._predictions: Int[Tensor, "n_nodes"] | None = None

    def fit(
        self,
        data: GraphData,
        embeddings: Float[Tensor, "n_nodes d_emb"],
        features: Float[Tensor, "n_nodes d_feat"],
    ) -> Self:
        n_nodes = data.labels.shape[0]
        num_classes = data.num_classes

        # Build one-hot seed labels; unlabelled nodes get uniform distribution.
        y = torch.full((n_nodes, num_classes), 1.0 / num_classes)
        y[data.train_idx] = torch.nn.functional.one_hot(
            data.labels[data.train_idx], num_classes=num_classes
        ).float()

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[data.train_idx] = True

        edge_index = data.graph.edge_index
        out: Float[Tensor, "n_nodes num_classes"] = self._lp(
            y, edge_index, mask=train_mask
        )
        self._predictions = out.argmax(dim=-1)
        return self

    def predict(
        self,
        embeddings: Float[Tensor, "n_nodes d_emb"],
        features: Float[Tensor, "n_nodes d_feat"],
    ) -> Int[Tensor, "n_nodes"]:
        if self._predictions is None:
            raise RuntimeError("LPClassifier.fit() must be called before predict().")
        return self._predictions
