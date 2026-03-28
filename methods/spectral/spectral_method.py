"""Single class for all 6 spectral embedding × classifier combinations."""

from __future__ import annotations

from typing import Literal
from jaxtyping import Float, Int

import torch
from sklearn.metrics import adjusted_rand_score

from data import GraphData
from methods.base import BaseMethod, ExperimentConfig
from methods.spectral.classifiers import LPClassifier, LRClassifier, SpectralClassifier
from methods.spectral.embeddings import (
    kcut_eigenspectrum,
    regularized_eigenspectrum,
    whole_eigenspectrum,
)

def get_spectral_embeddings(embedding_type: Literal["whole", "kcut", "regularized"], 
                            edge_index: Int[torch.Tensor, "2 num_edges"], 
                            num_nodes: int,
                            ) -> tuple[Float[torch.Tensor, "num_nodes n_eigenvectors"], Float[torch.Tensor, "n_eigenvectors"]]:
    
    if embedding_type == "whole":
        return whole_eigenspectrum(edge_index, num_nodes)
    elif embedding_type == "kcut":
        return kcut_eigenspectrum(edge_index, num_nodes)
    elif embedding_type == "regularized":
        return regularized_eigenspectrum(edge_index, num_nodes)
    else:
        raise ValueError(f"Invalid embedding_type: {embedding_type!r}")


class SpectralMethod(BaseMethod):
    """
    Spectral community detection with configurable embedding and classifier.

    Covers all 6 spectral benchmark methods by composing an embedding type
    with a downstream classifier. The embedding is computed on the full graph;
    the classifier is fit on train_idx nodes only.

    Parameters
    ----------
    config : ExperimentConfig
        config.n_eigenvectors must be set (not None) when embedding_type="kcut"
        or embedding_type="regularized".
    embedding_type : {"whole", "kcut", "regularized"}
        Which Laplacian spectrum variant to use as node features:
          "whole"       — full eigenspectrum of the normalised Laplacian
          "kcut"        — bottom-k eigenvectors (spectral k-way cut)
          "regularized" — regularized Laplacian spectrum (tau-shift)
    classifier_type : {"lr", "lp"}
        "lr" — logistic regression fit on train_idx embeddings
        "lp" — label propagation seeded from train_idx ground-truth labels

    Notes
    -----
    embedding_type and classifier_type are keyword-only to prevent silent
    argument-order bugs.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        embeddings: Float[torch.Tensor, "n_nodes n_eigenvectors"] | None = None,
        embedding_type: Literal["whole", "kcut", "regularized"],
        classifier_type: Literal["lr", "lp"],
    ) -> None:
        super().__init__(config)
        if embedding_type in ("kcut", "regularized") and config.n_eigenvectors is None:
            raise ValueError(
                f"config.n_eigenvectors must be set for embedding_type={embedding_type!r}"
            )
        self.embedding_type = embedding_type
        self.classifier_type = classifier_type
        self.classifier: SpectralClassifier = (
            LRClassifier(seed=config.seed)
            if classifier_type == "lr"
            else LPClassifier()
        )
        self.embeddings = embeddings

    def fit(self, data: GraphData) -> SpectralClassifier:
        """
        Compute spectral embedding on the full graph; fit classifier on train_idx.

        For classifier_type="lr":
            Fits a logistic regression on the embeddings of data.train_idx nodes.
        For classifier_type="lp":
            Initialises label propagation with ground-truth labels at data.train_idx.

        Parameters
        ----------
        data : GraphData

        Returns
        -------
        SpectralClassifier
        """
        num_nodes = data.graph.num_nodes
        edge_index: Int[torch.Tensor, "2 num_edges"] = data.graph.edge_index
        assert num_nodes is not None, "GraphData.graph.num_nodes must be set."

        if self.embeddings is None:
            self.embeddings, _ = get_spectral_embeddings(
                self.embedding_type, edge_index, num_nodes, #type: ignore
            )

        self.classifier.fit(data, self.embeddings, data.graph.x)
        return self.classifier

    def score(self, data: GraphData) -> dict[str, float]:
        """
        Predict community labels for data.val_idx and compute metrics.

        ARI computed via sklearn.metrics against data.labels[val_idx].
        relative_ARI is float("nan"); filled in at pipeline level.

        Parameters
        ----------
        data : GraphData

        Returns
        -------
        dict[str, float]
            Keys: "ARI", "relative_ARI".
        """
        if self.embeddings is None:
            raise RuntimeError("SpectralMethod.fit() must be called before score().")

        preds = self.classifier.predict(self.embeddings, data.graph.x)
        true = data.labels[data.val_idx].numpy()
        pred = preds[data.val_idx].numpy()

        return {
            "ARI": float(adjusted_rand_score(true, pred)),
            "relative_ARI": float("nan"), # TODO
        }
