"""Single class for all 6 spectral embedding × classifier combinations."""

from __future__ import annotations

from typing import Literal, Self
from jaxtyping import Float, Int

import torch
from sklearn.metrics import adjusted_rand_score

from data import GraphData
from methods.base import BaseMethod, ExperimentConfig
from methods.spectral.classifiers import LPClassifier, LRClassifier, RFClassifier, SpectralClassifier
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
    classifier_type : {"lr", "lp", "rf"}
        "lr" — logistic regression fit on train_idx embeddings
        "lp" — label propagation seeded from train_idx ground-truth labels
        "rf" — random forest fit on train_idx embeddings; requires config.n_estimators

    Notes
    -----
    embedding_type and classifier_type are keyword-only to prevent silent
    argument-order bugs.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        embedding_type: Literal["whole", "kcut", "regularized"],
        classifier_type: Literal["lr", "lp", "rf"],
    ) -> None:
        super().__init__(config)
        self.embedding_type = embedding_type
        self.classifier_type = classifier_type
        if classifier_type == "lr":
            self.classifier: SpectralClassifier = LRClassifier(seed=config.seed)
        elif classifier_type == "lp":
            self.classifier = LPClassifier()
        elif classifier_type == "rf":
            if config.n_estimators is None:
                raise ValueError("config.n_estimators must be set for classifier_type='rf'")
            self.classifier = RFClassifier(seed=config.seed, n_estimators=config.n_estimators)
        else:
            raise ValueError(f"Invalid classifier_type: {classifier_type!r}")
        self.embeddings: Float[torch.Tensor, "n_nodes n_eigenvectors"] | None = None

    def fit(
        self,
        data: GraphData,
        *,
        embeddings: Float[torch.Tensor, "n_nodes n_eigenvectors"] | None = None,
        study_name: str | None = None,
        optuna_storage_path: str | None = None,
    ) -> Self:
        """
        Fit classifier on train_idx using spectral embeddings.

        For classifier_type="lr":
            Fits a logistic regression on the embeddings of data.train_idx nodes.
        For classifier_type="lp":
            Initialises label propagation with ground-truth labels at data.train_idx.

        Parameters
        ----------
        data : GraphData
        embeddings : Float[Tensor, "n_nodes n_eigenvectors"] | None
            Precomputed spectral embeddings. If None, computed via get_spectral_embeddings.
        study_name : str | None
            Unused; present for API consistency with BaseMethod.
        optuna_storage_path : str | None
            Unused; present for API consistency with BaseMethod.

        Returns
        -------
        SpectralClassifier
        """
        num_nodes = data.graph.num_nodes
        edge_index: Int[torch.Tensor, "2 num_edges"] = data.graph.edge_index
        assert num_nodes is not None, "GraphData.graph.num_nodes must be set."

        if embeddings is None:
            embeddings, _ = get_spectral_embeddings(
                self.embedding_type, edge_index, num_nodes, #type: ignore
            )

        features = data.features
        assert features is not None, "GraphData.features must be set for SpectralMethod."
        
        self.classifier.fit(data, embeddings, features)
        self.embeddings = embeddings
        return self

    def score(
        self,
        data: GraphData,
        *,
        use_test_idx: bool = False,
    ) -> dict[str, float]:
        """
        Predict community labels for data.val_idx (or data.test_idx) and compute metrics.

        ARI computed via sklearn.metrics against data.labels[idx].
        Parameters
        ----------
        data : GraphData
        use_test_idx : bool
            If True, evaluate on data.test_idx instead of data.val_idx.

        Returns
        -------
        dict[str, float]
            Keys: "ARI".
        """
        if self.embeddings is None:
            raise RuntimeError("SpectralMethod.fit() must be called before score().")

        idx = data.test_idx if use_test_idx else data.val_idx
        features = data.graph.x if data.features is None else data.features
        preds = self.classifier.predict(self.embeddings, features)
        true = data.labels[idx].numpy()
        pred = preds[idx].numpy()

        return {"ARI": float(adjusted_rand_score(true, pred))}
