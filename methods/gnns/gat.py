"""Graph Attention Network (GAT) for transductive community detection."""

from __future__ import annotations

from typing import Self

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from sklearn.metrics import adjusted_rand_score
from torch_geometric.nn import GATv2Conv

from data import GraphData
from methods.base import BaseMethod, ExperimentConfig


class _GATModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        if num_layers == 1:
            self.convs = torch.nn.ModuleList(
                [GATv2Conv(in_channels, out_channels, heads=1, concat=False, dropout=dropout)]
            )
        else:
            self.convs = torch.nn.ModuleList(
                [GATv2Conv(in_channels, hidden_dim, heads=num_heads, concat=True, dropout=dropout)]
                + [
                    GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
                    for _ in range(num_layers - 2)
                ]
                + [GATv2Conv(hidden_dim * num_heads, out_channels, heads=1, concat=False, dropout=dropout)]
            )

    def forward(
        self,
        x: Float[torch.Tensor, "n_nodes in_channels"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
    ) -> Float[torch.Tensor, "n_nodes out_channels"]:
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GAT(BaseMethod):
    """
    Multi-head GAT for transductive node classification / community detection.

    Relevant config fields: hidden_dim, num_layers, lr, epochs, dropout, num_heads.

    Parameters
    ----------
    config : ExperimentConfig
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self._model: _GATModule | None = None

    def fit(
        self,
        data: GraphData,
        *,
        study_name: str | None = None,
        optuna_storage_path: str | None = None,
        **kwargs,
    ) -> Self:
        """
        Run the GAT training loop for config.epochs steps on data.train_idx nodes.

        Message passing uses the full graph; loss is computed only on train_idx.
        Uses config.num_heads attention heads per layer.

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
        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = data.features.to(device)
        edge_index = data.graph.edge_index.to(device)
        labels = data.labels.to(device)
        train_idx = data.train_idx.to(device)

        in_channels = x.size(1)
        self._model = _GATModule(
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            out_channels=cfg.num_classes,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg.lr)

        self._model.train()
        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            logits = self._model(x, edge_index)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

        self._model.eval()
        return self

    def score(
        self,
        data: GraphData,
        *,
        use_test_idx: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate GAT predictions on data.val_idx (or data.test_idx) nodes.

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
        device = next(self._model.parameters()).device
        x = data.features.to(device)
        edge_index = data.graph.edge_index.to(device)
        idx = data.test_idx if use_test_idx else data.val_idx

        with torch.no_grad():
            logits = self._model(x, edge_index)

        preds = logits.argmax(dim=-1).cpu()
        labels = data.labels.cpu()
        ari = adjusted_rand_score(labels[idx].numpy(), preds[idx].numpy())
        return {"ARI": float(ari)}
