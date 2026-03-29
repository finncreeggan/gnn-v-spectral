"""Graph Attention Network (GAT) for transductive community detection."""

from __future__ import annotations

from typing import Self

from data import GraphData
from methods.base import BaseMethod, ExperimentConfig

"""
class GNNEncoder(torch.nn.Module):
    def __init__(self, args: Config, in_channels: int, out_channels: int):
        super().__init__()
        h = args.hidden_channels
        conv_cls = GATv2Conv if args.conv == "gatv2" else GCNConv
        self.conv1 = conv_cls(in_channels, h)
        self.conv2 = conv_cls(h, out_channels)

    def forward(self,
                x: Float[torch.Tensor, "n_nodes in_channels"],
                edge_index: Int[torch.Tensor, "2 n_edges"],
                ) -> Float[torch.Tensor, "n_nodes out_channels"]:
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
"""


class GAT(BaseMethod):
    """
    Multi-head GAT for transductive node classification / community detection.

    Relevant config fields: hidden_dim, num_layers, lr, epochs, dropout, num_heads.

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
        Run the GAT training loop for config.epochs steps on data.train_idx nodes.

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
        return {}