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
        raise NotImplementedError

    def fit(self, data: GraphData) -> Self:
        """
        Run the GAT training loop for config.epochs steps on data.train_idx nodes.

        Uses config.num_heads attention heads per layer.

        Parameters
        ----------
        data : GraphData

        Returns
        -------
        Self
        """
        raise NotImplementedError

    def score(self, data: GraphData) -> dict[str, float]:
        """
        Evaluate GAT predictions on data.valid_idx nodes.

        ARI and NMI computed via sklearn.metrics.
        relative_ARI is float("nan"); filled in at pipeline level.

        Parameters
        ----------
        data : GraphData

        Returns
        -------
        dict[str, float]
            Keys: "ARI", "NMI", "relative_ARI".
        """
        raise NotImplementedError
