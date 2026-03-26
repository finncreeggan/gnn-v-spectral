# SBM (Stochastic Block Model) graph generation
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SBMConfig:
    """
    Configuration for generating one clean SBM graph.

    Parameters are chosen so the default graph is:
    - about 1000 nodes
    - 5 equal-sized communities
    - expected average degree around 25
    """

    community_sizes: tuple[int, ...] = (200, 200, 200, 200, 200)
    p_in: float = 0.08          # Moderate within-community density
    p_out: float = 0.011        # Set so expected average degree is ~25 for the default 5x200 setup
    self_loops: bool = False
    sparse: bool = True
    ensure_connected: bool = False
    max_attempts: int = 25

    @property
    def n_nodes(self) -> int:
        return sum(self.community_sizes)

    @property
    def num_communities(self) -> int:
        return len(self.community_sizes)

    @property
    def probability_matrix(self) -> list[list[float]]:
        """
        Build the block probability matrix expected by networkx.
        """
        k = self.num_communities
        probs: list[list[float]] = []
        for i in range(k):
            row = []
            for j in range(k):
                row.append(self.p_in if i == j else self.p_out)
            probs.append(row)
        return probs

    @property
    def expected_average_degree(self) -> float:
        """
        Approximate expected average degree for the SBM.

        For a node in community c:
            E[deg] = (size_c - 1) * p_in + sum_{d != c} size_d * p_out

        We average that over all nodes.
        """
        sizes = np.array(self.community_sizes, dtype=float)
        n = int(sizes.sum())
        expected_deg_sum = 0.0

        for i, size_i in enumerate(sizes):
            within = max(size_i - 1, 0) * self.p_in
            across = float(sizes.sum() - size_i) * self.p_out
            expected_deg_sum += size_i * (within + across)

        return expected_deg_sum / n if n > 0 else 0.0


def _validate_config(config: SBMConfig) -> None:
    if len(config.community_sizes) == 0:
        raise ValueError("community_sizes must contain at least one community.")

    if any(size <= 0 for size in config.community_sizes):
        raise ValueError("All community sizes must be positive integers.")

    if not (0.0 <= config.p_in <= 1.0):
        raise ValueError(f"p_in must be in [0, 1], got {config.p_in}.")

    if not (0.0 <= config.p_out <= 1.0):
        raise ValueError(f"p_out must be in [0, 1], got {config.p_out}.")

    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    if config.p_in <= config.p_out:
        raise ValueError(
            "For a community-detection benchmark, p_in should usually be greater than p_out."
        )


def _build_labels(community_sizes: tuple[int, ...]) -> np.ndarray:
    """
    Construct a planted community label array aligned with node ids 0..n-1.
    """
    labels = np.empty(sum(community_sizes), dtype=np.int64)
    start = 0
    for community_id, size in enumerate(community_sizes):
        end = start + size
        labels[start:end] = community_id
        start = end
    return labels


def _relabel_to_contiguous_ints(G: nx.Graph) -> nx.Graph:
    """
    Ensure node ids are exactly 0..n-1 in sorted order.
    This makes label alignment and downstream saving safer.
    """
    mapping = {node: new_id for new_id, node in enumerate(sorted(G.nodes()))}
    return nx.relabel_nodes(G, mapping, copy=True)


def _generate_single_sbm(config: SBMConfig, seed: int) -> nx.Graph:
    """
    Generate one SBM graph instance.
    """
    G = nx.stochastic_block_model(
        sizes=list(config.community_sizes),
        p=config.probability_matrix,
        seed=seed,
        selfloops=config.self_loops,
        sparse=config.sparse,
    )
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = _relabel_to_contiguous_ints(G)
    return G


def _generate_connected_sbm(config: SBMConfig, seed: int) -> tuple[nx.Graph, int]:
    """
    Try multiple seeds until we get a connected graph.
    Returns the graph and the actual seed used.
    """
    for offset in range(config.max_attempts):
        current_seed = seed + offset
        G = _generate_single_sbm(config, current_seed)
        if nx.is_connected(G):
            return G, current_seed

    raise RuntimeError(
        f"Failed to generate a connected SBM graph after {config.max_attempts} attempts "
        f"starting from seed={seed}."
    )


def generate_sbm(config: SBMConfig, seed: int) -> tuple[nx.Graph, np.ndarray, dict[str, Any]]:
    """
    Generate one clean SBM graph, its planted labels, and metadata.

    Returns
    -------
    G : nx.Graph
        Undirected graph with node ids 0..n-1.
    labels : np.ndarray
        1D integer array of planted community labels aligned to node ids.
    metadata : dict[str, Any]
        Family-level clean-graph metadata.
    """
    _validate_config(config)

    if config.ensure_connected:
        G, actual_seed = _generate_connected_sbm(config, seed)
    else:
        G = _generate_single_sbm(config, seed)
        actual_seed = seed

    labels = _build_labels(config.community_sizes)

    if G.number_of_nodes() != len(labels):
        raise RuntimeError(
            f"Label length {len(labels)} does not match node count {G.number_of_nodes()}."
        )

    metadata: dict[str, Any] = {
        "family": "sbm",
        "seed": actual_seed,
        "n_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_communities": config.num_communities,
        "community_sizes": list(config.community_sizes),
        "p_in": config.p_in,
        "p_out": config.p_out,
        "expected_average_degree": config.expected_average_degree,
        "self_loops": config.self_loops,
        "sparse": config.sparse,
        "ensure_connected": config.ensure_connected,
        "config": asdict(config),
    }

    return G, labels, metadata