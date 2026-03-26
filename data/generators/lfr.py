# LFR benchmark graph generation
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class LFRConfig:
    """
    Configuration for generating one clean LFR benchmark graph.

    Notes
    -----
    - tau1 controls the power-law exponent for the degree distribution.
    - tau2 controls the power-law exponent for community sizes.
    - mu is the mixing parameter: roughly the fraction of each node's edges
      that go outside its planted community.
    """

    n: int = 1000
    tau1: float = 3.0
    tau2: float = 1.5
    mu: float = 0.1  # Low mixing for a clear but nontrivial planted structure
    average_degree: int | None = 22  # Empirically calibrated to yield realized average degree ~25
    min_degree: int | None = None
    max_degree: int | None = None
    min_community: int = 50
    max_community: int = 300
    tol: float = 1.0e-7
    max_iters: int = 500
    ensure_connected: bool = False
    max_attempts: int = 25


def _validate_config(config: LFRConfig) -> None:
    if config.n <= 0:
        raise ValueError("n must be a positive integer.")

    if config.tau1 <= 1.0:
        raise ValueError("tau1 must be strictly greater than 1.")

    if config.tau2 <= 1.0:
        raise ValueError("tau2 must be strictly greater than 1.")

    if not (0.0 <= config.mu <= 1.0):
        raise ValueError("mu must be in [0, 1].")

    if (config.average_degree is None) == (config.min_degree is None):
        raise ValueError(
            "Exactly one of average_degree and min_degree must be specified."
        )

    if config.average_degree is not None and config.average_degree <= 0:
        raise ValueError("average_degree must be positive when provided.")

    if config.min_degree is not None and config.min_degree <= 0:
        raise ValueError("min_degree must be positive when provided.")

    if config.max_degree is not None and config.max_degree <= 0:
        raise ValueError("max_degree must be positive when provided.")

    if config.min_community <= 0:
        raise ValueError("min_community must be positive.")

    if config.max_community <= 0:
        raise ValueError("max_community must be positive.")

    if config.min_community > config.max_community:
        raise ValueError("min_community cannot exceed max_community.")

    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")


def _relabel_to_contiguous_ints(G: nx.Graph) -> nx.Graph:
    """
    Ensure node ids are exactly 0..n-1 in sorted order.
    This makes label alignment and downstream saving safer.
    """
    mapping = {node: new_id for new_id, node in enumerate(sorted(G.nodes()))}
    return nx.relabel_nodes(G, mapping, copy=True)


def _extract_partition_and_labels(G: nx.Graph) -> tuple[np.ndarray, list[int]]:
    """
    Extract a single planted community label for each node from NetworkX's
    LFR node attribute representation.

    Returns
    -------
    labels : np.ndarray
        1D integer label array aligned to node ids 0..n-1.
    community_sizes : list[int]
        Sizes of planted communities in label order.
    """
    node_to_commset: dict[int, frozenset[int]] = {}

    for node, data in G.nodes(data=True):
        if "community" not in data:
            raise RuntimeError(f"Node {node} is missing the 'community' attribute.")

        raw_community = data["community"]

        if not isinstance(raw_community, (set, frozenset)):
            raise RuntimeError(
                f"Unexpected community attribute type for node {node}: "
                f"{type(raw_community)}"
            )

        # Standard NetworkX LFR returns one community set per node.
        # If this ever turns out to be nested / overlapping, fail loudly so we
        # do not silently create invalid single-label targets.
        if len(raw_community) == 0:
            raise RuntimeError(f"Node {node} has an empty community attribute.")

        sample_member = next(iter(raw_community))
        if isinstance(sample_member, (set, frozenset)):
            raise RuntimeError(
                "Detected overlapping or nested community assignments, but this "
                "benchmark expects exactly one planted label per node."
            )

        community = frozenset(int(member) for member in raw_community)
        node_to_commset[int(node)] = community

    unique_communities = sorted(
        set(node_to_commset.values()),
        key=lambda comm: (min(comm), len(comm)),
    )

    community_to_label = {
        community: label for label, community in enumerate(unique_communities)
    }

    labels = np.empty(G.number_of_nodes(), dtype=np.int64)
    for node in range(G.number_of_nodes()):
        if node not in node_to_commset:
            raise RuntimeError(f"Missing community assignment for node {node}.")
        labels[node] = community_to_label[node_to_commset[node]]

    community_sizes = [len(comm) for comm in unique_communities]

    # Sanity checks: every node should belong to exactly one community and the
    # community sizes should sum to n.
    if labels.shape[0] != G.number_of_nodes():
        raise RuntimeError("Label array length does not match node count.")

    if sum(community_sizes) != G.number_of_nodes():
        raise RuntimeError(
            "Extracted community sizes do not sum to the number of nodes."
        )

    return labels, community_sizes


def _generate_single_lfr(config: LFRConfig, seed: int) -> nx.Graph:
    """
    Generate one LFR graph instance.
    """
    G = nx.LFR_benchmark_graph(
        n=config.n,
        tau1=config.tau1,
        tau2=config.tau2,
        mu=config.mu,
        average_degree=config.average_degree,
        min_degree=config.min_degree,
        max_degree=config.max_degree,
        min_community=config.min_community,
        max_community=config.max_community,
        tol=config.tol,
        max_iters=config.max_iters,
        seed=seed,
    )
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = _relabel_to_contiguous_ints(G)
    return G


def _generate_lfr_with_retries(config: LFRConfig, seed: int) -> tuple[nx.Graph, int]:
    """
    LFR generation can fail for some parameter/seed combinations, so retry
    across nearby seeds.
    """
    last_error: Exception | None = None

    for offset in range(config.max_attempts):
        current_seed = seed + offset
        try:
            G = _generate_single_lfr(config, current_seed)
            if config.ensure_connected and not nx.is_connected(G):
                continue
            return G, current_seed
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to generate a valid LFR graph after {config.max_attempts} attempts "
        f"starting from seed={seed}."
    ) from last_error


def generate_lfr(config: LFRConfig, seed: int) -> tuple[nx.Graph, np.ndarray, dict[str, Any]]:
    """
    Generate one clean LFR graph, its planted labels, and metadata.

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

    G, actual_seed = _generate_lfr_with_retries(config, seed)
    labels, community_sizes = _extract_partition_and_labels(G)

    metadata: dict[str, Any] = {
        "family": "lfr",
        "seed": actual_seed,
        "n_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_communities": len(community_sizes),
        "community_sizes": community_sizes,
        "tau1": config.tau1,
        "tau2": config.tau2,
        "mu": config.mu,
        "average_degree": config.average_degree,
        "min_degree": config.min_degree,
        "max_degree": config.max_degree,
        "min_community": config.min_community,
        "max_community": config.max_community,
        "tol": config.tol,
        "max_iters": config.max_iters,
        "ensure_connected": config.ensure_connected,
        "config": asdict(config),
    }

    return G, labels, metadata


__all__ = ["LFRConfig", "generate_lfr"]