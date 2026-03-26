# Random and targeted edge deletion perturbations
from __future__ import annotations

from typing import Any, Iterable

import networkx as nx
import numpy as np


Edge = tuple[int, int]


def _validate_graph(G: nx.Graph) -> None:
    expected_nodes = list(range(G.number_of_nodes()))
    actual_nodes = sorted(G.nodes())
    if actual_nodes != expected_nodes:
        raise ValueError(
            "Graph nodes must be contiguous integers 0..n-1 for perturbation utilities."
        )


def _validate_noise_frac(noise_frac: float) -> None:
    if not (0.0 <= noise_frac <= 1.0):
        raise ValueError(f"noise_frac must be in [0, 1], got {noise_frac}.")


def _canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def _sorted_canonical_edges(G: nx.Graph) -> list[Edge]:
    return sorted(_canonical_edge(int(u), int(v)) for u, v in G.edges())


def _num_edges_to_remove(G: nx.Graph, noise_frac: float) -> int:
    _validate_noise_frac(noise_frac)
    m = G.number_of_edges()
    return min(int(round(noise_frac * m)), m)


def get_random_deletion_order(G: nx.Graph, seed: int) -> list[Edge]:
    """
    Create one fixed random deletion order for the clean parent graph.

    This order is then reused across all random-noise levels so that
    increasing noise levels form a true perturbation chain.
    """
    _validate_graph(G)

    edges = _sorted_canonical_edges(G)
    rng = np.random.default_rng(seed)
    rng.shuffle(edges)
    return edges


def get_targeted_betweenness_deletion_order(G: nx.Graph) -> list[Edge]:
    """
    Create one fixed targeted deletion order based on node betweenness centrality.

    Strategy
    --------
    1. Compute node betweenness centrality on the clean parent graph.
    2. Sort nodes by decreasing centrality.
    3. Visit edges incident to high-betweenness nodes first.
    4. Within a node's neighborhood, prioritize neighbors with higher
       betweenness centrality.

    This gives a deterministic, monotonic deletion order tied to the clean parent.
    """
    _validate_graph(G)

    centrality = nx.betweenness_centrality(G)
    nodes_sorted = sorted(G.nodes(), key=lambda n: (-centrality[n], n))

    seen: set[Edge] = set()
    ordered_edges: list[Edge] = []

    for node in nodes_sorted:
        neighbors_sorted = sorted(
            G.neighbors(node),
            key=lambda nbr: (-centrality[nbr], nbr),
        )

        for nbr in neighbors_sorted:
            edge = _canonical_edge(int(node), int(nbr))
            if edge not in seen:
                seen.add(edge)
                ordered_edges.append(edge)

    if len(ordered_edges) != G.number_of_edges():
        raise RuntimeError(
            "Targeted deletion order does not cover all edges in the graph."
        )

    return ordered_edges


def apply_deletion_order(
    G: nx.Graph,
    deletion_order: list[Edge],
    noise_frac: float,
) -> tuple[nx.Graph, dict[str, Any]]:
    """
    Remove the first k edges from a fixed deletion order, where
    k = round(noise_frac * number_of_edges_in_clean_parent).
    """
    _validate_graph(G)
    _validate_noise_frac(noise_frac)

    clean_edges = set(_sorted_canonical_edges(G))
    if set(deletion_order) != clean_edges:
        raise ValueError(
            "deletion_order must be a permutation of the clean parent graph's edge set."
        )

    m_clean = G.number_of_edges()
    num_remove = _num_edges_to_remove(G, noise_frac)
    edges_to_remove = deletion_order[:num_remove]

    G_perturbed = G.copy()
    G_perturbed.remove_edges_from(edges_to_remove)

    metadata = {
        "num_edges_original": m_clean,
        "num_edges_removed": num_remove,
        "num_edges_remaining": G_perturbed.number_of_edges(),
        "removed_edge_fraction": (num_remove / m_clean) if m_clean > 0 else 0.0,
    }

    return G_perturbed, metadata


def apply_random_edge_deletion(
    G: nx.Graph,
    noise_frac: float,
    seed: int,
) -> tuple[nx.Graph, dict[str, Any]]:
    """
    Apply random edge deletion at one noise level.
    """
    deletion_order = get_random_deletion_order(G, seed=seed)
    G_perturbed, metadata = apply_deletion_order(G, deletion_order, noise_frac)
    metadata["perturbation_seed"] = seed
    metadata["noise_type"] = "random"
    return G_perturbed, metadata


def apply_targeted_betweenness_deletion(
    G: nx.Graph,
    noise_frac: float,
) -> tuple[nx.Graph, dict[str, Any]]:
    """
    Apply targeted edge deletion based on node betweenness centrality at one noise level.
    """
    deletion_order = get_targeted_betweenness_deletion_order(G)
    G_perturbed, metadata = apply_deletion_order(G, deletion_order, noise_frac)
    metadata["noise_type"] = "targeted_betweenness"
    return G_perturbed, metadata


def build_noise_chain(
    G_clean: nx.Graph,
    noise_type: str,
    noise_fracs: Iterable[float],
    seed: int | None = None,
) -> list[tuple[float, nx.Graph, dict[str, Any]]]:
    """
    Build a monotonic perturbation chain from one clean parent graph.

    Returns a list of:
        (noise_frac, perturbed_graph, perturbation_metadata)

    Notes
    -----
    - Uses one fixed deletion order per perturbation type.
    - All returned graphs are descendants of the same clean parent.
    - The clean graph itself is typically handled outside this function.
    """
    _validate_graph(G_clean)

    unique_noise_fracs = sorted(set(noise_fracs))
    for noise_frac in unique_noise_fracs:
        _validate_noise_frac(noise_frac)

    if noise_type == "random":
        if seed is None:
            raise ValueError("seed must be provided for random perturbation chains.")
        deletion_order = get_random_deletion_order(G_clean, seed=seed)

    elif noise_type == "targeted_betweenness":
        deletion_order = get_targeted_betweenness_deletion_order(G_clean)

    else:
        raise ValueError(
            f"Unsupported noise_type '{noise_type}'. "
            "Expected 'random' or 'targeted_betweenness'."
        )

    chain: list[tuple[float, nx.Graph, dict[str, Any]]] = []
    for noise_frac in unique_noise_fracs:
        G_perturbed, metadata = apply_deletion_order(
            G_clean,
            deletion_order,
            noise_frac,
        )
        metadata["noise_type"] = noise_type
        if seed is not None:
            metadata["perturbation_seed"] = seed

        chain.append((noise_frac, G_perturbed, metadata))

    return chain


__all__ = [
    "get_random_deletion_order",
    "get_targeted_betweenness_deletion_order",
    "apply_deletion_order",
    "apply_random_edge_deletion",
    "apply_targeted_betweenness_deletion",
    "build_noise_chain",
]