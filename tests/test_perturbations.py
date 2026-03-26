# Tests for data/generators/perturbations.py
from data.generators.perturbations import (
    apply_random_edge_deletion,
    apply_targeted_betweenness_deletion,
    build_noise_chain,
)
from data.generators.sbm import SBMConfig, generate_sbm


def edge_set(G):
    return {tuple(sorted((u, v))) for u, v in G.edges()}


def test_random_edge_deletion_removes_expected_number_of_edges():
    G, labels, _ = generate_sbm(SBMConfig(), seed=0)
    noise_frac = 0.10

    G_noisy, metadata = apply_random_edge_deletion(G, noise_frac=noise_frac, seed=123)

    expected_removed = round(noise_frac * G.number_of_edges())
    assert metadata["num_edges_removed"] == expected_removed
    assert G_noisy.number_of_edges() == G.number_of_edges() - expected_removed


def test_targeted_betweenness_deletion_removes_expected_number_of_edges():
    G, labels, _ = generate_sbm(SBMConfig(), seed=0)
    noise_frac = 0.10

    G_noisy, metadata = apply_targeted_betweenness_deletion(G, noise_frac=noise_frac)

    expected_removed = round(noise_frac * G.number_of_edges())
    assert metadata["num_edges_removed"] == expected_removed
    assert G_noisy.number_of_edges() == G.number_of_edges() - expected_removed


def test_random_noise_chain_is_monotonic():
    G, labels, _ = generate_sbm(SBMConfig(), seed=0)

    chain = build_noise_chain(
        G_clean=G,
        noise_type="random",
        noise_fracs=[0.05, 0.10, 0.20],
        seed=123,
    )

    edge_sets = [edge_set(Gp) for _, Gp, _ in chain]

    assert edge_sets[2].issubset(edge_sets[1])
    assert edge_sets[1].issubset(edge_sets[0])
    assert edge_sets[0].issubset(edge_set(G))


def test_targeted_noise_chain_is_monotonic():
    G, labels, _ = generate_sbm(SBMConfig(), seed=0)

    chain = build_noise_chain(
        G_clean=G,
        noise_type="targeted_betweenness",
        noise_fracs=[0.05, 0.10, 0.20],
    )

    edge_sets = [edge_set(Gp) for _, Gp, _ in chain]

    assert edge_sets[2].issubset(edge_sets[1])
    assert edge_sets[1].issubset(edge_sets[0])
    assert edge_sets[0].issubset(edge_set(G))