import numpy as np

from data.generators.sbm import SBMConfig, generate_sbm


def test_generate_sbm_basic():
    config = SBMConfig()
    G, labels, metadata = generate_sbm(config, seed=0)

    assert G.number_of_nodes() == 1000
    assert len(labels) == 1000
    assert metadata["family"] == "sbm"
    assert metadata["num_communities"] == 5
    assert metadata["p_in"] > metadata["p_out"]

    assert min(G.nodes()) == 0
    assert max(G.nodes()) == 999

    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64

    unique_labels = np.unique(labels)
    assert len(unique_labels) == 5
    assert set(unique_labels) == {0, 1, 2, 3, 4}

from data.generators.lfr import LFRConfig, generate_lfr


def test_generate_lfr_basic():
    config = LFRConfig()
    G, labels, metadata = generate_lfr(config, seed=0)

    assert G.number_of_nodes() == 1000
    assert len(labels) == 1000
    assert metadata["family"] == "lfr"
    assert metadata["num_communities"] >= 2

    assert min(G.nodes()) == 0
    assert max(G.nodes()) == 999

    assert sum(metadata["community_sizes"]) == 1000