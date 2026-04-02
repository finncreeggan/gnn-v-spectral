"""Usage example: instantiate all methods and run one end-to-end."""

from __future__ import annotations

from data import load_graph_data
from methods.registry import METHOD_REGISTRY, ExperimentConfig

# ---------------------------------------------------------------------------
# Load a real graph from the synthetic cache
# ---------------------------------------------------------------------------
data = load_graph_data(
    metadata_csv="data/cache/synthetic/metadata/graph_index_sbm.csv",
    graph_id="graph001_000_clean_sbm",
)

# ---------------------------------------------------------------------------
# Config — all fields populated so every method can be instantiated
# ---------------------------------------------------------------------------
config = ExperimentConfig(
    num_classes=data.num_classes,
    seed=0,
    hidden_dim=32,
    num_layers=2,
    lr=1e-3,
    epochs=50,
    dropout=0.0,
    num_heads=2,
    k_hops=2,
    n_estimators=100, # For Random Forest classifier
)

# ---------------------------------------------------------------------------
# Instantiate all 9 methods from the registry
# ---------------------------------------------------------------------------
methods = {name: ctor(config) for name, ctor in METHOD_REGISTRY.items()}
print("Instantiated methods:", list(methods.keys()))

# ---------------------------------------------------------------------------
# Run fit + score on "whole_lr" (spectral; GNN stubs raise NotImplementedError)
# ---------------------------------------------------------------------------
method = methods["gat"]
method.fit(data, embeddings=data.kcut_eigenspectrum)

val_score  = method.score(data)
test_score = method.score(data, use_test_idx=True)

print("val  score:", val_score)
print("test score:", test_score)
