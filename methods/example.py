"""Usage example: instantiate all methods and run one end-to-end."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from data import GraphData
from methods.registry import METHOD_REGISTRY, ExperimentConfig

# ---------------------------------------------------------------------------
# Synthetic graph: 30 nodes, 3 balanced classes, random undirected edges
# ---------------------------------------------------------------------------
torch.manual_seed(0)

NUM_NODES = 30
NUM_CLASSES = 3

src = torch.randint(0, NUM_NODES, (90,))
dst = torch.randint(0, NUM_NODES, (90,))
edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

graph = Data(edge_index=edge_index, num_nodes=NUM_NODES)
graph.x = torch.eye(NUM_NODES)  # identity features (matches load_graph_data fallback)

labels = torch.repeat_interleave(torch.arange(NUM_CLASSES), NUM_NODES // NUM_CLASSES)

perm = torch.randperm(NUM_NODES)
train_idx = perm[:21]   # 70 %
val_idx   = perm[21:26] # 15 %
test_idx  = perm[26:]   # 15 %

data = GraphData(
    graph=graph,
    graph_id="example",
    noise_fraction=0.0,
    num_classes=NUM_CLASSES,
    labels=labels,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
)

# ---------------------------------------------------------------------------
# Config — all fields populated so every method can be instantiated
# ---------------------------------------------------------------------------
config = ExperimentConfig(
    num_classes=NUM_CLASSES,
    seed=0,
    hidden_dim=32,
    num_layers=2,
    lr=1e-3,
    epochs=10,
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
method = methods["whole_lr"]
method.fit(data)

val_score  = method.score(data)
test_score = method.score(data, use_test_idx=True)

print("val  score:", val_score)
print("test score:", test_score)
