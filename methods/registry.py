"""Registry mapping method names to callables (class or partial).

All entries accept a single positional argument — config: ExperimentConfig —
and return a BaseMethod instance:

    method = METHOD_REGISTRY["kcut_lr"](config)
    method.fit(data).score(data)
"""

from __future__ import annotations

from functools import partial

from methods.base import ExperimentConfig  # noqa: F401  re-exported for convenience
from methods.gnns.gat import GAT
from methods.gnns.gcn import GCN
from methods.gnns.sgc import SGC
from methods.spectral.spectral_method import SpectralMethod

METHOD_REGISTRY: dict[str, type | partial] = {
    # Spectral: whole eigenspectrum
    "whole_lr":        partial(SpectralMethod, embedding_type="whole",        classifier_type="lr"),
    "whole_lp":        partial(SpectralMethod, embedding_type="whole",        classifier_type="lp"),
    "whole_rf":        partial(SpectralMethod, embedding_type="whole",        classifier_type="rf"),
    # Spectral: k-cut eigenspectrum
    "kcut_lr":         partial(SpectralMethod, embedding_type="kcut",         classifier_type="lr"),
    "kcut_lp":         partial(SpectralMethod, embedding_type="kcut",         classifier_type="lp"),
    "kcut_rf":         partial(SpectralMethod, embedding_type="kcut",         classifier_type="rf"),
    # Spectral: regularized eigenspectrum
    "regularized_lr":  partial(SpectralMethod, embedding_type="regularized",  classifier_type="lr"),
    "regularized_lp":  partial(SpectralMethod, embedding_type="regularized",  classifier_type="lp"),
    "regularized_rf":  partial(SpectralMethod, embedding_type="regularized",  classifier_type="rf"),
    # GNNs
    "sgc":             SGC,
    "gcn":             GCN,
    "gat":             GAT,
}
