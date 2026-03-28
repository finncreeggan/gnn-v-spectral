# Spectral embeddings: whole, k-cut and regularized eigenspectrum

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch_geometric.utils import degree, get_laplacian, to_scipy_sparse_matrix


def _normalized_laplacian(
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
) -> sp.spmatrix:
    """Symmetric normalized Laplacian as a scipy sparse matrix."""
    edge_index_L, edge_weight_L = get_laplacian(
        edge_index, normalization="sym", num_nodes=num_nodes
    )
    return to_scipy_sparse_matrix(edge_index_L, edge_weight_L, num_nodes=num_nodes)


def whole_eigenspectrum(
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
) -> tuple[Float[Tensor, "num_nodes num_nodes"], Float[Tensor, "num_nodes"]]:
    """
    Full eigendecomposition of the symmetric normalized Laplacian.

    Eigenvectors are sorted by ascending eigenvalue.

    Parameters
    ----------
    edge_index : Int[Tensor, "2 num_edges"]
    num_nodes : int

    Returns
    -------
    Float[Tensor, "num_nodes num_nodes"]
        Matrix whose columns are eigenvectors.
    """
    L = _normalized_laplacian(edge_index, num_nodes)
    eigenvals, V = eigh(L.toarray(), eigvals_only=False)
    return torch.from_numpy(V).float(), torch.from_numpy(eigenvals).float()


def kcut_eigenspectrum(
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
    sigma: float = 1e-5,
    *,
    n_eigenvectors: int,
) -> Float[Tensor, "num_nodes k"]:
    # TODO: FIX THIS!!! NOT CORRECT
    """
    Bottom-k eigenvectors of the symmetric normalized Laplacian, where k is
    selected automatically via the eigengap heuristic.

    Computes the ``n_eigenvectors`` smallest eigenvectors, then sets k to the
    index of the largest consecutive gap in the sorted eigenvalue sequence.

    Parameters
    ----------
    edge_index : Int[Tensor, "2 num_edges"]
    num_nodes : int
    n_eigenvectors : int
        Candidate pool size; k will satisfy 1 <= k <= n_eigenvectors.

    Returns
    -------
    Float[Tensor, "num_nodes k"]
    """
    L = _normalized_laplacian(edge_index, num_nodes)
    eigenvalues, V = eigsh(L, k=n_eigenvectors, which='LM', sigma=sigma, return_eigenvectors=True)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    V = V[:, order]
    k = int(np.argmax(np.diff(eigenvalues))) + 1
    return torch.from_numpy(V[:, :k]).float()


def regularized_eigenspectrum(
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
    *,
    n_eigenvectors: int,
    sigma: float = 1e-5,
) -> tuple[Float[Tensor, "num_nodes n_eigenvectors_plus_1"], Float[Tensor, "n_eigenvectors_plus_1"]]:
    """
    Bottom-(n_eigenvectors+1) eigenvectors of the Tikhonov-regularized
    symmetric normalized Laplacian.

    Regularization: ``L_tik = L + tau * I`` where ``tau = mean(degree)``.

    Parameters
    ----------
    edge_index : Int[Tensor, "2 num_edges"]
    num_nodes : int
    n_eigenvectors : int
        k in the formula ``eigsh(L_tik, k=n_eigenvectors+1)``.

    Returns
    -------
    Float[Tensor, "num_nodes n_eigenvectors_plus_1"]
    """
    L = _normalized_laplacian(edge_index, num_nodes)
    d = degree(edge_index[0], num_nodes=num_nodes).numpy() # TODO: Double check
    tau = float(np.mean(d))
    L_tik = L + tau * sp.eye(num_nodes)
    eigenvals, V = eigh(L_tik.toarray(), eigvals_only=False)
    return torch.from_numpy(V).float(), torch.from_numpy(eigenvals).float()
