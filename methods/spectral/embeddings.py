# Spectral embeddings: whole, k-cut and regularized eigenspectrum

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

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
    eigenvals, V = eigh(L.toarray(), eigvals_only=False) #type: ignore
    return torch.from_numpy(V).float(), torch.from_numpy(eigenvals).float()


def kcut_eigenspectrum(
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
    *,
    all_V: Float[Tensor, "num_nodes num_nodes"] | None = None,
    all_eigenvalues: Float[Tensor, "num_nodes"] | None = None,
) -> tuple[Float[Tensor, "num_nodes k"], Float[Tensor, "k"]]:
    
    # TODO: Read more into this!
    """
    Bottom-k eigenvectors of the symmetric normalized Laplacian, where k is
    selected automatically via the eigengap heuristic on the full spectrum.

    Finds the largest gap between consecutive eigenvalues (sorted ascending)
    at position k, then returns the first k eigenvectors.

    Parameters
    ----------
    edge_index : Int[Tensor, "2 num_edges"]
    num_nodes : int
    all_V : Float[Tensor, "num_nodes num_nodes"], optional
        Precomputed eigenvector matrix (columns sorted by ascending eigenvalue).
        If None, ``whole_eigenspectrum`` is called.
    all_eigenvalues : Float[Tensor, "num_nodes"], optional
        Precomputed eigenvalues sorted ascending. If None, ``whole_eigenspectrum``
        is called.

    Returns
    -------
    Float[Tensor, "num_nodes k"]
    Float[Tensor, "k"]
    """
    if all_V is None or all_eigenvalues is None:
        all_V, all_eigenvalues = whole_eigenspectrum(edge_index, num_nodes)

    eigenvalues_np = all_eigenvalues.numpy()
    k = int(np.argmax(np.diff(eigenvalues_np))) + 1
    return all_V[:, :k], all_eigenvalues[:k]


def regularized_eigenspectrum(
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
    L: sp.spmatrix | None = None,
) -> tuple[Float[Tensor, "num_nodes num_nodes"], Float[Tensor, "num_nodes"]]:
    """
    Full eigendecomposition of the Tikhonov-regularized symmetric normalized
    Laplacian.

    Regularization: ``L_tik = L + tau * I`` where ``tau = mean(degree)``.

    Parameters
    ----------
    edge_index : Int[Tensor, "2 num_edges"]
    num_nodes : int

    Returns
    -------
    Float[Tensor, "num_nodes num_nodes"]
        Matrix whose columns are eigenvectors.
    """
    if L is None:
        L = _normalized_laplacian(edge_index, num_nodes)
        
    d = degree(edge_index[0], num_nodes=num_nodes).numpy() # TODO: Double check
    tau = float(np.mean(d))
    L_tik = L + tau * sp.eye(num_nodes)
    eigenvals, V = eigh(L_tik.toarray(), eigvals_only=False)
    return torch.from_numpy(V).float(), torch.from_numpy(eigenvals).float()
