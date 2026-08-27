import inspect
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from src.basis_functions.basis_functions import BaseBasisFunction


def _build_basis(
    basis_cls: Type[BaseBasisFunction],
    loc: float,
    scale: float,
    prior_samples: np.ndarray,
    basis_kwargs: Dict[str, Any],
    posterior_samples_for_centers: Optional[np.ndarray] = None,
) -> BaseBasisFunction:
    """
    Construct a basis function, passing only the keyword arguments its
    constructor actually accepts -- so the same calling code works both for
    fixed-centre bases (e.g. FixedCentersRBFBasisFunction, which wants
    loc/scale) and data-driven bases (e.g. MaternBasisFunction,
    RBFBasisFunction, ... which pick their own centres/lengthscale from
    posterior_samples/prior_samples via kmeans/farthest/halton).
    """
    prior_col = np.asarray(prior_samples, dtype=float).reshape(-1, 1)
    posterior_col = (
        np.asarray(posterior_samples_for_centers, dtype=float).reshape(-1, 1)
        if posterior_samples_for_centers is not None
        else prior_col
    )

    candidate_kwargs = dict(basis_kwargs)
    candidate_kwargs.setdefault("loc", loc)
    candidate_kwargs.setdefault("scale", scale)
    candidate_kwargs.setdefault("prior_samples", prior_col)
    candidate_kwargs.setdefault("posterior_samples", posterior_col)

    accepted = inspect.signature(basis_cls.__init__).parameters
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    return basis_cls(**kwargs)


def _generalized_eigvals_max_batch(
    A_all: np.ndarray,
    A_c: np.ndarray,
    nugget: float = 1e-10,
) -> np.ndarray:
    """
    Largest generalised eigenvalue of A_j v = omega * A_c v for every node j.
    """
    K = A_c.shape[0]
    A_c = 0.5 * (A_c + A_c.T)
    min_eig = float(np.linalg.eigvalsh(A_c).min())
    if min_eig < nugget:
        A_c = A_c + (nugget - min_eig) * np.eye(K)

    L = np.linalg.cholesky(A_c)
    L_inv = np.linalg.inv(L)

    A_sym = 0.5 * (A_all + np.swapaxes(A_all, -1, -2))
    B = np.einsum("ij,njk,lk->nil", L_inv, A_sym, L_inv, optimize=True)
    eigvals = np.linalg.eigvalsh(B)  # (n, K), ascending order
    return eigvals[:, -1]


def compute_group_omega_max(
    posterior_samples: np.ndarray,
    loc: float,
    scale: float,
    prior_samples: np.ndarray,
    basis_cls: Type[BaseBasisFunction],
    basis_kwargs: Dict[str, Any],
    node_chunk_size: int = 1024,
) -> np.ndarray:
    """
    Per-node omega_max(A_j, A_c) for every scalar node (column) of
    `posterior_samples`, sharing one reference prior N(loc, scale^2) and one
    KEF basis (`basis_cls`, e.g. FixedCentersRBFBasisFunction with K fixed
    centres loc + {-2,-1,0,1,2}*scale).

    Per node j, the worst-case FD sensitivity over the local ball
    {Pi_j: lambda_j^T A_c lambda_j <= r_j} is r_j * omega_max_j (the infimum
    is 0, attained at lambda_j=0, since the FD objective is a PSD quadratic
    form); see eq. (per-node-sensitivity) in the paper. Multiply the returned
    array by r_j to get each node's contribution.

    posterior_samples: (m, n_nodes).
    prior_samples: (m_prior,) -- shared across nodes since A_c only depends
                   on (loc, scale, basis), not on any individual node's draws
                   (the basis itself is therefore also built once per group,
                   even for data-driven bases whose centres/lengthscale are
                   fit from prior_samples).

    Bases flagged `SEPARABLE = True` (e.g. FixedCentersRBFBasisFunction) treat
    every column of a (m, d) sample matrix as an independent scalar node, so
    all n_nodes can be pushed through one batched gradient() call per chunk.
    Other bases (e.g. MaternBasisFunction, RBFBasisFunction, ...) compute a
    single joint kernel value over the *whole* d-dimensional sample point, so
    they must be called once per node (d=1 each time); the basis fit itself
    (centres/lengthscale) still only happens once for the whole group.
    """
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    m, n_nodes = posterior_samples.shape

    basis = _build_basis(
        basis_cls, loc, scale, prior_samples, basis_kwargs,
        posterior_samples_for_centers=posterior_samples[:, 0],
    )

    grad_prior = basis.gradient(np.asarray(prior_samples, dtype=float).reshape(-1, 1))  # (m_prior, 1, K)
    m_prior = grad_prior.shape[0]
    A_c = np.einsum("mdk,mdl->kl", grad_prior, grad_prior) / m_prior  # (K, K)

    omega_max = np.empty(n_nodes, dtype=float)

    if getattr(basis_cls, "SEPARABLE", False):
        for start in range(0, n_nodes, node_chunk_size):
            end = min(start + node_chunk_size, n_nodes)
            grad = basis.gradient(posterior_samples[:, start:end])  # (m, c, K)
            A_chunk = np.einsum("mnk,mnl->nkl", grad, grad, optimize=True) / m  # (c, K, K)
            omega_max[start:end] = _generalized_eigvals_max_batch(A_chunk, A_c)
    else:
        for j in range(n_nodes):
            grad = basis.gradient(posterior_samples[:, j:j + 1])  # (m, 1, K)
            A_j = np.einsum("mdk,mdl->kl", grad, grad, optimize=True) / m  # (K, K)
            omega_max[j] = _generalized_eigvals_max_batch(A_j[None, :, :], A_c)[0]

    return omega_max


def compute_node_lambda_star(
    posterior_samples_col: np.ndarray,
    loc: float,
    scale: float,
    prior_samples: np.ndarray,
    basis_cls: Type[BaseBasisFunction],
    basis_kwargs: Dict[str, Any],
    radius_j: float,
    nugget: float = 1e-10,
) -> Tuple[np.ndarray, float, BaseBasisFunction]:
    """
    Exact worst-case KEF coefficient vector for a single scalar node, for use
    in plotting its candidate density (unlike `compute_group_omega_max`, which
    only returns the generalised eigenvalue, batched over many nodes and
    without recovering the eigenvector).

    Solves sup_{lambda: lambda^T A_c lambda <= radius_j} lambda^T A lambda in
    closed form: lambda_star = sqrt(radius_j) * v', where v' is the
    A_c-normalised eigenvector for the largest generalised eigenvalue of
    A v = omega A_c v -- the single-node version of
    OptimisationNonparametricBase.optimize_through_generalized_eigenvalue.

    Returns (lambda_star, omega_max, basis_function) so the caller can
    reconstruct the candidate log-density as `basis_function.evaluate(x) @
    lambda_star + prior_distribution.log_pdf(x)`.
    """
    basis = _build_basis(
        basis_cls, loc, scale, prior_samples, basis_kwargs,
        posterior_samples_for_centers=posterior_samples_col,
    )

    grad_post = basis.gradient(np.asarray(posterior_samples_col, dtype=float).reshape(-1, 1))
    grad_prior = basis.gradient(np.asarray(prior_samples, dtype=float).reshape(-1, 1))
    m = grad_post.shape[0]
    m_prior = grad_prior.shape[0]

    A = np.einsum("mdk,mdl->kl", grad_post, grad_post) / m
    A_c = np.einsum("mdk,mdl->kl", grad_prior, grad_prior) / m_prior
    A = 0.5 * (A + A.T)
    A_c = 0.5 * (A_c + A_c.T)

    K = A_c.shape[0]
    min_eig = float(np.linalg.eigvalsh(A_c).min())
    if min_eig < nugget:
        A_c = A_c + (nugget - min_eig) * np.eye(K)

    omega_vals, V = scipy_eigh(A, A_c)
    omega_max = float(omega_vals[-1])
    lam_prime = V[:, -1]  # A_c-normalised: lam_prime.T @ A_c @ lam_prime = 1

    lam_star = np.sqrt(max(radius_j, 0.0)) * lam_prime
    if lam_star[np.argmax(np.abs(lam_star))] < 0:
        lam_star = -lam_star

    return lam_star, omega_max, basis
