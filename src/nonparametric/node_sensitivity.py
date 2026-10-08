import inspect
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np

from src.nonparametric.basis_functions import BaseBasisFunction


def _build_basis(
    basis_cls: Type[BaseBasisFunction],
    loc: float,
    scale: float,
    prior_samples: np.ndarray,
    basis_kwargs: Dict[str, Any],
    posterior_samples_for_centers: Optional[np.ndarray] = None,
    center_prior_samples: Optional[np.ndarray] = None,
) -> BaseBasisFunction:
    """Construct a basis function, passing only the keyword arguments its constructor accepts."""
    centers_source = center_prior_samples if center_prior_samples is not None else prior_samples
    prior_col = np.asarray(centers_source, dtype=float).reshape(-1, 1)
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


def _ac_whitening_transform(
    A_c: np.ndarray,
    rel_tol: float = 1e-8,
    nugget: float = 1e-10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Whitening transform W with W^T A_c W = I onto A_c's well-conditioned eigen-subspace."""
    K = A_c.shape[0]
    ac_eigvals, ac_eigvecs = np.linalg.eigh(A_c)  # ascending
    max_eig = float(ac_eigvals[-1])
    threshold = max(rel_tol * max_eig, nugget)
    keep = ac_eigvals > threshold
    if not np.any(keep):
        keep[-1] = True  # always keep at least the dominant direction

    U = ac_eigvecs[:, keep]
    W = U / np.sqrt(ac_eigvals[keep])  # (K, K'): W^T A_c W = I_{K'}

    kept_min_eig = float(ac_eigvals[keep].min())
    diagnostics = {
        "ac_eigvals": ac_eigvals,
        "ac_min_eig": float(ac_eigvals[0]),  # raw smallest eigenvalue, incl. floating-point noise
        "ac_max_eig": max_eig,
        "ac_kept_min_eig": kept_min_eig,
        "ac_cond": float(max_eig / max(kept_min_eig, 1e-300)),  # condition number *within* the kept subspace
        "ac_rank_kept": int(keep.sum()),
        "ac_rank_total": K,
        "ac_threshold": threshold,
        "ac_truncated": bool(keep.sum() < K),
    }
    return W, diagnostics


def _generalized_eigvals_max_batch(
    A_all: np.ndarray,
    A_c: np.ndarray,
    nugget: float = 1e-10,
    rel_tol: float = 1e-8,
) -> np.ndarray:
    """
    Largest generalised eigenvalue of A_j v = omega * A_c v for every node j,
    solved by projecting onto A_c's numerically well-conditioned eigen-
    subspace rather than Cholesky-factorising A_c directly -- see
    `_ac_whitening_transform`.
    """
    A_c = 0.5 * (A_c + A_c.T)
    W, _ = _ac_whitening_transform(A_c, rel_tol=rel_tol, nugget=nugget)

    A_sym = 0.5 * (A_all + np.swapaxes(A_all, -1, -2))
    B = np.einsum("ki,nkl,lj->nij", W, A_sym, W, optimize=True)  # (n, K', K')
    eigvals = np.linalg.eigvalsh(B)  # (n, K'), ascending order
    return eigvals[:, -1]


def compute_group_omega_max(
    posterior_samples: np.ndarray,
    loc: float,
    scale: float,
    prior_samples: np.ndarray,
    basis_cls: Type[BaseBasisFunction],
    basis_kwargs: Dict[str, Any],
    center_prior_samples: Optional[np.ndarray] = None,
    rel_tol: float = 1e-8,
) -> np.ndarray:
    """Compute omega_max(A_j, A_c) for each scalar node of a group sharing one prior and KEF basis."""
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    m, n_nodes = posterior_samples.shape

    basis = _build_basis(
        basis_cls, loc, scale, prior_samples, basis_kwargs,
        posterior_samples_for_centers=posterior_samples[:, 0],
        center_prior_samples=center_prior_samples,
    )

    grad_prior = basis.gradient(np.asarray(prior_samples, dtype=float).reshape(-1, 1))  # (m_prior, 1, K)
    m_prior = grad_prior.shape[0]
    A_c = np.einsum("mdk,mdl->kl", grad_prior, grad_prior) / m_prior  # (K, K)

    omega_max = np.empty(n_nodes, dtype=float)
    for j in range(n_nodes):
        grad = basis.gradient(posterior_samples[:, j:j + 1])  # (m, 1, K)
        A_j = np.einsum("mdk,mdl->kl", grad, grad, optimize=True) / m  # (K, K)
        omega_max[j] = _generalized_eigvals_max_batch(A_j[None, :, :], A_c, rel_tol=rel_tol)[0]

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
    rel_tol: float = 1e-8,
    center_prior_samples: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, BaseBasisFunction, Dict[str, Any]]:
    """Compute the worst-case KEF coefficient vector and omega_max for a single scalar node."""
    basis = _build_basis(
        basis_cls, loc, scale, prior_samples, basis_kwargs,
        posterior_samples_for_centers=posterior_samples_col,
        center_prior_samples=center_prior_samples,
    )

    grad_post = basis.gradient(np.asarray(posterior_samples_col, dtype=float).reshape(-1, 1))
    grad_prior = basis.gradient(np.asarray(prior_samples, dtype=float).reshape(-1, 1))
    m = grad_post.shape[0]
    m_prior = grad_prior.shape[0]

    A = np.einsum("mdk,mdl->kl", grad_post, grad_post) / m
    A_c = np.einsum("mdk,mdl->kl", grad_prior, grad_prior) / m_prior
    A = 0.5 * (A + A.T)
    A_c = 0.5 * (A_c + A_c.T)

    W, diagnostics = _ac_whitening_transform(A_c, rel_tol=rel_tol, nugget=nugget)

    A_white = W.T @ A @ W
    A_white = 0.5 * (A_white + A_white.T)
    omega_vals, V_white = np.linalg.eigh(A_white)  # ascending
    omega_max = float(omega_vals[-1])
    y_star = V_white[:, -1]  # unit-norm in whitened coords: y_star.T @ y_star = 1
    lam_prime = W @ y_star   # lam_prime.T @ A_c @ lam_prime = 1 (A_c-normalised)

    lam_star = np.sqrt(max(radius_j, 0.0)) * lam_prime
    if lam_star[np.argmax(np.abs(lam_star))] < 0:
        lam_star = -lam_star

    return lam_star, omega_max, basis, diagnostics
