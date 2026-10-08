from typing import Dict, Optional
import numpy as np
from omegaconf import OmegaConf

from src.basis_functions.basis_functions import BaseBasisFunction
from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY


class OptimisationNonparametricBase:
    def __init__(
        self,
        posterior_estimator,
        prior_estimator,
        config: Dict,
        radius: float = 0.0,
        add_nuggets: bool = False,
        basis_function: Optional[BaseBasisFunction] = None,
    ):
        """
        Base class to handle nonparametric quadratic form optimization.

        `basis_function`, if given, is used as-is instead of being built from
        `config` (e.g. to force a basis whose centres are selected from a
        different sample set than `prior_estimator`/`posterior_estimator`).

        When built from `config`, prior-based centres are never chosen from
        `prior_estimator.samples` (those samples estimate the Fisher
        divergence). Instead, a fresh, independent i.i.d. draw of the same
        size from the reference prior is used as the pool to select centres
        from, decoupling centre selection from the FD estimate.
        """
        self.posterior_estimator = posterior_estimator
        self.prior_estimator = prior_estimator

        if basis_function is not None:
            self.basis_function = basis_function
        else:
            basis_cls_name = config["basis_funcs_type"]
            basis_cls = BASIS_FUNCTIONS_REGISTRY[basis_cls_name]
            basis_kwargs = config.get("basis_funcs_kwargs", {})
            basis_kwargs = OmegaConf.to_container(basis_kwargs, resolve=True)
            np.random.seed(27)
            basis_kwargs["prior_samples"] = self.prior_estimator.model.sample_from_base_prior(
                n_samples=len(self.prior_estimator.samples),
            )
            basis_kwargs["posterior_samples"] = self.posterior_estimator.samples
            self.basis_function = basis_cls(**basis_kwargs)

        self.A, self.b, self.c = self.posterior_estimator.compute_non_parametric_fisher_quadratic_form_prior_only(
            self.basis_function,
        )
        self.A_c, self.b_c, self.c_c = self.prior_estimator.compute_non_parametric_fisher_quadratic_form_prior_only(
            self.basis_function,
        )
        self.A = self._sym(self.A)
        self.A_c = self._sym(self.A_c)
        psd, rank = self._check_pd_psd_and_rank(self.A)
        print(f"A is {psd} with rank {rank}.")
        psd, rank = self._check_pd_psd_and_rank(self.A_c)
        print(f"A_c is {psd} with rank {rank}.")

        self.d = self.A_c.shape[0]

        if add_nuggets:
            avg_prior = np.trace(self.A_c) / max(self.d, 1)
            avg_obj = np.trace(self.A) / max(self.d, 1)
            eps_prior = max(1e-8, 1e-2 * avg_prior)
            eps_obj = max(1e-12, 1e-4 * avg_obj)
            self.A_c = self._sym(self.A_c + eps_prior * np.eye(self.d))
            self.A = self._sym(self.A + eps_obj * np.eye(self.d))
            psd, rank = self._check_pd_psd_and_rank(self.A)
            print(f"After nuggets A is {psd} with rank {rank}.")
            psd, rank = self._check_pd_psd_and_rank(self.A_c)
            print(f"After nuggets A_c is {psd} with rank {rank}.")

        self.r = self._compute_min_radius() + radius
        print(f"Radius {self.r}.")


    def _check_pd_psd_and_rank(
            self,
            A: np.ndarray,
            tol: float = 1e-10,
    ):
        """
        Check whether a symmetric matrix is positive definite (PD),
        positive semidefinite (PSD), or not PSD, and return its numerical rank.
        """
        eigvals = np.linalg.eigvalsh(A)
        rank = int(np.sum(eigvals > tol))

        if np.all(eigvals > tol):
            status = "PD"
        elif np.all(eigvals >= -tol):
            status = "PSD"
        else:
            status = "NOT_PSD"

        return status, rank

    @staticmethod
    def _pinv_psd(A: np.ndarray, rcond: float | None = None) -> np.ndarray:
        """
        Moore–Penrose pseudoinverse specialized for symmetric PSD/Hermitian:
        eigen-decompose, invert eigenvalues above tolerance.
        """
        w, V = np.linalg.eigh(A)

        if rcond is None:
            rcond = np.finfo(A.dtype).eps * max(A.shape) * max(w.max(), 1.0)
        w_inv = np.zeros_like(w)
        mask = w > rcond
        w_inv[mask] = 1.0 / w[mask]
        return (V * w_inv) @ V.T

    def _evaluate_qf(self, lam: np.ndarray) -> float:
        return lam @ self.A @ lam + self.b @ lam + self.c

    def _evaluate_constraint_qf(self, lam: np.ndarray) -> float:
        return lam @ self.A_c @ lam + self.b_c @ lam + self.c_c

    def _compute_min_radius(self) -> float:
        quad_term = -0.25 * float(self.b_c.T @ self._pinv_psd(self.A_c) @ self.b_c)
        min_val = float(self.c_c + quad_term)
        print(f"Computed min radius threshold: {min_val}.")
        return max(0.0, min_val)

    # -------------------------
    # Helpers for dual methods
    # -------------------------


    @staticmethod
    def _sym(A: np.ndarray) -> np.ndarray:
        return 0.5 * (A + A.T)

    @staticmethod
    def _canonical_sign(lam: np.ndarray) -> np.ndarray:
        """Make the largest-magnitude component positive to break sign ambiguity."""
        return lam if lam[np.argmax(np.abs(lam))] >= 0 else -lam


    # -------------------------
    # Optimisation methods
    # -------------------------


    def optimize_through_generalized_eigenvalue(self, nugget: float = 1e-10, rel_tol: float | None = None):
        """
        Solve the QCQP via the generalised eigenvalue problem.

        Valid when the base measure g = π_ref, which makes b = b_c = 0 and
        c = c_c = 0.  The problem then simplifies to:

            sup_{λᵀ A_c λ ≤ r} λᵀ A λ  =  r · ω_max

        where ω_max is the largest generalised eigenvalue of  A λ' = ω A_c λ'.
        The optimal solution is  λ_star = sqrt(r) · λ'_star,  with λ'_star the
        A_c-normalised eigenvector for ω_max.  Complexity O(K³).

        Args:
            nugget: regularisation added to A_c when it is not PD.
            rel_tol: if given, instead of shifting A_c by a nugget and
                Cholesky-factorising it, solve on A_c's numerically
                well-conditioned eigen-subspace (eigenvalues above
                max(rel_tol * max_eig(A_c), nugget)) -- see
                src.optimization.bnn_node_sensitivity._ac_whitening_transform.
                Use this when A_c is near-singular (e.g. centres clustered
                where few prior samples fall), where the nugget shift
                inflates the ratio along the near-null directions.
        """
        from scipy.linalg import eigh as scipy_eigh
        import warnings

        b_tol = 1e-6 * max(float(np.trace(np.abs(self.A))), 1.0)
        if float(np.linalg.norm(self.b)) > b_tol:
            warnings.warn(
                f"||b|| = {np.linalg.norm(self.b):.2e} is non-negligible (tol {b_tol:.2e}). "
                "Generalised eigenvalue method assumes b = b_c = 0 (g = π_ref)."
            )

        A = self._sym(self.A)
        A_c = self._sym(self.A_c)

        if rel_tol is not None:
            from src.optimization.bnn_node_sensitivity import _ac_whitening_transform

            W, ac_diag = _ac_whitening_transform(A_c, rel_tol=rel_tol, nugget=nugget)  # W^T A_c W = I
            print(f"A_c whitened onto its well-conditioned subspace: rank {ac_diag['ac_rank_kept']}/{self.d}.")
            omega_vals, Y = np.linalg.eigh(self._sym(W.T @ A @ W))  # ascending
            omega_star = float(omega_vals[-1])
            lam_prime = W @ Y[:, -1]  # A_c-normalised: lam_prime.T @ A_c @ lam_prime = 1
        else:
            min_eig = float(np.linalg.eigvalsh(A_c).min())
            if min_eig < nugget:
                shift = nugget - min_eig
                A_c = A_c + shift * np.eye(self.d)
                print(f"A_c shifted by {shift:.2e} to ensure PD for generalised eigenvalue solver.")

            # scipy_eigh solves A v = ω A_c v and returns A_c-orthonormal eigenvectors
            # in ascending eigenvalue order; the optimum is the last column.
            omega_vals, V = scipy_eigh(A, A_c)

            omega_star = float(omega_vals[-1])
            lam_prime = V[:, -1]  # A_c-normalised: lam_prime.T @ A_c @ lam_prime = 1

        lam_star = np.sqrt(max(self.r, 0.0)) * lam_prime
        lam_star = self._canonical_sign(lam_star)

        primal_value = self._evaluate_qf(lam_star)
        constraint_value = self._evaluate_constraint_qf(lam_star)

        return {
            "lambda_star": lam_star,
            "omega_star": omega_star,
            "primal_value": primal_value,
            "constraint_value": constraint_value,
            "theoretical_value": float(self.r) * omega_star,
        }
