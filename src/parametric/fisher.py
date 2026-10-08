from typing import Tuple, List, Sequence, Dict

import numpy as np
from scipy.stats import norm

from src.common.bayesian_model.base import BayesianModel
from src.common.fisher import PosteriorFDBase


class PosteriorFDParametric(PosteriorFDBase):
    def __init__(self, model: "BayesianModel"):
        """Fisher divergence at posterior samples for parametric exponential-family candidate priors."""
        super().__init__(model=model)

        # Candidate prior (exp family decomposition), evaluated at the same samples
        # grad_T: (m, paramdim, natparamdim)
        # grad_log_g: (m, paramdim)
        self.grad_T = self.model.prior_candidate.grad_sufficient_statistics(self.samples)
        self.grad_log_g = self.model.prior_candidate.grad_log_base_measure(self.samples)
        self.eta = self.model.prior_candidate.natural_parameters()

    def update_candidate(self) -> None:
        self.grad_T = self.model.prior_candidate.grad_sufficient_statistics(self.samples)
        self.grad_log_g = self.model.prior_candidate.grad_log_base_measure(self.samples)
        self.eta = self.model.prior_candidate.natural_parameters()
        self.beta = self.model.loss_lr

    def _v_prior_only(self) -> np.ndarray:
        """Return v_i = s_ref(θ_i) - grad_log_g(θ_i) at the posterior samples, shape (m, paramdim)."""
        return self.score_prior_ref - self.grad_log_g

    def _delta_score_prior_only(self, eta: np.ndarray) -> np.ndarray:
        """Return the score differences v_i - grad_T(θ_i) @ eta at the posterior samples, shape (m, paramdim)."""
        v = self._v_prior_only()
        gradT_eta = np.einsum("idp,p->id", self.grad_T, eta)
        return v - gradT_eta

    def estimate_fisher_prior_only(self) -> float:
        self.update_candidate()
        diff = self._delta_score_prior_only(self.eta)
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def fd_prior_only_given_eta(self, eta: np.ndarray) -> float:
        """Compute the prior-only empirical FD at the natural parameter vector eta."""
        self.update_candidate()
        eta = np.asarray(eta, dtype=float).reshape(-1)
        v = self._v_prior_only()
        gradT_eta = np.einsum("idp,p->id", self.grad_T, eta)
        diff = v - gradT_eta
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def compute_fisher_quadratic_form_prior_only(self) -> Tuple:
        """
        Prior-only perturbation quadratic in eta.
        """
        self.update_candidate()
        A = self._compute_A_prior_only()
        b = self._compute_b_prior_only()
        c = self._compute_c_prior_only()
        return A, b, c

    def compute_prior_only_qf_per_component(
            self,
            component_names: List[str],
            theta_blocks: Sequence[Sequence[int]],
            eta_blocks: Sequence[Sequence[int]],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """Build per-component prior-only quadratic forms (A_j, b_j, c_j) from theta and eta blocks."""
        assert len(component_names) == len(theta_blocks) == len(eta_blocks)

        self.update_candidate()
        v = self._v_prior_only()
        G = self.grad_T
        m = self.m

        out = {}
        for name, th_block, et_block in zip(component_names, theta_blocks, eta_blocks):
            th_block = list(th_block)
            et_block = list(et_block)
            v_b = v[:, th_block]
            G_b = G[:, th_block, :][:, :, et_block]
            A_j = np.einsum("itp,itq->pq", G_b, G_b) / m
            b_j = (-2.0 / m) * np.einsum("itp,it->p", G_b, v_b)
            c_j = float(np.mean(np.sum(v_b * v_b, axis=1)))
            out[name] = (A_j, b_j, c_j)

        return out

    # -------------------------------
    # Learning-rate-only perturbation
    # -------------------------------

    def estimate_fisher_lr_only(self) -> float:
        """Compute the learning-rate-only FD (beta - beta_ref) * mean ||g_i||^2 with the prior fixed."""
        self.beta = self.model.loss_lr
        diff = float(np.mean(np.sum(self.g * self.g, axis=1)))
        return (self.beta - self.beta_ref)**2 * diff


    # -------------------------
    # Copula perturbations
    # -------------------------

    def fd_gaussian_copula_given_lambda(
            self,
            lam: float,
            idx_g0: int = 0,
            idx_nu: int = 2,
            eps: float = 1e-10,
            lower: np.ndarray = None,
            upper: np.ndarray = None,
            apply_z_transform: bool = True,
    ) -> float:
        """Compute the empirical Fisher divergence of a Gaussian copula perturbation with correlation lam."""
        lam = float(lam)
        if abs(lam) >= 1.0:
            return np.inf

        u_g0 = np.asarray(self.samples[:, idx_g0], dtype=float)
        u_nu = np.asarray(self.samples[:, idx_nu], dtype=float)

        if apply_z_transform:
            if lower is not None and upper is not None:
                u_g0 = (u_g0 - lower[idx_g0]) / (upper[idx_g0] - lower[idx_g0])
                u_nu = (u_nu - lower[idx_nu]) / (upper[idx_nu] - lower[idx_nu])

            u_g0 = np.clip(u_g0, eps, 1.0 - eps)
            u_nu = np.clip(u_nu, eps, 1.0 - eps)

            z_g0 = norm.ppf(u_g0)
            z_nu = norm.ppf(u_nu)
        else:
            z_g0 = u_g0
            z_nu = u_nu

        phi_z_g0 = norm.pdf(z_g0)
        phi_z_nu = norm.pdf(z_nu)

        denom = 1.0 - lam ** 2

        dlogc_dg0 = lam * (z_nu - lam * z_g0) / (denom * phi_z_g0)
        dlogc_dnu = lam * (z_g0 - lam * z_nu) / (denom * phi_z_nu)

        sq_norm = dlogc_dg0 ** 2 + dlogc_dnu ** 2

        return float(np.mean(sq_norm))
