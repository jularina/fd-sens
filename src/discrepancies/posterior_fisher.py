from src.bayesian_model.base import BayesianModel
from src.basis_functions.basis_functions import BaseBasisFunction

from typing import Tuple, List, Sequence, Dict
import numpy as np
from scipy.stats import norm


class PosteriorFDBase:
    def __init__(self, model: "BayesianModel"):
        """
        Base class for Fisher divergence computed at posterior samples.

        Holds posterior samples, reference prior score, and loss gradient.
        Subclasses add either a parametric candidate prior (PosteriorFDParametric)
        or a nonparametric basis-function representation (PosteriorFDNonParametric).
        """
        self.model = model
        self.samples: np.ndarray = self.model.posterior_samples_init
        self.m = self.samples.shape[0]

        # Reference prior score at the samples — shape: (m, paramdim)
        self.score_prior_ref = self.model.prior_init.grad_log_pdf(self.samples)

        # Loss gradient sum_j ∇_θ l(θ_i, x_j) — shape: (m, paramdim)
        self.g = self.model.loss_score(self.samples, multiply_by_lr=False)
        self.beta_ref = self.model.loss_lr_init
        self.beta = self.model.loss_lr

    # -------------------------
    # Shared quadratic-form helpers (require self.grad_T and _v_prior_only)
    # -------------------------

    def _compute_A_prior_only(self) -> np.ndarray:
        return np.einsum("idp,idq->pq", self.grad_T, self.grad_T) / self.m

    def _compute_b_prior_only(self) -> np.ndarray:
        v = self._v_prior_only()
        term = np.einsum("idp,id->p", self.grad_T, v)
        return (-2.0 / self.m) * term

    def _compute_c_prior_only(self) -> float:
        v = self._v_prior_only()
        return float(np.sum(v * v) / self.m)

    # -------------------------------
    # Learning-rate-only perturbation
    # -------------------------------

    def estimate_fisher_lr_only(self) -> float:
        """
        Learning-rate-only perturbation with eta fixed to eta_ref (and prior fixed to ref).
        Then:
            s_ref(θ_i) - s_beta(θ_i) = (beta - beta_ref) * g_i
        so FD = (beta - beta_ref) (1/m) sum_i || g_i||^2
        """
        self.beta = self.model.loss_lr
        diff = float(np.mean(np.sum(self.g * self.g, axis=1)))
        return (self.beta - self.beta_ref)**2 * diff

    def compute_fisher_quadratic_form_lr_only(
        self,
    ) -> Tuple[float, float, float]:
        """
        Learning-rate-only perturbation quadratic in beta:
            FD(beta) = A beta^2 + b beta + c
        with:
            A = (1/m) sum_i ||g_i||^2
            b = -2 beta_ref A
            c = beta_ref^2 A
        """
        self.beta = self.model.loss_lr
        A = float(np.mean(np.sum(self.g * self.g, axis=1)))
        b = float(-2.0 * self.beta_ref * A)
        c = float((self.beta_ref ** 2) * A)
        return A, b, c

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
        """
        Empirical Fisher divergence for Gaussian copula perturbation.

        Computes
            E_{theta ~ Pi_ref} || ∇_theta log c_lam(u_G0, u_nu) ||^2.

        Parameters
        ----------
        lam : float
            Gaussian copula correlation parameter. Must satisfy |lam| < 1.
        idx_g0 : int
            Column index of the G0 coordinate in self.samples.
        idx_nu : int
            Column index of the nu coordinate in self.samples.
        eps : float
            Boundary clipping level for numerical stability after F_i transform.
        lower, upper : np.ndarray, optional
            Per-dimension prior bounds. When provided, samples are mapped to (0,1)
            via F_i(theta) = (theta - lower[i]) / (upper[i] - lower[i]) before
            applying Phi^{-1}. If None, samples are assumed to already be in (0,1).
            Only used when apply_z_transform=True.
        apply_z_transform : bool
            If True (default), rescale samples to (0,1) using lower/upper and apply
            Phi^{-1} to obtain z-scores. If False, samples are used directly as
            z-scores without any transform.
        """
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


class PosteriorFDParametric(PosteriorFDBase):
    def __init__(self, model: "BayesianModel"):
        """
        Fisher divergence at posterior samples in the parametric (exponential-family)
        representation. Requires model.prior_candidate to be set.
        """
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
        """
        v_i = s_{pi_ref}(θ_i) - grad_log_g(θ_i)
        Shape: (m, paramdim)
        """
        return self.score_prior_ref - self.grad_log_g

    def _delta_score_prior_only(self, eta: np.ndarray) -> np.ndarray:
        """
        delta_i = s_ref(θ_i) - s_candidate(θ_i)
               = (s_{pi_ref}(θ_i) - grad_log_g(θ_i)) - grad_T(θ_i) @ eta
        Shape: (m, paramdim)
        """
        v = self._v_prior_only()
        gradT_eta = np.einsum("idp,p->id", self.grad_T, eta)
        return v - gradT_eta

    def estimate_fisher_prior_only(self) -> float:
        self.update_candidate()
        diff = self._delta_score_prior_only(self.eta)
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def fd_prior_only_given_eta(self, eta: np.ndarray) -> float:
        """
        Black-box objective: compute \hat{rho}^FD_m for prior-only perturbations,
        evaluated at the provided natural parameter vector eta.
        """
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
        """
        Build per-component quadratic forms:
            Q_j(eta_j) = eta_j^T A_j eta_j + b_j^T eta_j + c_j
        where each component j uses only theta coordinates in theta_blocks[j]
        and only natural-parameter coordinates in eta_blocks[j].
        """
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


class PosteriorFDNonParametric(PosteriorFDBase):
    def __init__(self, model: "BayesianModel"):
        """
        Fisher divergence at posterior samples in the nonparametric representation.
        No candidate prior required. The base measure g is the reference prior.
        """
        super().__init__(model=model)
        # ∇_θ log g(θ_i) where g = prior_init is the base measure in the nonparametric family
        self.grad_log_g = self.model.prior_init.grad_log_pdf(self.samples)

    def _v_prior_only(self) -> np.ndarray:
        return self.score_prior_ref - self.grad_log_g

    def compute_non_parametric_fisher_quadratic_form_prior_only(
        self,
        basis_func: "BaseBasisFunction",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Quadratic form using nonparametric basis functions.
        """
        self.grad_T = self._compute_grad_basis_function_for_prior(basis_func)

        A = self._compute_A_prior_only()
        b = self._compute_b_prior_only()
        c = self._compute_c_prior_only()

        return A, b, c

    def _compute_grad_basis_function_for_prior(
        self,
        basis_func: "BaseBasisFunction",
    ) -> np.ndarray:
        """
        Returns grad_phi with shape (m, paramdim, K).
        """
        return basis_func.gradient(self.samples)
