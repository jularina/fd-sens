from src.common.bayesian_model.base import BayesianModel
from src.common.fisher import PosteriorFDBase
from src.nonparametric.basis_functions import BaseBasisFunction

from typing import Tuple
import numpy as np


class PriorFDBase:
    def __init__(self, model: "BayesianModel"):
        """
        Base class for Fisher divergence between priors.

        Holds prior samples and the reference prior score. PriorFDNonParametric
        adds the basis-function representation of the candidate prior.
        """
        self.model = model
        self.samples: np.ndarray = self.model.prior_samples_init
        self.m = int(self.samples.shape[0])

        # Reference prior score at the samples — shape: (m, paramdim)
        self.score_prior_ref = self.model.prior_init.grad_log_pdf(self.samples)

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


class PriorFDNonParametric(PriorFDBase):
    def __init__(self, model: "BayesianModel"):
        """
        Fisher divergence between priors in the nonparametric representation.
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
        gradT(theta_i) = Jacobian of T(theta) = [phi_1, ..., phi_K]^T at samples.

        Expected shape: (m, paramdim, K)
        """
        return basis_func.gradient(self.samples)


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
