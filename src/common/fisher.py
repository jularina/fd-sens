import numpy as np

from src.common.bayesian_model.base import BayesianModel


class PosteriorFDBase:
    def __init__(self, model: "BayesianModel"):
        """Store posterior samples, reference prior score and loss gradient for Fisher divergence estimation."""
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
