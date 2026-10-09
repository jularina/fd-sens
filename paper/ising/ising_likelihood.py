import numpy as np

from paper.common.losses.base import BaseLoss


class IsingLikelihoodGivenGrads(BaseLoss):
    """Unnormalised Ising model likelihood on an l x l grid, with precomputed log-likelihood gradients."""

    def __init__(self):
        self.grad_log_likelihood = None

    def grad_log_pdf(self) -> np.ndarray:
        return self.grad_log_likelihood
