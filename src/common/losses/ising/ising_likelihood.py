import numpy as np

from src.common.losses.base import BaseLoss


class IsingLikelihoodGivenGrads(BaseLoss):
    """
    Ising Model Likelihood (ignoring normalization constant).
    4-neighbour l x l grid, where d = l*l.
    """

    def __init__(self):
        self.grad_log_likelihood = None

    def grad_log_pdf(self) -> np.ndarray:
        return self.grad_log_likelihood
