import numpy as np
from typing import Union

from .base import BaseDistribution


class HalfCauchy(BaseDistribution):
    """
    Univariate Half-Cauchy distribution on [0, ∞) with scale γ > 0.

    Parameters
    ----------
    gamma : float
        Scale (> 0)
    """

    def __init__(self, gamma: float):
        assert gamma > 0, "Scale must be positive."
        self.gamma = gamma
        self._norm_const = 2.0 / (np.pi * self.gamma)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return self.gamma * np.abs(np.random.standard_cauchy(size=n_samples))

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        pdf = np.zeros_like(x, dtype=np.float64)
        mask = x >= 0
        z = x[mask] / self.gamma
        pdf[mask] = self._norm_const / (1.0 + z**2)
        return pdf

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        log_pdf = np.full_like(x, -np.inf, dtype=np.float64)
        mask = x >= 0
        z = x[mask] / self.gamma
        log_pdf[mask] = np.log(2.0) - np.log(np.pi) - np.log(self.gamma) - np.log1p(z**2)
        return log_pdf

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        grad = np.zeros_like(x, dtype=np.float64)
        mask = x >= 0
        dx = x[mask]
        grad[mask] = -2.0 * dx / (self.gamma**2 + dx**2)
        return grad

    def natural_parameters(self) -> np.ndarray:
        raise NotImplementedError(
            "Half-Cauchy is not an exponential-family distribution; natural parameters are undefined.")

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Cauchy is not an exponential-family distribution; sufficient statistics are undefined.")

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Cauchy is not an exponential-family distribution; base measure is undefined.")

