import numpy as np
from typing import Union
from .base import BaseDistribution


class ChiSquared(BaseDistribution):
    """Univariate Chi-Squared distribution with k > 0 degrees of freedom."""

    def __init__(self, k: int):
        assert k > 0, "Degrees of freedom must be positive."
        self.k = k
        self._norm_const = 1.0 / (2 ** (self.k / 2) * np.math.gamma(self.k / 2))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.chisquare(self.k, size=n_samples)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        pdf = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        x_pos = x[mask]
        pdf[mask] = self._norm_const * (x_pos ** (self.k / 2 - 1)) * np.exp(-x_pos / 2)
        return pdf

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        log_pdf = np.full_like(x, -np.inf, dtype=np.float64)
        mask = x > 0
        x_pos = x[mask]
        log_pdf[mask] = (
            (self.k / 2 - 1) * np.log(x_pos)
            - x_pos / 2
            - (self.k / 2) * np.log(2)
            - np.log(np.math.gamma(self.k / 2))
        )
        return log_pdf

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Gradient of log-pdf wrt x.
        """
        x = np.asarray(x)
        grad = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        x_pos = x[mask]
        grad[mask] = (self.k / 2 - 1) / x_pos - 0.5
        return grad

    def natural_parameters(self) -> np.ndarray:
        """Return the natural parameters (k/2 - 1, -1/2) of the Chi-Squared exponential family."""
        eta1 = self.k / 2 - 1
        eta2 = -0.5
        return np.array([eta1, eta2])

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Return gradients (1/x, 1) of the sufficient statistics (log x, x), zero outside the support."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n = x.shape[0]
        grad = np.zeros((n, 2), dtype=np.float64)
        mask = x > 0
        grad[mask, 0] = 1.0 / x[mask]
        grad[mask, 1] = 1.0
        return grad[:, None, :]

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """
        Base measure for Chi-Squared is constant wrt x, so gradient is zero.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        grad = np.zeros_like(x)
        return grad[:, None]

