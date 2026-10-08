import math

import numpy as np
from typing import Union

from .base import BaseDistribution


class Gamma(BaseDistribution):
    """Univariate Gamma distribution on (0, ∞) with shape α > 0 and scale θ > 0."""

    def __init__(self, alpha: float, theta: float):
        assert alpha > 0, "Shape must be positive."
        assert theta > 0, "Scale must be positive."
        self.alpha = alpha
        self.theta = theta
        self._log_norm_const = - (self.alpha * np.log(self.theta) + math.lgamma(self.alpha))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.gamma(self.alpha, self.theta, size=n_samples).reshape(-1, 1)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        out = np.zeros_like(x)
        mask = x > 0
        xm = x[mask]
        log_pdf_vals = (
            (self.alpha - 1.0) * np.log(xm)
            - xm / self.theta
            + self._log_norm_const
        )
        out[mask] = np.exp(log_pdf_vals)
        return out

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        out = np.full_like(x, -np.inf, dtype=np.float64)
        mask = x > 0
        xm = x[mask]
        out[mask] = (
            (self.alpha - 1.0) * np.log(xm)
            - xm / self.theta
            + self._log_norm_const
        )
        return out

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Return the gradient of the log-pdf w.r.t. x, (α - 1)/x - 1/θ for x > 0 and 0 otherwise."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        grad = np.zeros_like(x)
        mask = x > 0
        xm = x[mask]
        grad[mask] = (self.alpha - 1.0) / xm - 1.0 / self.theta
        return grad

    def natural_parameters(self) -> np.ndarray:
        """Return the natural parameters (α - 1, -1/θ) of the Gamma exponential family."""
        return np.array([self.alpha - 1.0, -1.0 / self.theta], dtype=np.float64)

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Return gradients (1/x, 1) of the sufficient statistics (log x, x), shape (N, 1, 2), zero for x ≤ 0."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        N = x.shape[0]
        grad = np.zeros((N, 1, 2), dtype=np.float64)
        mask = (x > 0)[:, 0]
        grad[mask, 0, 0] = 1.0 / x[mask, 0]
        grad[mask, 0, 1] = 1.0
        return grad

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """Return the zero gradient of the log base measure, shape (N, 1)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        return np.zeros_like(x, dtype=np.float64)

