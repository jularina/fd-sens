import math

import numpy as np
from typing import Union

from .base import BaseDistribution


class InverseGamma(BaseDistribution):
    """Univariate Inverse-Gamma distribution on (0, ∞) with shape α > 0 and scale β > 0."""

    def __init__(self, alpha: float, beta: float):
        assert alpha > 0, "Shape must be positive."
        assert beta > 0, "Scale must be positive."
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._log_norm_const = self.alpha * np.log(self.beta) - math.lgamma(self.alpha)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        # If X ~ Gamma(alpha, 1/beta) then 1/X ~ InverseGamma(alpha, beta)
        return (1.0 / np.random.gamma(self.alpha, 1.0 / self.beta, size=n_samples)).reshape(-1, 1)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        out = np.zeros_like(x)
        mask = x > 0
        xm = x[mask]
        log_pdf_vals = (
            self._log_norm_const
            - (self.alpha + 1.0) * np.log(xm)
            - self.beta / xm
        )
        out[mask] = np.exp(log_pdf_vals)
        return out

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        out = np.full_like(x, -np.inf, dtype=np.float64)
        mask = x > 0
        xm = x[mask]
        out[mask] = (
            self._log_norm_const
            - (self.alpha + 1.0) * np.log(xm)
            - self.beta / xm
        )
        return out

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        ∂/∂x log f(x) = −(α + 1)/x + β/x²  for x > 0; 0 otherwise.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        grad = np.zeros_like(x)
        mask = x > 0
        xm = x[mask]
        grad[mask] = -(self.alpha + 1.0) / xm + self.beta / (xm * xm)
        return grad

    def natural_parameters(self) -> np.ndarray:
        """
        η₁ = −(α + 1),  η₂ = −β
        """
        return np.array([-(self.alpha + 1.0), -self.beta], dtype=np.float64)

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Return gradients (1/x, −1/x²) of the sufficient statistics (log x, 1/x), shape (N, 1, 2)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        N = x.shape[0]
        grad = np.zeros((N, 1, 2), dtype=np.float64)
        mask = (x > 0)[:, 0]
        xm = x[mask, 0]
        grad[mask, 0, 0] = 1.0 / xm
        grad[mask, 0, 1] = -1.0 / (xm * xm)
        return grad

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """Return the zero gradient of the log base measure, shape (N, 1)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        return np.zeros_like(x, dtype=np.float64)

