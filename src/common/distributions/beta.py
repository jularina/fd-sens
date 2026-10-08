import numpy as np
from typing import Union
from scipy.special import betaln
from .base import BaseDistribution


class Beta(BaseDistribution):
    """Univariate Beta distribution with shape parameters alpha, beta > 0 on (0, 1)."""

    def __init__(self, alpha: Union[float, str], beta: Union[float, str]):

        def parse_param(p):
            if isinstance(p, str):
                return float(p)
            return float(p)

        self.alpha = parse_param(alpha)
        self.beta = parse_param(beta)

        assert self.alpha > 0, "alpha must be positive."
        assert self.beta > 0, "beta must be positive."

        self._log_norm_const = -betaln(self.alpha, self.beta)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.beta(self.alpha, self.beta, size=n_samples).reshape(-1, 1)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)

        out = np.zeros_like(x, dtype=float)
        mask = (x > 0.0) & (x < 1.0)

        out[mask] = np.exp(
            self._log_norm_const
            + (self.alpha - 1.0) * np.log(x[mask])
            + (self.beta - 1.0) * np.log(1.0 - x[mask])
        )

        return out

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)

        out = np.full_like(x, -np.inf, dtype=float)
        mask = (x > 0.0) & (x < 1.0)

        out[mask] = (
            self._log_norm_const
            + (self.alpha - 1.0) * np.log(x[mask])
            + (self.beta - 1.0) * np.log(1.0 - x[mask])
        )

        return out

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Score function ∇ log p(x)
        """
        x = np.asarray(x, dtype=float)

        grad = np.zeros_like(x)

        mask = (x > 0.0) & (x < 1.0)

        grad[mask] = (
            (self.alpha - 1.0) / x[mask]
            - (self.beta - 1.0) / (1.0 - x[mask])
        )

        # outside support → could also set np.nan if you prefer
        return grad

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """
        Base measure is Lebesgue → gradient zero.
        """
        return np.zeros_like(x, dtype=float)

    def natural_parameters(self) -> np.ndarray:
        """Return the natural parameters (alpha - 1, beta - 1) of the Beta exponential family."""
        return np.array([self.alpha - 1.0, self.beta - 1.0])

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Return the gradients (1/x, -1/(1-x)) of the sufficient statistics (log x, log(1-x))."""
        x = np.asarray(x, dtype=float)

        grad = np.zeros((x.shape[0], 2))

        grad[:, 0] = 1.0 / x.squeeze()
        grad[:, 1] = -1.0 / (1.0 - x.squeeze())

        return grad

