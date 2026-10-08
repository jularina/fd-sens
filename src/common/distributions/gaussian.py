import numpy as np
from numpy.linalg import inv, det
from typing import Union, Dict

from .base import BaseDistribution
from src.common.utils.typing import ArrayLike
from src.common.utils.checkers import is_symmetric_and_psd


class Gaussian(BaseDistribution):
    """Univariate Gaussian distribution with mean mu and standard deviation sigma."""

    def __init__(self, mu: float, sigma: float):
        assert sigma > 0, "Standard deviation must be positive."
        self.mu = mu
        self.sigma = sigma
        self.var = sigma ** 2
        self._norm_const = 1.0 / np.sqrt(2 * np.pi * self.var)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma, size=n_samples).reshape(-1, 1)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        return self._norm_const * np.exp(-((x - self.mu) ** 2) / (2 * self.var))

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        return -0.5 * np.log(2 * np.pi * self.var) - ((x - self.mu) ** 2) / (2 * self.var)

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        return -(x - self.mu) / self.var

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        grad_T1 = np.ones_like(x[..., 0])
        grad_T2 = 2 * x[..., 0]
        grad = np.stack([grad_T1, grad_T2], axis=-1)
        return grad[..., None, :]

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def natural_parameters(self) -> np.ndarray:
        eta1 = self.mu / self.var
        eta2 = -0.5 / self.var
        return np.array([eta1, eta2])

    @property
    def parameters_dict(self) -> Dict[str, float]:
        return {"mu": self.mu, "sigma": self.sigma}


class MultivariateGaussian(BaseDistribution):
    """Multivariate Gaussian distribution with mean vector mu and covariance matrix cov."""

    def __init__(self, mu: ArrayLike, cov: ArrayLike):
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)
        self.dim = self.mu.shape[0]
        assert self.cov.shape == (self.dim, self.dim), "Covariance shape mismatch."
        assert is_symmetric_and_psd(self.cov), "Covariance is not p.s.d."

        self.cov_inv = inv(self.cov)
        self._norm_const = 1.0 / np.sqrt((2 * np.pi) ** self.dim * det(self.cov))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.multivariate_normal(self.mu, self.cov, size=n_samples)

    def pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        x = np.atleast_2d(x)
        dx = x - self.mu
        exponent = np.einsum("...i,ij,...j->...", dx, self.cov_inv, dx)
        return self._norm_const * np.exp(-0.5 * exponent)

    def log_pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        x = np.atleast_2d(x)
        dx = x - self.mu
        exponent = np.einsum("...i,ij,...j->...", dx, self.cov_inv, dx)
        return np.log(self._norm_const) - 0.5 * exponent

    def grad_log_pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        x = np.atleast_2d(x)
        dx = x - self.mu
        return - np.einsum("ij,...j->...i", self.cov_inv, dx)

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.zeros_like(x)

    def natural_parameters(self) -> np.ndarray:
        eta1 = self.cov_inv @ self.mu
        eta2 = -0.5 * self.cov_inv.flatten()
        return np.concatenate([eta1, eta2])

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        batch_shape = x.shape[:-1]
        d = x.shape[-1]

        grad_T = np.zeros(batch_shape + (d, d + d**2))
        grad_T[..., :, :d] = np.eye(d)

        for i in range(d):
            for j in range(d):
                grad_T[..., i, d + i * d + j] = x[..., j]
                grad_T[..., j, d + i * d + j] += x[..., i]

        return grad_T

    @property
    def parameters_dict(self) -> Dict[str, np.ndarray]:
        return {"mu": self.mu, "cov": self.cov}
