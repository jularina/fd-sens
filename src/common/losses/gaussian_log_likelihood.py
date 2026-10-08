import numpy as np

from src.common.utils.typing import ArrayLike
from src.common.losses.base import BaseLoss


class GaussianLogLikelihood(BaseLoss):
    """
    Univariate Gaussian distribution.
    """

    def __init__(self, mu: float, sigma: float):
        assert sigma > 0, "Standard deviation must be positive."
        self.mu = mu
        self.sigma = sigma
        self.var = sigma ** 2
        self._norm_const = 1.0 / np.sqrt(2 * np.pi * self.var)

    def grad_log_pdf(self, theta: ArrayLike, x_bar: float, observations_num: int) -> np.ndarray:
        return observations_num * (x_bar-theta) / self.var


class MultivariateGaussianLogLikelihood(BaseLoss):
    """
    Multivariate Gaussian log-likelihood with full covariance.
    """

    def __init__(self, mu: ArrayLike, cov: ArrayLike):
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)

        assert self.cov.shape[0] == self.cov.shape[1], "Covariance must be square."
        assert self.cov.shape[0] == self.mu.shape[0], "Covariance and mean dimension mismatch."

        if not np.all(np.linalg.eigvals(self.cov) > 0):
            raise ValueError("Covariance matrix must be positive definite.")

        self.dim = self.mu.shape[0]
        self.cov_inv = np.linalg.inv(self.cov)
        self.det_cov = np.linalg.det(self.cov)
        self._norm_const = 1.0 / np.sqrt((2 * np.pi) ** self.dim * self.det_cov)

    def grad_log_pdf(self, theta: ArrayLike, x_bar: ArrayLike, observations_num: int) -> np.ndarray:
        """
        Gradient of the log-likelihood w.r.t. parameter x (mean vector).

        Parameters
        ----------
        theta : np.ndarray
            Current parameter (mean vector), shape (d,)
        x_bar : np.ndarray
            Empirical mean of the data, shape (d,)
        observations_num : int
            Number of data points

        Returns
        -------
        grad : np.ndarray
            Gradient vector of shape (d,)
        """
        diff = x_bar - theta
        result = diff @ self.cov_inv.T

        return observations_num * result


class GaussianARLogLikelihood(BaseLoss):
    """
    AR(K) Gaussian log-likelihood:
        y_t ~ Normal(alpha + sum_{k=1}^K beta_k * y_{t-k}, sigma)
    Parameter vector order: [alpha, beta1, ..., betaK, gamma]
        gamma = sigma        if scale == "sigma"
        gamma = log(sigma)   if scale == "log_sigma"
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self._has_data = False
        self.K = 0
        self.n = 0  # n = T - K

        self.Sy = None                  # sum y_t
        self.Syy = None                 # sum y_t^2
        self.Sx = None                  # sum x_t (K-dim) where x_tk = y_{t-k-1}
        self.Sxx = None                 # sum x_t x_t^T (KxK)
        self.Sxy = None                 # sum x_t y_t (K-dim)

    def set_data(self, y: ArrayLike, K: int) -> None:
        y = np.asarray(y, float).reshape(-1)
        T = y.shape[0]
        if not (1 <= K < T):
            raise ValueError("Require T > K >= 1.")
        self.K = int(K)

        # Targets and lagged design
        Y = y[K:T]  # (n,)
        X = np.column_stack([y[K - k: T - k] for k in range(1, K + 1)])  # (n, K)

        # Sufficient stats over t = K..T-1
        self.n = Y.shape[0]
        self.Sy = float(Y.sum())
        self.Syy = float(Y @ Y)
        self.Sx = X.sum(axis=0)  # (K,)
        self.Sxx = X.T @ X  # (K,K)
        self.Sxy = X.T @ Y  # (K,)
        self._has_data = True

    def _ensure(self):
        if not self._has_data:
            raise RuntimeError("Data not set. Call set_data(y, K).")

    def _S_T_Q(self, alpha: np.ndarray, beta: np.ndarray):
        # alpha: (m,1), beta: (m,K)
        n, Sy, Sx, Sxx, Sxy, Syy = self.n, self.Sy, self.Sx, self.Sxx, self.Sxy, self.Syy

        beta_Sx = beta @ Sx.reshape(-1, 1)  # (m,1)
        S = Sy - n * alpha - beta_Sx  # (m,1)

        T = Sxy[None, :] - alpha * Sx[None, :] - (beta @ Sxx)  # (m,K)

        beta_Sxy = (beta * Sxy[None, :]).sum(axis=1, keepdims=True)  # (m,1)
        beta_Sx = (beta * Sx[None, :]).sum(axis=1, keepdims=True)  # (m,1)
        quad = np.einsum('mi,ij,mj->m', beta, Sxx, beta).reshape(-1, 1)  # (m,1)

        Q = (Syy
             - 2 * alpha * Sy
             - 2 * beta_Sxy
             + n * alpha ** 2
             + 2 * alpha * beta_Sx
             + quad)  # (m,1)
        return S, T, Q

    def grad_log_pdf(self, theta: ArrayLike, *_ignored) -> np.ndarray:
        self._ensure()
        th = np.asarray(theta, float)
        if th.ndim == 1:
            th = th[None, :]
        K = self.K

        alpha = th[:, [0]]  # (m,1)
        beta = th[:, 1:1 + K]  # (m,K)
        gamma = th[:, [1 + K]]  # (m,1)

        S, T, Q = self._S_T_Q(alpha, beta)
        sigma = gamma
        sigma2 = sigma * sigma  # (m,1)
        g_alpha = S / sigma2  # (m,1)
        g_beta = T / sigma2  # (m,K)  (row-wise divide)
        g_gamma = -self.n / sigma + Q / (sigma * sigma2)  # (m,1)

        return np.concatenate([g_alpha, g_beta, g_gamma], axis=1)  # (m, 2+K)


class GaussianLogLikelihoodWithGivenGrads(BaseLoss):
    """
    Gaussian log-likelihood with given grads for the posterior samples.
    """

    def __init__(self):
        self.grad_log_likelihood = None

    def grad_log_pdf(self) -> np.ndarray:
        return self.grad_log_likelihood
