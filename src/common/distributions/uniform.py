import warnings
import numpy as np
from typing import Union
from .base import BaseDistribution


class Uniform(BaseDistribution):
    """
    Univariate Uniform distribution.

    Parameters
    ----------
    low : float or str
        Lower bound of the uniform distribution.
    high : float or str
        Upper bound of the uniform distribution.

    Notes
    -----
    If either bound is infinite, pdf/log_pdf/grad_log_pdf/grad_log_base_measure/sampling
    will not be available.
    """

    def __init__(self, low: Union[float, str], high: Union[float, str]):
        def parse_bound(b):
            if isinstance(b, str):
                if b == "inf":
                    return np.inf
                elif b == "-inf":
                    return -np.inf
                else:
                    return float(b)
            return float(b)

        self.low = parse_bound(low)
        self.high = parse_bound(high)

        assert self.low < self.high, "Lower bound must be less than upper bound."

        self._finite_bounds = np.isfinite(self.low) and np.isfinite(self.high)

        if not self._finite_bounds:
            warnings.warn(
                "At least one bound is infinite; sampling, pdf, log_pdf, "
                "grad_log_pdf, and grad_log_base_measure will not be available.",
                UserWarning
            )
        else:
            self._width = self.high - self.low
            self._log_pdf_const = -np.log(self._width)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        if not self._finite_bounds:
            raise NotImplementedError("Cannot sample from an infinite-bounds uniform.")
        return np.random.uniform(self.low, self.high, size=n_samples).reshape(-1, 1)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        if not self._finite_bounds:
            raise NotImplementedError("pdf is undefined for infinite bounds.")
        x = np.asarray(x)
        return np.where((x >= self.low) & (x <= self.high), 1.0 / self._width, 0.0)

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        if not self._finite_bounds:
            raise NotImplementedError("log_pdf is undefined for infinite bounds.")
        x = np.asarray(x)
        return np.where((x >= self.low) & (x <= self.high), self._log_pdf_const, -np.inf)

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        if not self._finite_bounds:
            raise NotImplementedError("grad_log_pdf is undefined for infinite bounds.")
        x = np.asarray(x)
        grad = np.zeros_like(x, dtype=float)
        # grad[(x < self.low) | (x > self.high)] = np.nan
        return grad

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        if not self._finite_bounds:
            raise NotImplementedError("grad_log_base_measure is undefined for infinite bounds.")
        return np.zeros_like(x, dtype=float)

    def natural_parameters(self) -> np.ndarray:
        raise NotImplementedError("Uniform distribution is not in a standard exponential family form.")

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Uniform distribution has no standard sufficient statistic gradients.")

