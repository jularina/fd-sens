from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class BaseDistribution(ABC):
    """
    Abstract base class for distributions.
    """

    @abstractmethod
    def sample(self, n_samples: int = 1) -> np.ndarray:
        pass

    @abstractmethod
    def pdf(self, x: ArrayLike) -> np.ndarray:
        pass

    @abstractmethod
    def log_pdf(self, x: ArrayLike) -> np.ndarray:
        pass

    @abstractmethod
    def grad_log_pdf(self, x: ArrayLike) -> np.ndarray:
        pass

    @abstractmethod
    def grad_log_base_measure(self, x: ArrayLike) -> np.ndarray:
        pass

    @abstractmethod
    def natural_parameters(self) -> np.ndarray:
        pass

    @abstractmethod
    def grad_sufficient_statistics(self, x: ArrayLike) -> np.ndarray:
        pass

    def augmented_natural_parameters(self) -> np.ndarray:
        return np.append(self.natural_parameters(), 1.0)
