import numpy as np
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """
    Abstract base class for log-likelihood (loss) objects.
    """

    @abstractmethod
    def grad_log_pdf(
        self,
    ) -> np.ndarray:

        raise NotImplementedError