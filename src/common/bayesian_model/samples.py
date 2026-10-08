from typing import Any, Callable, Optional, Union

import numpy as np

from src.common.distributions.composite import CompositeProduct
from src.common.utils.typing import ArrayLike


def _as_product(dist: Any) -> Any:
    """Wrap a single distribution as a one-component CompositeProduct, leaving products unchanged."""
    return dist if isinstance(dist, CompositeProduct) else CompositeProduct(distributions={"theta": dist})


class PosteriorSamplesModel:
    """Reference model given directly by reference-posterior samples, a reference prior and an optional loss gradient."""

    def __init__(
        self,
        posterior_samples: ArrayLike,
        base_prior: Any,
        candidate_prior: Optional[Any] = None,
        loss_grad: Optional[Union[ArrayLike, Callable[[np.ndarray], np.ndarray]]] = None,
        loss_lr: float = 1.0,
        prior_samples: Optional[ArrayLike] = None,
    ):
        """Store the reference-posterior draws, priors, loss gradient and (optional) reference-prior draws."""
        samples = np.asarray(posterior_samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        self.posterior_samples_init = samples
        if samples.shape[1] == 1:
            # A single parameter is wrapped as a one-component product so every family has the same array shapes.
            base_prior = _as_product(base_prior)
            candidate_prior = None if candidate_prior is None else _as_product(candidate_prior)
        self.prior_init = base_prior
        self.prior_candidate = candidate_prior
        self.prior = candidate_prior
        self.loss_lr_init = float(loss_lr)
        self.loss_lr = float(loss_lr)
        self._loss_grad = loss_grad

        if prior_samples is None and hasattr(base_prior, "sample"):
            prior_samples = base_prior.sample(len(samples))
        if prior_samples is not None:
            prior_samples = np.asarray(prior_samples, dtype=float)
            if prior_samples.ndim == 1:
                prior_samples = prior_samples.reshape(-1, 1)
        self.prior_samples_init = prior_samples

    def sample_from_base_prior(self, n_samples: int = 1000) -> np.ndarray:
        """Draw samples from the reference prior."""
        return np.asarray(self.prior_init.sample(n_samples), dtype=float).reshape(n_samples, -1)

    def set_lr_parameter(self, lr: float) -> None:
        """Set the learning rate that scales the loss term."""
        self.loss_lr = float(lr)

    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """Return the loss gradient at x (zeros if no loss gradient was given, as in prior-only analyses)."""
        x = np.asarray(x, dtype=float)
        if self._loss_grad is None:
            grad = np.zeros_like(x)
        elif callable(self._loss_grad):
            grad = np.asarray(self._loss_grad(x), dtype=float).reshape(x.shape)
        else:
            grad = np.asarray(self._loss_grad, dtype=float).reshape(x.shape)
        return self.loss_lr * grad if multiply_by_lr else grad
