from typing import Any, Callable, Optional, Union

import numpy as np

from src.common.distributions.composite import CompositeProduct
from src.common.utils.typing import ArrayLike


def _as_product(dist: Any) -> Any:
    """Wrap a single distribution as a one-component CompositeProduct, leaving products unchanged."""
    return dist if isinstance(dist, CompositeProduct) else CompositeProduct(distributions={"theta": dist})


class PosteriorSamplesModel:
    """Reference model given directly by reference-posterior samples, a reference prior, an optional loss gradient and
    optional per-observation log-likelihood gradients."""

    def __init__(
        self,
        posterior_samples: ArrayLike,
        base_prior: Optional[Any] = None,
        candidate_prior: Optional[Any] = None,
        loss_grad: Optional[Union[ArrayLike, Callable[[np.ndarray], np.ndarray]]] = None,
        loss_lr: float = 1.0,
        prior_samples: Optional[ArrayLike] = None,
        data_grad: Optional[Union[ArrayLike, Callable[[np.ndarray], np.ndarray]]] = None,
    ):
        """Store the reference-posterior draws, priors, loss gradient, (optional) reference-prior draws and
        (optional) per-observation log-likelihood gradients."""
        samples = np.asarray(posterior_samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        self.posterior_samples_init = samples
        if samples.shape[1] == 1:
            # A single parameter is wrapped as a one-component product so every family has the same array shapes.
            base_prior = None if base_prior is None else _as_product(base_prior)
            candidate_prior = None if candidate_prior is None else _as_product(candidate_prior)
        self.prior_init = base_prior
        self.prior_candidate = candidate_prior
        self.prior = candidate_prior
        self.loss_lr_init = float(loss_lr)
        self.loss_lr = float(loss_lr)
        self._loss_grad = loss_grad
        self._data_grad = data_grad

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

    def data_score(self, x: ArrayLike) -> np.ndarray:
        """Return the per-observation log-likelihood gradients at x, shape (m, n, d): entry [j, i] is
        grad_theta log l(x_j; obs_i). Used by data-removal sensitivity; the observations may be groups of points."""
        if self._data_grad is None:
            raise ValueError("Data-removal sensitivity needs data_grad (per-observation log-likelihood gradients).")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        grad = self._data_grad(x) if callable(self._data_grad) else self._data_grad
        grad = np.asarray(grad, dtype=float)
        if grad.ndim == 2 and x.shape[1] == 1:
            grad = grad[:, :, None]  # (m, n) for a single parameter
        if grad.ndim != 3 or grad.shape[0] != x.shape[0] or grad.shape[2] != x.shape[1]:
            raise ValueError(
                f"data_grad must have shape (m, n, d) = ({x.shape[0]}, n, {x.shape[1]}); got {grad.shape}.")
        return grad
