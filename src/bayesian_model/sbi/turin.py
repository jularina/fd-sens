import copy
from abc import ABC
from typing import Any, Dict, Tuple
import numpy as np
import torch
from scipy.stats import norm

from src.distributions.composite import CompositeProduct
from src.utils.distributions import DISTRIBUTION_MAP
from src.utils.files_operations import instantiate_from_target_str
from src.utils.distributions import is_basedistribution_like
from src.losses.gaussian_log_likelihood import GaussianLogLikelihoodWithGivenGrads


class TurinBayesianModel(ABC):
    """
    Bayesian model for Turin radio propagation experiment.
    Parameters vector order used here: [theta_1, theta_2, theta_3, theta_4].
    """

    def __init__(self, data_config: Any):
        self.true_dgp = data_config.true_dgp
        self.loss_lr: float = data_config.loss_lr
        self.loss: GaussianLogLikelihoodWithGivenGrads = data_config.loss
        self.prior: Any = data_config.candidate_prior
        self.prior_init: Any = data_config.base_prior
        self.prior_candidate: Any = data_config.candidate_prior
        self.loss_lr_init: float = data_config.loss_lr
        self.archive_path = data_config.archive_path

        (self.observations, self.posterior_samples_init, self.likelihood_grads, self.prior_samples_init) = self._prepare_data()
        self.loss.grad_log_likelihood = self.likelihood_grads
        self.observations_num = self.observations.shape[0]
        self.x_bar: np.ndarray = np.mean(self.observations, axis=0).reshape(-1, 1)
        self.m = self.posterior_samples_init.shape[0]
        self.m_prior = len(self.prior_samples_init)

    def back_to_prior_init(self, *, deep: bool = True):
        """
        Reset the current prior to the initial/base prior.

        Args:
            deep: If True (default), use a deep copy so future mutations of
                  `self.prior` do not affect `self.prior_init`.
        Returns:
            self (for chaining)
        """
        self.prior = copy.deepcopy(self.prior_init) if deep else self.prior_init
        return self

    def back_to_prior_candidate(self, *, deep: bool = True):
        """
        Reset the current prior to the candidate prior.

        Args:
            deep: If True (default), use a deep copy so future mutations of
                  `self.prior` do not affect `self.prior_candidate`.
        Returns:
            self (for chaining)
        """
        self.prior = copy.deepcopy(self.prior_candidate) if deep else self.prior_candidate
        return self

    @staticmethod
    def _theta_to_z_probit(
        theta: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        eps: float = 1e-9,
    ) -> np.ndarray:
        """
        Componentwise probit transform from theta in (lower, upper) to z in R^d.
        """
        theta = np.asarray(theta, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        denom = upper - lower
        if np.any(denom <= 0):
            raise ValueError("All entries of (upper - lower) must be positive.")

        u = (theta - lower) / denom
        u = np.clip(u, eps, 1.0 - eps)
        z = norm.ppf(u)
        return z

    @staticmethod
    def _theta_to_uniform_0_1(
        theta: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        eps: float = 1e-9,
    ) -> np.ndarray:
        """
        Componentwise transform from theta in (lower, upper) to U(0,1).
        """
        theta = np.asarray(theta, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        denom = upper - lower
        if np.any(denom <= 0):
            raise ValueError("All entries of (upper - lower) must be positive.")

        u = (theta - lower) / denom
        u = np.clip(u, eps, 1.0 - eps)

        return u

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ckpt = torch.load(self.archive_path, map_location="cpu")

        lower = np.array([c.low for c in self.prior_init.components], dtype=np.float64)
        upper = np.array([c.high for c in self.prior_init.components], dtype=np.float64)
        prior_samples = ckpt["theta_nle"].cpu().numpy()
        posterior_samples = ckpt["posterior_samples_z"].cpu().numpy()  # ckpt["posterior_samples_nle"].cpu().numpy()
        # prior_samples = np.clip(prior_samples, lower, upper)
        # posterior_samples = np.clip(posterior_samples, lower, upper)

        observations = ckpt["obs_x"].cpu().numpy()
        likelihood_grads = ckpt["likelihood_grads"].cpu().numpy()

        return observations, posterior_samples, likelihood_grads, prior_samples

    def sample_from_base_prior(self, n_samples: int = 1000) -> np.ndarray:
        return self.prior_init.sample(n_samples)

    def set_composite_prior_parameters(self, components: Dict[str, Any], combine_rule: str = "product",
                                       reset_from_init: bool = True, ) -> None:
        if combine_rule != "product":
            raise NotImplementedError("Only product composites are supported right now.")

        base = self.prior_init if reset_from_init else self.prior

        if not isinstance(base, CompositeProduct):
            raise TypeError("Expected CompositeProduct as base prior.")

        base_names = list(base.names)
        base_map = {n: copy.deepcopy(c) for n, c in zip(base.names, base.components)}
        new_map: Dict[str, Any] = dict(base_map)

        for name, spec in components.items():
            if is_basedistribution_like(spec):
                new_map[name] = spec
                continue

            if isinstance(spec, dict) and "_target_" in spec:
                kwargs = {k: v for k, v in spec.items() if k != "_target_"}
                new_map[name] = instantiate_from_target_str(spec["_target_"], kwargs)
                continue

            if isinstance(spec, dict) and "family" in spec:
                fam = spec["family"]
                params = spec.get("params", {k: v for k, v in spec.items() if k != "family"})
                cls = DISTRIBUTION_MAP.get(fam)
                if cls is None:
                    raise ValueError(f"Unknown family '{fam}'. Available: {list(DISTRIBUTION_MAP.keys())}")
                new_map[name] = cls(**params)
                continue

            if isinstance(spec, dict) and name in base_map:
                cls = base_map[name].__class__
                new_map[name] = cls(**spec)
                continue

            raise ValueError(
                f"Component '{name}' must be a BaseDistribution instance, "
                f"a dict with '_target_', a dict with 'family'/params, or bare params matching a base component."
            )

        ordered_map: Dict[str, Any] = {n: new_map[n] for n in base_names if n in new_map}
        for n, v in new_map.items():
            if n not in ordered_map:
                ordered_map[n] = v

        self.prior = CompositeProduct(distributions=ordered_map)

    def set_lr_parameter(self, lr: float) -> None:
        self.loss_lr = lr

    def prior_score(self, x: np.ndarray) -> np.ndarray:
        return self.prior.grad_log_pdf(x)

    def reference_prior_score(self, x) -> np.ndarray:
        """Compute gradient of reference log prior."""
        return self.prior_init.grad_log_pdf(x)

    def loss_score(self, x: np.ndarray, multiply_by_lr: bool = True) -> np.ndarray:
        grad = self.loss.grad_log_pdf()
        return self.loss_lr * grad if multiply_by_lr else grad

    def reference_loss_score(self, x: np.ndarray, multiply_by_lr: bool = True) -> np.ndarray:
        """Compute gradient of reference log loss."""
        grad = self.loss.grad_log_pdf()
        return self.loss_lr_init * grad if multiply_by_lr else grad

    def posterior_score(self, x: np.ndarray) -> np.ndarray:
        prior = self.prior_score(x)
        loss = self.loss_score(x)
        return prior + loss

    def jacobian_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        return self.prior.grad_sufficient_statistics(x)

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        return self.prior.grad_log_base_measure(x)

    def reference_prior_score(self, x) -> np.ndarray:
        """Compute gradient of reference log prior."""
        return self.prior_init.grad_log_pdf(x)
