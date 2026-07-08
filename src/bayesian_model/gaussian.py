import copy
from typing import Dict, Any

import numpy as np

from bayesian_model.base import BayesianModelExtended
from src.distributions.composite import CompositeProduct
from src.utils.distributions import DISTRIBUTION_MAP
from src.utils.files_operations import instantiate_from_target_str
from src.utils.distributions import is_basedistribution_like


class SimpleGaussianModel(BayesianModelExtended):
    """
    Univariate Gaussian likelihood with Gaussian or LogNormal prior.
    """

    def __init__(self, data_config):
        super().__init__(data_config)

    def compute_posterior_params(self) -> tuple:
        """
        Compute the conjugate posterior Normal(mu_n, sigma_n^2) w.r.t. the reference
        prior (prior_init), and cache mu_n/sigma_n2 on the model.
        """
        sigma_n2 = 1 / (self.observations_num / self.loss.var + 1 / self.prior_init.var)
        mu_n = sigma_n2 * (self.observations_num * self.x_bar / self.loss.var +
                           self.prior_init.mu / self.prior_init.var)

        self.mu_n = mu_n
        self.sigma_n2 = sigma_n2
        return mu_n, sigma_n2

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Sample from conjugate posterior: Normal(mu_n, sigma_n^2).
        """
        mu_n, sigma_n2 = self.compute_posterior_params()
        sigma_n = np.sqrt(sigma_n2)

        return np.random.normal(mu_n, sigma_n, size=(n_samples, 1))


class MultivariateGaussianModel(BayesianModelExtended):
    """
    Multivariate Gaussian likelihood with Gaussian prior on the mean.
    """

    def __init__(self, data_config):
        super().__init__(data_config)
        self.dim = self.observations.shape[1]

    def compute_posterior_params(self) -> tuple:
        """
        Compute the closed-form posterior for the mean with known covariance,
        w.r.t. the reference prior (prior_init), and cache mu_n/Sigma_n on the model.
        """
        mu0, Sigma0 = self.prior_init.mu, self.prior_init.cov
        Sigma_obs = self.loss.cov

        Sigma_obs_inv = np.linalg.inv(Sigma_obs)
        Sigma0_inv = np.linalg.inv(Sigma0)

        Sigma_n_inv = self.observations_num * Sigma_obs_inv + Sigma0_inv
        Sigma_n = np.linalg.inv(Sigma_n_inv)

        mu_n = Sigma_n @ (self.observations_num * Sigma_obs_inv @ self.x_bar + Sigma0_inv @ mu0)

        self.mu_n = mu_n
        self.Sigma_n = Sigma_n
        return mu_n, Sigma_n

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Closed-form posterior for mean with known covariance.
        """
        mu_n, Sigma_n = self.compute_posterior_params()
        return np.random.multivariate_normal(mu_n, Sigma_n, size=n_samples)

    def set_composite_prior_parameters(self, components: Dict[str, Any], combine_rule: str = "product", reset_from_init: bool = True,) -> None:
        if combine_rule != "product":
            raise NotImplementedError("Only product composites are supported right now.")

        # --- choose a clean base to overlay on ---
        base = self.prior_init if reset_from_init else self.prior

        if not isinstance(base, CompositeProduct):
            raise TypeError("Expected CompositeProduct as base prior.")

        # Deep-copy base to avoid mutating prior_init
        base_names = list(base.names)
        base_map = {n: copy.deepcopy(c) for n, c in zip(base.names, base.components)}

        # Start from the clean base and overlay user-specified components
        new_map: Dict[str, Any] = dict(base_map)

        for name, spec in components.items():
            # 1) already-instantiated distribution
            if is_basedistribution_like(spec):
                new_map[name] = spec
                continue

            # 2) hydra-style instantiation
            if isinstance(spec, dict) and "_target_" in spec:
                kwargs = {k: v for k, v in spec.items() if k != "_target_"}
                new_map[name] = instantiate_from_target_str(spec["_target_"], kwargs)
                continue

            # 3) family/params shorthand
            if isinstance(spec, dict) and "family" in spec:
                fam = spec["family"]
                params = spec.get("params", {k: v for k, v in spec.items() if k != "family"})
                cls = DISTRIBUTION_MAP.get(fam)
                if cls is None:
                    raise ValueError(f"Unknown family '{fam}'. Available: {list(DISTRIBUTION_MAP.keys())}")
                new_map[name] = cls(**params)
                continue

            # 4) bare params: reuse the *base* component's class (not the current self.prior)
            if isinstance(spec, dict) and name in base_map:
                cls = base_map[name].__class__
                new_map[name] = cls(**spec)
                continue

            raise ValueError(
                f"Component '{name}' must be a BaseDistribution instance, "
                f"a dict with '_target_', a dict with 'family'/params, or bare params matching a base component."
            )

        # Preserve base ordering; append any truly new names at the end
        ordered_map: Dict[str, Any] = {n: new_map[n] for n in base_names if n in new_map}
        for n, v in new_map.items():
            if n not in ordered_map:
                ordered_map[n] = v

        # Finalize
        self.prior = CompositeProduct(distributions=ordered_map)
