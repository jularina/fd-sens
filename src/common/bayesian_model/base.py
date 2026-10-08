import copy
from abc import ABC
from typing import Any, Dict, Type
import numpy as np
import os
import warnings
from hydra.utils import get_original_cwd
from typing import Optional

from src.common.utils.typing import ArrayLike
from src.common.utils.files_operations import load_numpy_array


def _resolve_path(path: str) -> str:
    """Resolve a relative path against the launch directory, inside or outside a Hydra run."""
    if os.path.isabs(path):
        return path
    try:
        root = get_original_cwd()
    except ValueError:
        root = os.getcwd()
    return os.path.join(root, path)


class BayesianModel(ABC):
    def __init__(self, data_config: Any):
        """
        Base class for Bayesian models.
        """
        self.true_dgp = data_config.true_dgp
        self.loss_lr: float = data_config.loss_lr
        self.loss: Any = data_config.loss
        self.prior_candidate: Any = getattr(data_config, "candidate_prior", None)
        self.prior: Any = self.prior_candidate
        self.prior_init: Any = data_config.base_prior
        self.loss_lr_init: float = data_config.loss_lr_init
        self.m: int = data_config.posterior_samples_num
        self.m_prior: int = data_config.prior_samples_num


    def back_to_prior_candidate(self, *, deep: bool = True):
        """Reset the current prior to the candidate prior (deep-copied by default) and return self."""
        self.prior = copy.deepcopy(self.prior_candidate) if deep else self.prior_candidate
        return self

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Draw samples from the posterior distribution.
        """
        pass

    def sample_from_base_prior(self, n_samples: int = 1000) -> np.ndarray:
        """Draw samples from the reference prior."""
        return self.prior_init.sample(n_samples)

    def set_prior_parameters(self, params: Dict[str, Any], distribution_cls: Type) -> None:
        """Set the prior by instantiating distribution_cls with the given parameters."""
        self.prior = distribution_cls(**params)

    def set_candidate_prior_parameters(self, params: Dict[str, Any], distribution_cls: Type) -> None:
        """Set the candidate prior by instantiating distribution_cls with the given parameters."""
        self.prior_candidate = distribution_cls(**params)

    def set_lr_parameter(self, lr: float) -> None:
        """Set the learning rate that scales the loss term."""
        self.loss_lr = lr


    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """Compute gradient of log likelihood (scaled by learning rate)."""
        grad = self.loss.grad_log_pdf(x, self.x_bar, self.observations_num)
        return self.loss_lr * grad if multiply_by_lr else grad


class BayesianModelExtended(BayesianModel):
    def __init__(self, data_config: Any):
        """
        Base class for Extended Bayesian models.
        """
        super().__init__(data_config)
        # Prepare observations
        self.observations = self._prepare_observations(data_config)
        self.observations_num = self.observations.shape[0]
        self.x_bar: np.ndarray = np.mean(self.observations, axis=0)
        self.posterior_samples_init = self._prepare_array_from_presaved_samples(
            getattr(data_config, "posterior_samples_path", None),
            name="posterior"
        )
        self.m = len(self.posterior_samples_init)
        self.prior_samples_init = self._prepare_array_from_presaved_samples(
            getattr(data_config, "prior_samples_path", None),
            name="prior"
        )
        try:
            self.m_prior = len(self.prior_samples_init)
        except:
            self.m_prior = None

    def _prepare_observations(self, data_config: Any) -> np.ndarray:
        """
        Prepare observations: either provided directly, loaded from file, or sampled from true_dgp.
        """
        obs = getattr(data_config, "observations", None)
        obs_path = getattr(data_config, "observations_path", None)

        if obs is not None and obs_path is not None:
            warnings.warn("Both observations and observations_path provided; using observations.")

        # Load from path if given
        if obs is None and obs_path is not None:
            obs = load_numpy_array(_resolve_path(obs_path))

        # Sample from true_dgp if no data provided
        if obs is None:
            if self.true_dgp is None:
                raise ValueError("Provide data.observations(_path) or set data.true_dgp to sample from.")
            observations_num = int(getattr(data_config, "observations_num", 0))
            if observations_num <= 0:
                raise ValueError("data.observations_num must be > 0 when sampling from true_dgp.")
            obs = self.true_dgp.sample(observations_num)

        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)  # enforce (n, d) shape
        return obs

    def _prepare_array_from_presaved_samples(self, path: Optional[str], name: str) -> Optional[np.ndarray]:
        """Load an array of samples from path, or sample from the posterior/prior if no path is given."""
        if path is None and name == "posterior":
            try:
                print("Posterior samples path not provided. Trying to sample from posterior.")
                arr = self.sample_posterior()
                return arr
            except:
                raise Exception("Was not able to sample from posterior.")
        elif path is None and name == "prior":
            try:
                print("Prior samples path not provided. Trying to sample from prior.")
                arr = self.sample_from_base_prior()
                return arr
            except:
                raise Exception("Was not able to sample from prior.")

        path = _resolve_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name.capitalize()} samples file not found: {path}")

        arr = load_numpy_array(path)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
