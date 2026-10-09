from abc import ABC
from typing import Any, Tuple
import numpy as np
import torch

from paper.common.losses.gaussian_log_likelihood import GaussianLogLikelihoodWithGivenGrads


class TurinBayesianModel(ABC):
    """Bayesian model for the Turin radio propagation experiment with parameters [theta_1, ..., theta_4]."""

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

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ckpt = torch.load(self.archive_path, map_location="cpu")

        prior_samples = ckpt["theta_nle"].cpu().numpy()
        posterior_samples = ckpt["posterior_samples_z"].cpu().numpy()  # ckpt["posterior_samples_nle"].cpu().numpy()

        observations = ckpt["obs_x"].cpu().numpy()
        likelihood_grads = ckpt["likelihood_grads"].cpu().numpy()

        return observations, posterior_samples, likelihood_grads, prior_samples

    def loss_score(self, x: np.ndarray, multiply_by_lr: bool = True) -> np.ndarray:
        grad = self.loss.grad_log_pdf()
        return self.loss_lr * grad if multiply_by_lr else grad
