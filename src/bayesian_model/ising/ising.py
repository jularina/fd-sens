import numpy as np

from src.bayesian_model.base import BayesianModelExtended


class IsingBayesianModel(BayesianModelExtended):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.likelihood_grads = self._prepare_array_from_presaved_samples(
            getattr(data_config, "pseudoliklelhood_grads_path", None),
            name="pseudoliklelhood_grads"
        )
        self.loss.grad_log_likelihood = self.likelihood_grads

    def loss_score(self, x: np.ndarray, multiply_by_lr: bool = True) -> np.ndarray:
        grad = self.loss.grad_log_pdf()
        return self.loss_lr * grad if multiply_by_lr else grad

