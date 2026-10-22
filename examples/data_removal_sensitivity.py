"""FD sensitivity to removing at most 10% of the observations of a Gaussian location model with two outliers."""
import numpy as np

from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.data_removal.sensitivity import data_removal_sensitivity

rng = np.random.default_rng(123)
noise_sd, prior_mu, prior_sd = 1.0, 0.0, 2.0
y = np.concatenate([rng.normal(loc=1.0, scale=noise_sd, size=28), [5.0, 6.0]])   # observations 28 and 29 are outliers
var_n = 1.0 / (len(y) / noise_sd ** 2 + 1.0 / prior_sd ** 2)
posterior = rng.normal(var_n * (y.sum() / noise_sd ** 2 + prior_mu / prior_sd ** 2), np.sqrt(var_n), size=4000)


def data_grad(theta: np.ndarray) -> np.ndarray:
    """Per-observation log-likelihood gradients (y_i - theta) / sigma^2 at each draw, shape (m, n, 1)."""
    return ((y[None, :] - theta) / noise_sd ** 2)[:, :, None]


model = PosteriorSamplesModel(posterior_samples=posterior, data_grad=data_grad)   # the prior cancels: not needed

result = data_removal_sensitivity(model, alpha=0.1)   # remove at most floor(0.1 * 30) = 3 observations
print(result)
# The worst-case removals of size 1 and 2 are the outliers; result.A[i, i] is the FD of removing y_i alone
