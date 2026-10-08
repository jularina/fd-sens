"""FDsens learning-rate sensitivity for a Gaussian location model."""
import numpy as np

from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.distributions.gaussian import Gaussian
from src.parametric.sensitivity import lr_sensitivity

rng = np.random.default_rng(123)
y = rng.normal(loc=1.0, scale=1.0, size=30)
var_n = 1.0 / (len(y) + 1.0 / 2.0 ** 2)
posterior = rng.normal(var_n * y.sum(), np.sqrt(var_n), size=2000)


def loss_grad(theta: np.ndarray) -> np.ndarray:
    """Gradient of the unscaled loss 0.5 * sum((y - theta)^2) at each posterior draw."""
    return len(y) * theta - y.sum()


model = PosteriorSamplesModel(
    posterior_samples=posterior,
    base_prior=Gaussian(mu=0.0, sigma=2.0),
    loss_grad=loss_grad,
    loss_lr=1.0,                                   # reference learning rate
)

print(lr_sensitivity(model, lower=0.5, upper=1.5))
