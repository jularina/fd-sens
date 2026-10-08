"""FDsens+ prior sensitivity over an FD ball around the reference prior for a Gaussian location model."""
import numpy as np

from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.distributions.gaussian import Gaussian
from src.nonparametric.sensitivity import nonparametric_prior_sensitivity

np.random.seed(0)  # the basis centres and reference-prior draws use NumPy's global generator
rng = np.random.default_rng(123)
y = rng.normal(loc=3.0, scale=2.0, size=100)
var_n = 1.0 / (len(y) / 4.0 + 1.0 / 4.0 ** 2)
posterior = rng.normal(var_n * (y.sum() / 4.0 + 2.0 / 4.0 ** 2), np.sqrt(var_n), size=5000)

model = PosteriorSamplesModel(
    posterior_samples=posterior,
    base_prior=Gaussian(mu=2.0, sigma=4.0),        # reference prior; also used to draw prior samples
)

result = nonparametric_prior_sensitivity(
    model,
    radius=1.0,
    basis="MaternBasisFunction",
    basis_kwargs={"num_basis_functions": 100, "method": "kmeans", "nu": 3.5},
)
print(result)
# The worst-case prior is pi_ref(theta) * exp(sum_k lambda_k kappa(c_k, theta)) with lambda = result.lambda_sup
# and centres c_k = result.basis_function.centers; result.basis_function.evaluate(theta) gives the kappa terms.
