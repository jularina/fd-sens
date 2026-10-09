"""FDsens prior sensitivity for a Gaussian location model, starting from your own posterior samples."""
import numpy as np

from interpretation.plots import plot_ecdf, plot_kde, plot_quantiles, save_sensitivity_result
from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.distributions.gaussian import Gaussian
from src.parametric.sensitivity import prior_sensitivity

rng = np.random.default_rng(123)
y = rng.normal(loc=1.0, scale=1.0, size=30)

# Reference model: theta ~ N(0, 2^2), y_i ~ N(theta, 1). Use draws from any sampler (Stan, PyMC, NumPyro, ...);
# here the conjugate posterior is sampled exactly.
var_n = 1.0 / (len(y) + 1.0 / 2.0 ** 2)
posterior = rng.normal(var_n * y.sum(), np.sqrt(var_n), size=2000)

model = PosteriorSamplesModel(
    posterior_samples=posterior,
    base_prior=Gaussian(mu=0.0, sigma=2.0),        # reference prior
    candidate_prior=Gaussian(mu=0.0, sigma=1.0),   # candidate family; only the family matters
)

# Candidates N(mu, s^2) with natural parameters eta_1 = mu / s^2 and eta_2 = -1 / (2 s^2).
result = prior_sensitivity(model, natural_box={"theta": {"eta_1": [-1.0, 1.0], "eta_2": [-2.0, -0.05]}})
print(result)

mu_max, var_max = result.lambda_max[0] * (-0.5 / result.lambda_max[1]), -0.5 / result.lambda_max[1]
print(f"Worst-case prior: N({mu_max:.3f}, {np.sqrt(var_max):.3f}^2)")

# Refit under the worst-case prior (conjugate here, so sampled exactly) and compare it with the reference posterior.
var_c = 1.0 / (len(y) + 1.0 / var_max)
candidate_posterior = rng.normal(var_c * (y.sum() + mu_max / var_max), np.sqrt(var_c), size=2000)

draws = {"reference": posterior, "candidate": candidate_posterior}
output_dir = "interpretation/output/gaussian_location_prior"
save_sensitivity_result(result, output_dir)
print(plot_quantiles(draws, output_dir, variables=["theta"]))
plot_kde(draws, output_dir, variables=["theta"])
plot_ecdf(draws, output_dir, variables=["theta"])
print("Wrote the result and plots to", output_dir)
