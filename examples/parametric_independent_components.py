"""FDsens prior sensitivity decomposed over independent prior components (mean and scale of a Gaussian)."""
import numpy as np

from interpretation.plots import plot_component_shares, save_sensitivity_result
from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.distributions.composite import CompositeProduct
from src.common.distributions.gamma import Gamma
from src.common.distributions.gaussian import Gaussian
from src.parametric.sensitivity import prior_sensitivity

rng = np.random.default_rng(1)
y = rng.normal(loc=1.0, scale=1.5, size=50)
base_prior = CompositeProduct(distributions={"mu": Gaussian(mu=0.0, sigma=5.0), "sigma": Gamma(alpha=2.0, theta=1.0)})


def log_posterior(theta: np.ndarray) -> float:
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf
    log_lik = -len(y) * np.log(sigma) - 0.5 * np.sum((y - mu) ** 2) / sigma ** 2
    return log_lik - 0.5 * mu ** 2 / 25.0 + np.log(sigma) - sigma


# A short random-walk Metropolis run stands in for your sampler of choice.
theta, current, draws = np.array([0.0, 1.0]), None, []
current = log_posterior(theta)
for i in range(30000):
    proposal = theta + rng.normal(scale=0.2, size=2)
    candidate = log_posterior(proposal)
    if np.log(rng.uniform()) < candidate - current:
        theta, current = proposal, candidate
    if i >= 5000 and i % 5 == 0:
        draws.append(theta.copy())
posterior = np.array(draws)  # columns in the same order as the prior components: (mu, sigma)

model = PosteriorSamplesModel(
    posterior_samples=posterior,
    base_prior=base_prior,
    candidate_prior=CompositeProduct(distributions={"mu": Gaussian(0.0, 1.0), "sigma": Gamma(1.0, 1.0)}),
)

# Gaussian: eta = (mu / s^2, -1 / (2 s^2)); Gamma(alpha, theta): eta = (alpha - 1, -1 / theta).
natural_box = {
    "mu": {"eta_1": [-0.5, 0.5], "eta_2": [-0.5, -0.01]},
    "sigma": {"eta_1": [0.5, 2.0], "eta_2": [-2.0, -0.5]},
}
result = prior_sensitivity(model, natural_box, independent=True)
print(result)

output_dir = "interpretation/output/independent_components"
save_sensitivity_result(result, output_dir)
plot_component_shares(result, output_dir)
print("Wrote the result and component-share plot to", output_dir)
