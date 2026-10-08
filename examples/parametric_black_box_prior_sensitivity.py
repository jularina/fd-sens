"""FDsens prior sensitivity to a non-exponential-family candidate (Student-t) via black-box optimisation."""
import numpy as np

from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.distributions.gaussian import Gaussian
from src.parametric.sensitivity import prior_sensitivity_black_box

rng = np.random.default_rng(123)
y = rng.normal(loc=1.0, scale=1.0, size=30)
var_n = 1.0 / (len(y) + 1.0 / 2.0 ** 2)
posterior = rng.normal(var_n * y.sum(), np.sqrt(var_n), size=2000)

model = PosteriorSamplesModel(posterior_samples=posterior, base_prior=Gaussian(mu=0.0, sigma=2.0))


def student_t_score(draws: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Score of a Student-t prior with location lam[0], scale lam[1] and lam[2] degrees of freedom."""
    loc, scale, df = lam
    z = draws - loc
    return -(df + 1.0) * z / (df * scale ** 2 + z ** 2)


# Box on lambda = (location, scale, degrees of freedom), in the family's own parametrisation.
result = prior_sensitivity_black_box(
    model, student_t_score, lower=[-1.0, 1.0, 2.0], upper=[1.0, 3.0, 30.0], maxiter=200, n_restarts=3,
)
print(result)
