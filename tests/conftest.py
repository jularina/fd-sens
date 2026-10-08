import numpy as np
import pytest

from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.distributions.composite import CompositeProduct
from src.common.distributions.gaussian import Gaussian

PRIOR_MU, PRIOR_SD, NOISE_SD = 2.0, 4.0, 2.0


def conjugate_posterior(y: np.ndarray, rng: np.random.Generator, m: int) -> np.ndarray:
    """Draw m samples from the conjugate posterior of a Gaussian mean under the N(2, 4^2) prior."""
    var_n = 1.0 / (len(y) / NOISE_SD ** 2 + 1.0 / PRIOR_SD ** 2)
    mean_n = var_n * (y.sum() / NOISE_SD ** 2 + PRIOR_MU / PRIOR_SD ** 2)
    return rng.normal(mean_n, np.sqrt(var_n), size=m)


@pytest.fixture(scope="module")
def gaussian_location():
    """Univariate Gaussian location model with exact reference-posterior samples."""
    rng = np.random.default_rng(0)
    y = rng.normal(3.0, NOISE_SD, size=100)
    posterior = conjugate_posterior(y, rng, 20000)
    model = PosteriorSamplesModel(
        posterior_samples=posterior,
        base_prior=Gaussian(mu=PRIOR_MU, sigma=PRIOR_SD),
        candidate_prior=Gaussian(mu=0.0, sigma=1.0),
        loss_grad=lambda theta: (len(y) * theta - y.sum()) / NOISE_SD ** 2,
    )
    return model, y


@pytest.fixture(scope="module")
def two_independent_locations():
    """Two independent Gaussian location models sharing one factorised reference prior."""
    rng = np.random.default_rng(1)
    columns = [conjugate_posterior(rng.normal(c, NOISE_SD, size=100), rng, 20000) for c in (3.0, -1.0)]
    model = PosteriorSamplesModel(
        posterior_samples=np.column_stack(columns),
        base_prior=CompositeProduct(distributions={
            "a": Gaussian(mu=PRIOR_MU, sigma=PRIOR_SD), "b": Gaussian(mu=PRIOR_MU, sigma=PRIOR_SD),
        }),
        candidate_prior=CompositeProduct(distributions={"a": Gaussian(0.0, 1.0), "b": Gaussian(0.0, 1.0)}),
    )
    return model
