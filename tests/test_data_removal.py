import itertools

import numpy as np
import pytest

from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.data_removal.fisher_divergence import PosteriorFDDataRemoval
from src.data_removal.optimization import OptimizationCornerPointsDataRemoval, number_of_corner_points
from src.data_removal.sensitivity import data_removal_sensitivity, removal_budget

from tests.conftest import NOISE_SD, PRIOR_MU, PRIOR_SD, conjugate_posterior


@pytest.fixture(scope="module")
def gaussian_location_data():
    """Gaussian location model with per-observation scores (y_i - theta) / sigma^2."""
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.normal(3.0, NOISE_SD, size=10), [12.0]])
    posterior = conjugate_posterior(y, rng, 40000)
    model = PosteriorSamplesModel(
        posterior_samples=posterior,
        data_grad=lambda theta: ((y[None, :] - theta) / NOISE_SD ** 2)[:, :, None],
    )
    return model, y


def _closed_form_fd(removed, y):
    """E[(sum_{i in S} (y_i - theta))^2] / sigma^4 under the conjugate posterior."""
    var_n = 1.0 / (len(y) / NOISE_SD ** 2 + 1.0 / PRIOR_SD ** 2)
    mean_n = var_n * (y.sum() / NOISE_SD ** 2 + PRIOR_MU / PRIOR_SD ** 2)
    s, k = y[list(removed)].sum(), len(removed)
    return ((s - k * mean_n) ** 2 + k ** 2 * var_n) / NOISE_SD ** 4


def test_fd_estimate_matches_closed_form(gaussian_location_data):
    model, y = gaussian_location_data
    estimator = PosteriorFDDataRemoval(model)
    A = estimator.compute_fisher_quadratic_form()
    for removed in [(10,), (0, 3), (1, 4, 10)]:
        v = np.zeros(len(y))
        v[list(removed)] = 1.0
        assert estimator.fd_given_removal(removed) == pytest.approx(_closed_form_fd(removed, y), rel=0.02)
        assert v @ A @ v == pytest.approx(estimator.fd_given_removal(removed), rel=1e-10)


def test_corner_points_match_brute_force():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(9, 4))
    A = B @ B.T
    A[0, 1] = A[1, 0] = -2.0  # negative off-diagonals are allowed
    K = 4
    per_size, everything = OptimizationCornerPointsDataRemoval(A, chunk_size=7).evaluate_all_corners(K, keep_all=True)
    for k in range(1, K + 1):
        vals = [A[np.ix_(s, s)].sum() for s in itertools.combinations(range(9), k)]
        assert per_size[k]["fd"] == pytest.approx(max(vals))
        assert A[np.ix_(per_size[k]["removed"], per_size[k]["removed"])].sum() == pytest.approx(max(vals))
        assert len(everything[k - 1][1]) == len(vals)


def test_sensitivity_is_max_over_sizes_and_picks_outlier(gaussian_location_data):
    model, _ = gaussian_location_data
    result = data_removal_sensitivity(model, alpha=0.3)
    assert result.max_removed == 3
    assert result.n_evaluated == number_of_corner_points(11, 3)
    assert result.sensitivity == pytest.approx(max(e["fd"] for e in result.per_size.values()))
    assert list(result.per_size[1]["removed"]) == [10]


def test_groups_sum_observation_scores(gaussian_location_data):
    model, _ = gaussian_location_data
    groups = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9], [10]]
    grouped = PosteriorFDDataRemoval(model, groups=groups)
    single = PosteriorFDDataRemoval(model)
    assert grouped.fd_given_removal([0, 3]) == pytest.approx(single.fd_given_removal([0, 1, 2, 10]))
    result = data_removal_sensitivity(model, max_removed=2, groups=groups, names=["a", "b", "c", "d"])
    assert result.n_units == 4


def test_removal_budget():
    assert removal_budget(100, alpha=0.29) == 29
    assert removal_budget(7, max_removed=6) == 6
    with pytest.raises(ValueError):
        removal_budget(7, alpha=0.5, max_removed=3)
    with pytest.raises(ValueError):
        removal_budget(7)
