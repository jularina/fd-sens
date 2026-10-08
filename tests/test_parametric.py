import itertools

import numpy as np
import pytest

from src.parametric.fisher_divergence import PosteriorFDParametric
from src.parametric.sensitivity import lr_sensitivity, prior_sensitivity

from tests.conftest import NOISE_SD, PRIOR_MU, PRIOR_SD

BOX = {"theta": {"eta_1": [-1.0, 1.0], "eta_2": [-0.5, -0.02]}}


def _gaussian_from_eta(eta_1, eta_2):
    var = -0.5 / eta_2
    return eta_1 * var, var


def _closed_form_fd(eta, y):
    """Exact prior-only FD for a Gaussian candidate, using the analytic conjugate posterior moments."""
    var_n = 1.0 / (len(y) / NOISE_SD ** 2 + 1.0 / PRIOR_SD ** 2)
    mean_n = var_n * (y.sum() / NOISE_SD ** 2 + PRIOR_MU / PRIOR_SD ** 2)
    mu_c, var_c = _gaussian_from_eta(*eta)
    # Score difference a * theta + b, so E[(a theta + b)^2] follows from the posterior moments.
    a = 1.0 / var_c - 1.0 / PRIOR_SD ** 2
    b = PRIOR_MU / PRIOR_SD ** 2 - mu_c / var_c
    return a ** 2 * (var_n + mean_n ** 2) + 2 * a * b * mean_n + b ** 2


def test_fd_estimate_matches_closed_form(gaussian_location):
    model, y = gaussian_location
    estimator = PosteriorFDParametric(model)
    for eta in [(-1.0, -0.5), (0.5, -0.1), (0.125, -0.03125)]:
        assert estimator.fd_prior_only_given_eta(np.array(eta)) == pytest.approx(_closed_form_fd(eta, y), rel=0.02, abs=1e-6)


def test_quadratic_corner_maximum_matches_grid(gaussian_location):
    model, _ = gaussian_location
    result = prior_sensitivity(model, BOX)
    estimator = PosteriorFDParametric(model)
    grid = [
        estimator.fd_prior_only_given_eta(np.array([e1, e2]))
        for e1, e2 in itertools.product(np.linspace(-1.0, 1.0, 41), np.linspace(-0.5, -0.02, 41))
    ]
    assert result.fd_max == pytest.approx(max(grid), rel=1e-9)
    assert result.fd_min <= min(grid) + 1e-9
    assert result.sensitivity == pytest.approx(result.fd_max - result.fd_min)


def test_reference_prior_inside_box_gives_zero_minimum(gaussian_location):
    model, _ = gaussian_location
    result = prior_sensitivity(model, BOX)
    eta_ref = np.array([PRIOR_MU / PRIOR_SD ** 2, -0.5 / PRIOR_SD ** 2])
    assert result.fd_min == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(result.lambda_min, eta_ref, atol=1e-4)


def test_black_box_agrees_with_quadratic_route(gaussian_location):
    model, _ = gaussian_location
    quadratic = prior_sensitivity(model, BOX)
    black_box = prior_sensitivity(model, BOX, method="black_box", maxiter=100)
    assert black_box.sensitivity == pytest.approx(quadratic.sensitivity, rel=1e-3)


def test_independent_decomposition_matches_joint(two_independent_locations):
    box = {name: BOX["theta"] for name in ("a", "b")}
    joint = prior_sensitivity(two_independent_locations, box)
    split = prior_sensitivity(two_independent_locations, box, independent=True)
    assert split.sensitivity == pytest.approx(joint.sensitivity, rel=1e-8)
    assert sum(c["sensitivity_share"] for c in split.components.values()) == pytest.approx(1.0)


def test_natural_box_must_match_dimension(two_independent_locations):
    with pytest.raises(ValueError):
        prior_sensitivity(two_independent_locations, BOX)


def test_lr_sensitivity_matches_closed_form(gaussian_location):
    model, y = gaussian_location
    result = lr_sensitivity(model, lower=0.5, upper=1.5)
    theta = model.posterior_samples_init[:, 0]
    mean_sq_grad = np.mean(((len(y) * theta - y.sum()) / NOISE_SD ** 2) ** 2)
    assert result.sensitivity == pytest.approx(0.25 * mean_sq_grad)
    assert result.fd_min == 0.0
    assert result.lambda_min[0] == 1.0


def test_lr_sensitivity_reference_outside_interval(gaussian_location):
    model, _ = gaussian_location
    result = lr_sensitivity(model, lower=1.2, upper=2.0)
    assert result.lambda_min[0] == 1.2
    assert result.lambda_max[0] == 2.0
    assert result.fd_min > 0.0
