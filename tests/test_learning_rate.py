import numpy as np
import pytest

from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.learning_rate.sensitivity import lr_sensitivity

from tests.conftest import NOISE_SD


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


def test_lr_sensitivity_needs_no_prior(gaussian_location):
    model, _ = gaussian_location
    no_prior = PosteriorSamplesModel(posterior_samples=model.posterior_samples_init, loss_grad=model._loss_grad)
    assert lr_sensitivity(no_prior, 0.5, 1.5).sensitivity == pytest.approx(lr_sensitivity(model, 0.5, 1.5).sensitivity)
