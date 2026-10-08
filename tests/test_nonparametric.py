import numpy as np
import pytest

from src.nonparametric.basis_functions import MaternBasisFunction
from src.nonparametric.node_sensitivity import _ac_whitening_transform
from src.nonparametric.fisher_divergence import PosteriorFDNonParametric, PriorFDNonParametric
from src.nonparametric.optimization import OptimisationNonparametricBase
from src.nonparametric.sensitivity import nonparametric_prior_sensitivity

from tests.conftest import NOISE_SD, PRIOR_MU, PRIOR_SD

BASIS_KWARGS = {"num_basis_functions": 30, "method": "kmeans", "nu": 3.5}


def _optimizer(model, radius):
    np.random.seed(0)
    basis = MaternBasisFunction(
        posterior_samples=model.posterior_samples_init,
        prior_samples=model.sample_from_base_prior(len(model.prior_samples_init)),
        **BASIS_KWARGS,
    )
    return OptimisationNonparametricBase(
        PosteriorFDNonParametric(model), PriorFDNonParametric(model), config={},
        radius=radius, basis_function=basis,
    )


def test_quadratic_forms_are_positive_semidefinite(gaussian_location):
    model, _ = gaussian_location
    optimizer = _optimizer(model, radius=1.0)
    assert np.linalg.eigvalsh(optimizer.A).min() > -1e-10
    assert np.linalg.eigvalsh(optimizer.A_c).min() > -1e-10


def test_generalised_eigenvalue_solution_is_the_constrained_maximum(gaussian_location):
    model, _ = gaussian_location
    optimizer = _optimizer(model, radius=1.0)
    res = optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)
    lam = res["lambda_star"]
    assert lam @ optimizer.A_c @ lam == pytest.approx(1.0, rel=1e-6)
    assert lam @ optimizer.A @ lam == pytest.approx(res["theoretical_value"], rel=1e-6)

    # No feasible direction in A_c's well-conditioned subspace does better.
    W, _ = _ac_whitening_transform(optimizer.A_c, rel_tol=1e-8)
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = W @ rng.normal(size=W.shape[1])
        v = v / np.sqrt(v @ optimizer.A_c @ v)
        assert v @ optimizer.A @ v <= res["theoretical_value"] * (1 + 1e-6)


def test_sensitivity_is_linear_in_radius(gaussian_location):
    model, _ = gaussian_location
    np.random.seed(0)
    small = nonparametric_prior_sensitivity(model, radius=1.0, basis_kwargs=BASIS_KWARGS)
    np.random.seed(0)
    large = nonparametric_prior_sensitivity(model, radius=3.0, basis_kwargs=BASIS_KWARGS)
    assert large.sensitivity == pytest.approx(3.0 * small.sensitivity, rel=1e-8)
    assert small.normalised_sensitivity == pytest.approx(small.sensitivity)


def test_sieve_stays_below_closed_form_sensitivity(gaussian_location):
    model, y = gaussian_location
    np.random.seed(0)
    result = nonparametric_prior_sensitivity(model, radius=1.0, basis_kwargs=BASIS_KWARGS)
    var_lik = NOISE_SD ** 2 / len(y)
    # ||l||_inf / Z_ref for the conjugate Gaussian location model (Theorem 1).
    m_closed = np.sqrt((var_lik + PRIOR_SD ** 2) / var_lik) * np.exp(
        0.5 * (y.mean() - PRIOR_MU) ** 2 / (var_lik + PRIOR_SD ** 2)
    )
    assert 0.0 < result.normalised_sensitivity <= 1.05 * m_closed


def test_independent_components_sum_to_total(two_independent_locations):
    np.random.seed(0)
    result = nonparametric_prior_sensitivity(
        two_independent_locations, radius=0.5, basis_kwargs=BASIS_KWARGS,
        independent=True, component_names=["a", "b"],
    )
    total = sum(c["sensitivity"] for c in result.components.values())
    assert result.sensitivity == pytest.approx(total)
    assert result.radius == pytest.approx(1.0)
    assert sum(c["sensitivity_share"] for c in result.components.values()) == pytest.approx(1.0)
