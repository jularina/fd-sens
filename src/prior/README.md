# Prior sensitivity (FDsens and FDsens+)

Global sensitivity of a posterior to the choice of prior, measured by the Fisher divergence (FD) between the reference
posterior and the posterior under a candidate prior. The likelihood is shared, so its score cancels and the FD is
the expected squared difference of the prior scores under the reference posterior.

- **Parametric (FDsens)**: candidate priors within an exponential family, over a box of natural parameters.
  Methodology: [*A computationally-tractable measure of global sensitivity for sampling-based Bayesian
  inference*](https://arxiv.org/abs/2605.28099).
- **Nonparametric (FDsens+)**: all priors within an FD ball around the reference prior, approximated by a kernel
  exponential family sieve and solved exactly through a generalised eigenvalue problem.

See [`GETTING_STARTED.md`](../../GETTING_STARTED.md) for the method.

## Usage

```python
import numpy as np
from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.distributions.gaussian import Gaussian
from src.prior.parametric.sensitivity import prior_sensitivity

rng = np.random.default_rng(123)
y = rng.normal(loc=1.0, scale=1.0, size=30)
var_n = 1.0 / (len(y) + 1.0 / 2.0 ** 2)
posterior_samples = rng.normal(var_n * y.sum(), np.sqrt(var_n), size=2000)   # exact conjugate posterior draws

model = PosteriorSamplesModel(
    posterior_samples=posterior_samples,
    base_prior=Gaussian(mu=0.0, sigma=2.0),         # reference prior
    candidate_prior=Gaussian(mu=0.0, sigma=1.0),    # candidate family (its parameter values are not used)
)
# eta_1 = mu / s^2 and eta_2 = -1 / (2 s^2) are the natural parameters
result = prior_sensitivity(model, natural_box={"theta": {"eta_1": [-1.0, 1.0], "eta_2": [-2.0, -0.05]}})
print(result)
#> FD prior sensitivity
#>   optimisation: quadratic_corner
#>   sensitivity: 31.2854
#>   minimum FD:  0 at lambda_min = [ 0.    -0.125]
#>   maximum FD:  31.2854 at lambda_max = [-1. -2.]
```

`result.sensitivity` is the global sensitivity, `result.lambda_max` the worst-case prior that produces it.

The model needs `base_prior` (the reference prior). Any object with `grad_log_pdf(x) -> (m, d)` works, and FDsens+
additionally uses `sample(n)`. For several parameters with independent priors, use a `CompositeProduct` whose
components are in column order.

## Parametric prior sensitivity (FDsens)

### Exponential-family priors

```python
prior_sensitivity(model, natural_box, method="quadratic", independent=False, **black_box_kwargs)
```

- `model.candidate_prior`: the exponential-family candidate family.
- `natural_box`: one entry per parameter: `{name: {"eta_1": (lower, upper), "eta_2": (lower, upper)}}`,
  the box $\Gamma$ of candidate natural parameters.
- `method="quadratic"` (default): brute-force corner points optimisation.
- `method="black_box"`: the same exponential-family FD optimised globally (`scipy` dual annealing) instead of through
  the quadratic form; pass e.g. `maxiter=150, n_restarts=5`.
- `independent=True`: when the priors factorise over parameters, the sensitivity is the sum of per-parameter
  sensitivities. `result.components` holds each block's `sensitivity`, `fd_min`, `fd_max`, `lambda_min`,
  `lambda_max` and `sensitivity_share`.

Reference priors need not be exponential families or match the candidate family (e.g. `HalfCauchy`, `Uniform`,
`ChiSquared` from [`src/common/distributions/`](../common/distributions)).

### Non-exponential-family priors

```python
prior_sensitivity_black_box(model, score_prior_candidate, lower, upper, score_prior_ref=None,
                            method="dual_annealing", seed=0, maxiter=200, n_restarts=1)
```

- `score_prior_candidate(draws, lam)`: the gradient of the candidate log-prior density at the posterior draws, an
  `(m, d)` array, for hyperparameters `lam` (a 1-d array);
  ```python
  def student_t_score(draws, lam):
      loc, scale, df = lam
      z = draws - loc
      return -(df + 1.0) * z / (df * scale ** 2 + z ** 2)

  result = prior_sensitivity_black_box(model, student_t_score, lower=[-1.0, 1.0, 2.0], upper=[1.0, 3.0, 30.0])
  ```
- `lower` / `upper`: the box on `lam`;
- `score_prior_ref(draws)`: the reference prior's score; defaults to the `grad_log_pdf` of `base_prior`.

Both return an `FDSensitivityResult` with `sensitivity`, `fd_min`, `fd_max`, `lambda_min`, `lambda_max` (and
`components` when `independent=True`).

## Nonparametric prior sensitivity (FDsens+)

```python
nonparametric_prior_sensitivity(model, radius, basis="MaternBasisFunction", basis_kwargs=None, independent=False)
```

- `radius`: the radius $r$ of the FD ball.
- `basis` / `basis_kwargs`: the kernel of the sieve, e.g. `{"num_basis_functions": 100, "method": "kmeans", "nu": 3.5}`
  for a Matérn-7/2 kernel with $K=100$ centres placed by k-means on reference-prior draws. Available bases:
  `MaternBasisFunction`, `RBFBasisFunction`, `MaternBasisFunctionMultidim`, `FixedCentersRBFBasisFunctionMultidim`.
- `independent=True`: for factorised priors; `radius` may then be one value per parameter.
- `model.prior_samples`: reference-prior draws; drawn from `base_prior` if omitted.

The result (`NonparametricSensitivityResult`) holds `sensitivity`, `normalised_sensitivity`, `radius` and
`lambda_sup`. The worst-case prior is $\pi_{\text{ref}}(\theta) \exp\big(\sum_k \lambda_k \kappa(c_k, \theta)\big)$,
with $\lambda$ = `result.lambda_sup` and centres `result.basis_function.centers`.

### Plots

[`interpretation/plots.py`](../../interpretation/plots.py):

- `plot_quantiles(draws, output_dir, variables=None)`: a posterior-quantile table (`.csv`) and a median/90%-interval
  plot (`.png`) comparing the reference and worst-case fits;
- `plot_kde(draws, output_dir, variables=None)` / `plot_ecdf(draws, output_dir, variables=None)`: kernel density and
  empirical CDF comparisons, one panel per parameter;
- `plot_component_shares(result, output_dir)`: for a result computed with `independent=True`, a 100%-stacked bar of
  each parameter's share of the total sensitivity;
- `save_sensitivity_result(result, output_dir)`: the result as JSON.

## Files

| Path | Contents |
| --- | --- |
| [`parametric/sensitivity.py`](parametric/sensitivity.py) | `prior_sensitivity()`, `prior_sensitivity_black_box()` |
| [`parametric/fisher_divergence.py`](parametric/fisher_divergence.py) | `PosteriorFDParametric`: the FD quadratic form in the natural parameters |
| [`parametric/optimization.py`](parametric/optimization.py) | Corner-point, QP and black-box optimisers over the natural-parameter box |
| [`nonparametric/sensitivity.py`](nonparametric/sensitivity.py) | `nonparametric_prior_sensitivity()` and `NonparametricSensitivityResult` |
| [`nonparametric/basis_functions.py`](nonparametric/basis_functions.py) | Kernels of the sieve and `BASIS_FUNCTIONS_REGISTRY` |
| [`nonparametric/fisher_divergence.py`](nonparametric/fisher_divergence.py) | Posterior and prior FD quadratic forms in the basis coefficients |
| [`nonparametric/optimization.py`](nonparametric/optimization.py) | The generalised-eigenvalue solver |
| [`nonparametric/node_sensitivity.py`](nonparametric/node_sensitivity.py) | Per-parameter (and per-group) FDsens+ for factorised priors |

## Examples

Results and plots are written to `outputs/examples/prior/`.

| Script | Analysis |
| --- | --- |
| [`examples/prior/parametric_prior_sensitivity.py`](../../examples/prior/parametric_prior_sensitivity.py) | FDsens prior sensitivity (the Usage example), with quantile, KDE and ECDF plots |
| [`examples/prior/parametric_independent_components.py`](../../examples/prior/parametric_independent_components.py) | FDsens decomposition over independent prior components, with a component-share plot |
| [`examples/prior/parametric_black_box_prior_sensitivity.py`](../../examples/prior/parametric_black_box_prior_sensitivity.py) | FDsens for a non-exponential-family candidate |
| [`examples/prior/nonparametric_prior_sensitivity.py`](../../examples/prior/nonparametric_prior_sensitivity.py) | FDsens+ sensitivity over an FD ball |

## Contributing

New distributions go under [`src/common/distributions/`](../common/distributions), deriving from `BaseDistribution`
and implementing `sample`, `pdf`, `log_pdf`, `grad_log_pdf`, `natural_parameters`, `grad_sufficient_statistics`.

New kernels go in [`nonparametric/basis_functions.py`](nonparametric/basis_functions.py) with `evaluate` /
`gradient` returning `(m, d, K)` arrays, and an entry in `BASIS_FUNCTIONS_REGISTRY`.
