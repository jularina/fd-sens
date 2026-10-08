# fd-sens

Fisher-divergence (FD) global sensitivity analysis for sampling-based Bayesian inference.

- **FDsens** (parametric): sensitivity to prior hyperparameters within an exponential family, or to the learning rate of
  a generalised posterior. Methodology: [*A computationally-tractable measure of global sensitivity for sampling-based
  Bayesian inference*](https://arxiv.org/abs/2605.28099).
- **FDsens+** (nonparametric): sensitivity over an FD ball of priors around the reference prior, approximated by a kernel
  exponential family sieve and solved exactly through a generalised eigenvalue problem.

See [`GETTING_STARTED.md`](GETTING_STARTED.md) for the method and [`configs/README.md`](configs/README.md)
for writing configs. An R/Stan implementation of FDsens is available at [fd-sens-r](https://github.com/jularina/fd-sens-r).

## Installation

The package is not on PyPI; clone the repository and install its dependencies with [PDM](https://pdm-project.org)
(Python 3.12):

```sh
git clone https://github.com/jularina/fd-sens.git
cd fd-sens
pdm install            # add -G test to also install pytest
```

## Quickstart (Gaussian location model)

Reference model: $\theta \sim \mathcal N(0, 2^2)$, $y_i \sim \mathcal N(\theta, 1)$. Take draws from the reference
posterior with any sampler (Stan, PyMC, NumPyro, ...) and measure sensitivity to the prior's hyperparameters:

```python
import numpy as np
from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.distributions.gaussian import Gaussian
from src.parametric.sensitivity import prior_sensitivity

rng = np.random.default_rng(123)
y = rng.normal(loc=1.0, scale=1.0, size=30)
var_n = 1.0 / (len(y) + 1.0 / 2.0 ** 2)
posterior = rng.normal(var_n * y.sum(), np.sqrt(var_n), size=2000)   # exact conjugate posterior draws

model = PosteriorSamplesModel(
    posterior_samples=posterior,
    base_prior=Gaussian(mu=0.0, sigma=2.0),         # reference prior
    candidate_prior=Gaussian(mu=0.0, sigma=1.0),    # candidate family (its parameter values are not used)
)
result = prior_sensitivity(model, natural_box={"theta": {"eta_1": [-1.0, 1.0], "eta_2": [-2.0, -0.05]}})
print(result)
#> FD prior sensitivity
#>   optimisation: quadratic_corner
#>   sensitivity: 31.2854
#>   minimum FD:  4.16334e-17 at lambda_min = [ 0.    -0.125]
#>   maximum FD:  31.2854 at lambda_max = [-1. -2.]
```

`eta_1 = mu / s^2` and `eta_2 = -1 / (2 s^2)` are the natural parameters of the candidate $\mathcal N(\mu, s^2)$ (see
[Exponential-family priors](#exponential-family-priors)). `result.sensitivity` is the largest change in the posterior
over the box, `result.lambda_max` the worst-case prior that produces it (here $\mathcal N(-0.25, 0.5^2)$), and the
minimum sits at the reference prior, $\eta = (0, -1/8)$.

The full script is [`examples/parametric_prior_sensitivity.py`](examples/parametric_prior_sensitivity.py).

## Repository contents

| Path | What's there |
| --- | --- |
| [`src/common/`](src/common) | Shared building blocks: reference models (incl. `PosteriorSamplesModel` for your own draws), distributions, losses, the posterior FD base class, utilities. |
| [`src/parametric/`](src/parametric) | FDsens: `prior_sensitivity()`, `lr_sensitivity()`, the quadratic-form FD estimator and the optimisers. |
| [`src/nonparametric/`](src/nonparametric) | FDsens+: `nonparametric_prior_sensitivity()`, basis functions, FD quadratic forms and the generalised-eigenvalue solver. |
| [`examples/`](examples) | Runnable scripts for each analysis, including one driven by a Hydra config. |
| [`configs/`](configs) | Hydra configs: [`configs/examples/`](configs/examples) for the examples, `configs/paper/` for the paper experiments; the rules are in [`configs/README.md`](configs/README.md). |
| [`tests/`](tests) | Unit tests (`pytest`) for the parametric and nonparametric estimators and for every config. |
| [`paper/`](paper) | Scripts and plotting code reproducing the figures of both papers. |

## Functionality

### Your model: `PosteriorSamplesModel`

```python
PosteriorSamplesModel(posterior_samples, base_prior, candidate_prior=None, loss_grad=None, loss_lr=1.0, prior_samples=None)
```

- `posterior_samples`: `(m, d)` draws from the reference posterior, one column per parameter.
- `base_prior`: the reference prior. Any object with `grad_log_pdf(x) -> (m, d)` works; FDsens+ additionally uses `sample(n)`.
  For several parameters with independent priors use a `CompositeProduct` whose components are in column order.
- `candidate_prior`: the exponential-family candidate family (FDsens prior sensitivity only).
- `loss_grad`: gradient of the unscaled loss at each draw, an `(m, d)` array or a function of the draws (learning-rate
  sensitivity only).
- `loss_lr`: the reference learning rate.
- `prior_samples`: reference-prior draws for FDsens+; drawn from `base_prior` if omitted.

Models can also be built from a config, see [Config-driven use](#config-driven-use).

### Prior sensitivity (FDsens)

```python
prior_sensitivity(model, natural_box, method="quadratic", independent=False, **black_box_kwargs)
```

- `natural_box`: one entry per parameter, in column order: `{name: {"eta_1": (lower, upper), "eta_2": (lower, upper)}}`,
  the box $\Gamma$ of candidate natural parameters.
- `method="quadratic"` (default): the FD is a convex quadratic form in the natural parameters, so the maximum is found by
  enumerating the box corners and the minimum by a convex QP. Cost grows as $4^d$ corners.
- `method="black_box"`: global optimisation (`scipy` dual annealing by default) of the FD over the box; pass e.g.
  `maxiter=150, n_restarts=5`.
- `independent=True`: when the priors factorise over parameters, the sensitivity is the sum of per-parameter
  sensitivities, each solved on its own 2-d box ($4d$ corners instead of $4^d$). `result.components` holds each block's
  `sensitivity`, `fd_min`, `fd_max`, `lambda_min`, `lambda_max` and `sensitivity_share`; see
  [`examples/parametric_independent_components.py`](examples/parametric_independent_components.py).

#### Exponential-family priors

Each candidate component is a one-dimensional exponential family with two natural parameters:

| Family | Class | Natural parameters $(\eta_1, \eta_2)$ | Valid box |
| --- | --- | --- | --- |
| Normal $(\mu, s)$ | `Gaussian(mu, sigma)` | $(\mu/s^2,\; -1/(2s^2))$ | $\eta_2 < 0$ |
| Gamma $(\alpha, \theta)$ | `Gamma(alpha, theta)` | $(\alpha - 1,\; -1/\theta)$ | $\eta_1 > -1,\ \eta_2 < 0$ |
| Inverse-Gamma $(\alpha, \beta)$ | `InverseGamma(alpha, beta)` | $(-(\alpha+1),\; -\beta)$ | $\eta_1 < -1,\ \eta_2 < 0$ |
| Beta $(\alpha, \beta)$ | `Beta(alpha, beta)` | $(\alpha - 1,\; \beta - 1)$ | $\eta_1, \eta_2 > -1$ |

Reference priors need not be exponential families or match the candidate family (e.g. `HalfCauchy`, `Uniform`,
`ChiSquared` from [`src/common/distributions/`](src/common/distributions)).

### Learning-rate sensitivity (FDsens)

```python
lr_sensitivity(model, lower, upper, lr_ref=None)
```

With the prior fixed, $\mathrm{FD}(\lambda) = (\lambda - \lambda_\mathrm{ref})^2\,\mathbb E\|\nabla_\theta \ell\|^2$, so no
optimisation is needed. The model needs `loss_grad`; see
[`examples/parametric_lr_sensitivity.py`](examples/parametric_lr_sensitivity.py).

### Nonparametric prior sensitivity (FDsens+)

```python
nonparametric_prior_sensitivity(model, radius, basis="MaternBasisFunction", basis_kwargs=None, independent=False)
```

- `radius`: the radius $r$ of the FD ball $\{\Pi : \mathrm{FD}(\Pi_\mathrm{ref}\|\Pi) \le r\}$. The sensitivity is linear in $r$,
  so `result.normalised_sensitivity` ($S/r$) summarises the whole family of balls.
- `basis` / `basis_kwargs`: the kernel of the sieve, e.g. `{"num_basis_functions": 100, "method": "kmeans", "nu": 3.5}`
  for a Matérn-7/2 kernel with $K=100$ centres placed by k-means on reference-prior draws. Available bases:
  `MaternBasisFunction`, `RBFBasisFunction`, `MaternBasisFunctionMultidim`, `FixedCentersRBFBasisFunctionMultidim`.
- `independent=True`: for factorised priors, solves one problem per parameter (`radius` may then be one value per
  parameter); `result.components` holds the per-parameter results.

The result also contains `lambda_sup`, the coefficients of the worst-case prior
$\pi_K(\theta) \propto \pi_\mathrm{ref}(\theta)\exp(\sum_k \lambda_k \kappa(\bar\theta_k, \theta))$, and the fitted
`basis_function` (its `centers` and `evaluate`). See
[`examples/nonparametric_prior_sensitivity.py`](examples/nonparametric_prior_sensitivity.py).

### Config-driven use

Models, priors, data and neighbourhoods can be declared in a Hydra YAML config and instantiated with
`instantiate(cfg.model, data_config=cfg.data)`:

```sh
PYTHONPATH=. pdm run python examples/from_config.py                                         # FDsens
PYTHONPATH=. pdm run python examples/from_config.py --config-name gaussian_location_nonparam  # FDsens+
```

[`configs/README.md`](configs/README.md) explains the required keys and how to write a config for a new model.

### Algorithm

1. **Fit the reference model** with any sampler and collect `(m, d)` reference-posterior draws.
2. **Wrap it** in `PosteriorSamplesModel` (or a config-defined model), giving the reference prior and, depending on the
   analysis, the candidate family or the loss gradient.
3. **Choose the analysis and the neighbourhood:**
   - prior hyperparameters (FDsens): a natural-parameter box per parameter, then `prior_sensitivity(...)`;
   - learning rate (FDsens): an interval, then `lr_sensitivity(...)`;
   - nonparametric prior perturbations (FDsens+): a radius and a kernel, then `nonparametric_prior_sensitivity(...)`.
4. **Read the result**: `sensitivity` is $\widehat S_m^{\mathrm{FD}}$; `lambda_max` / `lambda_min` (FDsens) or `lambda_sup`
   (FDsens+) describe the worst-case and least-sensitive choices; `components` gives per-parameter shares when
   `independent=True`.

### Examples

| Script | Analysis |
| --- | --- |
| [`examples/parametric_prior_sensitivity.py`](examples/parametric_prior_sensitivity.py) | FDsens prior sensitivity (Quickstart) |
| [`examples/parametric_independent_components.py`](examples/parametric_independent_components.py) | FDsens decomposition over independent prior components (Normal mean, Gamma scale) |
| [`examples/parametric_lr_sensitivity.py`](examples/parametric_lr_sensitivity.py) | FDsens learning-rate sensitivity |
| [`examples/nonparametric_prior_sensitivity.py`](examples/nonparametric_prior_sensitivity.py) | FDsens+ sensitivity over an FD ball |
| [`examples/from_config.py`](examples/from_config.py) | Both, from [`configs/examples/`](configs/examples) |

### Tests

```sh
pdm install -G test
pdm run pytest
```

The tests check the estimators against closed forms on a conjugate Gaussian model, the optimisers against brute-force
search, and that every config's `_target_` and basis names resolve.

## Contributing

New candidate families are welcome: add a class under [`src/common/distributions/`](src/common/distributions) deriving
from `BaseDistribution` and implement `sample`, `pdf`, `log_pdf`, `grad_log_pdf`, `natural_parameters`,
`grad_sufficient_statistics` (shape `(m, 1, 2)`) and `grad_log_base_measure`, then register it in
[`src/common/utils/distributions.py`](src/common/utils/distributions.py). New kernels go in
[`src/nonparametric/basis_functions.py`](src/nonparametric/basis_functions.py) with `evaluate` / `gradient` returning
`(m, d, K)` arrays, and an entry in `BASIS_FUNCTIONS_REGISTRY`.

## Reproducing the papers

The `paper/` directory contains the experiment scripts, organized into one folder per experimental setting.
In each folder, `run_parametric.py` belongs to the parametric (FDsens) paper and `run_nonparametric.py` to the
nonparametric (FDsens+) paper, which uses kernel exponential family neighbourhoods solved through a generalised
eigenvalue problem.

### `paper/toy/`
Toy Gaussian experiments and finite-sample complexity comparisons.
- `run_parametric.py` — sensitivity analysis on univariate/multivariate Gaussian models; generates FD sensitivity curves and comparison plots against the mean, KL and Wasserstein-2 measures.
- `run_nonparametric.py` — FDsens+ on the Gaussian location model: worst-case priors across radii (1d/2d), runtimes, sensitivity vs number of centres, closed-form estimation errors, kernel and centre choices, and the parametric vs nonparametric comparison.

### `paper/ising/`
Generalised Bayesian inference for the Ising model with pseudolikelihood and discrete Fisher divergence losses.
- `run_parametric.py` — `main()` computes FD learning-rate sensitivity grids for the three learning-rate calibration methods; `create_combined_plots()` produces the paper figures.

### `paper/posteriordb/`
Real-data experiments using models from the PosteriorDB benchmark.
- `run_parametric.py` — z-scale prior sensitivity and posterior predictives for the Kilpisjarvi AR(5) model; run end to end with `run_parametric_predictive.sh` (requires R with `rstan`). The optimiser runtime comparison (full corner enumeration, per-component decomposition, black-box dual annealing) runs separately with `playground.stage=timing` and is replotted from the saved timings with `playground.stage=plot_timing`.
- `run_nonparametric.py` — FDsens vs FDsens+ per-parameter sensitivity shares for the Kilpisjarvi AR(5) model in z-scale.

### `paper/sbi/`
Experiments on the Turin channel model fitted via simulation-based inference (SBI).
- `run_parametric.py` — FD sensitivity to Gaussian-copula prior dependence for the Turin SBI model.

### `paper/bnn/`
Bayesian neural networks on UCI regression datasets (posterior samples from the `bnn_priors` code of Fortuin et al., 2022).
- `run_nonparametric.py` — per-parameter FDsens+ sensitivity: Boston weight heatmap, all-dataset runs, and the layer-wise sensitivity table.

### `paper/illustrative/`
- `run_nonparametric.py` — schematic of the sieve approximation and the Monte Carlo constraint estimate (`sieve_and_mc.pdf`); run with `python -m paper.illustrative.run_nonparametric`.

## Citing

```bibtex
@article{Odnoblyudova2026,
  author  = {Odnoblyudova, A. and Dellaporta, C. and Briol, F-X.},
  journal = {arXiv:2605.28099},
  title   = {{A computationally-tractable measure of global sensitivity for sampling-based Bayesian inference}},
  year    = {2026}
}
```
