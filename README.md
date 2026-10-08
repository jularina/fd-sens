# Bayesian sensitivity analysis toolkit

A Python-based toolkit for **global Bayesian sensitivity analysis** using the **Fisher Divergence (FD)**.

---

## Features

- **Distributions**
  - Gaussian (univariate and multivariate)
  - Gamma, Inverse-Gamma
  - Beta
  - Cauchy, Half-Cauchy
  - Uniform
  - Chi-squared
  - Composite product of independent marginals
  - Extensible to custom distributions

- **FD computations**
  - FD between posterior samples and candidate posterior
  - FD between prior samples and candidate prior

- **Optimization tools for sensitivity analysis**
  - Parametric corner-point search over prior and loss parameters

- **Plotting utilities**
  - Prior vs posterior samples
  - FD sensitivity surfaces across parameter grids
  - Comparisons across methods and models

- **Configuration-driven**
  [Hydra](https://hydra.cc/) for flexible experiment configuration via YAML files.

---

## Main components

`src/` is split into three packages. `parametric` and `nonparametric` both build on `common` and never import each other.

```
src/
  common/         shared by both methods
    bayesian_model/   reference models (Gaussian location, Ising, arK, Kilpisjarvi, Turin, BNN)
    distributions/    priors with scores and exponential-family decompositions
    losses/           likelihoods / generalised-Bayes losses and their gradients
    fisher.py         PosteriorFDBase: reference-posterior samples, scores and quadratic-form helpers
    utils/            config loading, JSON I/O, distribution registry
  parametric/     FDsens: exponential-family candidate priors
    fisher.py         PosteriorFDParametric: FD as a convex quadratic form in the natural parameters
    corner_points.py  corner enumeration, convex QP, black-box baseline, Gaussian-copula perturbations
  nonparametric/  FDsens+: kernel exponential family (sieve) neighbourhoods
    basis_functions.py  Matern/RBF bases and the BASIS_FUNCTIONS_REGISTRY
    fisher.py           prior/posterior FD quadratic forms in the basis coefficients
    optimization.py     worst case via the generalised eigenvalue problem
    node_sensitivity.py per-parameter sensitivity for factorised priors (BNN, Kilpisjarvi)
    loaders.py          per-parameter view of the Kilpisjarvi model
```

Plotting lives with the experiments in `paper/`: each folder has `plots_parametric.py` / `plots_nonparametric.py`
next to its `run_*.py`, and `paper/plot_utils.py` holds the shared matplotlib style.

### 1. Bayesian model — `src/common/bayesian_model/`

The abstract base [`BayesianModel`](src/common/bayesian_model/base.py) defines the interface: it holds a prior and a likelihood, exposes score functions, and handles posterior/prior sampling. Concrete subclasses implement model-specific closed-form posteriors:

- [`SimpleGaussianModel`](src/common/bayesian_model/gaussian.py) — univariate Gaussian likelihood with Gaussian prior.
- [`MultivariateGaussianModel`](src/common/bayesian_model/gaussian.py) — multivariate Gaussian likelihood with Gaussian prior on the mean.

### 2. Fisher divergence

- [`PosteriorFDBase`](src/common/fisher.py) — holds the reference-posterior samples and scores shared by both estimators.
- [`PosteriorFDParametric`](src/parametric/fisher.py) — FD between the reference and a candidate exponential-family prior, as a quadratic form in its natural parameters; also learning-rate and Gaussian-copula perturbations.
- [`PriorFDNonParametric`](src/nonparametric/fisher.py) / [`PosteriorFDNonParametric`](src/nonparametric/fisher.py) — constraint and objective quadratic forms in the kernel-exponential-family coefficients.

### 3. Optimisation

- [`OptimizationCornerPointsUnivariateGaussian`](src/parametric/corner_points.py) / [`OptimizationCornerPointsMultivariateGaussian`](src/parametric/corner_points.py) — corner-point search over a box of Gaussian prior hyperparameters (toy experiments).
- [`OptimizationCornerPointsCompositePrior`](src/parametric/corner_points.py) — composite independent-marginal priors: corner enumeration of the convex quadratic form (full or per component), convex QP for the infimum, black-box dual annealing baseline, and Gaussian-copula perturbations.
- [`OptimisationNonparametricBase`](src/nonparametric/optimization.py) — FDsens+ worst-case prior through the generalised eigenvalue problem.
- [`compute_group_omega_max`](src/nonparametric/node_sensitivity.py) — FDsens+ per-parameter sensitivities for factorised priors.

---

## Contributions

We are happy with any help in adding distributions and models to the project!

---

## License

---

## Config structure

Experiments are configured via YAML files loaded by Hydra. Below is an example for a univariate Gaussian model:

```yaml
data:
  base_prior:
    _target_: src.common.distributions.gaussian.Gaussian
    mu: 2
    sigma: 4
  true_dgp:
    _target_: src.common.distributions.gaussian.Gaussian
    mu: 3
    sigma: 2
  loss:
    _target_: src.common.losses.gaussian_log_likelihood.GaussianLogLikelihood
    mu: 3
    sigma: 2
  loss_lr: 1.0
  observations_num: 100
  posterior_samples_num: 1000
  prior_samples_num: 1000

model:
  _target_: src.common.bayesian_model.gaussian.SimpleGaussianModel

fd:
  optimize:
    prior:
      Gaussian:
        parameters_box_range:
          ranges:
            mu: [-10, 10]
            sigma: [2, 5]
          nums:
            mu: 21
            sigma: 4
    loss:
      GaussianLogLikelihood:
        parameters_box_range:
          ranges:
            lr: [0.5, 2.5]
          nums:
            lr: 9

flags:
  plots:
    output_dir: outputs/paper/plots/fisher/univariate
```

---

## Paper experiments

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
