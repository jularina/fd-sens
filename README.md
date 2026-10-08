# Bayesian sensitivity analysis toolkit

A Python-based toolkit for **global Bayesian sensitivity analysis** using the **Fisher Divergence (FD)**.

---

## Features

- **Distributions**
  - Gaussian (univariate and multivariate)
  - Log-normal
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

Analysis is structured around three building blocks that are composed in each experiment script.

### 1. Bayesian model — `src/bayesian_model/`

The abstract base [`BayesianModel`](src/bayesian_model/base.py) defines the interface: it holds a prior and a likelihood, exposes score functions, and handles posterior/prior sampling. Concrete subclasses implement model-specific closed-form posteriors:

- [`SimpleGaussianModel`](src/bayesian_model/gaussian.py) — univariate Gaussian likelihood with Gaussian or Log-normal prior.
- [`MultivariateGaussianModel`](src/bayesian_model/gaussian.py) — multivariate Gaussian likelihood with Gaussian prior on the mean.

### 2. Fisher Divergence — `src/discrepancies/`

FD is computed separately for the prior and the posterior.

- [`PriorFDBase`](src/discrepancies/prior_fisher.py) — FD between prior samples and a candidate prior; uses the exponential family score decomposition.
- [`PosteriorFDBase`](src/discrepancies/posterior_fisher.py) — FD for the posterior, combining the reference prior score with the candidate prior's natural statistics evaluated on posterior samples.

### 3. Optimizer — `src/optimization/`

Given a discrepancy object, the optimizer searches for the worst-case prior (or loss learning rate) over a user-specified parameter box.

- [`OptimizationCornerPointsUnivariateGaussian`](src/optimization/corner_points_fisher.py) / [`OptimizationCornerPointsMultivariateGaussian`](src/optimization/corner_points_fisher.py) — corner-point search over a box of Gaussian prior hyperparameters (toy experiments).
- [`OptimizationCornerPointsCompositePrior`](src/optimization/corner_points_fisher.py) — composite independent-marginal priors: corner enumeration of the convex quadratic form (full or per component), convex QP for the infimum, black-box dual annealing baseline, and Gaussian-copula perturbations.

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
    _target_: src.distributions.gaussian.Gaussian
    mu: 2
    sigma: 4
  true_dgp:
    _target_: src.distributions.gaussian.Gaussian
    mu: 3
    sigma: 2
  loss:
    _target_: src.losses.gaussian_log_likelihood.GaussianLogLikelihood
    mu: 3
    sigma: 2
  loss_lr: 1.0
  observations_num: 100
  posterior_samples_num: 1000
  prior_samples_num: 1000

model:
  _target_: src.bayesian_model.gaussian.SimpleGaussianModel

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
Scripts with a `_nonparam` suffix belong to the nonparametric (FDsens+) paper, which uses kernel exponential family
neighbourhoods solved through a generalised eigenvalue problem; the others belong to the parametric (FDsens) paper.

### `paper/toy/`
Toy Gaussian experiments and finite-sample complexity comparisons.
- `run_toy_fisher.py` — sensitivity analysis on univariate/multivariate Gaussian models; generates FD sensitivity curves and comparison plots against the mean, KL and Wasserstein-2 measures.
- `run_toy_fisher_nonparam.py` — FDsens+ on the Gaussian location model: worst-case priors across radii (1d/2d), runtimes, sensitivity vs number of centres, closed-form estimation errors, kernel and centre choices, and the parametric vs nonparametric comparison.

### `paper/ising/`
Generalised Bayesian inference for the Ising model with pseudolikelihood and discrete Fisher divergence losses.
- `run_ising_fisher.py` — `main()` computes FD learning-rate sensitivity grids for the three learning-rate calibration methods; `create_combined_plots()` produces the paper figures.

### `paper/posteriordb/`
Real-data experiments using models from the PosteriorDB benchmark.
- `run_ark_fisher.py` — optimisation runtime comparison (convex corner enumeration, per-component decomposition, black-box dual annealing) for the arK model.
- `run_ark_kilpisjarvi.py` — z-scale prior sensitivity and posterior predictives for the Kilpisjarvi AR(5) model; run end to end with `run_kilpisjarvi_param_predictive.sh` (requires R with `rstan`).
- `run_ark_kilpisjarvi_nonparam.py` — FDsens vs FDsens+ per-parameter sensitivity shares for the Kilpisjarvi AR(5) model in z-scale.

### `paper/sbi/`
Experiments on the Turin channel model fitted via simulation-based inference (SBI).
- `run_turin_fisher.py` — FD sensitivity to Gaussian-copula prior dependence for the Turin SBI model.

### `paper/bnn/`
Bayesian neural networks on UCI regression datasets (posterior samples from the `bnn_priors` code of Fortuin et al., 2022).
- `run_bnn_uci_fisher_nonparam.py` — per-parameter FDsens+ sensitivity: Boston weight heatmap, all-dataset runs, and the layer-wise sensitivity table.
