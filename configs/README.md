# Paper configs

The paper experiments in [`paper/`](../paper) are driven by [Hydra](https://hydra.cc/) YAML configs. For your own
model you do not need any of this: pass your posterior draws to `PosteriorSamplesModel` in Python (see the
[README](../README.md)).

A config declares the reference model (prior, likelihood or loss,
data and posterior samples) and the neighbourhood to search. Every `_target_` is the import path of a class, which
Hydra instantiates with the remaining keys of that block as keyword arguments.

| Folder | Used by |
| --- | --- |
| [`paper/toy/`](paper/toy), [`paper/real/`](paper/real) | The paper scripts in [`paper/`](../paper). |
| [`plots/`](plots) | Shared matplotlib settings (`overleaf_plots_settings.yaml`) for the paper figures. |

## Minimal config

```yaml
defaults:                         # silences Hydra's own logging
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

hydra:                            # run in the launch directory; no outputs/<date>/<time> folders
  output_subdir: null
  run:
    dir: .
  job:
    chdir: false

data:
  base_prior:                     # reference prior Pi_ref
    _target_: src.common.distributions.gaussian.Gaussian
    mu: 2
    sigma: 4
  candidate_prior:                # FDsens only: candidate exponential family (values unused)
    _target_: src.common.distributions.gaussian.Gaussian
    mu: 0
    sigma: 1
  loss:                           # likelihood or loss, providing grad_log_pdf
    _target_: paper.common.losses.gaussian_log_likelihood.GaussianLogLikelihood
    mu: 0
    sigma: 2
  loss_lr: 1.0                    # current learning rate
  loss_lr_init: 1.0               # reference learning rate
  true_dgp: null                  # distribution to simulate data from when observations_path is null
  observations_num: 100
  observations_path: data/my_model/observations.npy
  posterior_samples_path: data/my_model/posterior_samples.npy
  prior_samples_path: null        # null: draw from base_prior
  posterior_samples_num: 5000
  prior_samples_num: 5000

model:
  _target_: paper.common.bayesian_model.gaussian.SimpleGaussianModel
```

The script then builds the model with

```python
model = instantiate(cfg.model, data_config=cfg.data)
```

## Rules

1. **`data` is required** and must contain `base_prior`, `loss`, `loss_lr`, `loss_lr_init`, `true_dgp`,
   `posterior_samples_num` and `prior_samples_num` (use `null` for unused ones). `candidate_prior` is required for
   FDsens prior sensitivity only.
2. **`model._target_`** is a model class taking `data_config`. Classes deriving from
   `BayesianModelExtended` ([`paper/common/bayesian_model/base.py`](../paper/common/bayesian_model/base.py)) load
   `observations_path`, `posterior_samples_path` and `prior_samples_path`; when a posterior path is `null` the model
   must implement `sample_posterior()`, and when a prior path is `null` draws come from `base_prior`.
3. **Paths** are `.npy` files; relative paths are resolved from the directory you launch from. Arrays are
   `(n, d)`, one row per sample and one column per parameter.
4. **Several parameters with independent priors:** use a `CompositeProduct` whose component order matches the
   posterior-sample columns:

   ```yaml
   base_prior:
     _target_: src.common.distributions.composite.CompositeProduct
     distributions:
       alpha:
         _target_: src.common.distributions.gaussian.Gaussian
         mu: 0
         sigma: 5
       sigma:
         _target_: src.common.distributions.cauchy.HalfCauchy
         gamma: 1
   ```

5. **FDsens neighbourhood:** one natural-parameter box per parameter, keyed by name, in column order (see the README for
   each family's natural parameters and valid ranges):

   ```yaml
   sensitivity:
     natural_box:
       theta:
         eta_1: [-1.0, 1.0]
         eta_2: [-0.5, -0.02]
     lr_interval: [0.5, 1.5]
   ```

6. **FDsens+ neighbourhood:** the radius and the kernel; `basis` must be a key of `BASIS_FUNCTIONS_REGISTRY` in
   [`src/nonparametric/basis_functions.py`](../src/nonparametric/basis_functions.py), and `basis_kwargs` are passed to
   that class:

   ```yaml
   nonparametric:
     radius: 1.0
     basis: MaternBasisFunction
     basis_kwargs:
       num_basis_functions: 100     # K
       method: kmeans               # kmeans | halton | random
       nu: 3.5                      # Matern-7/2
   ```

7. **Overriding from the command line:** another config in the same folder is selected with `--config-name`, and any
   existing key can be overridden, e.g.
   `pdm run python paper/toy/run_nonparametric.py --config-name univariate_gaussian_nonparam data.observations_num=50`.

`pdm run pytest tests/test_configs.py` checks that every `_target_` and basis name in `configs/` resolves.

## Adding a model class

A config model is a class taking `data_config`, usually a subclass of `BayesianModelExtended`
([`paper/common/bayesian_model/base.py`](../paper/common/bayesian_model/base.py)). Implement `sample_posterior()` if
it should sample its own posterior, and override `loss_score(x, multiply_by_lr=True)` if your loss's `grad_log_pdf` does
not take `(x, x_bar, n)`; see [`paper/common/bayesian_model/gaussian.py`](../paper/common/bayesian_model/gaussian.py).
