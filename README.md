# fd-sens

Fisher-divergence (FD) global sensitivity analysis for sampling-based Bayesian inference.

## Sensitivity analyses

| Analysis | What varies | Entry point | Documentation |
| --- | --- | --- | --- |
| **Prior** | Prior hyperparameters (FDsens), or all priors in an FD ball around the reference (FDsens+) | `prior_sensitivity()`, `prior_sensitivity_black_box()`, `nonparametric_prior_sensitivity()` | [`src/prior/README.md`](src/prior/README.md) |
| **Learning rate** | The learning rate of a generalised posterior | `lr_sensitivity()` | [`src/learning_rate/README.md`](src/learning_rate/README.md) |
| **Data removal** | Removing at most a fraction $\alpha$ of the observations (or groups) | `data_removal_sensitivity()` | [`src/data_removal/README.md`](src/data_removal/README.md) |

[`GETTING_STARTED.md`](GETTING_STARTED.md) introduces the method. An R/Stan implementation of FDsens is available at
[fd-sens-r](https://github.com/jularina/fd-sens-r).

## Installation

Clone the repository and install its dependencies with [PDM](https://pdm-project.org) (Python 3.12):

```sh
git clone https://github.com/jularina/fd-sens.git
cd fd-sens
pdm install            # add -G test to also install pytest
pdm run python examples/prior/parametric_prior_sensitivity.py
pdm run pytest         # needs -G test
```

## Workflow

1. **Fit the reference model** with any sampler (Stan, PyMC, NumPyro, your own MCMC) and collect `(m, d)`
   reference-posterior draws.
2. **Wrap the draws** in `PosteriorSamplesModel` with the inputs the analysis needs:

   ```python
   from src.common.bayesian_model.samples import PosteriorSamplesModel

   PosteriorSamplesModel(posterior_samples, base_prior=None, candidate_prior=None, loss_grad=None, loss_lr=1.0, prior_samples=None, data_grad=None)
   ```

   | Argument | Meaning | Prior | Learning rate | Data removal |
   | --- | --- | :-: | :-: | :-: |
   | `posterior_samples` | `(m, d)` reference-posterior draws | ✓ | ✓ | ✓ |
   | `base_prior` | reference prior, with `grad_log_pdf` (and `sample` for FDsens+) | ✓ | | |
   | `candidate_prior` | exponential-family candidate family | FDsens | | |
   | `prior_samples` | reference-prior draws (drawn from `base_prior` if omitted) | FDsens+ | | |
   | `loss_grad`, `loss_lr` | gradient of the unscaled loss, and the reference learning rate | | ✓ | |
   | `data_grad` | per-observation (or per-group) log-likelihood gradients, `(m, n, d)` | | | ✓ |

3. **Run the analysis**
4. **Interpret the result**

## Repository layout

```
src/
  common/          shared: PosteriorSamplesModel, distributions, FD base class, result class, utilities
  prior/           prior sensitivity: parametric/ (FDsens) and nonparametric/ (FDsens+)
  learning_rate/   learning-rate sensitivity
  data_removal/    data-removal sensitivity
examples/          runnable scripts, one folder per analysis: prior/, learning_rate/, data_removal/
interpretation/    plots.py: tables, plots and JSON summaries of a result
tests/             pytest unit tests, one file per analysis, plus the paper configs
paper/             paper experiments: model classes, losses, run scripts and plotting code
configs/           Hydra configs for the paper experiments, see configs/README.md
data/              datasets for the paper experiments and examples
outputs/           generated results (not tracked): examples/<analysis>/, paper/, presentation/
```

## Citing

```bibtex
@article{Odnoblyudova2026,
  author  = {Odnoblyudova, A. and Dellaporta, C. and Briol, F-X.},
  journal = {arXiv:2605.28099},
  title   = {{A computationally-tractable measure of global sensitivity for sampling-based Bayesian inference}},
  year    = {2026}
}
```
