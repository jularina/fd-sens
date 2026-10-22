# Data-removal sensitivity (FDsens-data)

Global sensitivity of a posterior to removing a small fraction of the data, measured by the Fisher divergence (FD)
to the full-data reference posterior. One run of the sampler on the full data is enough to score every removal.

## Usage

```python
from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.data_removal.sensitivity import data_removal_sensitivity

model = PosteriorSamplesModel(posterior_samples=draws, data_grad=data_grad)
result = data_removal_sensitivity(model, alpha=0.1)
print(result)
```

```python
data_removal_sensitivity(model, alpha=None, max_removed=None, groups=None, names=None, whiten=False, method="corner", max_evaluations=10_000_000, keep_all=False)
```

- `model.data_grad`: per-observation (or per-group) log-likelihood gradients at the draws, given as an `(m, n, d)`
  array or as a function of the `(m, d)` draws. `(m, n)` is accepted for a single parameter.
- `alpha` / `max_removed`: give exactly one. `alpha` removes at most $\lfloor \alpha n \rfloor$ units;
  `max_removed` sets $K$ directly.
- `groups`: optional list of disjoint index lists. Per-observation scores are summed within each group, and whole
  groups are removed.
- `names`: labels for the observations or groups, used in the output and the plots.
- `whiten`: measure the FD in posterior-standardised coordinates.
- `keep_all=True`: keep every evaluated subset and its FD (`result.subsets`, `result.fd_values`, one entry per
  removal size), e.g. for `plot_removal_budget`.

The result (`DataRemovalSensitivityResult`) holds:

| Field | Meaning |
| --- | --- |
| `sensitivity`, `sensitivity_se` | maximal FD over all removals of at most $K$ units, and its Monte Carlo standard error |
| `removed`, `removed_names` | the worst-case removal |
| `per_size[k]` | for each $k = 1, \ldots, K$: `fd`, `removed`, `removed_names`, `se` of the best removal of exactly $k$ units |
| `A` | the $n \times n$ (or $G \times G$) matrix of the quadratic form |
| `n_evaluated` | number of corner points $N(n, K)$ |

The minimum FD is 0, attained at the reference (nothing removed), so the sensitivity equals the maximal FD.

### Plots

[`interpretation/plots.py`](../../interpretation/plots.py):

- `plot_removal_interactions(result, output_dir)`: single-removal FDs $A_{ii}$ and the normalised cross terms
  $A_{ik} / \sqrt{A_{ii} A_{kk}}$;
- `plot_removal_budget(result, output_dir)`: the optimal FD for each removal size, compared with removing the $k$
  individually most influential units (and all subsets when `keep_all=True`);
- `save_sensitivity_result(result, output_dir)`: the result as JSON.

Both plot functions take `plot_cfg=load_plot_config("configs/plots/overleaf_plots_settings.yaml")`
(from `src.common.utils.files_operations`). They then use its fonts, LaTeX rendering, dpi, figure size and the
`data_removal_colors` palette (dark blue, dark grey-purple, grey, graphite, ...). Both write `.pdf` files.

## Files

| File | Contents |
| --- | --- |
| [`fisher_divergence.py`](fisher_divergence.py) | `PosteriorFDDataRemoval`: scores at the draws, the matrix $A$, FD and its standard error for a given removal |
| [`optimization.py`](optimization.py) | `OptimizationCornerPointsDataRemoval`: exhaustive corner-point evaluation, chunked by removal size |
| [`sensitivity.py`](sensitivity.py) | `data_removal_sensitivity()` and `DataRemovalSensitivityResult` |

## Examples

| Script | Analysis |
| --- | --- |
| [`examples/data_removal/data_removal_sensitivity.py`](../../examples/data_removal/data_removal_sensitivity.py) | Gaussian location model with two outliers, $\alpha = 0.1$ |
| [`examples/data_removal/data_removal_sensitivity_bao.py`](../../examples/data_removal/data_removal_sensitivity_bao.py) | Baryon acoustic oscillation (BAO) cosmology dataset: removing up to 6 of the 7 independent BAO blocks |

### BAO

[`paper/bao/bao.py`](../../paper/bao/bao.py) holds the DESI DR2 BAO likelihood (data in [`data/bao/`](../../data/bao)).

```sh
pdm run python examples/data_removal/data_removal_sensitivity_bao.py
```

This takes a few seconds and writes results to
`outputs/examples/data_removal/bao_LCDM/`. Example output:

```
FD data-removal sensitivity
  optimisation: corner_points (127 corner points)
  removal budget: at most 6 of 7
  sensitivity: 2.15877 (se 0.017)
  worst-case removal: BGS, LRG1, LRG3+ELG1, ELG2, QSO, Lya
  per removal size:
    k=1: FD=0.515708 removing LRG2
    k=2: FD=0.893005 removing BGS, LRG1
    k=3: FD=1.39142 removing BGS, LRG1, Lya
    k=4: FD=1.86763 removing LRG2, LRG3+ELG1, ELG2, Lya
    k=5: FD=1.91819 removing BGS, LRG1, LRG3+ELG1, ELG2, Lya
    k=6: FD=2.15877 removing BGS, LRG1, LRG3+ELG1, ELG2, QSO, Lya
```
