# Data-removal sensitivity (FDsens-data)

Global sensitivity of a posterior to removing a small fraction of the data, measured by the Fisher divergence (FD)
to the full-data reference posterior. One run of the sampler on the full data is enough to score every removal.

## Method

For weights $w \in \{0,1\}^n$ ($w_i = 0$ removes $x_i$) the weighted posterior is
$\tilde\pi_w(\theta) \propto \pi(\theta) \exp\big(\sum_i w_i \log \ell(\theta; x_i)\big)$, and
$\tilde\pi_{\text{ref}} = \tilde\pi_{\mathbf 1_n}$. Removing at most $K = \lfloor \alpha n \rfloor$ points gives the
sensitivity

$$
S_{\mathrm{FD}}(W_\alpha) = \max_{w \in W_\alpha} \mathrm{FD}(\tilde\Pi_{\text{ref}} \,\|\, \tilde\Pi_w).
$$

With $J(\theta) = [\nabla_\theta \log \ell(\theta; x_1), \ldots, \nabla_\theta \log \ell(\theta; x_n)] \in
\mathbb{R}^{d \times n}$ the score difference is $J(\theta)\, v$, where $v = \mathbf 1_n - w$ marks the removed points.
The prior score cancels, so the prior is not needed. Given reference-posterior draws
$\tilde\theta_1, \ldots, \tilde\theta_m$,

$$
\hat\rho_{\mathrm{FD}}(\tilde\Pi_w) = v^\top A\, v, \qquad
A = \frac{1}{m} \sum_{j=1}^m J(\tilde\theta_j)^\top J(\tilde\theta_j) \succeq 0,
$$

and the estimated sensitivity is the cardinality-constrained binary quadratic program
$\max_{v \in \{0,1\}^n,\ \mathbf 1^\top v \le K} v^\top A v$. $A_{ii}$ is the FD of removing $x_i$ alone, and
$A_{ik}$ shows whether removing $x_i$ and $x_k$ together reinforces ($> 0$) or masks ($< 0$) their effects.

**Optimisation (corner points).** $v \mapsto v^\top A v$ is convex, so its maximum over the relaxation
$\{v \in [0,1]^n : \mathbf 1^\top v \le K\}$ is attained at a vertex, i.e. a binary $v$ with at most $K$ ones. All
$N(n, K) = \sum_{k=0}^{K} \binom{n}{k}$ vertices are evaluated, which is exact. This is the only optimiser for now:
it handles grouped data (e.g. $G = 7$ BAO blocks, 127 subsets) and small $K$
(e.g. $n = 30$, $K = 3$: 4,526 subsets). Larger problems are refused above `max_evaluations`.

**Grouped data.** When the likelihood factorises over groups $\mathcal G_1, \ldots, \mathcal G_G$ (studies, sites,
survey blocks), whole groups are removed. The group scores are
$\gamma_{\mathcal G_g}(\theta) = \nabla_\theta \log \ell(\theta; x_{\mathcal G_g})$, and the same quadratic form
applies with $A_{\mathcal G}$ ($G \times G$). Either pass the group scores directly as `data_grad`, which you must do
when points within a group are correlated (as in BAO), or pass per-observation scores together with `groups`,
which sums the scores within each group.

**Whitening (optional).** With `whiten=True` the FD is measured in $z = L^{-1}(\theta - \bar\theta)$, where
$L L^\top$ is the posterior covariance. Scores become $L^\top \nabla_\theta \log \ell$, so parameters on very
different scales (e.g. $\omega_b \sim 10^{-2}$ and $H_0 \sim 10^2$) contribute comparably.

## Usage

```python
from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.data_removal.sensitivity import data_removal_sensitivity

model = PosteriorSamplesModel(posterior_samples=draws, data_grad=data_grad)   # base_prior is optional here
result = data_removal_sensitivity(model, alpha=0.1)
print(result)
```

```python
data_removal_sensitivity(model, alpha=None, max_removed=None, groups=None, names=None, whiten=False,
                         method="corner", max_evaluations=10_000_000, keep_all=False)
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

## Files

| File | Contents |
| --- | --- |
| [`fisher_divergence.py`](fisher_divergence.py) | `PosteriorFDDataRemoval`: scores at the draws, the matrix $A$, FD and its standard error for a given removal |
| [`optimization.py`](optimization.py) | `OptimizationCornerPointsDataRemoval`: exhaustive corner-point evaluation, chunked by removal size |
| [`sensitivity.py`](sensitivity.py) | `data_removal_sensitivity()` and `DataRemovalSensitivityResult` |

## Examples

| Script | Analysis |
| --- | --- |
| [`examples/data_removal_sensitivity.py`](../../examples/data_removal_sensitivity.py) | Gaussian location model with two outliers, $\alpha = 0.1$ |
| [`examples/data_removal_sensitivity_bao.py`](../../examples/data_removal_sensitivity_bao.py) | DESI DR2 BAO ($\Lambda$CDM): removing up to 6 of the 7 independent BAO blocks |

### BAO

[`paper/bao/bao.py`](../../paper/bao/bao.py) holds the DESI DR2 BAO likelihood (data in [`data/bao/`](../../data/bao)).
The 13 measurements form $G = 7$ independent Gaussian blocks, one with a single point and six with two correlated
points, and inform $\theta = (\omega_b, \omega_c, H_0)$. The block scores are

$$
\gamma_{\mathcal G_g}(\theta) = J_{M_g}(\theta)^\top \Sigma_g^{-1}\big(D_g - M_g(\theta)\big),
$$

with the Jacobian $J_{M_g}$ of the theory prediction computed by complex-step differentiation (`block_scores`). The
reference posterior is sampled with an affine-invariant ensemble sampler (`run_posterior`).

```sh
pdm run python examples/data_removal_sensitivity_bao.py
```

This takes a few seconds and writes `sensitivity_result.json`, `removal_interactions.png` and `removal_budget.png` to
`outputs/data_removal/bao_LCDM/`. Example output (whitened FD):

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

Because of the cross terms, the optimal pair (BGS + LRG1) does not include the single most influential block (LRG2).
