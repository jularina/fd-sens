# Learning-rate sensitivity (FDsens)

Global sensitivity of a generalised posterior $\tilde\pi_\beta(\theta) \propto \pi(\theta)\exp(-\beta\, L(\theta))$ to
the learning rate $\beta$, measured by the Fisher divergence (FD) to the reference posterior at $\beta_{\text{ref}}$.
The prior is shared, so its score cancels and

$$
\mathrm{FD}(\tilde\Pi_{\beta_{\text{ref}}} \,\|\, \tilde\Pi_\beta)
= (\beta - \beta_{\text{ref}})^2\, \mathbb{E}_{\theta \sim \tilde\Pi_{\beta_{\text{ref}}}}\big[\|\nabla_\theta L(\theta)\|^2\big],
$$

so the maximum over an interval is at one of its end points and the minimum at the reference (or the nearest end
point). Methodology: [*A computationally-tractable measure of global sensitivity for sampling-based Bayesian
inference*](https://arxiv.org/abs/2605.28099); see also [`GETTING_STARTED.md`](../../GETTING_STARTED.md).

## Usage

```python
from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.learning_rate.sensitivity import lr_sensitivity

model = PosteriorSamplesModel(posterior_samples=draws, loss_grad=loss_grad, loss_lr=1.0)
result = lr_sensitivity(model, lower=0.5, upper=1.5)
print(result)
```

```python
lr_sensitivity(model, lower, upper, lr_ref=None)
```

- `model.loss_grad`: gradient of the unscaled loss $\nabla_\theta L$ at the draws, given as an `(m, d)` array or as
  a function of the draws.
- `model.loss_lr`: the reference learning rate $\beta_{\text{ref}}$ (overridden by `lr_ref`).
- `lower` / `upper`: the interval of candidate learning rates.

No prior is needed. The result (`FDSensitivityResult`) holds `sensitivity`, `fd_min`, `fd_max`, and the learning
rates `lambda_min`, `lambda_max` attaining them.

## Files

| File | Contents |
| --- | --- |
| [`sensitivity.py`](sensitivity.py) | `lr_sensitivity()` |

## Examples

| Script | Analysis |
| --- | --- |
| [`examples/learning_rate/lr_sensitivity.py`](../../examples/learning_rate/lr_sensitivity.py) | Gaussian location model with a squared-error loss, $\beta \in [0.5, 1.5]$ |
