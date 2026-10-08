# Getting started with FD-based global sensitivity analysis

## Introduction

Bayesian inference can be sensitive to choices made by the modeller, such as the parameters of the prior or the weight assigned to the likelihood or loss. A sensitivity analysis asks whether plausible alternative choices would lead to substantially different posterior conclusions.

This package performs **global Bayesian sensitivity analysis** using the Fisher divergence (FD). The method is designed to answer three practical questions:

1. How much can the posterior change over the specified range of modelling choices?
2. Which choice produces the largest change?
3. Which choice produces the smallest change?

It comes in two flavours: **FDsens**, where the alternatives are prior hyperparameters within a parametric family (or learning rates), and **FDsens+**, where the alternatives are all priors within an FD ball around the reference prior.

## Reference Bayesian model

The starting point is a reference Bayesian model of your choice. It consists of a reference prior $\Pi_{\mathrm{ref}}$, a reference likelihood or loss, and the resulting reference posterior $\widetilde\Pi_{\mathrm{ref}}$.
Sensitivity is assessed relative to this model.

## Candidate models

The user specifies which modelling choices of the reference Bayesian model may vary and gives a plausible set of values, denoted by $\Gamma$.
Each $\lambda\in\Gamma$ defines a candidate posterior $\widetilde\Pi^\lambda$.
For example, $\Gamma$ could contain:

- a range of prior means or scales (FDsens);
- a range of learning rates (FDsens);
- all priors within a given FD distance of the reference prior (FDsens+).

The neighbourhood $\Gamma$ determines the scope of the sensitivity analysis. It should therefore contain alternatives that are scientifically or practically plausible.

## How global sensitivity is measured

The method compares the reference and candidate posteriors using the Fisher divergence:

$$
\mathrm{FD}(\widetilde\Pi_{\mathrm{ref}}\|\widetilde\Pi^\lambda)
=\mathbb E_{\theta\sim\widetilde\Pi_{\mathrm{ref}}}\left[\left\|
s_{\widetilde\pi_{\mathrm{ref}}}(\theta)-s_{\widetilde\pi^\lambda}(\theta)\right\|^2\right],
$$

where $s_p = \nabla_\theta \log p$ is the score.
Intuitively, the FD measures how differently the two posteriors behave across the region occupied by the reference posterior.
A value of zero means that the candidate and reference posteriors coincide. Larger values indicate larger changes to the posterior.
Scores do not depend on normalising constants, so neither the evidence nor the normalised prior is ever needed. When only the prior changes, the likelihood cancels and the score difference is just the prior score difference.

For any candidate $\lambda$, the FD is estimated using the reference-posterior samples:

$$
\widehat{\mathrm{FD}}_m(\widetilde\Pi_{\mathrm{ref}}\|\widetilde\Pi^\lambda)=\frac{1}{m}\sum_{i=1}^m\left\|s_{\widetilde\pi_{\mathrm{ref}}}(\theta_i)-s_{\widetilde\pi^\lambda}(\theta_i)\right\|^2.
$$

The same reference samples are reused for every candidate. Evaluating a new candidate therefore requires score evaluations but not a new posterior fit.

For the global sensitivity value, the method searches $\Gamma$ for the candidate with the largest estimated FD and the candidate with the smallest estimated FD. The global sensitivity value is their difference:

$$
\widehat{S}_m^{\mathrm{FD}}(\Gamma)=\sup_{\lambda\in\Gamma}\widehat{\mathrm{FD}}_m(\widetilde\Pi_{\mathrm{ref}}\|\widetilde\Pi^\lambda)-\inf_{\lambda\in\Gamma}\widehat{\mathrm{FD}}_m(\widetilde\Pi_{\mathrm{ref}}\|\widetilde\Pi^\lambda).
$$

## FDsens: parametric neighbourhoods

If the candidate priors form an exponential family $\pi(\theta\mid\eta)\propto h(\theta)\exp(\eta^\top T(\theta))$ and $\Gamma$ is a box of natural parameters $\eta$, the estimated FD is a **convex quadratic form** in $\eta$. Its maximum over the box is attained at a corner, so it is found by enumerating the corners, and its minimum is a convex quadratic programme. When the priors of different parameters are independent, the sensitivity is the sum of per-parameter sensitivities, which keeps the problem small in high dimensions.

For learning-rate sensitivity of a generalised posterior $\widetilde\pi^\lambda \propto \exp(-\lambda\,\ell)\,\pi$, the FD reduces to $(\lambda-\lambda_{\mathrm{ref}})^2\,\mathbb E\|\nabla_\theta\ell\|^2$, so no optimisation is needed.

Natural parametrisations matter: the FD is convex in $\eta$ but generally not in, say, the mean and standard deviation, so specify the box in natural parameters (the README lists them per family).

## FDsens+: nonparametric neighbourhoods

FDsens+ takes as neighbourhood every prior within FD radius $r$ of the reference prior,

$$
\mathcal Q_r = \{\Pi : \mathrm{FD}(\Pi_{\mathrm{ref}}\|\Pi) \le r\},
$$

which captures perturbations no single parametric family can, such as skewness, heavy tails or multimodality.
The neighbourhood is approximated by a **sieve** of kernel exponential-family tilts of the reference prior,

$$
\pi_K(\theta)\propto \pi_{\mathrm{ref}}(\theta)\exp\Big(\sum_{k=1}^K \lambda_k\,\kappa(\bar\theta_k,\theta)\Big),
$$

with $K$ kernel centres $\bar\theta_k$. Both the FD objective (estimated with $m$ posterior samples) and the FD constraint (estimated with $l$ prior samples) are quadratic forms in $\lambda$, $\lambda^\top \hat A\lambda$ and $\lambda^\top\hat A_c\lambda$, so the worst case is the leading **generalised eigenvector** of $\hat A v = \mu \hat A_c v$ and

$$
\widehat S^{\mathrm{FD}}_m = r\,\mu_{\sup}.
$$

The sensitivity is linear in $r$; $\mu_{\sup} = \widehat S/r$ is reported as the normalised sensitivity. The cost is $O((m+l)\,d\,K^2)$ for the matrices plus $O(K^3)$ for the eigenvalue problem, and the sieve becomes richer as $K$ grows. In practice:

- **kernel:** a Matérn kernel with smoothness above 1 (Matérn-7/2 by default) or a Gaussian RBF kernel; the lengthscale defaults to the median heuristic on prior samples;
- **centres:** placed by k-means on fresh reference-prior draws, so that they cover where the prior has mass;
- **independent priors:** with a factorised reference prior, solve one small problem per parameter (`independent=True`).

### Choosing the radius

Pick $r$ so that $\mathcal Q_r$ contains the alternatives you care about, which makes FDsens+ conservative relative to them:

- **relative to an exponential-family box** $\Gamma$ (FDsens): take $r \ge \max_{\text{corners } v}\ \mathbb E_{\theta\sim\Pi_{\mathrm{ref}}}\|\nabla T(\theta)(v-\eta_{\mathrm{ref}})\|^2$;
- **relative to Huber contamination** $(1-\varepsilon)\Pi_{\mathrm{ref}} + \varepsilon U$, $U\in\mathcal U$: take $r \ge \sup_{U\in\mathcal U}\mathrm{FD}(\Pi_{\mathrm{ref}}\|U)$.

Because the result is linear in $r$, comparing normalised sensitivities across parameters or models does not require choosing $r$ at all.

## Requirements
- samples $\theta_{1:m}$ from the reference posterior $\widetilde\Pi_{\mathrm{ref}}$;
- the score of the reference prior (and, for learning-rate sensitivity, the gradient of the loss at the samples);
- **FDsens:** the candidate exponential family and a box $\Gamma$ of its natural parameters;
- **FDsens+:** a radius $r$, a kernel and samples from the reference prior (or a way to draw them).

## Interpreting the output
- **global sensitivity value:** the range of posterior change over the neighbourhood;
- **worst-case hyperparameters $\lambda_{\sup}$:** the modelling choice producing the greatest posterior change (for FDsens+, the coefficients of the worst-case tilt);
- **least-sensitive hyperparameters $\lambda_{\inf}$:** the choice producing the smallest posterior change (FDsens);
- **maximum and minimum estimated FD values:** the endpoints used to calculate global sensitivity;
- **per-parameter shares:** with independent priors, how much each parameter contributes to the total.

Raw sensitivity values are hard to interpret on their own; compare them across neighbourhoods, parameters, priors or models.

## References
The parametric methodology is described in [*A computationally-tractable measure of global sensitivity for sampling-based Bayesian inference*](https://arxiv.org/abs/2605.28099).
