# Getting started with FD-based global sensitivity analysis

## Introduction

Bayesian inference can be sensitive to choices made by the modeller, such as the parameters of the prior or the weight assigned to the likelihood or loss. A sensitivity analysis asks whether plausible alternative choices would lead to substantially different posterior conclusions.

This approach performs **global Bayesian sensitivity analysis** using the Fisher divergence (FD). The method is designed to answer three practical questions:

1. How much can the posterior change over the specified range of modelling choices?
2. Which hyperparameter choice produces the largest change?
3. Which choice produces the smallest change?

## Reference Bayesian model

The starting point is a reference Bayesian model of your choice. It consists of a reference prior, a reference likelihood or loss, and the resulting reference posterior $\widetilde\Pi_{\mathrm{ref}}$.
Sensitivity is assessed relative to this model.

## Candidate models

The user specifies which modelling choices of the reference Bayesian model may vary and gives a plausible set of values, denoted by $\Gamma$.
Each $\lambda\in\Gamma$ defines a candidate posterior $\widetilde\Pi^\lambda$.
For example, $\Gamma$ could contain:

- a range of prior means or scales;
- a range of learning rates;
- parameters controlling dependence in a joint prior;
- etc.

The neighbourhood $\Gamma$ determines the scope of the sensitivity analysis. It should therefore contain alternatives that are scientifically or practically plausible.

## How global sensitivity is measured

The method compares the reference and candidate posteriors using the Fisher divergence:

$$
\mathrm{FD}(\widetilde\Pi_{\mathrm{ref}}\|\widetilde\Pi^\lambda)
=\mathbb E_{\theta\sim\widetilde\Pi_{\mathrm{ref}}}\left[\left\|
s_{\widetilde\pi_{\mathrm{ref}}}(\theta)-s_{\widetilde\pi^\lambda}(\theta)\right\|^2\right].
$$

Intuitively, the FD measures how differently the two posteriors behave across the region occupied by the reference posterior.
A value of zero means that the candidate and reference posteriors coincide. Larger values indicate larger changes to the posterior.

For any candidate $\lambda$, the FD is estimated using the reference-posterior samples:

$$
\widehat{\mathrm{FD}}_m(\widetilde\Pi_{\mathrm{ref}}\|\widetilde\Pi^\lambda)=\frac{1}{m}\sum_{i=1}^m\left\|s_{\widetilde\pi_{\mathrm{ref}}}(\theta_i)-s_{\widetilde\pi^\lambda}(\theta_i)\right\|^2.
$$

The same reference samples are reused for every candidate. Evaluating a new candidate therefore requires score evaluations but not a new posterior fit.

For the global sensitivity value, the method searches $\Gamma$ for the candidate with the largest estimated FD and the candidate with the smallest estimated FD. The global sensitivity value is their difference:

$$
\widehat{S}_m^{\mathrm{FD}}(\Gamma)=\sup_{\lambda\in\Gamma}\widehat{\mathrm{FD}}_m(\widetilde\Pi_{\mathrm{ref}}\|\widetilde\Pi^\lambda)-\inf_{\lambda\in\Gamma}\widehat{\mathrm{FD}}_m(\widetilde\Pi_{\mathrm{ref}}\|\widetilde\Pi^\lambda).
$$

## Requirements
- samples $\theta_{1:m}$ from the reference posterior $\widetilde\Pi_{\mathrm{ref}}$;
- the score of the reference posterior;
- the score of the candidate posterior $\widetilde\Pi^{\lambda}$;
- set $\Gamma$ of hyperparameter $\lambda$ bounds.

## Interpreting the output
- **global sensitivity value:** the range of posterior change over the neighbourhood;
- **worst-case hyperparameters $\lambda_{\sup}$:** the modelling choice producing the greatest posterior change;
- **least-sensitive hyperparameters $\lambda_{\inf}$:** the choice producing the smallest posterior change;
- **maximum and minimum estimated sensitivity values:** the endpoints used to calculate global sensitivity.


## References
The complete methodology is described in [*A computationally-tractable measure of global sensitivity for sampling-based Bayesian inference*](https://arxiv.org/abs/2605.28099).