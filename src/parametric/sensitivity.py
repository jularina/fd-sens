from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing

from src.common.fisher_divergence import PosteriorFDBase
from src.parametric.fisher_divergence import PosteriorFDParametric
from src.parametric.optimization import OptimizationCornerPointsCompositePrior


@dataclass
class FDSensitivityResult:
    """Global FD sensitivity together with the hyperparameters attaining its maximum and minimum."""

    sensitivity: float
    fd_min: float
    fd_max: float
    lambda_min: np.ndarray
    lambda_max: np.ndarray
    analysis: str
    method: str
    components: Optional[Dict[str, Dict[str, Any]]] = None

    def __str__(self) -> str:
        lines = [
            f"FD {self.analysis} sensitivity",
            f"  optimisation: {self.method}",
            f"  sensitivity: {self.sensitivity:.6g}",
            f"  minimum FD:  {self.fd_min:.6g} at lambda_min = {np.round(self.lambda_min, 6)}",
            f"  maximum FD:  {self.fd_max:.6g} at lambda_max = {np.round(self.lambda_max, 6)}",
        ]
        if self.components:
            lines.append("  components:")
            for name, comp in self.components.items():
                lines.append(f"    {name}: sensitivity={comp['sensitivity']:.6g} ({comp['sensitivity_share']:.1%})")
        return "\n".join(lines)


def _eta_components(natural_box: Dict[str, Dict[str, Sequence[float]]], dim: int) -> list:
    """Convert {name: {"eta_1": (lo, hi), "eta_2": (lo, hi)}} into the optimiser's component list."""
    if len(natural_box) != dim:
        raise ValueError(
            f"natural_box has {len(natural_box)} components but the posterior samples have {dim} columns; "
            "give one entry per parameter, in column order."
        )
    comps = []
    for name, box in natural_box.items():
        eta_1 = [float(v) for v in box["eta_1"]]
        eta_2 = [float(v) for v in box["eta_2"]]
        if eta_1[0] > eta_1[1] or eta_2[0] > eta_2[1]:
            raise ValueError(f"Box for '{name}' must be given as (lower, upper).")
        comps.append({"name": name, "eta_range": {"eta_1": eta_1, "eta_2": eta_2}})
    return comps


def prior_sensitivity(
    model,
    natural_box: Dict[str, Dict[str, Sequence[float]]],
    method: str = "quadratic",
    independent: bool = False,
    **black_box_kwargs,
) -> FDSensitivityResult:
    """Parametric (FDsens) prior sensitivity over a box of exponential-family natural parameters."""
    if model.prior_candidate is None:
        raise ValueError("Prior sensitivity needs model.candidate_prior (the exponential-family candidate family).")
    dim = model.posterior_samples_init.shape[1]
    comps = _eta_components(natural_box, dim)
    names = [c["name"] for c in comps]

    estimator = PosteriorFDParametric(model=model)
    optimizer = OptimizationCornerPointsCompositePrior(estimator, {"eta_components": comps}, loss_config={})

    if method == "black_box":
        if independent:
            raise ValueError("independent=True is only available with method='quadratic'.")
        res = optimizer.black_box_optimize_prior_box_global(**black_box_kwargs)
        return FDSensitivityResult(
            sensitivity=res.S_hat, fd_min=res.val_inf, fd_max=res.val_sup,
            lambda_min=res.eta_inf, lambda_max=res.eta_sup, analysis="prior", method="black_box",
        )
    if method != "quadratic":
        raise ValueError("method must be 'quadratic' or 'black_box'.")

    if not independent:
        corners, eta_max = optimizer.evaluate_all_prior_corners()
        eta_min, fd_min = optimizer.minimize_prior_full_qp()
        fd_max = float(corners[0][1])
        return FDSensitivityResult(
            sensitivity=fd_max - fd_min, fd_min=fd_min, fd_max=fd_max,
            lambda_min=eta_min, lambda_max=np.asarray(eta_max), analysis="prior", method="quadratic_corner",
        )

    corners, eta_max = optimizer.evaluate_all_prior_corners_per_component(component_names=names)
    eta_min, fd_min = optimizer.minimize_prior_per_component_qp(names)
    components = {}
    for name in names:
        comp_max = float(corners[name][0][1])
        components[name] = {
            "sensitivity": comp_max - fd_min[name],
            "fd_min": fd_min[name],
            "fd_max": comp_max,
            "lambda_min": eta_min[name],
            "lambda_max": eta_max[name],
        }
    total = sum(c["sensitivity"] for c in components.values())
    for comp in components.values():
        comp["sensitivity_share"] = comp["sensitivity"] / total if total > 0 else 0.0
    return FDSensitivityResult(
        sensitivity=total,
        fd_min=float(sum(c["fd_min"] for c in components.values())),
        fd_max=float(sum(c["fd_max"] for c in components.values())),
        lambda_min=np.concatenate([components[n]["lambda_min"] for n in names]),
        lambda_max=np.concatenate([components[n]["lambda_max"] for n in names]),
        analysis="prior", method="quadratic_corner", components=components,
    )


def lr_sensitivity(model, lower: float, upper: float, lr_ref: Optional[float] = None) -> FDSensitivityResult:
    """Learning-rate sensitivity over [lower, upper], using FD(lr) = (lr - lr_ref)^2 E||grad loss||^2."""
    if lower > upper:
        raise ValueError("lower must not exceed upper.")
    estimator = PosteriorFDBase(model=model)
    lr_ref = estimator.beta_ref if lr_ref is None else float(lr_ref)
    mean_sq_grad = float(np.mean(np.sum(estimator.g * estimator.g, axis=1)))

    def fd(lr):
        return (lr - lr_ref) ** 2 * mean_sq_grad

    lr_max = lower if fd(lower) >= fd(upper) else upper
    lr_min = min(max(lr_ref, lower), upper)
    return FDSensitivityResult(
        sensitivity=fd(lr_max) - fd(lr_min), fd_min=fd(lr_min), fd_max=fd(lr_max),
        lambda_min=np.array([lr_min]), lambda_max=np.array([lr_max]),
        analysis="learning_rate", method="closed_form",
    )


def _global_minimise(func, bounds, method: str, seed: int, maxiter: int, n_restarts: int):
    """Minimise func over a box with scipy's dual annealing or differential evolution, keeping the best restart."""
    if method not in ("dual_annealing", "differential_evolution"):
        raise ValueError("method must be 'dual_annealing' or 'differential_evolution'.")
    best = None
    for r in range(max(1, n_restarts)):
        if method == "dual_annealing":
            res = dual_annealing(func, bounds=bounds, seed=seed + r, maxiter=maxiter)
        else:
            res = differential_evolution(func, bounds=bounds, seed=seed + r, maxiter=maxiter, polish=True)
        if best is None or res.fun < best.fun:
            best = res
    return best


def prior_sensitivity_black_box(
    model,
    score_prior_candidate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lower: Sequence[float],
    upper: Sequence[float],
    score_prior_ref: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    method: str = "dual_annealing",
    seed: int = 0,
    maxiter: int = 200,
    n_restarts: int = 1,
) -> FDSensitivityResult:
    """FDsens prior sensitivity for any candidate family, given its prior score, by global optimisation over a box."""
    lower = np.atleast_1d(np.asarray(lower, dtype=float))
    upper = np.atleast_1d(np.asarray(upper, dtype=float))
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("lower and upper must have the same length and satisfy lower <= upper.")
    draws = model.posterior_samples_init
    score_ref = model.prior_init.grad_log_pdf(draws) if score_prior_ref is None else score_prior_ref(draws)
    score_ref = np.asarray(score_ref, dtype=float).reshape(draws.shape)

    def fd(lam: np.ndarray) -> float:
        diff = score_ref - np.asarray(score_prior_candidate(draws, np.asarray(lam, dtype=float))).reshape(draws.shape)
        return float(np.mean(np.sum(diff * diff, axis=1)))

    bounds = list(zip(lower, upper))
    kwargs = dict(method=method, maxiter=maxiter, n_restarts=n_restarts)
    res_max = _global_minimise(lambda lam: -fd(lam), bounds, seed=seed, **kwargs)
    res_min = _global_minimise(fd, bounds, seed=seed + n_restarts, **kwargs)
    lam_max, lam_min = np.asarray(res_max.x, dtype=float), np.asarray(res_min.x, dtype=float)
    fd_max, fd_min = fd(lam_max), fd(lam_min)
    return FDSensitivityResult(
        sensitivity=fd_max - fd_min, fd_min=fd_min, fd_max=fd_max,
        lambda_min=lam_min, lambda_max=lam_max, analysis="prior", method=f"black_box ({method})",
    )
