from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from src.nonparametric.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.nonparametric.fisher_divergence import PosteriorFDNonParametric, PriorFDNonParametric
from src.nonparametric.node_sensitivity import compute_node_lambda_star
from src.nonparametric.optimization import OptimisationNonparametricBase

DEFAULT_BASIS_KWARGS = {"num_basis_functions": 100, "method": "kmeans", "nu": 3.5}


@dataclass
class NonparametricSensitivityResult:
    """FDsens+ sensitivity over an FD ball of kernel-exponential-family priors and its worst-case tilt."""

    sensitivity: float
    normalised_sensitivity: float
    radius: float
    lambda_sup: Optional[np.ndarray]
    basis_function: Any
    components: Optional[Dict[str, Dict[str, Any]]] = None

    def __str__(self) -> str:
        lines = [
            "FDsens+ nonparametric prior sensitivity",
            f"  radius r:               {self.radius:.6g}",
            f"  sensitivity:            {self.sensitivity:.6g}",
            f"  normalised sensitivity: {self.normalised_sensitivity:.6g}",
        ]
        if self.components:
            lines.append("  components:")
            for name, comp in self.components.items():
                lines.append(f"    {name}: sensitivity={comp['sensitivity']:.6g} ({comp['sensitivity_share']:.1%})")
        return "\n".join(lines)


def nonparametric_prior_sensitivity(
    model,
    radius: Union[float, Sequence[float]],
    basis: str = "MaternBasisFunction",
    basis_kwargs: Optional[Dict[str, Any]] = None,
    independent: bool = False,
    component_names: Optional[Sequence[str]] = None,
    rel_tol: float = 1e-8,
) -> NonparametricSensitivityResult:
    """FDsens+ prior sensitivity over the FD ball of radius r, solved via the generalised eigenvalue problem."""
    basis_cls = BASIS_FUNCTIONS_REGISTRY[basis]
    kwargs = dict(DEFAULT_BASIS_KWARGS if basis_kwargs is None else basis_kwargs)
    posterior = model.posterior_samples_init
    prior_samples = model.prior_samples_init
    if prior_samples is None:
        raise ValueError("FDsens+ needs reference-prior samples (model.prior_samples_init) or a base prior with sample().")
    # Centres and lengthscale come from a fresh prior draw, independent of the draws used for the constraint.
    center_samples = model.sample_from_base_prior(len(prior_samples))

    if not independent:
        radius = float(radius)
        basis_function = basis_cls(**{**kwargs, "prior_samples": center_samples, "posterior_samples": posterior})
        optimizer = OptimisationNonparametricBase(
            PosteriorFDNonParametric(model), PriorFDNonParametric(model), config={},
            radius=radius, basis_function=basis_function,
        )
        # Restrict to A_c's well-conditioned eigen-subspace, as the per-component route does.
        res = optimizer.optimize_through_generalized_eigenvalue(rel_tol=rel_tol)
        return NonparametricSensitivityResult(
            sensitivity=float(res["theoretical_value"]), normalised_sensitivity=float(res["omega_star"]),
            radius=float(optimizer.r), lambda_sup=res["lambda_star"], basis_function=basis_function,
        )

    dim = posterior.shape[1]
    names = list(component_names) if component_names is not None else [f"theta{j}" for j in range(dim)]
    radii = np.broadcast_to(np.asarray(radius, dtype=float), (dim,))
    components = {}
    for j, name in enumerate(names):
        lam_star, omega, basis_j, _ = compute_node_lambda_star(
            posterior_samples_col=posterior[:, j],
            loc=0.0,
            scale=1.0,
            prior_samples=prior_samples[:, j],
            basis_cls=basis_cls,
            basis_kwargs=kwargs,
            radius_j=float(radii[j]),
            rel_tol=rel_tol,
            center_prior_samples=center_samples[:, j],
        )
        components[name] = {
            "sensitivity": float(radii[j]) * omega,
            "normalised_sensitivity": omega,
            "radius": float(radii[j]),
            "lambda_sup": lam_star,
            "basis_function": basis_j,
        }
    total = sum(c["sensitivity"] for c in components.values())
    for comp in components.values():
        comp["sensitivity_share"] = comp["sensitivity"] / total if total > 0 else 0.0
    total_radius = float(np.sum(radii))
    return NonparametricSensitivityResult(
        sensitivity=total, normalised_sensitivity=total / total_radius if total_radius > 0 else 0.0,
        radius=total_radius, lambda_sup=None, basis_function=None, components=components,
    )
