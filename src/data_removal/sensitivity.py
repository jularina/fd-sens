from dataclasses import dataclass
from math import floor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data_removal.fisher_divergence import PosteriorFDDataRemoval
from src.data_removal.optimization import OptimizationCornerPointsDataRemoval, number_of_corner_points


@dataclass
class DataRemovalSensitivityResult:
    """FD sensitivity to removing at most `max_removed` observations (or groups) and the worst-case removal."""

    sensitivity: float
    sensitivity_se: float
    removed: np.ndarray
    removed_names: List[str]
    max_removed: int
    n_units: int
    names: List[str]
    A: np.ndarray
    per_size: Dict[int, Dict[str, Any]]
    n_evaluated: int
    method: str
    whiten: bool
    subsets: Optional[List[np.ndarray]] = None
    fd_values: Optional[List[np.ndarray]] = None

    def __str__(self) -> str:
        lines = [
            "FD data-removal sensitivity",
            f"  optimisation: {self.method} ({self.n_evaluated} corner points)",
            f"  removal budget: at most {self.max_removed} of {self.n_units}",
            f"  sensitivity: {self.sensitivity:.6g} (se {self.sensitivity_se:.2g})",
            f"  worst-case removal: {', '.join(self.removed_names) if self.removed_names else '(none)'}",
            "  per removal size:",
        ]
        for k, entry in self.per_size.items():
            lines.append(f"    k={k}: FD={entry['fd']:.6g} removing {', '.join(entry['removed_names'])}")
        return "\n".join(lines)


def removal_budget(n_units: int, alpha: Optional[float] = None, max_removed: Optional[int] = None) -> int:
    """K = floor(alpha * n) from a fraction alpha in (0, 1), or the given max_removed; exactly one must be set."""
    if (alpha is None) == (max_removed is None):
        raise ValueError("Give exactly one of alpha (fraction removed) or max_removed (number removed).")
    if alpha is not None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1).")
        return int(floor(alpha * n_units + 1e-9))  # guard against e.g. 0.29 * 100 = 28.999...
    if not 0 <= int(max_removed) <= n_units:
        raise ValueError(f"max_removed must lie in [0, {n_units}].")
    return int(max_removed)


def data_removal_sensitivity(
    model,
    alpha: Optional[float] = None,
    max_removed: Optional[int] = None,
    groups: Optional[Sequence[Sequence[int]]] = None,
    names: Optional[Sequence[str]] = None,
    whiten: bool = False,
    method: str = "corner",
    max_evaluations: int = 10_000_000,
    keep_all: bool = False,
) -> DataRemovalSensitivityResult:
    """Sensitivity to removing at most floor(alpha * n) observations: max_{v in {0,1}^n, 1^T v <= K} v^T A v.

    `model.data_score` gives the per-observation log-likelihood gradients; with `groups` they are summed within each
    group and whole groups are removed. The FD is 0 at the reference (nothing removed), so the sensitivity is the
    maximal FD.
    """
    if method != "corner":
        raise ValueError("method must be 'corner' (exhaustive corner-point evaluation).")
    estimator = PosteriorFDDataRemoval(model, groups=groups, whiten=whiten)
    n = estimator.n
    names = [str(x) for x in names] if names is not None else [str(i) for i in range(n)]
    if len(names) != n:
        raise ValueError(f"names has {len(names)} entries but there are {n} observations/groups.")
    K = removal_budget(n, alpha=alpha, max_removed=max_removed)

    n_evaluated = number_of_corner_points(n, K)
    if n_evaluated > max_evaluations:
        raise ValueError(
            f"Corner-point enumeration needs {n_evaluated:.3g} evaluations (n={n}, K={K}), above "
            f"max_evaluations={max_evaluations:.3g}; reduce alpha, group the data, or raise max_evaluations."
        )

    A = estimator.compute_fisher_quadratic_form()
    optimizer = OptimizationCornerPointsDataRemoval(A)
    per_size, everything = optimizer.evaluate_all_corners(K, keep_all=keep_all)

    for entry in per_size.values():
        entry["removed_names"] = [names[i] for i in entry["removed"]]
        entry["se"] = estimator.fd_standard_error(entry["removed"])

    if per_size:
        k_best = max(per_size, key=lambda k: per_size[k]["fd"])
        best = per_size[k_best]
        sensitivity, removed, se = max(best["fd"], 0.0), best["removed"], best["se"]
    else:  # K = 0: only the reference posterior is allowed
        sensitivity, removed, se = 0.0, np.array([], dtype=int), 0.0

    return DataRemovalSensitivityResult(
        sensitivity=float(sensitivity), sensitivity_se=float(se), removed=removed,
        removed_names=[names[i] for i in removed], max_removed=K, n_units=n, names=names, A=A,
        per_size=per_size, n_evaluated=n_evaluated, method="corner_points", whiten=whiten,
        subsets=[s for s, _ in everything] if everything is not None else None,
        fd_values=[v for _, v in everything] if everything is not None else None,
    )
