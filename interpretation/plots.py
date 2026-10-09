"""Helpers for interpreting a sensitivity result: compare reference and candidate posterior draws (typically the
candidate at `lambda_max`) through quantiles, kernel density estimates and empirical CDFs, plot each component's
share of the sensitivity, and save the result as JSON.

The draw-comparison functions take `draws`, a dict mapping a fit name to an `(m, d)` array of posterior draws,
e.g. {"reference": reference_draws, "candidate": candidate_draws}, with columns named by `variables`. Each function
writes one file (and, for `plot_quantiles()`, also a CSV) into `output_dir`, which is created if needed.
"""
import dataclasses
import json
import os
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Categorical slots in fixed order: blue for the first fit (typically the reference), orange for the second
# (typically the candidate); further fits take the next slots.
FIT_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Sequential blue ramp (light to dark) for ordered blocks; starts at a step that still reads on white.
SHARE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#1f1f1e", "#5f5e58", "#e6e5df"
DEFAULT_PROBS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _style_axes(ax) -> None:
    """Recessive axes: no top/right spines, light grid behind the data, secondary-ink ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _fit_colors(fit_names: Sequence[str]) -> Dict[str, str]:
    if len(fit_names) > len(FIT_PALETTE):
        raise ValueError(f"At most {len(FIT_PALETTE)} fits can be compared in one plot.")
    return dict(zip(fit_names, FIT_PALETTE))


def _as_columns(draws: Dict[str, Any], variables: Optional[Sequence[str]]) -> Dict[str, np.ndarray]:
    """Validate `draws` and return {fit: (m, d) array}, checking every fit has the same number of columns."""
    if not isinstance(draws, dict) or not draws:
        raise ValueError("`draws` must be a non-empty dict, e.g. {'reference': ..., 'candidate': ...}.")
    out = {}
    for name, arr in draws.items():
        arr = np.asarray(arr, dtype=float)
        out[name] = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    widths = {a.shape[1] for a in out.values()}
    if len(widths) != 1:
        raise ValueError("Every fit in `draws` must have the same number of columns.")
    if variables is not None and len(variables) != widths.pop():
        raise ValueError("`variables` must name every column of the draws.")
    return out


def _variable_names(columns: Dict[str, np.ndarray], variables: Optional[Sequence[str]]) -> list:
    d = next(iter(columns.values())).shape[1]
    if variables is not None:
        return list(variables)
    return ["theta"] if d == 1 else [f"theta[{j}]" for j in range(d)]


def _panels(n: int):
    """One small-multiple panel per variable, at most three per row."""
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat[:n]


def _save(fig, output_dir: str, file_stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{file_stem}.png"), dpi=200, bbox_inches="tight", facecolor="white")


def summarise_quantiles(
    draws: Dict[str, Any], variables: Optional[Sequence[str]] = None, probs: Sequence[float] = DEFAULT_PROBS,
) -> pd.DataFrame:
    """Posterior mean, sd and quantiles: one row per fit x variable."""
    columns = _as_columns(draws, variables)
    names = _variable_names(columns, variables)
    rows = []
    for fit, arr in columns.items():
        for j, var in enumerate(names):
            x = arr[:, j]
            row = {"fit": fit, "variable": var, "mean": x.mean(), "sd": x.std(ddof=1)}
            row.update({f"{100 * p:g}%": q for p, q in zip(probs, np.quantile(x, probs))})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_quantiles(
    draws: Dict[str, Any],
    output_dir: str,
    variables: Optional[Sequence[str]] = None,
    probs: Sequence[float] = DEFAULT_PROBS,
    file_stem: str = "quantiles",
) -> pd.DataFrame:
    """Write `<file_stem>.csv` (the quantile table) and `<file_stem>.png` (median and 90% interval per fit and
    variable, one colour per fit) into `output_dir`, and return the table."""
    columns = _as_columns(draws, variables)
    names = _variable_names(columns, variables)
    table = summarise_quantiles(columns, names, probs)
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(os.path.join(output_dir, f"{file_stem}.csv"), index=False)

    colors = _fit_colors(list(columns))
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.7 * len(names) * max(1, len(columns) / 2)))
    offsets = np.linspace(0.15, -0.15, len(columns)) if len(columns) > 1 else [0.0]
    for (fit, arr), dy in zip(columns.items(), offsets):
        low, mid, high = np.quantile(arr, [0.05, 0.5, 0.95], axis=0)
        y = np.arange(len(names))[::-1] + dy
        ax.hlines(y, low, high, color=colors[fit], linewidth=2, label=fit)
        ax.plot(mid, y, "o", color=colors[fit], markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    ax.set_yticks(np.arange(len(names))[::-1], names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("value (median, 90% interval)", color=TEXT_SECONDARY)
    _style_axes(ax)
    ax.grid(False, axis="y")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, output_dir, file_stem)
    plt.close(fig)
    return table


def plot_kde(
    draws: Dict[str, Any], output_dir: str, variables: Optional[Sequence[str]] = None, file_stem: str = "kde",
):
    """Write `<file_stem>.png`: one kernel density panel per variable, one curve per fit. Returns the figure."""
    columns = _as_columns(draws, variables)
    names = _variable_names(columns, variables)
    colors = _fit_colors(list(columns))
    fig, axes = _panels(len(names))
    for j, (ax, var) in enumerate(zip(axes, names)):
        lo = min(arr[:, j].min() for arr in columns.values())
        hi = max(arr[:, j].max() for arr in columns.values())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        grid = np.linspace(lo - pad, hi + pad, 400)
        for fit, arr in columns.items():
            density = gaussian_kde(arr[:, j])(grid)
            ax.fill_between(grid, density, color=colors[fit], alpha=0.15, linewidth=0)
            ax.plot(grid, density, color=colors[fit], linewidth=2, label=fit)
        ax.set_title(var, color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
        _style_axes(ax)
    axes[0].set_ylabel("density", color=TEXT_SECONDARY)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY,
               loc="upper left", bbox_to_anchor=(1.0, 0.95))
    fig.tight_layout()
    _save(fig, output_dir, file_stem)
    plt.close(fig)
    return fig


def plot_ecdf(
    draws: Dict[str, Any], output_dir: str, variables: Optional[Sequence[str]] = None, file_stem: str = "ecdf",
):
    """Write `<file_stem>.png`: one empirical CDF panel per variable, one curve per fit. Returns the figure."""
    columns = _as_columns(draws, variables)
    names = _variable_names(columns, variables)
    colors = _fit_colors(list(columns))
    fig, axes = _panels(len(names))
    for j, (ax, var) in enumerate(zip(axes, names)):
        for fit, arr in columns.items():
            x = np.sort(arr[:, j])
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=colors[fit], linewidth=2, label=fit)
        ax.set_title(var, color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.set_ylim(0, 1.02)
        _style_axes(ax)
    axes[0].set_ylabel("empirical CDF", color=TEXT_SECONDARY)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY,
               loc="upper left", bbox_to_anchor=(1.0, 0.95))
    fig.tight_layout()
    _save(fig, output_dir, file_stem)
    plt.close(fig)
    return fig


def plot_component_shares(result, output_dir: str, file_stem: str = "component_shares"):
    """Write `<file_stem>.png`: a 100%-stacked bar of each component's share of the total sensitivity, for a result
    computed with `independent=True` (FDsens or FDsens+). Returns the figure."""
    if not getattr(result, "components", None):
        raise ValueError("The result has no components; compute it with independent=True.")
    names = list(result.components)
    shares = np.array([100.0 * result.components[n]["sensitivity_share"] for n in names])
    if not np.isfinite(shares).all() or shares.sum() <= 0:
        raise ValueError("Shares are undefined (total sensitivity is zero), so they cannot be plotted.")
    if len(names) > len(SHARE_RAMP):
        raise ValueError(f"At most {len(SHARE_RAMP)} components can be shown; group the rest.")
    idx = np.round(np.linspace(0, len(SHARE_RAMP) - 1, len(names))).astype(int) if len(names) > 1 else [2]
    colors = [SHARE_RAMP[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7, 1.6))
    left = 0.0
    for name, share, color in zip(names, shares, colors):
        ax.barh(0, share, left=left, color=color, height=0.6, edgecolor="white", linewidth=2, label=name)
        if share >= 6:
            text_color = "white" if color in SHARE_RAMP[2:] else TEXT_PRIMARY
            ax.text(left + share / 2, 0, f"{share:.1f}%", ha="center", va="center", fontsize=9,
                    fontweight="bold", color=text_color)
        left += share
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("share of total sensitivity (%)", color=TEXT_SECONDARY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, ncol=min(len(names), 6),
              loc="lower left", bbox_to_anchor=(0.0, 1.0))
    _save(fig, output_dir, file_stem)
    plt.close(fig)
    return fig


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return type(x).__name__  # e.g. the basis-function object of an FDsens+ result


def save_sensitivity_result(result, output_dir: str, file_stem: str = "sensitivity_result") -> Dict[str, Any]:
    """Write `<file_stem>.json` with the result's fields (sensitivity, FD extremes, lambdas, components, ...)
    and return the written dict. Objects that are not plain values are recorded by their class name."""
    fields = ({f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
              if dataclasses.is_dataclass(result) else vars(result))
    values = _jsonable(fields)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{file_stem}.json"), "w") as f:
        json.dump(values, f, indent=2)
    return values
