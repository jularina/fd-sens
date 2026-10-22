"""Helpers for interpreting a sensitivity result: compare reference and candidate posterior draws (typically the
candidate at `lambda_max`) through quantiles, kernel density estimates and empirical CDFs, plot each component's
share of the sensitivity, plot the FD of data removals (`plot_removal_single()`, `plot_removal_interactions()`,
`plot_removal_budget()`), and
save the result as JSON.

The draw-comparison functions take `draws`, a dict mapping a fit name to an `(m, d)` array of posterior draws,
e.g. {"reference": reference_draws, "candidate": candidate_draws}, with columns named by `variables`. Each function
writes one file (and, for `plot_quantiles()`, also a CSV) into `output_dir`, which is created if needed.
"""
import dataclasses
import json
import os
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

plt.switch_backend("Agg")

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


# Data-removal colours by role, used when no plot config is given; `plot.color_palette.data_removal_colors` in
# configs/plots/overleaf_plots_settings.yaml overrides them in the same order.
REMOVAL_PALETTE = ["#1f3a5f", "#4b4453", "#8a8d91", "#2f3136", "#5c6b7a", "#c5c7cc"]
REMOVAL_ROLES = ["main", "contrast", "background", "text", "secondary", "light"]


def _removal_style(plot_cfg) -> Dict[str, Any]:
    """Colours by role, rcParams, font size and figure size for the data-removal plots, from an optional plot config
    (e.g. `load_plot_config("configs/plots/overleaf_plots_settings.yaml")`). Every figure is drawn at exactly the
    configured size, with all text at the configured font size."""
    colors = list(REMOVAL_PALETTE)
    rc, size, fs, tight, formats = {}, (3.5, 2.0), 10.0, True, ["pdf"]
    if plot_cfg is not None:
        cfg = plot_cfg.plot
        custom = list(getattr(cfg.color_palette, "data_removal_colors", None) or [])
        colors[:len(custom)] = custom[:len(colors)]
        fs = float(cfg.font.size)
        rc = {
            "font.size": fs,
            "font.family": cfg.font.family,
            "text.usetex": bool(cfg.font.use_tex),
            "text.latex.preamble": r"\usepackage{amsmath}",
            "savefig.dpi": cfg.figure.dpi,
        }
        size = (float(cfg.figure.size.width), float(cfg.figure.size.height))
        tight = bool(getattr(cfg.figure, "tight_layout", True))
    rc.update({"axes.labelsize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs})
    return {"colors": dict(zip(REMOVAL_ROLES, colors)), "rc": rc, "size": size, "fs": fs, "tight": tight,
            "formats": formats}


def _tex_safe(text: str) -> str:
    """Escape characters that LaTeX treats specially, for labels drawn with text.usetex."""
    for ch in ("_", "%", "&", "#"):
        text = text.replace(ch, "\\" + ch)
    return text


def _removal_names(names: Sequence[str], style: Dict[str, Any]) -> list:
    return [_tex_safe(x) for x in names] if style["rc"].get("text.usetex", False) else list(names)


def _save_formats(fig, output_dir: str, file_stem: str, style: Dict[str, Any], layout: bool = True) -> None:
    if layout and style["tight"]:
        fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for fmt in style["formats"]:
        fig.savefig(os.path.join(output_dir, f"{file_stem}.{fmt}"), bbox_inches="tight", facecolor="white")


def _style_removal_axes(ax, colors: Dict[str, str]) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors["light"])
    ax.grid(True, color=colors["light"], linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=colors["text"])


def _budget_curves(result):
    """Removal sizes K, the size k <= K of the best removal for each budget of at most K units, its FD, and the FD of
    removing the K units with the largest A_ii."""
    A = np.asarray(result.A)
    diag = np.diag(A)
    ks = np.array(sorted(result.per_size))
    if ks.size == 0:
        raise ValueError("The result has no removals (max_removed = 0), so there is nothing to plot.")
    best_k = [max((k for k in ks if k <= K), key=lambda k: result.per_size[k]["fd"]) for K in ks]
    best = np.array([result.per_size[k]["fd"] for k in best_k])
    top_k = np.array([A[np.ix_(idx, idx)].sum() for idx in (np.argsort(-diag)[:k] for k in ks)])
    return ks, best_k, best, top_k


def plot_removal_single(result, output_dir: str, file_stem: str = "removal_single", plot_cfg=None):
    """Write `<file_stem>.pdf` for a data-removal result: the FD of removing each observation (group) alone, A_ii,
    sorted from largest to smallest.

    With `plot_cfg` (a loaded plot config) the fonts, LaTeX rendering, dpi, figure size and the
    `data_removal_colors` come from it. Returns the figure."""
    style = _removal_style(plot_cfg)
    c, fs = style["colors"], style["fs"]
    names = _removal_names(result.names, style)
    diag = np.diag(np.asarray(result.A))
    order = np.argsort(-diag)
    n = len(names)

    with plt.rc_context(style["rc"]):
        fig, ax = plt.subplots(figsize=style["size"])
        ax.barh(range(n), diag[order][::-1], color=c["main"], height=0.6)
        ax.set_yticks(range(n), [names[i] for i in order][::-1])
        for i, v in enumerate(diag[order][::-1]):
            ax.annotate(f"{v:.2f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=fs, color=c["text"])
        ax.set_xlabel(r"$A_{ii}$", color=c["text"])
        _style_removal_axes(ax, c)
        ax.grid(False, axis="y")
        ax.set_xlim(0, diag.max() * 1.35 if diag.max() > 0 else 1.0)  # room for the value labels
        _save_formats(fig, output_dir, file_stem, style)
    plt.close(fig)
    return fig


def plot_removal_interactions(result, output_dir: str, file_stem: str = "removal_interactions", plot_cfg=None):
    """Write `<file_stem>.pdf` for a data-removal result: the normalised cross terms A_ik / sqrt(A_ii A_kk), i > k,
    as a lower-triangular heatmap (dark blue: removals reinforce, dark grey-purple: they cancel). The values are in
    `result.A`.

    With `plot_cfg` (a loaded plot config) the fonts, LaTeX rendering, dpi, figure size and the
    `data_removal_colors` come from it. Returns the figure."""
    style = _removal_style(plot_cfg)
    c = style["colors"]
    names = _removal_names(result.names, style)
    A = np.asarray(result.A)
    n = len(names)
    diag = np.diag(A)
    scale = np.sqrt(np.outer(diag, diag))
    corr = np.divide(A, scale, out=np.zeros_like(A), where=scale > 0)
    if n < 2:
        raise ValueError("Cross terms need at least two observations (groups).")
    # The matrix is symmetric: keep the strict lower triangle, rows 1..n-1 against columns 0..n-2.
    corr[np.triu_indices(n)] = np.nan
    corr = corr[1:, :-1]
    vmax = max(float(np.nanmax(np.abs(corr))), 1e-12)
    cmap = LinearSegmentedColormap.from_list("removal", [c["contrast"], "#ffffff", c["main"]])
    cmap.set_bad("white")

    with plt.rc_context(style["rc"]):
        fig, ax = plt.subplots(figsize=style["size"])
        im = ax.imshow(corr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n - 1), names[:-1], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(n - 1), names[1:])
        ax.tick_params(colors=c["text"])
        for spine in ax.spines.values():
            spine.set_color(c["light"])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$A_{ik}/\sqrt{A_{ii}A_{kk}}$", color=c["text"])
        cb.outline.set_edgecolor(c["light"])
        cb.ax.tick_params(colors=c["text"])
        _save_formats(fig, output_dir, file_stem, style)
    plt.close(fig)
    return fig


def _fit_labels_in_axes(fig, ax, labels, tight: bool, max_iter: int = 20) -> None:
    """Raise the upper y-limit until every annotation in `labels` ends below the top of the axes, redoing the
    tight layout (which moves the axes) at each step when `tight`."""
    for _ in range(max_iter):
        if tight:
            fig.tight_layout()
        fig.canvas.draw()
        top = ax.get_window_extent().y1
        overflow = max(lab.get_window_extent().y1 for lab in labels) - top
        if overflow <= 0:
            return
        lo, hi = ax.get_ylim()
        height = ax.get_window_extent().height
        ax.set_ylim(lo, hi + (hi - lo) * 1.05 * overflow / height)


def plot_removal_budget(result, output_dir: str, file_stem: str = "removal_budget", plot_cfg=None):
    """Write `<file_stem>.pdf` for a data-removal result: the sensitivity for each budget of at most K removed
    observations (groups) against removing the K individually most influential ones (largest A_ii). With
    `keep_all=True` every evaluated subset is also shown, at its size. Each optimal point is labelled with the
    units it removes, one name per line, at 0.6 times the configured font size.

    With `plot_cfg` (a loaded plot config) the fonts, LaTeX rendering, dpi, figure size and the
    `data_removal_colors` come from it. Returns the figure."""
    style = _removal_style(plot_cfg)
    c = style["colors"]
    label_fs = max(6.0, 0.6 * style["fs"])  # the removed-set labels only; axes, ticks and legend use the config size
    names = _removal_names(result.names, style)
    ks, best_k, best, top_k = _budget_curves(result)

    with plt.rc_context(style["rc"]):
        fig, ax = plt.subplots(figsize=style["size"])
        if result.fd_values is not None:
            rng = np.random.default_rng(0)
            for k, vals in zip(ks, result.fd_values):
                vals = np.asarray(vals)
                ax.scatter(k + rng.uniform(-0.15, 0.15, len(vals)), vals, s=6, color=c["background"], alpha=0.45,
                           linewidth=0, label="all subsets" if k == ks[0] else None, zorder=1)
        ax.plot(ks, top_k, "--o", color=c["contrast"], markersize=3, linewidth=1.2, label=r"top-$K$", zorder=3)
        ax.plot(ks, best, "-o", color=c["main"], markersize=4, linewidth=1.6, label="optimal", zorder=4)
        labels = []
        for K, k, val in zip(ks, best_k, best):
            # One name per line; combined names such as "LRG3+ELG1" break after the "+" to stay narrow.
            label = "\n".join(names[i].replace("+", "+\n") for i in result.per_size[k]["removed"])
            labels.append(ax.annotate(label, (K, val), textcoords="offset points", xytext=(0, 3), ha="center",
                                      va="bottom", fontsize=label_fs, color=c["text"], linespacing=0.95))
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        ax.set_ylim(0, 1.05 * best.max())
        _fit_labels_in_axes(fig, ax, labels, style["tight"])  # also does the final layout
        ax.set_xlabel("max $K$", color=c["text"])
        ax.set_ylabel(r"$\hat{\rho}_{\mathrm{FD}}$", color=c["text"])
        _style_removal_axes(ax, c)
        ax.legend(frameon=False, labelcolor=c["text"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                  handlelength=1.2, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.1)
        _save_formats(fig, output_dir, file_stem, style, layout=False)
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
