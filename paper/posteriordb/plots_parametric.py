import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from src.common.utils.distributions import DISTRIBUTION_MAP as _DIST_MAP_EXTPROJ
from paper.plot_utils import apply_plot_rc, save_fig


def _make_pdf(family: str, params: Dict[str, float]):
    """Prefer your DISTRIBUTION_MAP, otherwise fallback for Gaussian/Gamma."""
    fam = family.strip()
    Dist = _DIST_MAP_EXTPROJ[fam]
    dist = Dist(**params)

    def f(x):
        y = dist.pdf(x)
        return np.asarray(y).squeeze()
    return f


def _sample_param_sets(
    ranges: Dict[str, List[float]],
    n: int,
    rng: np.random.Generator,
):
    """
    Sample `n` parameter dictionaries uniformly from a box defined by `ranges`.
    """
    keys = list(ranges)
    bounds = np.asarray([ranges[k] for k in keys], dtype=float)
    lows, highs = bounds[:, 0], bounds[:, 1]

    samples = rng.uniform(low=lows, high=highs, size=(n, len(keys)))
    return [
        {k: float(v) for k, v in zip(keys, row)}
        for row in samples
    ]


def plot_priors_z_scale_one_panel(
    worst_corners_z: Dict[str, Tuple[float, float]],
    mu_z_range: Tuple[float, float],
    sigma_z_range: Tuple[float, float],
    plot_cfg=None,
    output_dir: str = ".",
    filename: str = "priors_z_scale.pdf",
    title: str = r"$\alpha, \beta_{1\cdots 5}, \sigma$",
    sample_n: int = 50,
    seed: int = 27,
    cloud_color: str = "#7c397d",
    x_range: Tuple[float, float] = (-4.0, 4.0),
):
    """Plot reference, neighbourhood and worst-case priors for all components on one z-scale panel."""
    if plot_cfg is not None:
        apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    col_ref = "black"
    col_red = "red"
    alpha_cloud = 0.08
    group_styles = ["-.", "-", ":", "--"]

    if plot_cfg is not None and hasattr(plot_cfg.plot.figure, "size"):
        figsize = (plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        dpi = plot_cfg.plot.figure.dpi
    else:
        figsize = (6.0, 3.2)
        dpi = 200

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    lo, hi = x_range
    z = np.linspace(lo, hi, 500)
    pdf_gauss = lambda m, s: _make_pdf("Gaussian", {"mu": m, "sigma": s})(z)

    box = {"mu": mu_z_range, "sigma": sigma_z_range}
    for p in _sample_param_sets(box, sample_n, rng):
        ax.plot(z, pdf_gauss(p["mu"], p["sigma"]), linewidth=0.9, alpha=alpha_cloud, color=cloud_color)
    ax.plot(z, pdf_gauss(0.0, 1.0), linestyle="--", color=col_ref, linewidth=1.0)

    groups: Dict[Tuple[float, float], List[str]] = {}
    for name, (mu_z, sig_z) in worst_corners_z.items():
        groups.setdefault((round(float(mu_z), 10), round(float(sig_z), 10)), []).append(name)
    for gi, ((mu_z, sig_z), _names) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        ax.plot(z, pdf_gauss(mu_z, sig_z), color=col_red,
                linestyle=group_styles[gi % len(group_styles)], linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\pi$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if plot_cfg is not None and getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    save_fig(fig, output_dir, filename, plot_cfg)


def plot_complexity_bar(
    plot_cfg,
    output_dir: str,
    prefix: str = "ark_param",
    filename: str | None = None,
    use_log10: bool = True,
    qf_full_time_sec=0.0,
    qf_decomp_time_sec=0.0,
    black_box_time_sec=0.0,
):
    try:
        apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f"{prefix}_complexity_bar.pdf"

    labels = [
        "BBO",
        r"$\Gamma_\text{box}$",
        r"$\prod_{j=1}^{d_\Theta}\Gamma_{\text{box}_j}$",
    ]

    def _to_array(v):
        if np.isscalar(v):
            return np.array([float(v)])
        return np.maximum(np.asarray(v, dtype=float), 1e-12)

    data = [_to_array(black_box_time_sec), _to_array(qf_full_time_sec), _to_array(qf_decomp_time_sec)]

    ylab = r"Time (sec.)"

    try:
        palette = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(palette) < len(labels):
        reps = int(np.ceil(len(labels) / len(palette)))
        palette = (palette * reps)[:len(labels)]

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (10, 5),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )

    ax = fig.add_subplot(1, 1, 1)

    if use_log10:
        ax.set_yscale("log")

    print(
        f"Bbox={np.mean(data[0])},  our approach avg. time={np.min(data[1])}, decomposed approach avg. time={np.min(data[2])}")

    for i, (d, color) in enumerate(zip(data, palette)):
        ax.boxplot(
            d,
            positions=[i + 1],
            patch_artist=True,
            widths=0.5,
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.75),
            medianprops=dict(color=color, linewidth=1.0, linestyle="--"),
            whiskerprops=dict(color=color, alpha=0.8),
            capprops=dict(color=color, alpha=0.8),
            flierprops=dict(markerfacecolor=color, markeredgecolor=color, alpha=0.6),
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize="small")
    ax.tick_params(axis="x", length=0, labelcolor="none")

    for i, (d, label) in enumerate(zip(data, labels)):
        if i == 0:
            ax.annotate(label, xy=(i + 1, np.min(d)), xytext=(0, -10),
                        textcoords="offset points", ha="center", va="top", fontsize="small")
        else:
            ax.annotate(label, xy=(i + 1, np.max(d)), xytext=(0, 10),
                        textcoords="offset points", ha="center", va="bottom", fontsize="small")

    ax.set_ylabel(ylab)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()

    try:
        save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")


def plot_component_sensitivity_bar(
    plot_cfg,
    output_dir: str,
    names: list,
    contributions: dict,
    display_names: dict | None = None,
    order: list | None = None,
    group_tail_betas: bool = False,
    tail_beta_keys: list | None = None,
    filename: str | None = None,
    prefix: str = "kilpisjarvi",
    ylabel: str | None = None,
    width_scale: float = 1.0,
):
    try:
        apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f"{prefix}_component_sensitivity.pdf"

    import matplotlib.colors as mcolors

    _display = display_names or {}

    # optionally collapse tail betas into one block
    if group_tail_betas:
        group_keys = tail_beta_keys or ["beta3", "beta4", "beta5"]
        group_val = sum(contributions.get(k, 0.0) for k in group_keys)
        group_label = ", ".join(_display.get(k, k) for k in group_keys)
        _group_key = "__grouped_betas__"
        contributions = {k: v for k, v in contributions.items() if k not in group_keys}
        contributions[_group_key] = group_val
        names = [k for k in names if k not in group_keys] + [_group_key]
        _display = dict(_display)
        _display[_group_key] = group_label

    percentages = contributions

    # display order: low to high from bottom to top; default to names order
    display_order = order if order is not None else list(names)

    # color per component based on contribution rank (highest → #450314, lowest → #5b9bd5)
    c_high = np.array(mcolors.to_rgb("#450314"))
    c_low = np.array(mcolors.to_rgb("#5b9bd5"))
    ranked = sorted(names, key=lambda k: contributions[k])
    n = len(ranked)
    color_map = {
        k: mcolors.to_hex(c_low + t * (c_high - c_low))
        for k, t in zip(ranked, np.linspace(0, 1, n))
    }

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width * width_scale, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (3, 6),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )
    ax = fig.add_subplot(1, 1, 1)

    bottom = 0.0
    min_label_pct = 4.0
    for k in display_order:
        color = color_map[k]
        pct = percentages[k]
        ax.bar(0, pct, bottom=bottom, color=color, alpha=0.5, edgecolor="white", linewidth=0.5, width=0.6)
        if pct >= min_label_pct:
            label = _display.get(k, k)
            ax.text(
                0, bottom + pct / 2,
                f"{label} {pct:.1f}%",
                ha="center", va="center",
                fontsize="x-small", color="black",
                usetex=False,
            )
        bottom += pct

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"], usetex=False)
    ax.set_ylabel(ylabel if ylabel is not None else r"$\widehat{S}_m^{\mathrm{FD}}(\Gamma_j)$ \%")

    ax.tick_params(axis="x", length=0, labelcolor="none")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()

    try:
        save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")


def plot_acf_comparison(
    plot_cfg,
    output_dir: str,
    acf_ref: np.ndarray,
    acf_corner: np.ndarray,
    ref_color: str = "#7c397d",
    corner_color: str = "#5b9bd5",
    ref_label: str = "Reference",
    corner_label: str = "Corner",
    filename: str = "kilpisjarvi_acf_comparison.pdf",
):
    try:
        apply_plot_rc(plot_cfg)
    except Exception:
        pass

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (6, 4),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )

    lags = np.arange(len(acf_ref))
    ax.plot(lags, acf_ref, color=ref_color, linewidth=1.5, label=ref_label)
    ax.plot(lags, acf_corner, color=corner_color, linewidth=1.5, label=corner_label)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")

    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    try:
        save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")


def plot_posterior_predictive_with_data(
    plot_cfg,
    output_dir: str,
    x_years: np.ndarray,
    y_uncentered: np.ndarray,
    x_pred_years: np.ndarray,
    pred_mean: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    pred_label: str,
    filename: str,
    pred_color: str | None = None,
    ylim=None,
    show_ylabel: bool = True,
):
    """Plot uncentred observations by year with posterior predictive bands overlaid."""
    try:
        apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    x_years = np.asarray(x_years, dtype=float)
    y_uncentered = np.asarray(y_uncentered, dtype=float)
    x_pred_years = np.asarray(x_pred_years, dtype=float)

    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    obs_color = colors[0]
    if pred_color is None:
        pred_color = colors[1]

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    ax.plot(
        x_years, y_uncentered,
        linestyle="None", marker="x", markersize=3.0, markeredgewidth=1.0,
        color="black", zorder=1,
    )

    ax.fill_between(x_pred_years, pred_lo, pred_hi, color=pred_color, alpha=0.25, zorder=2)
    ax.plot(x_pred_years, pred_mean, linewidth=1.5, color=pred_color, label=pred_label, zorder=3)

    ax.set_xlabel("Year")
    if show_ylabel:
        ax.set_ylabel("Temperature (°C)")
    else:
        ax.set_ylabel("Temperature (°C)", color="none")
        ax.tick_params(axis="y", labelcolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    # ax.legend(frameon=False, loc="upper right")

    try:
        save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")


