import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from typing import Dict, Tuple, List
from scipy.special import logsumexp

from utils.distributions import DISTRIBUTION_MAP as _DIST_MAP_EXTPROJ


def _deep_get(cfg, path, default=None):
    """Safe nested getter that works with dicts or OmegaConf/objects."""
    if cfg is None:
        return default
    cur = cfg
    for key in path.split('.'):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{type1cm}",
    })


def _new_fig_ax(plot_cfg):
    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    return fig, ax


def _save_fig(fig, output_dir: str, filename: str, plot_cfg):
    os.makedirs(output_dir, exist_ok=True)
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)


def _palette(plot_cfg, n: int) -> List[str]:
    base = list(plot_cfg.plot.color_palette.colors)
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]


def _latex_name(plot_cfg, key: str, default: str) -> str:
    try:
        return getattr(plot_cfg.plot.param_latex_names, key)
    except Exception:
        return default


def _auto_x_range_from_ref(family: str, ref_params: Dict[str, float]) -> Tuple[float, float]:
    low = family.strip().lower()

    # Gaussian / Normal
    if low in ("gaussian", "normal"):
        mu = float(ref_params["mu"])
        s = float(ref_params["sigma"])
        return (mu - 4.0 * s, mu + 4.0 * s)

    if low in ("halfcauchy"):
        gamma = float(ref_params.get("gamma", ref_params.get("scale", 1.0)))
        return (0.0, 10.0 * gamma)

    # LogNormal with parameters mu_log, sigma_log (accept "sigma-log" typo too)
    if low == "lognormal":
        mu_log = float(ref_params["mu_log"])
        sigma_log = float(ref_params.get("sigma_log", ref_params.get("sigma-log")))
        lo = np.exp(mu_log - 5.0 * sigma_log)
        hi = np.exp(mu_log + 5.0 * sigma_log)
        # Ensure strictly positive lower bound
        return (max(1e-8, lo), hi)

    # Gamma with shape alpha and scale theta; support x > 0
    if low == "gamma":
        alpha = float(ref_params["alpha"])
        theta = float(ref_params["theta"])
        mean = alpha * theta
        sd = np.sqrt(alpha) * theta
        return (0.0, max(1e-8, mean + 6.0 * sd))

    # Fallback
    return (-5.0, 5.0)


def _linspace_pad(xmin, xmax, n=1200, pad=0.05):
    if not np.isfinite([xmin, xmax]).all():
        xmin, xmax = -5.0, 5.0
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    span = xmax - xmin
    return np.linspace(xmin - pad*span, xmax + pad*span, n)


def _make_pdf(family: str, params: Dict[str, float]):
    fam = family.strip()
    Dist = _DIST_MAP_EXTPROJ[fam]
    dist = Dist(**params)

    def f(x):
        y = dist.pdf(x)
        return np.asarray(y).squeeze()
    return f


def _parse_family_from_target(target: str) -> str:
    return target.split(".")[-1]


def _get_base_prior_spec(cfg) -> Tuple[str, Dict[str, float]]:
    d = _deep_get(cfg, "data.base_prior", None)
    target = d.get("_target_", "")
    family = _parse_family_from_target(target)
    params = {k: float(v) for k, v in d.items() if k != "_target_"}
    return family, params


def _sample_param_sets(ranges: Dict[str, List[float]], n: int, rng: np.random.Generator):
    keys = list(ranges.keys())
    lows = np.array([ranges[k][0] for k in keys], dtype=float)
    highs = np.array([ranges[k][1] for k in keys], dtype=float)
    out = []
    for _ in range(n):
        u = rng.random(len(keys))
        vals = lows + (highs - lows) * u
        out.append({k: float(v) for k, v in zip(keys, vals)})
    return out


def plot_theta_prior(
    worst_corner: Dict,
    cfg,
    plot_cfg,
    output_dir: str,
    sample_n: int = 30,
    seed: int = 123,
    filename: str = None,
):
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    col_ref = "black"
    col_red = "red"
    alpha_cloud = 0.18
    palette_full = list(plot_cfg.plot.color_palette.colors)

    def _cloud_colors(n: int, skip_first: int = 2) -> List[str]:
        base = palette_full[skip_first:] if skip_first < len(palette_full) else palette_full
        reps = int(np.ceil(n / len(base)))
        return (base * reps)[:n]

    fam_ref, ref_params = _get_base_prior_spec(cfg)
    ranges = cfg.ksd.optimize.prior.Gamma["parameters_box_range"]["ranges"]
    fam_ms = worst_corner["family"]
    ms_params = worst_corner["params"]

    X = np.linspace(0, 15, 200)

    pdf_ref = _make_pdf(fam_ref, ref_params)
    pdf_ms = _make_pdf(fam_ms,  ms_params)

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi
    )
    ax = fig.add_subplot(111)

    ax.plot(X, pdf_ref(X), linestyle="--", color=col_ref, linewidth=1.2,
            label=plot_cfg.plot.param_latex_names["baseprior"])
    for (p, c) in zip(_sample_param_sets(ranges, sample_n, rng), _cloud_colors(sample_n, 2)):
        pdf_c = _make_pdf(fam_ms, p)
        ax.plot(X, pdf_c(X), linewidth=0.9, alpha=alpha_cloud, color=c)
    ax.plot(X, pdf_ms(X), color=col_red, linewidth=1.8, label=r"$\Pi$ at " +
            plot_cfg.plot.param_latex_names["argoptimisationProblemParam"])

    ax.set_ylabel(plot_cfg.plot.param_latex_names.priorsimple)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=2)

    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_lr_vs_ksd(
    lr_grid,
    plot_cfg,
    output_dir: str,
    filename: str = "sbi_experiment_turin_lr_scan.pdf",
    xlabel: str = None,
    title: str = None,
):
    _apply_plot_rc(plot_cfg)

    pairs = []
    for item in lr_grid:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            lr, ksd = float(item[0]), float(item[1])
            pairs.append((lr, ksd))
        elif isinstance(item, dict):
            lr_key = "lr" if "lr" in item else ("learning_rate" if "learning_rate" in item else None)
            ksd_key = "ksd" if "ksd" in item else ("value" if "value" in item else None)
            if lr_key is None or ksd_key is None:
                continue
            pairs.append((float(item[lr_key]), float(item[ksd_key])))

    if not pairs:
        raise ValueError("plot_lr_vs_ksd: empty or unrecognized lr_grid format.")

    # sort by lr
    pairs.sort(key=lambda t: t[0])
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)

    # ---- figure ----
    W = plot_cfg.plot.figure.size.width
    H = plot_cfg.plot.figure.size.height
    fig, ax = plt.subplots(1, 1, figsize=(W, H), dpi=plot_cfg.plot.figure.dpi)

    # line + markers
    ax.plot(xs, ys, marker="o", linewidth=1.5, color=plot_cfg.plot.color_palette.colors[1], markersize=4)

    # aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # remove y-axis ticks and labels completely
    # ax.set_yticks([])
    ax.set_ylabel(plot_cfg.plot.param_latex_names["estimatedKSDposteriorsShort"])
    # ax.tick_params(axis="y", left=False, right=False, labelleft=False)

    # labels
    if xlabel is None:
        xlabel = r"lr"
    ax.set_xlabel(xlabel)

    xmin, xmax = float(xs.min()), float(xs.max())
    if xmin > 0 and (xmax / max(xmin, 1e-20) > 1e3):
        ax.set_xscale("log")

    if title:
        ax.set_title(title)

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_lr_vs_ksd_multi(
    lr_grids,
    ds,
    beta_refs,
    plot_cfg,
    output_dir: str,
    filename: str = "ising_experiment_multi_d_lr_vs_ksd.pdf",
    xlabel: str = None,
    legend: bool = True,
    ylbl: str = "estimatedKSDposteriorsShort",
):
    _apply_plot_rc(plot_cfg)

    W = plot_cfg.plot.figure.size.width
    H = plot_cfg.plot.figure.size.height
    fig, ax = plt.subplots(1, 1, figsize=(W, H), dpi=plot_cfg.plot.figure.dpi)

    colors = plot_cfg.plot.color_palette.colors
    for i, (grid, d) in enumerate(zip(lr_grids, ds)):
        xs = np.array(grid[:, 0], dtype=float)
        ys = np.array(grid[:, 1], dtype=float)
        idx = np.argsort(xs)
        xs, ys = xs[idx], ys[idx]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.0,
                color=colors[i % len(colors)], label=fr"{d}")

        max_idx = np.argmax(ys)
        beta_ref = beta_refs[i]
        y_ref = np.interp(beta_ref, xs, ys)
        ax.plot(xs[max_idx], ys[max_idx], marker="*", color="red", markersize=6, zorder=5)
        ax.plot(beta_ref, y_ref, marker="x", color="black", markersize=6, zorder=5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(plot_cfg.plot.param_latex_names[ylbl])
    ax.set_xlabel(xlabel if xlabel is not None else r"lr")

    xmin = min(grid[:, 0].min() for grid in lr_grids)
    xmax = max(grid[:, 0].max() for grid in lr_grids)
    if xmin > 0 and (xmax / max(xmin, 1e-20) > 1e3):
        ax.set_xscale("log")

    if legend:
        ax.legend()

    _save_fig(fig, output_dir, filename, plot_cfg)


def _make_figure(plot_cfg):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    fig.subplots_adjust(
        left=0.24,
        right=0.98,
        bottom=0.20,
        top=0.98,
    )
    return fig, ax


def plot_lr_vs_loss_multi(
    lr_grids,
    losses,
    beta_refs,
    plot_cfg,
    output_dir: str,
    filename: str,
    xlabel: str = r"$\beta$",
    legend: bool = True,
    ylbl: str = "estimatedFDposteriorsQuadraticForm",
    logy: bool = False,
    method: str = "",
):
    _apply_plot_rc(plot_cfg)

    fig, ax = _make_figure(plot_cfg)

    colors = plot_cfg.plot.color_palette.colors
    line_styles = ["-", "--", "-."]

    xmin_candidates = []
    xmax_candidates = []
    y_refs = []

    for i, (grid, loss_name, beta_ref) in enumerate(zip(lr_grids, losses, beta_refs)):
        xs = np.array(grid[:, 0], dtype=float)
        ys = np.array(grid[:, 1], dtype=float)

        idx = np.argsort(xs)
        xs, ys = xs[idx], ys[idx]

        left = max(beta_ref - 0.3, 0.01)
        right = beta_ref + 0.3

        mask = (xs >= left) & (xs <= right)
        xs_plot = xs[mask]
        ys_plot = ys[mask]

        ax.plot(
            xs_plot,
            ys_plot,
            linewidth=1.8,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i % len(colors)],
            label=loss_name,
        )

        max_idx = np.argmax(ys_plot)
        y_ref = np.interp(beta_ref, xs_plot, ys_plot)
        y_refs.append(y_ref)

        ax.plot(
            xs_plot[max_idx],
            ys_plot[max_idx],
            marker="*",
            color="red",
            markersize=6,
            zorder=5,
        )
        ax.plot(
            beta_ref,
            y_ref,
            marker="x",
            color="black",
            markersize=6,
            zorder=5,
        )

        xmin_candidates.append(left)
        xmax_candidates.append(right)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(min(y_refs), 10 ** 7)

    ax.set_xlim(0.0, 1.0)

    if method != "syring":
        ax.tick_params(axis="y", labelleft=False)

    if method == "syring":
        ax.set_ylabel(plot_cfg.plot.param_latex_names[ylbl])
        if legend:
            ax.legend()

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_lr_vs_method_multi(
    lr_grids,
    methods,
    beta_refs,
    plot_cfg,
    output_dir: str,
    filename: str,
    xlabel: str = r"$\beta$",
    legend: bool = True,
    ylbl: str = "estimatedFDposteriorsQuadraticForm",
    logy: bool = False,
    loss: str = "",
    ylim=None,
    lr_bars=None,
):
    _apply_plot_rc(plot_cfg)

    fig, ax = _make_figure(plot_cfg)

    colors = plot_cfg.plot.color_palette.colors
    line_styles = ["-", "--", "-."]

    y_refs = []

    for i, (grid, method_name, beta_ref) in enumerate(zip(lr_grids, methods, beta_refs)):
        if loss == "ksd" and method_name == "Lyddon et.al.":
            continue
        xs = np.array(grid[:, 0], dtype=float)
        ys = np.array(grid[:, 1], dtype=float)

        idx = np.argsort(xs)
        xs, ys = xs[idx], ys[idx]

        left = max(beta_ref - 0.05, 0.01)
        right = beta_ref + 0.05

        mask = (xs >= left) & (xs <= right)
        xs_plot = xs[mask]
        ys_plot = ys[mask]

        ax.plot(
            xs_plot,
            ys_plot,
            linewidth=1.8,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i % len(colors)],
            label=method_name,
        )

        y_ref = np.interp(beta_ref, xs_plot, ys_plot)
        y_refs.append(y_ref)

        # ax.plot(
        #     beta_ref,
        #     y_ref,
        #     marker="x",
        #     color="black",
        #     markersize=6,
        #     zorder=5,
        # )

    if loss == "ksd" and lr_bars is not None:
        ax.axvline(x=lr_bars[0], color="red", linewidth=1.0, linestyle="--")
        ax.axvline(x=lr_bars[1], color="red", linewidth=1.0, linestyle="--")
        ax.axvspan(lr_bars[0], lr_bars[1], color="grey", alpha=0.08)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if loss == "dfd":
        ax.set_xlabel(xlabel)

    if logy:
        ax.set_yscale("log")

    if ylim is not None:
        ax.set_ylim(ylim)

    # if loss == "ksd":
    #     ax.set_xlim(0.0, 1.0)
    # else:
    #     ax.set_xlim(0.0, 1.0)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    if loss == "dfd":
        from matplotlib.ticker import ScalarFormatter
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-4, -4))
        ax.yaxis.set_major_formatter(fmt)

    ax.set_ylabel(plot_cfg.plot.param_latex_names[ylbl])

    if legend:
        ax.legend()

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_sdp_densities_only(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-2, 5),
    resolution: int = 200,
    filename="toy_gaussian_model_nonparametric_optimisation_densities.pdf",
    ylbl: str = "estimatedKSDposteriorsShort",
) -> None:
    """
    Plot true prior density and normalized SDP densities (no samples).
    Matches styling (legend, colors, fonts, LaTeX) used in plot_sdp_comparisons_multiple_radii.
    Saves a single-panel figure to PDF.
    """
    os.makedirs(output_dir, exist_ok=True)

    _apply_plot_rc(plot_cfg)

    # Figure
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Grid + features
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    Phi_x = basis_function.evaluate(x)  # (resolution, D)

    # True prior density
    prior_density = prior_distribution.pdf(x).flatten()

    # Colors (cycle through palette for each radius curve)
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]  # fallback

    # Names/labels map
    names = plot_cfg.plot.param_latex_names
    ksd_label = names.get(ylbl)
    xlabel = names.get("theta")
    ylabel = names.get("nonparametric_prior", "Density")
    true_label = names.get("baseprior", "True Prior Density")
    approx_sym = r"$\approx$"
    geq_sym = r"$\geq$"

    # Plot SDP densities (normalized on grid)
    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f = (Phi_x @ psi).flatten()
        logZ = logsumexp(f) + np.log(dx)
        p_hat = np.exp(f - logZ)
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label:.2f} ({int(ksd)})"
        ax.plot(
            x.flatten(),
            p_hat,
            label=label,
            linewidth=1.5,
            color=color,
        )

    # Plot true prior density (on top/last or first as you prefer)
    ax.plot(
        x.flatten(),
        prior_density,
        label=true_label,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    centers = np.array(basis_function.centers).flatten()
    # ax.scatter(centers, np.zeros_like(centers), color="grey", marker="o", s=10, alpha=0.6, label=None)
    y_min = -0.08 * max(prior_density)  # 5% below axis
    ax.scatter(
        centers,
        np.full_like(centers, y_min),
        color="grey",
        marker="o",
        s=10,
        alpha=0.8,
        label=None,
    )

    # Axes labels & styling
    ax.set_xlabel(xlabel, fontsize=plot_cfg.plot.font["size"])
    ax.set_ylabel(ylabel, fontsize=plot_cfg.plot.font["size"])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend styling identical to your other function
    leg = ax.legend(
        title=ksd_label,
        loc="center left",
        bbox_to_anchor=(1.02, 0.6),
        frameon=False,
        labelspacing=0.3,
        handlelength=0.2,
        handletextpad=0.3,
        borderpad=0.1,
    )
    for t in leg.get_texts():
        t.set_wrap(True)
    leg.get_title().set_ha("right")
    leg._legend_box.align = "right"

    plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"]*0.8)
    plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"]*0.8)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    # Save and close
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_posterior_histograms(
    samples_by_loss: Dict[str, Dict[str, np.ndarray]],
    loss_labels: Dict[str, str],
    method_labels: Dict[str, str],
    plot_cfg,
    output_dir: str,
    dnum: int = 1000,
    ref_samples_by_loss: Dict[str, Dict[str, np.ndarray]] = None,
):
    """Create one figure per loss with three overlapping KDE density lines (one per LR method)."""
    from scipy.stats import gaussian_kde
    _apply_plot_rc(plot_cfg)
    colors = _palette(plot_cfg, 3)
    line_styles = ["-", "--", "-."]
    os.makedirs(output_dir, exist_ok=True)

    for loss, loss_label in loss_labels.items():
        fig, ax = plt.subplots(
            figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
            dpi=plot_cfg.plot.figure.dpi,
        )

        ref_labeled = False
        for i, (method, method_label) in enumerate(method_labels.items()):
            if loss == "ksd" and method == "lyddon":
                continue
            color = colors[i]
            ls = line_styles[i]
            samples = samples_by_loss.get(loss, {}).get(method)
            if samples is None:
                continue
            flat = samples.flatten()
            xs = np.linspace(3.0, 7.0, 300)
            ax.plot(xs, gaussian_kde(flat)(xs), color=color, linestyle=ls, linewidth=1.8, label=method_label)

            if ref_samples_by_loss is not None:
                ref = ref_samples_by_loss.get(loss, {}).get(method)
                if ref is not None:
                    ref_label = "Reference" if not ref_labeled else None
                    ax.plot(xs, gaussian_kde(ref.flatten())(xs),
                            color="black", linewidth=1.2, linestyle="--", label=ref_label)
                    ref_labeled = True

        ax.set_xlim(3.0, 7.0)
        ax.set_xlabel(r"$\theta$")
        ax.set_title(loss_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        leg = ax.legend(frameon=False, loc="center right")
        plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"] * 0.8)

        filename = f"ising-posterior-histograms-{loss}--posteriors.pdf"
        _save_fig(fig, output_dir, filename, plot_cfg)


def plot_posterior_per_combination(
    samples_by_loss: Dict[str, Dict[str, np.ndarray]],
    ref_samples_by_loss: Dict[str, Dict[str, np.ndarray]],
    loss_labels: Dict[str, str],
    method_labels: Dict[str, str],
    plot_cfg,
    output_dir: str,
    dnum: int = 1000,
):
    """One figure per (loss, method) combination showing reference and worst-case KDE."""
    from scipy.stats import gaussian_kde
    _apply_plot_rc(plot_cfg)
    colors = _palette(plot_cfg, 3)
    line_styles = ["-", "--", "-."]
    os.makedirs(output_dir, exist_ok=True)
    xs = np.linspace(3.0, 7.0, 300)
    methods_list = list(method_labels.keys())

    for loss, loss_label in loss_labels.items():
        for method, method_label in method_labels.items():
            if loss == "ksd":
                xs = np.linspace(10.0, 120.0, 300)
            if loss == "ksd" and method == "lyddon":
                continue

            i = methods_list.index(method)
            color = colors[i]
            ls = line_styles[i]

            fig, ax = plt.subplots(
                figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
                dpi=plot_cfg.plot.figure.dpi,
            )

            ref = ref_samples_by_loss.get(loss, {}).get(method)
            if ref is not None:
                ax.plot(xs, gaussian_kde(ref.flatten())(xs),
                        color="black", linestyle="--", linewidth=1.4, label="Reference")

            samples = samples_by_loss.get(loss, {}).get(method)
            if samples is not None:
                flat = samples.flatten()
                ax.hist(flat, bins=20, color=color, alpha=0.3,
                        edgecolor="white", linewidth=0.5, density=True, label="Worst-case")
                kde_wc = gaussian_kde(flat)
                ax.plot(xs, kde_wc(xs), color=color, linestyle=ls, linewidth=1.4)

            if loss == "ksd":
                ax.set_xlim(10, 120)
            else:
                ax.set_xlim(3.7, 7.0)
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel("Density")
            ax.set_title(f"{loss_label}, {method_label}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            leg = ax.legend(frameon=False, loc="center right")
            plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"] * 0.8)

            filename = f"ising-posterior-{loss}-{method}-{dnum}-worstcase-reference-posterior.pdf"
            _save_fig(fig, output_dir, filename, plot_cfg)


def plot_loss_gradient_times_density(
    X: np.ndarray,
    ising_grads,
    loss: str,
    samples_by_method: Dict[str, np.ndarray],
    method_labels: Dict[str, str],
    theta_min: float,
    theta_max: float,
    n_theta: int,
    plot_cfg,
    output_dir: str,
    filename: str,
):
    """
    For each method, plot grad_loss(theta) * KDE_method(theta) on a single axis.
    Produces one figure per loss with 3 lines (one per method).
    """
    import torch
    from scipy.stats import gaussian_kde

    _apply_plot_rc(plot_cfg)

    theta_grid = np.linspace(theta_min, theta_max, n_theta)
    param_tensor = torch.tensor(theta_grid, dtype=torch.float32).view(-1, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        if loss == "pseudolikelihood":
            grad_vals = -ising_grads.grad_pseudologlikelihood(param_tensor, X_tensor).squeeze().numpy()
        else:
            grad_vals = ising_grads.grad_dfd_loss(param_tensor, X_tensor).squeeze().numpy()

    finite_mask = np.isfinite(grad_vals)

    fig, ax = _make_figure(plot_cfg)

    colors = _palette(plot_cfg, 3)
    line_styles = ["-", "--", "-."]

    for i, (method, label) in enumerate(method_labels.items()):
        samples = samples_by_method.get(method)
        if samples is None or samples.size < 2:
            continue
        kde = gaussian_kde(samples.flatten())
        density = kde(theta_grid)
        product = grad_vals ** 2 * density
        ax.plot(theta_grid[finite_mask], product[finite_mask],
                color=colors[i], linestyle=line_styles[i], linewidth=1.5, label=label)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if loss == "dfd":
        ax.set_xlabel(r"$\theta$")
    if loss == "pseudolikelihood":
        ax.set_ylabel(r"$(\nabla_\theta l^{\mathrm{PL}}(\theta))^2\,\tilde{\pi}_\mathrm{ref}(\theta)$")
    else:
        ax.set_ylabel(r"$(\nabla_\theta l^{\mathrm{DFD}}(\theta))^2\,\tilde{\pi}_\mathrm{ref}(\theta)$")
    # ax.legend(frameon=False, loc="lower right")

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_loss_gradient_vs_theta(
    X: np.ndarray,
    ising_grads,
    loss: str,
    samples_by_method: Dict[str, np.ndarray],
    method_labels: Dict[str, str],
    theta_min: float,
    theta_max: float,
    n_theta: int,
    plot_cfg,
    output_dir: str,
    filename: str,
):
    """
    Plot PL or DFD loss gradient vs theta.
    """
    import torch

    _apply_plot_rc(plot_cfg)

    theta_grid = np.linspace(theta_min, theta_max, n_theta)
    param_tensor = torch.tensor(theta_grid, dtype=torch.float32).view(-1, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        if loss == "pseudolikelihood":
            grad_vals = -ising_grads.grad_pseudologlikelihood(param_tensor, X_tensor).squeeze().numpy()
        else:
            grad_vals = ising_grads.grad_dfd_loss(param_tensor, X_tensor).squeeze().numpy()

    fig, ax1 = _make_figure(plot_cfg)

    ax1.plot(theta_grid, grad_vals ** 2, color="black", linewidth=1.5)
    ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if loss == "dfd":
        ax1.set_xlabel(r"$\theta$")
    if loss == "pseudolikelihood":
        ax1.set_ylabel(r"$(\nabla_\theta l^{\mathrm{PL}}(\theta))^2$")
        _fmt = ScalarFormatter(useMathText=True)
        _fmt.set_powerlimits((5, 5))
        ax1.yaxis.set_major_formatter(_fmt)
    else:
        ax1.set_ylabel(r"$(\nabla_\theta l^{\mathrm{DFD}}(\theta))^2$")
        _fmt = ScalarFormatter(useMathText=True)
        _fmt.set_powerlimits((2, 2))
        ax1.yaxis.set_major_formatter(_fmt)

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_basis_colored_by_eigenvector(
    basis_function,
    eigenvector: np.ndarray,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-3, 12),
    resolution: int = 300,
    filename: str = "basis_colored_by_eigenvector.pdf",
    cmap_name: str = "coolwarm",
    linewidth_base: float = 1.5,
    linewidth_scale: float = 1.0,
    show_colorbar: bool = True,
) -> None:
    """
    Plot each basis function φ_k(x) colored by the corresponding eigenvector entry (length K).

    Args:
        basis_function: object with .evaluate(x)->(N,K) and .centers (optional).
        eigenvector: shape (K,), entries used to color each basis curve.
        plot_cfg: your existing plot config (used by _apply_plot_rc).
        output_dir: directory to save the PDF.
        domain: (xmin, xmax) for plotting grid.
        resolution: number of grid points.
        filename: output PDF filename.
        cmap_name: matplotlib colormap name (use a diverging cmap for +/- weights).
        linewidth_base: base linewidth for curves.
        linewidth_scale: additional scaling by |eigenvector| (0 disables scaling).
        show_colorbar: whether to draw a colorbar for eigenvector values.

    Saves:
        A single-panel PDF showing all K basis curves colored by eigenvector entries.
    """
    from matplotlib import cm, colors
    os.makedirs(output_dir, exist_ok=True)

    # Apply your global plot styling (must exist in your code base)
    _apply_plot_rc(plot_cfg)

    # Prepare grid and evaluate basis
    x = np.linspace(domain[0], domain[1], resolution)[:, None]  # (N,1)
    Phi_x = basis_function.evaluate(x)  # (N, K)
    if Phi_x.ndim == 3 and Phi_x.shape[1] == 1:
        Phi_x = Phi_x[:, 0, :]
    N, K = Phi_x.shape

    eigenvector = np.asarray(eigenvector).reshape(-1)
    if eigenvector.shape[0] != K:
        raise ValueError(f"eigenvector length {eigenvector.shape[0]} does not match number of basis functions K={K}.")

    # Figure
    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Build colormap normalizer centered at 0 for +/- weights
    vmax = np.max(np.abs(eigenvector)) if np.any(eigenvector) else 1.0
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    # (Optional) draw centers along the baseline for reference
    centers = np.array(getattr(basis_function, "centers", [])).flatten()
    # Estimate a baseline a bit below visible y-range using a rough amplitude proxy
    # (use the max absolute value over all basis to scale baseline offset)
    amp_proxy = float(np.max(np.abs(Phi_x))) if Phi_x.size else 1.0
    y_min = -0.08 * amp_proxy
    if centers.size > 0:
        ax.scatter(
            centers,
            np.full_like(centers, y_min),
            color="grey",
            marker="o",
            s=10,
            alpha=0.8,
            label=None,
        )

    # Plot each basis function colored by its eigenvector entry
    for k in range(K):
        yk = Phi_x[:, k]
        w = eigenvector[k]
        color = cmap(norm(w))

        # Optional: make lines a bit thicker for larger |weight|
        lw = linewidth_base + linewidth_scale * (abs(w) / vmax if vmax > 0 else 0.0)

        ax.plot(
            x.flatten(),
            yk,
            color=color,
            linewidth=lw,
            alpha=0.95,
        )

    # Labels & axes styling
    names = plot_cfg.plot.param_latex_names
    xlabel = names.get("theta", r"$\theta$")
    ylabel = names.get("basis_functions", "Basis Functions")
    ax.set_xlabel(xlabel, fontsize=plot_cfg.plot.font["size"])
    ax.set_ylabel(ylabel, fontsize=plot_cfg.plot.font["size"])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colorbar to explain mapping eigenvector value -> color
    if show_colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(names.get("eigenvector_value", "Eigenvector Entry"), rotation=90)

    # Tight layout if configured
    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    # Save and close
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
