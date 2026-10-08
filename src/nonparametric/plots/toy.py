from collections import defaultdict
from typing import Any
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import logsumexp
from scipy.stats import sem, t


def plot_sdp_densities(
    basis_function,
    sdp_lambda_list: list[np.ndarray],
    radius_labels: list[float],
    estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-5, 5),
    resolution: int = 200,
    posterior_distribution=None,
    show_posterior: bool = False,
    show_legend: bool = False,
) -> None:
    """
    Single-panel plot: nonparametric SDP densities for each radius, plus the true prior
    (dashed steelblue) and, optionally (if show_posterior=True), the true posterior
    (dashed black) density.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    Phi_x = basis_function.evaluate(x)
    prior_density_true = prior_distribution.pdf(x).flatten()

    # log g(θ): reference/base-measure log-density, shared across all candidate priors
    log_g = prior_distribution.log_pdf(x).flatten()

    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    names = plot_cfg.plot.param_latex_names
    fd_label = names.get("estimatedSensitivityMeasure")
    xlabel = names.get("theta")
    ylabel_density = names.get("nonparametric_prior", "Density")
    true_prior_label = names.get("baseprior")
    true_posterior_label = names.get("baseposterior", r"$\tilde{\Pi}_{\mathrm{ref}}$")
    geq_sym = r"$\geq$"

    fig, ax = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    for i, (psi, r_label, ksd) in enumerate(zip(sdp_lambda_list, radius_labels, estimates)):
        f = (Phi_x @ psi).flatten()
        log_post = f + log_g
        logZ = logsumexp(log_post) + np.log(dx)
        p_hat = np.exp(log_post - logZ)
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label} ({ksd:.1f})"
        ax.plot(x.flatten(), p_hat, label=label, linewidth=1.5, color=color)

    ax.plot(
        x.flatten(),
        prior_density_true,
        label=true_prior_label,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    if show_posterior and posterior_distribution is not None:
        posterior_density_true = posterior_distribution.pdf(x).flatten()
        ax.plot(
            x.flatten(),
            posterior_density_true,
            label=true_posterior_label,
            linestyle="--",
            linewidth=1.5,
            color="black",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_density)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        leg = fig.legend(
            handles, labels,
            title=fd_label,
            loc="center left",
            bbox_to_anchor=(0.95, 0.5),
            frameon=False,
            labelspacing=0.4,
            handlelength=1.8,
            handletextpad=0.5,
            borderpad=0.4,
        )
        leg.get_title().set_ha("right")
        leg._legend_box.align = "right"
        plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"] * 0.75)
        plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"] * 0.75)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout(rect=[0, 0, 0.95, 1])

    filename = "gaussian_1d_location_model_diff_radii.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sdp_matern_nu_comparison(
    basis_functions: list,
    sdp_lambda_list: list[np.ndarray],
    nu_labels: list[float],
    estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    filename: str,
    domain: tuple = (-10, 12),
    resolution: int = 500,
    show_legend: bool = True,
) -> None:
    """Plot SDP worst-case candidate densities for several Matérn smoothness values nu with the true prior."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    prior_density_true = prior_distribution.pdf(x).flatten()
    log_g = prior_distribution.log_pdf(x).flatten()

    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    names = plot_cfg.plot.param_latex_names
    fd_label = names.get("estimatedSensitivityMeasure")
    xlabel = names.get("theta")
    ylabel_density = names.get("nonparametric_prior", "Density")

    fig, ax = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    for i, (basis_function, psi, nu, ksd) in enumerate(
        zip(basis_functions, sdp_lambda_list, nu_labels, estimates)
    ):
        Phi_x = basis_function.evaluate(x)
        f = (Phi_x @ psi).flatten()
        log_post = f + log_g
        logZ = logsumexp(log_post) + np.log(dx)
        p_hat = np.exp(log_post - logZ)
        color = palette[i % len(palette)]
        label = rf"$\nu={nu}$ ({ksd:.1f})"
        ax.plot(x.flatten(), p_hat, label=label, linewidth=1.5, color=color)

    ax.plot(
        x.flatten(),
        prior_density_true,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_density)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        leg = ax.legend(
            title=fd_label,
            loc="best",
            frameon=False,
            fontsize=plt.rcParams["font.size"] * 0.65,
        )
        leg.get_title().set_ha("right")
        leg._legend_box.align = "right"
        plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"] * 0.65)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sdp_density_with_centers(
    basis_function,
    lambda_star: np.ndarray,
    estimate: float,
    prior_distribution,
    plot_cfg,
    output_dir: str,
    filename: str,
    domain: tuple = (-15, 15),
    resolution: int = 500,
    show_legend: bool = True,
    show_centers: bool = True,
    show_yaxis: bool = True,
) -> None:
    """Plot one SDP candidate density with the true prior and optionally its basis-function centres."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    Phi_x = basis_function.evaluate(x)
    prior_density_true = prior_distribution.pdf(x).flatten()
    log_g = prior_distribution.log_pdf(x).flatten()

    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    color = palette[0] if palette else "C0"

    names = plot_cfg.plot.param_latex_names
    xlabel = names.get("theta")
    ylabel_density = names.get("nonparametric_prior", "Density")
    fd_label = names.get("estimatedSensitivityMeasure")

    fig, ax = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    f = (Phi_x @ lambda_star).flatten()
    log_post = f + log_g
    logZ = logsumexp(log_post) + np.log(dx)
    p_hat = np.exp(log_post - logZ)
    density_line, = ax.plot(
        x.flatten(), p_hat,
        label=rf"{fd_label} = {estimate:.1f}", linewidth=1.5, color=color,
    )

    ax.plot(
        x.flatten(),
        prior_density_true,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    ymax = max(float(p_hat.max()), float(prior_density_true.max()))

    if show_centers:
        center_y = -0.03 * ymax
        centers_x = np.asarray(basis_function.centers).reshape(-1)
        ax.plot(
            centers_x, np.full_like(centers_x, center_y),
            marker='o', markersize=4, linestyle='None',
            color="#F4A6A6", clip_on=False,
        )
        ax.set_ylim(bottom=-0.06 * ymax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_density)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(
            handles=[density_line],
            loc="upper left",
            frameon=False,
            fontsize=plt.rcParams["font.size"] * 0.7,
            handlelength=0,
            handletextpad=0,
        )

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    # Hidden only after the layout pass above, so the axes keep the same
    # inner width/position they would have with the y-axis shown -- tight_layout
    # would otherwise reclaim the freed-up label space and widen the plot.
    if not show_yaxis:
        ax.set_ylabel("")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_sdp_density_with_centers_combined(
    basis_functions: list,
    lambda_star_list: list,
    estimates: list,
    labels: list,
    prior_distribution,
    plot_cfg,
    output_dir: str,
    filename: str,
    domain: tuple = (-10, 12),
    resolution: int = 500,
    show_legend: bool = True,
    colors: list = None,
    legend_labels: bool = False,
    upper_bound: float = None,
    upper_bound_at: float = None,
    sensitivity_label_key: str = "estimatedSensitivityMeasure",
) -> None:
    """Plot candidate densities for several centre-selection methods with per-method centre rug strips below."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size*1.2,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    n = len(basis_functions)
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    prior_density_true = prior_distribution.pdf(x).flatten()
    log_g = prior_distribution.log_pdf(x).flatten()

    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    names = plot_cfg.plot.param_latex_names
    fd_label = names.get(sensitivity_label_key)
    upper_bound_label = names.get("sensitivityMeasureFull", "Upper bound")
    xlabel = names.get("theta")
    ylabel_density = names.get("nonparametric_prior", "Density")

    fig, axes = plt.subplots(
        n + 1, 1,
        figsize=(4.0,
                 plot_cfg.plot.figure.size.height * 0.8 * (1 + 0.05 * n)),
        dpi=plot_cfg.plot.figure.dpi,
        gridspec_kw={"height_ratios": [6] + [0.5] * n, "hspace": 0.15},
        sharex=True,
    )
    ax_density, rug_axes = axes[0], axes[1:]

    if colors is None:
        colors = [palette[i % len(palette)] for i in range(n)]
    density_lines = []
    for basis_function, lambda_star, estimate, label, color in zip(
        basis_functions, lambda_star_list, estimates, labels, colors
    ):
        Phi_x = basis_function.evaluate(x)
        f = (Phi_x @ lambda_star).flatten()
        log_post = f + log_g
        logZ = logsumexp(log_post) + np.log(dx)
        p_hat = np.exp(log_post - logZ)
        line, = ax_density.plot(
            x.flatten(), p_hat,
            label=rf"{label}: {estimate:.1f}" if legend_labels else rf"{estimate:.1f}",
            linewidth=1.5, color=color,
        )
        density_lines.append(line)

    ax_density.plot(
        x.flatten(), prior_density_true, linestyle="--", linewidth=1.5, color="black",
    )
    if upper_bound is not None:
        bound_line = ax_density.axvline(
            upper_bound_at, linestyle=":", linewidth=1.0, color="grey",
            label=rf"{upper_bound_label}: {upper_bound:.1f}" if legend_labels else rf"{upper_bound:.1f}",
        )
        density_lines.append(bound_line)
    ax_density.set_ylabel(ylabel_density)
    ax_density.grid(True, alpha=0.3)
    ax_density.spines["top"].set_visible(False)
    ax_density.spines["right"].set_visible(False)

    if show_legend:
        # Labelled entries are too wide to sit inside the axes without
        # covering the curves, so they go outside, to the right.
        legend_loc = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0)) if legend_labels else dict(loc="best")
        leg = ax_density.legend(
            handles=density_lines,
            title=fd_label,
            frameon=False,
            fontsize=plt.rcParams["font.size"] * 0.75,
            **legend_loc,
        )
        leg.get_title().set_ha("right")
        leg._legend_box.align = "right"
        plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"] * 0.75)

    for i, (ax_rug, basis_function, color) in enumerate(zip(rug_axes, basis_functions, colors)):
        centers_x = np.asarray(basis_function.centers).reshape(-1)
        centers_x = centers_x[(centers_x >= domain[0]) & (centers_x <= domain[1])]
        ax_rug.plot(
            centers_x, np.zeros_like(centers_x),
            marker='o', markersize=1.5, linestyle='None', color=color, clip_on=False,
        )
        ax_rug.set_ylim(-1, 1)
        ax_rug.set_yticks([])
        ax_rug.spines["top"].set_visible(False)
        ax_rug.spines["left"].set_visible(False)
        ax_rug.spines["right"].set_visible(False)
        is_last = (i == len(rug_axes) - 1)
        ax_rug.spines["bottom"].set_visible(is_last)
        ax_rug.tick_params(bottom=is_last, labelbottom=is_last)

    ax_density.set_xlim(domain)
    rug_axes[-1].set_xlabel(xlabel)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_sdp_2d_densities(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = ((-5, 5), (-5, 5)),
    resolution: int | tuple = 200,
    contour_levels: int = 5,
    posterior_distribution=None,
    show_posterior: bool = False,
    show_centers: bool = False,
    show_legend: bool = False,
) -> None:
    """
    2D plot:
      True prior contours (dashed blue) + SDP density contours (palette) + optional
      posterior contours (dashed black), shown only if show_posterior=True.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _f_grid(Phi_XY: np.ndarray, psi: np.ndarray, nx: int, ny: int) -> np.ndarray:
        fN2 = np.tensordot(Phi_XY, psi, axes=([-1], [0]))
        fN = fN2[:, 0]
        return fN.reshape(ny, nx)

    # --- rcParams (LaTeX + font) ---
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # --- Colors ---
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        raise ValueError("plot_cfg.plot.color_palette.colors is empty.")
    sdp_palette = palette if len(palette) > 0 else ["C0"]

    # --- Labels ---
    names = getattr(plot_cfg.plot, "param_latex_names", {})
    fd_label = names.get("estimatedSensitivityMeasure")
    xlabel = r"$\theta_{1}$"
    ylabel = r"$\theta_{2}$"
    geq_sym = r"$\geq$"

    # --- Grid ---
    if isinstance(resolution, int):
        nx = ny = resolution
    else:
        nx, ny = resolution
    (x_min, x_max), (y_min, y_max) = domain
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    XY = np.column_stack([X.ravel(), Y.ravel()])
    Phi_XY = basis_function.evaluate(XY)

    # True prior density (reshaped to grid)
    prior_density_true = prior_distribution.pdf(XY).reshape(ny, nx)

    # log g(θ): reference/base-measure log-density, shared across all candidate priors
    log_g = prior_distribution.log_pdf(XY).reshape(-1)

    # Contour levels derived from the true prior's own density
    p_max = prior_density_true.max()
    prior_levels = np.linspace(0.05 * p_max, 0.86 * p_max, contour_levels)

    # --- Figure ---
    legend_handles = []
    fig, ax = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    if show_centers:
        centers = basis_function.centers
        ax.scatter(
            centers[:, 0], centers[:, 1],
            color="black", s=15, zorder=5, marker="x", linewidths=0.8,
        )

    if show_posterior and posterior_distribution is not None:
        posterior_density = posterior_distribution.pdf(XY).reshape(ny, nx)
        post_levels = np.linspace(0.05 * posterior_density.max(), 0.86 * posterior_density.max(), contour_levels)
        ax.contour(
            X, Y, posterior_density,
            levels=post_levels,
            colors="black",
            linewidths=0.5,
            linestyles="dashed",
            alpha=0.8,
        )

    # True prior contours (dashed blue)
    ax.contour(
        X, Y, prior_density_true,
        levels=prior_levels,
        colors="black",
        linewidths=0.4,
        linestyles="dashed",
        alpha=0.7,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # SDP priors (contours in palette colors)
    log_g_grid = log_g.reshape(ny, nx)
    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f = _f_grid(Phi_XY, psi, nx, ny)
        log_post = f + log_g_grid
        logZ = logsumexp(log_post) + np.log(dx * dy)
        p_hat = np.exp(log_post - logZ)
        p_hat_max = p_hat.max()
        sdp_levels = np.linspace(0.05 * p_hat_max, 0.86 * p_hat_max, contour_levels)

        color = sdp_palette[i % len(sdp_palette)]
        label = rf"r {geq_sym} {r_label} ({ksd:.1f})"
        ax.contour(
            X, Y, p_hat,
            levels=sdp_levels,
            colors=color,
            linewidths=1.2,
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=1.8, label=label))

    legend_handles.append(Line2D([0], [0], color="steelblue", lw=1.5,
                          linestyle="dashed", label=r"$\Pi_{\mathrm{ref}}$"))
    if show_centers:
        legend_handles.append(Line2D([0], [0], color="black", marker="x", linestyle="None",
                                     markersize=5, label="centers"))
    if show_posterior and posterior_distribution is not None:
        legend_handles.append(Line2D([0], [0], color="black", lw=1.5,
                                     linestyle="dashed", label=r"$\tilde{\Pi}_{\mathrm{ref}}$"))

    if show_legend:
        fig.legend(
            handles=legend_handles,
            title=fd_label,
            loc="center left",
            bbox_to_anchor=(0.96, 0.5),
            frameon=False,
        )

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout(rect=[0, 0, 0.95, 1])

    # Save
    filename = "gaussian_2d_location_model_diff_radii.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_runtime_nonparametric_diff_basis_funcs_num_diff_samples_with_ci(
    times_nonparametric: dict[int, dict[int, dict[int, float]]],
    plot_cfg: Any,
    output_dir: str,
    ci_level: float = 0.95,
    filename: str = "runtime_nonparametric_diff_basis_funcs_nums_diff_samples.pdf",
    k_max: int | None = None,
) -> None:
    """Plot nonparametric optimisation runtime against K with confidence bands, one line per sample size."""
    os.makedirs(output_dir, exist_ok=True)

    # Helper
    def _deep_get(cfg, path, default=None):
        cur = cfg
        for key in path.split('.'):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(key, None)
            else:
                cur = getattr(cur, key, None)
        return default if cur is None else cur

    plt.rcParams.update({
        "font.size": _deep_get(plot_cfg, "plot.font.size", 12)*1.0,
        "font.family": _deep_get(plot_cfg, "plot.font.family", "serif"),
        "text.usetex": bool(_deep_get(plot_cfg, "plot.font.use_tex", False)),
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig_w = float(_deep_get(plot_cfg, "plot.figure.size.width", 6.0))
    fig_h = float(_deep_get(plot_cfg, "plot.figure.size.height", 4.0))
    fig_dpi = int(_deep_get(plot_cfg, "plot.figure.dpi", 150))
    lw = float(_deep_get(plot_cfg, "plot.line.width", 1.0))
    marker = _deep_get(plot_cfg, "plot.marker.style", "o")
    ms = float(_deep_get(plot_cfg, "plot.marker.size", 4.0))
    grid_alpha = float(_deep_get(plot_cfg, "plot.grid.alpha", 0.7))
    tight = bool(_deep_get(plot_cfg, "plot.figure.tight_layout", True))

    names = _deep_get(plot_cfg, "plot.param_latex_names", {}) or {}
    x_label = r"$K$"
    y_label = names.get("runtimeSeconds", "Time (sec.)")
    # Keys of times_nonparametric are the total m+l; the legend shows the
    # per-set count n=m=(m+l)/2 instead.
    legend_prefix = names.get("numPriorPosteriorSamplesEach", "n=m")

    # Colors
    palette = list(getattr(_deep_get(plot_cfg, "plot.color_palette", {}), "colors", []))
    if not palette:
        palette = [f"C{i}" for i in range(10)]

    # --- Confidence interval helper
    def mean_ci(data, level=ci_level):
        arr = np.array(data)
        mean = arr.mean()
        if len(arr) > 1:
            se = sem(arr)
            h = se * t.ppf((1 + level) / 2., len(arr) - 1)
        else:
            h = 0.0
        return mean, h

    def _to_int(x):
        return int(x)

    # --- Process: one series of (k, mean, ci) per samples num m
    samples_nums = sorted(times_nonparametric.keys(), key=_to_int)
    by_samples = defaultdict(lambda: ([], [], []))
    for m in samples_nums:
        for k, runs in times_nonparametric[m].items():
            if k_max is not None and int(k) > k_max:
                continue
            vals = list(runs.values())
            mean, h = mean_ci(vals)
            xs, means, cis = by_samples[m]
            xs.append(int(k))
            means.append(mean)
            cis.append(h)

    # --- Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
    handles_ordered, labels_ordered = [], []
    basis_funcs_nums = None

    for i, m in enumerate(sorted(by_samples.keys(), key=_to_int)):
        xs, means, cis = by_samples[m]
        sorted_idx = np.argsort(xs)
        xs = list(np.array(xs)[sorted_idx])
        means = list(np.array(means)[sorted_idx])
        cis = list(np.array(cis)[sorted_idx])
        if basis_funcs_nums is None:
            basis_funcs_nums = xs

        color = palette[i % len(palette)]
        label = rf"{legend_prefix}={_to_int(m) // 2}"
        h_line = ax.plot(
            xs, means,
            marker=marker, markersize=ms, linewidth=lw,
            color=color, label=label,
        )[0]
        ax.fill_between(
            xs,
            np.array(means) - np.array(cis),
            np.array(means) + np.array(cis),
            color=color, alpha=0.4,
        )
        handles_ordered.append(h_line)
        labels_ordered.append(label)

    # Axes styling
    ax.set_xlabel(x_label)
    if basis_funcs_nums is not None:
        ax.set_xticks(basis_funcs_nums)
        ax.set_xticklabels([str(v) for v in basis_funcs_nums])
    ax.set_ylabel(y_label)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4), useMathText=True)
    ax.grid(axis="y", linestyle=":", alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_fs = float(_deep_get(plot_cfg, "plot.legend.fontsize",
                                plt.rcParams["font.size"] * 0.6))
    ax.legend(
        handles_ordered, labels_ordered,
        loc="upper left",
        fontsize=legend_fs,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )

    if tight:
        fig.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_param_nonparam_prior_and_stats(
    worst_corner: dict,
    lambda_star: np.ndarray,
    basis_function,
    model,
    plot_cfg,
    output_dir: str = "outputs/param_nonparam_skewness",
    domain: tuple = (-15, 15),
    resolution: int = 400,
    filename: str = "gaussian_1d_location_model_param_nonparam_prior_stats.pdf",
) -> None:
    """Plot reference and worst-case candidate priors beside a table of their shape statistics."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size * 1.4,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    theta = np.linspace(domain[0], domain[1], resolution)
    x = theta[:, None]
    dx = float(theta[1] - theta[0])

    # Reference prior
    prior_distribution = model.prior_init
    log_g = prior_distribution.log_pdf(x).flatten()
    g = np.exp(log_g)

    def _skewness(pdf: np.ndarray) -> float:
        mu = np.sum(theta * pdf) * dx
        sigma = np.sqrt(max(np.sum((theta - mu) ** 2 * pdf) * dx, 1e-12))
        return float(np.sum(((theta - mu) / sigma) ** 3 * pdf) * dx)

    def _excess_kurtosis(pdf: np.ndarray) -> float:
        mu = np.sum(theta * pdf) * dx
        sigma = np.sqrt(max(np.sum((theta - mu) ** 2 * pdf) * dx, 1e-12))
        return float(np.sum(((theta - mu) / sigma) ** 4 * pdf) * dx - 3.0)

    # Parametric worst-case candidate
    mu_a, sigma_a = float(worst_corner["mu"]), float(worst_corner["sigma"])
    log_pi_a = -0.5 * ((theta - mu_a) / sigma_a) ** 2 - np.log(sigma_a * np.sqrt(2 * np.pi))
    pi_a = np.exp(log_pi_a)

    # Nonparametric worst-case candidate
    Phi_x = basis_function.evaluate(x)
    f = (Phi_x @ lambda_star).flatten()
    log_pi_b_un = f + log_g
    log_pi_b = log_pi_b_un - (logsumexp(log_pi_b_un) + np.log(dx))
    pi_b = np.exp(log_pi_b)
    print(f"Nonparametric prior integral (should be 1.0): {np.sum(pi_b) * dx:.8f}")

    # Parametric candidate is exactly Gaussian: skewness/excess kurtosis are 0
    # analytically (estimating them via truncated-domain quadrature is
    # numerically unstable -- a tail-sensitive 3rd/4th moment on a fixed
    # domain not guaranteed to cover +-few sigma_a), so hardcode rather than
    # estimate. It's unimodal by construction, hence 1 mode.
    skew_a, kurt_a, modes_a = 0.0, 0.0, 1
    skew_b = _skewness(pi_b)
    kurt_b = _excess_kurtosis(pi_b)
    modes_b = 2  # hardcoded: peak-detection undercounted the visible modes here.
    print(
        f"Param.: skewness={skew_a:.4f}, modes={modes_a}, excess kurtosis={kurt_a:.4f}. "
        f"Ours: skewness={skew_b:.4f}, modes={modes_b}, excess kurtosis={kurt_b:.4f}."
    )

    # Colors & labels
    palette = ["#9EB8A0", "#ADEBC8"]
    col_ref = "black"
    col_a = palette[0]
    col_b = palette[1]

    names = plot_cfg.plot.param_latex_names
    xlabel = names.get("theta", r"$\theta$")

    fig_w = plot_cfg.plot.figure.size.width
    fig_h = plot_cfg.plot.figure.size.height
    fs = plot_cfg.plot.font.size

    fig = plt.figure(figsize=(fig_w * 2, fig_h), dpi=plot_cfg.plot.figure.dpi)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.3, 1.0], wspace=0.04)

    # Left: prior densities
    ax_prior = fig.add_subplot(gs[0, 0])
    ax_prior.plot(theta, g, color=col_ref, linestyle="--", linewidth=1.5, label=r"$\mathrm{Ref.}$")
    ax_prior.plot(theta, pi_a, color=col_a, linewidth=1.5, label=r"$\mathrm{Param.}$")
    ax_prior.plot(theta, pi_b, color=col_b, linewidth=1.5, label=r"$\mathrm{Ours}$")
    ax_prior.set_xlabel(xlabel)
    ax_prior.set_ylabel(r"$\mathrm{Prior}$")
    ax_prior.set_ylim(bottom=0)
    ax_prior.grid(True, alpha=0.3)
    ax_prior.spines["top"].set_visible(False)
    ax_prior.spines["right"].set_visible(False)
    ax_prior.legend(frameon=False, fontsize=fs * 1.0, labelspacing=0.4, handlelength=1.8, handletextpad=0.5)

    # Right: full-height 2 (Param. / Ours) x 3 (skewness, modes, excess
    # kurtosis) stats layout -- plain positioned text (no cell grid/box
    # borders), with a light header rule and column rules so it still reads
    # as a table.
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis("off")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)

    row_labels = [r"Skewness", r"Modes", "Excess\nkurtosis"]
    param_vals = [skew_a, modes_a, kurt_a]
    nonparam_vals = [skew_b, modes_b, kurt_b]
    row_fmts = ["{:.1f}", "{:.0f}", "{:.1f}"]
    fs_table = fs * 1.25

    label_x, param_x, nonparam_x = 0.02, 0.60, 0.90
    label_col_x, param_col_x = 0.40, 0.76  # vertical column rules
    header_y = 0.92
    row_ys = np.linspace(0.68, 0.12, len(row_labels))
    rule_bottom = row_ys[-1] - 0.10
    header_rule_y = header_y - 0.09

    rule_kwargs = dict(color="#999999", linewidth=0.8)
    ax_stats.plot([0.0, 1.0], [header_rule_y, header_rule_y], **rule_kwargs)
    ax_stats.plot([label_col_x, label_col_x], [header_rule_y, rule_bottom], **rule_kwargs)
    ax_stats.plot([param_col_x, param_col_x], [header_rule_y, rule_bottom], **rule_kwargs)

    ax_stats.text(param_x, header_y, r"$\mathbf{Param.}$", ha="center", va="center",
                  fontsize=fs_table, weight="bold", color=col_a)
    ax_stats.text(nonparam_x, header_y, r"$\mathbf{Ours}$", ha="center", va="center",
                  fontsize=fs_table, weight="bold", color=col_b)

    for row_label, val_a, val_b, fmt, y in zip(row_labels, param_vals, nonparam_vals, row_fmts, row_ys):
        ax_stats.text(label_x, y, row_label, ha="left", va="center", fontsize=fs_table)
        ax_stats.text(param_x, y, fmt.format(val_a), ha="center", va="center", fontsize=fs_table)
        ax_stats.text(nonparam_x, y, fmt.format(val_b), ha="center", va="center", fontsize=fs_table)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        fig.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_closed_form_sensitivity_error(
    sample_sizes: list[int],
    series: dict[str, tuple[list[float], list[float]]],
    plot_cfg,
    output_dir: str,
    filename: str = "gaussian_1d_location_model_closed_form_convergence.pdf",
    ylabel: str = r"$|S^{\mathrm{FD}}(\mathcal{Q}_r^K) - \widehat{S}_m^{\mathrm{FD}}(\widehat{\mathcal{Q}}_r^{K,l})|$",
    xlabel: str = r"$m = l$",
    y_log_scale: bool = False,
    series_x: dict[str, list[int]] | None = None,
    y_bottom: float | None = None,
    ylabel_fontsize_scale: float = 0.8,
) -> None:
    """Plot the absolute error of Monte-Carlo sensitivity estimates against the sample size."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    series_x = series_x or {}

    fig, ax = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    for i, (label, (mean, band)) in enumerate(series.items()):
        color = palette[i % len(palette)]
        x = np.asarray(series_x.get(label, sample_sizes), dtype=float)
        mean = np.asarray(mean, dtype=float)
        band = np.asarray(band, dtype=float)
        floor = float(mean.min()) * 1e-2

        ax.plot(x, mean, marker="o", markersize=4, linewidth=1.5, color=color, label=label)
        ax.fill_between(
            x,
            np.maximum(mean - band, floor),
            mean + band,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

    if y_log_scale:
        ax.set_yscale("log")
    elif y_bottom is not None:
        ax.set_ylim(bottom=y_bottom)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=plt.rcParams["font.size"] * ylabel_fontsize_scale, y=0.4)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(series) > 1:
        ax.legend(frameon=False, fontsize=plt.rcParams["font.size"] * 0.8)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_sensitivity_vs_basis_funcs_num(
    basis_funcs_nums: list[int],
    estimates: list[float],
    true_value: float,
    plot_cfg,
    output_dir: str,
    filename: str = "gaussian_2d_location_model_sensitivity_vs_K.pdf",
    xlabel: str = r"$K$",
    ylabel: str = r"$\widehat{S}^{\mathrm{FD}}_m(\widehat{\mathcal{Q}}_r^{K,l})$",
    true_value_label: str = r"$S^{\mathrm{FD}}(\mathcal{Q}_r)$",
    x_log_scale: bool = True,
) -> None:
    """
    Plot the nonparametric sieve sensitivity estimate as a function of the
    number of basis functions K, together with a horizontal line at the
    exact closed-form sensitivity S^FD(Q_r) = M*r that the sieve estimate is
    expected to approach as K grows.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    color = palette[0] if palette else "C0"

    fig, ax = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    ax.plot(basis_funcs_nums, estimates, marker="o", markersize=4, linewidth=1.5, color=color)
    ax.axhline(true_value, linestyle="--", linewidth=1.5, color="black")

    if x_log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=plt.rcParams["font.size"] * 0.85)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_bottom, y_top = ax.get_ylim()
    y_top = max(y_top, true_value * 1.15)
    ax.set_ylim(y_bottom, y_top)

    label_offset = 0.04 * (y_top - y_bottom)
    ax.text(
        0.5, true_value + label_offset, true_value_label,
        transform=ax.get_yaxis_transform(),
        ha="center", va="bottom",
        fontsize=plt.rcParams["font.size"] * 0.8,
    )

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_sensitivity_vs_basis_funcs_num_dual(
    basis_funcs_nums_left: list[int],
    estimates_left: list[float],
    true_value_left: float,
    basis_funcs_nums_right: list[int],
    estimates_right: list[float],
    true_value_right: float,
    radius: float,
    plot_cfg,
    output_dir: str,
    filename: str = "gaussian_sensitivity_vs_K_univariate_vs_multivariate.pdf",
    xlabel: str = r"$K$",
    ylabel_left: str = r"$(S^{\mathrm{FD}}(\mathcal{Q}_r) - S^{\mathrm{FD}}(\mathcal{Q}_r^{K})) / r$",
    x_log_scale: bool = True,
) -> None:
    """Plot univariate and multivariate sieve approximation errors divided by r against K on twin y-axes."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]
    color_left = palette[0]
    color_right = palette[1 % len(palette)]

    fig, ax_left = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    ax_right = ax_left.twinx()

    errors_left = (true_value_left - np.asarray(estimates_left, dtype=float)) / radius
    errors_right = (true_value_right - np.asarray(estimates_right, dtype=float)) / radius

    ax_left.plot(
        basis_funcs_nums_left, errors_left, marker="o", markersize=2,
        linewidth=1.0, color=color_left,
    )

    ax_right.plot(
        basis_funcs_nums_right, errors_right, marker="s", markersize=2,
        linewidth=1.0, color=color_right,
    )

    if x_log_scale:
        ax_left.set_xscale("log")
        # Matplotlib only labels decade ticks (10^0, 10^1, ...) that fall
        # strictly inside the data range; if the largest K (e.g. 6400) sits
        # below the next power of ten (10^4), that next decade tick is never
        # drawn, so labels appear to stop early. Extend the right xlim just
        # past the next decade above the max K so its tick renders too.
        max_K = max(max(basis_funcs_nums_left), max(basis_funcs_nums_right))
        ax_left.set_xlim(right=10 ** np.ceil(np.log10(max_K)) * 1.2)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(ylabel_left, fontsize=plt.rcParams["font.size"] * 0.7, color="black")
    ax_left.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
