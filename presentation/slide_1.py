import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.common.utils.files_operations import load_plot_config


# ------------------------------------------------------------
# Paper-style helpers
# ------------------------------------------------------------
def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size":           plot_cfg.plot.font.size,
        "font.family":         plot_cfg.plot.font.family,
        "text.usetex":         plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{type1cm}",
    })


def _make_figure(plot_cfg):
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        )
    )
    fig.subplots_adjust(left=0.28, right=0.99, bottom=0.24, top=0.99)
    return fig, ax


def _save_fig(fig, output_dir, filename, plot_cfg):
    os.makedirs(output_dir, exist_ok=True)
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)


def _style_ax(ax, xlabel=r"$\theta$", ylabel="density"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)


# ------------------------------------------------------------
# Density helpers
# ------------------------------------------------------------
def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def normalize_density(x, y):
    return y / np.trapz(y, x)


# ------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------
LINESTYLES = ["-", "--", ":"]
LABELS_PRIOR = [r"$A$", r"$B$", r"$C$"]
LABELS_POST  = [r"$A$", r"$B$", r"$C$"]


def plot_prior(theta, priors, colors, plot_cfg, output_dir, prefix):
    fig, ax = _make_figure(plot_cfg)
    for p, color, ls, label in zip(priors, colors, LINESTYLES, LABELS_PRIOR):
        ax.plot(theta, p, linewidth=1.5, color=color, ls=ls, label=label)
    _style_ax(ax, ylabel=rf"$\pi(\theta)$")
    ax.legend(frameon=False, ncol=1, loc="best")
    _save_fig(fig, output_dir, f"{prefix}_prior.pdf", plot_cfg)


def plot_likelihood(theta, likelihood, y_obs, sigma_y, colors, plot_cfg, output_dir, prefix):
    fig, ax = _make_figure(plot_cfg)
    ax.plot(theta, likelihood, linewidth=1.5, color=colors[1])
    _style_ax(ax, ylabel=r"$p_{\theta}(x_{1:n})$")
    _save_fig(fig, output_dir, f"{prefix}_likelihood.pdf", plot_cfg)


def plot_posterior(theta, posteriors, colors, plot_cfg, output_dir, prefix):
    fig, ax = _make_figure(plot_cfg)
    for p, color, ls, label in zip(posteriors, colors, LINESTYLES, LABELS_POST):
        ax.plot(theta, p, linewidth=1.5, color=color, ls=ls, label=label)
    _style_ax(ax, ylabel=r"$\tilde{\pi}(\theta|x_{1:n})$")
    ax.legend(frameon=False, ncol=1, loc="best")
    _save_fig(fig, output_dir, f"{prefix}_posterior.pdf", plot_cfg)


def create_prior_posterior_animation(
    theta, priors, posteriors, colors, plot_cfg, output_dir, prefix,
    fps=30, hold_seconds=2.0, draw_seconds=1.0,
):
    """Create an MP4 where prior/posterior pairs are drawn one by one left-to-right.

    For each pair the line is animated from x-min to x-max, then held for
    `hold_seconds` before the next pair starts drawing.
    """
    import imageio.v3 as iio

    n_hold = max(1, int(fps * hold_seconds))
    n_draw = max(2, int(fps * draw_seconds))
    n_pts  = len(theta)
    n_pairs = len(priors)

    dpi = getattr(plot_cfg.plot.figure, "dpi", 150)

    fig, (ax_prior, ax_post) = plt.subplots(
        1, 2,
        figsize=(
            plot_cfg.plot.figure.size.width * 2.1,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=dpi,
    )
    fig.subplots_adjust(wspace=0.45)

    y_prior_max = max(np.max(p) for p in priors)
    y_post_max  = max(np.max(p) for p in posteriors)
    for ax in (ax_prior, ax_post):
        ax.set_xlim(theta[0], theta[-1])
    ax_prior.set_ylim(0, y_prior_max * 1.15)
    ax_post.set_ylim(0,  y_post_max  * 1.15)

    _style_ax(ax_prior, ylabel=r"$\pi(\theta)$")
    _style_ax(ax_post,  ylabel=r"$\tilde{\pi}(\theta|x_{1:n})$")

    # Pre-create all lines with no data; fill in incrementally
    lines_prior, lines_post = [], []
    for _, _, color, ls, label in zip(priors, posteriors, colors, LINESTYLES, LABELS_PRIOR):
        lp, = ax_prior.plot([], [], linewidth=1.5, color=color, ls=ls, label=label)
        lq, = ax_post.plot( [], [], linewidth=1.5, color=color, ls=ls, label=label)
        lines_prior.append(lp)
        lines_post.append(lq)

    legends = [None, None]

    def _update_legend():
        for ax, lines, leg_idx in [(ax_prior, lines_prior, 0), (ax_post, lines_post, 1)]:
            visible_handles = [l for l in lines if len(l.get_xdata()) > 0]
            if legends[leg_idx] is not None:
                legends[leg_idx].remove()
                legends[leg_idx] = None
            if visible_handles:
                legends[leg_idx] = ax.legend(
                    handles=visible_handles, frameon=False, ncol=1, loc="best"
                )

    def _capture():
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{prefix}_animation.mp4")

    fig.canvas.draw()
    frames = []

    for pair_idx in range(n_pairs):
        lp = lines_prior[pair_idx]
        lq = lines_post[pair_idx]
        p  = priors[pair_idx]
        post = posteriors[pair_idx]

        # Draw the line from left to right
        for k in range(1, n_draw + 1):
            end = max(2, int(n_pts * k / n_draw))
            lp.set_data(theta[:end], p[:end])
            lq.set_data(theta[:end], post[:end])
            _update_legend()
            frames.append(_capture())

        # Ensure the full line is set, then hold
        lp.set_data(theta, p)
        lq.set_data(theta, post)
        _update_legend()
        hold_frame = _capture()
        for _ in range(n_hold):
            frames.append(hold_frame.copy())

    iio.imwrite(path, np.stack(frames), fps=fps, quality=9)

    plt.close(fig)
    print(f"Saved animation to {path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
@hydra.main(version_base="1.1", config_path="../configs/presentation/", config_name="slide_1")
def main(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "slide_1")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    _apply_plot_rc(plot_cfg)
    colors = list(plot_cfg.plot.color_palette.colors)

    theta = np.linspace(-2.5, 6.5, 5000)

    # ---------------------------------------------------------------
    # Three unimodal priors — all centered at 0, visually similar near
    # the mode (peaks differ by <10%), but with progressively heavier
    # tails via scale mixtures (same mean → always unimodal).
    #
    # At the likelihood location θ≈3.5 their densities differ by
    # factors of ~6× (A→B) and ~15× (A→C), producing:
    #   Posterior A: concentrated near ~2.4 (light tail resists data)
    #   Posterior B: bimodal — uncertainty between prior and data
    #   Posterior C: concentrated near ~3.4 (heavy tail follows data)
    #
    #   Prior A: N(0, 1.00)                          — light tail
    #   Prior B: 0.93·N(0,0.98) + 0.07·N(0,4.5)     — moderate tail
    #   Prior C: 0.80·N(0,0.92) + 0.20·N(0,4.5)     — heavy tail
    # ---------------------------------------------------------------
    prior_A_unnorm = gaussian_pdf(theta, 0, 1.00)
    prior_B_unnorm = (
        0.93 * gaussian_pdf(theta, 0, 0.98)
        + 0.07 * gaussian_pdf(theta, 0, 4.5)
    )
    prior_C_unnorm = (
        0.80 * gaussian_pdf(theta, 0, 0.92)
        + 0.20 * gaussian_pdf(theta, 0, 4.5)
    )

    prior_A = normalize_density(theta, prior_A_unnorm)
    prior_B = normalize_density(theta, prior_B_unnorm)
    prior_C = normalize_density(theta, prior_C_unnorm)

    # Gaussian likelihood in the tail (direct observation y ~ N(θ, σ_y²))
    y_obs, sigma_y = 3.5, 0.7
    likelihood_unnorm = gaussian_pdf(theta, y_obs, sigma_y)
    likelihood = normalize_density(theta, likelihood_unnorm)

    posterior_A = normalize_density(theta, prior_A * likelihood_unnorm)
    posterior_B = normalize_density(theta, prior_B * likelihood_unnorm)
    posterior_C = normalize_density(theta, prior_C * likelihood_unnorm)

    priors     = [prior_A,     prior_B,     prior_C]
    posteriors = [posterior_A, posterior_B, posterior_C]

    plot_prior(theta, priors, colors, plot_cfg, output_dir, prefix)
    plot_likelihood(theta, likelihood, y_obs, sigma_y, colors, plot_cfg, output_dir, prefix)
    plot_posterior(theta, posteriors, colors, plot_cfg, output_dir, prefix)
    create_prior_posterior_animation(theta, priors, posteriors, colors, plot_cfg, output_dir, prefix, draw_seconds=2.5)

    print(f"Saved plots to {output_dir} with prefix '{prefix}'")


# ------------------------------------------------------------
# 2-D helpers
# ------------------------------------------------------------
def gaussian_pdf_2d(X, Y, mu_x, mu_y, sigma):
    """Isotropic bivariate Gaussian evaluated on a meshgrid."""
    return (
        np.exp(-0.5 * (((X - mu_x) / sigma) ** 2 + ((Y - mu_y) / sigma) ** 2))
        / (2 * np.pi * sigma ** 2)
    )


def plot_densities_2d(xx, yy, densities, labels, colors, ylabel_str, plot_cfg, output_dir, filename, linestyle="-", show_legend=True):
    from matplotlib.lines import Line2D
    n = len(densities)
    w = plot_cfg.plot.figure.size.width
    h = plot_cfg.plot.figure.size.height
    fig, axes = plt.subplots(1, n, figsize=(w * n * 0.85, h * 1.2), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(wspace=0.4)
    xlim = (xx[0, 0], xx[0, -1])
    ylim = (yy[0, 0], yy[-1, 0])
    for i, (ax, Z, color, label) in enumerate(zip(axes, densities, colors, labels)):
        levels = np.linspace(Z.max() * 0.02, Z.max(), 14)
        if linestyle == "-":
            cmap = LinearSegmentedColormap.from_list("mono", ["#ffffff", color])
            ax.contourf(xx, yy, Z, levels=levels, cmap=cmap, extend="min")
            ax.contour(xx, yy, Z, levels=levels, colors=[color], linewidths=0.5, alpha=0.45, linestyles="-")
            if show_legend:
                prior_label = rf"$\pi(\theta|\lambda_{i+1})$"
                ax.legend(
                    handles=[Line2D([0], [0], color=color, lw=0.5, ls="-", label=prior_label)],
                    frameon=False, loc="upper right",
                )
        else:
            post_levels = np.linspace(Z.max() * 0.02, Z.max(), 7)
            ax.contour(xx, yy, Z, levels=post_levels, colors=[color], linewidths=1.0, linestyles=linestyle)
            if show_legend:
                post_label = rf"$\tilde{{\pi}}^{{\lambda^{i+1}}}(\theta|x_{{1:n}})$"
                ax.legend(
                    handles=[Line2D([0], [0], color=color, lw=1.0, ls=linestyle, label=post_label)],
                    frameon=False, loc="upper right",
                )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_observations_2d(obs, plot_cfg, output_dir, filename, xx=None, yy=None):
    w = plot_cfg.plot.figure.size.width
    h = plot_cfg.plot.figure.size.height
    fig, ax = plt.subplots(figsize=(w, h))
    ax.scatter(obs[:, 0], obs[:, 1], marker="x", s=40, linewidths=1.2, color="black", zorder=3)
    if xx is not None and yy is not None:
        ax.set_xlim(xx[0, 0], xx[0, -1])
        ax.set_ylim(yy[0, 0], yy[-1, 0])
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_prior_posterior_2d(xx, yy, priors, posteriors, colors, plot_cfg, output_dir, filename, labels=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    n = len(priors)
    w = plot_cfg.plot.figure.size.width
    h = plot_cfg.plot.figure.size.height
    fig, axes = plt.subplots(1, n, figsize=(w * n * 0.85, h * 1.2), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(wspace=0.4)
    if labels is None:
        labels = [r"$A$", r"$B$", r"$C$"]
    xlim = (xx[0, 0], xx[0, -1])
    ylim = (yy[0, 0], yy[-1, 0])
    for i, (ax, prior, posterior, color, label) in enumerate(zip(axes, priors, posteriors, colors, labels)):
        cmap = LinearSegmentedColormap.from_list("mono", ["#ffffff", color])
        prior_levels = np.linspace(prior.max() * 0.02, prior.max(), 14)
        post_levels  = np.linspace(posterior.max() * 0.02, posterior.max(), 7)
        ax.contourf(xx, yy, prior, levels=prior_levels, cmap=cmap, extend="min")
        ax.contour(xx, yy, prior,     levels=prior_levels, colors=[color], linewidths=0.5, alpha=0.45, linestyles="-")
        ax.contour(xx, yy, posterior, levels=post_levels,  colors=[color], linewidths=1.0, linestyles="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        lam = rf"$\lambda_{i+1}$"
        ax.legend(
            handles=[
                Line2D([0], [0], color=color, lw=0.5, ls="-",  label=rf"$\pi(\theta|\lambda_{i+1})$"),
                Line2D([0], [0], color=color, lw=1.0, ls="--", label=rf"$\tilde{{\pi}}^{{\lambda_{i+1}}}(\theta|x_{{1:n}})$"),
            ],
            frameon=False, loc="upper right",
        )
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")

        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _save_fig(fig, output_dir, filename, plot_cfg)


# ------------------------------------------------------------
# Main 2-D
# ------------------------------------------------------------
def main_2d() -> None:
    """2-D analogue of main(): three bivariate scale-mixture priors × one
    2-D Gaussian likelihood, yielding qualitatively different posteriors.

    Prior A: isotropic N(0, I)                                — light tail
    Prior B: 0.93·N(0, 0.98²I) + 0.07·N(0, 4.5²I)           — moderate tail
    Prior C: 0.80·N(0, 0.92²I) + 0.20·N(0, 4.5²I)           — heavy tail

    Observation placed in the tail at (3.5, 3.5), sigma_y = 0.7:
      Posterior A  — concentrated close to origin (prior dominates)
      Posterior B  — bimodal / uncertain between prior and data
      Posterior C  — pulled toward the observation (tail follows data)
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_config_path = os.path.join(project_root, "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(project_root, "outputs/presentation/slide_1")
    plot_cfg = load_plot_config(plot_config_path)
    _apply_plot_rc(plot_cfg)
    colors = list(plot_cfg.plot.color_palette.colors)

    prefix = "slide_1_2d"

    # 2-D grid covering the same range as the 1-D case
    n_grid = 400
    t = np.linspace(-2.5, 6.5, n_grid)
    dx = dy = t[1] - t[0]
    XX, YY = np.meshgrid(t, t)

    # --- priors (unnormalized for posterior computation) ---
    prior_A_un = gaussian_pdf_2d(XX, YY, 0, 0, 1.00)
    prior_B_un = (
        0.85 * gaussian_pdf_2d(XX, YY, 0, 0, 0.98)
        + 0.15 * gaussian_pdf_2d(XX, YY, 0, 0, 4.5)
    )
    prior_C_un = (
        0.62 * gaussian_pdf_2d(XX, YY, 0, 0, 0.88)
        + 0.38 * gaussian_pdf_2d(XX, YY, 0, 0, 4.5)
    )

    # normalize for display
    prior_A = prior_A_un / (np.sum(prior_A_un) * dx * dy)
    prior_B = prior_B_un / (np.sum(prior_B_un) * dx * dy)
    prior_C = prior_C_un / (np.sum(prior_C_un) * dx * dy)

    # --- likelihood ---
    # 5 observations drawn around (2.5, 2.5) for display as crosses.
    # Likelihood is modelled as N(θ | ȳ, σ_y²) — sigma_y is kept at its
    # original value so the prior still meaningfully shapes each posterior.
    rng = np.random.default_rng(42)
    sigma_y, n_obs = 0.7, 5
    obs = rng.normal(loc=[2.5, 2.5], scale=0.35, size=(n_obs, 2))
    y_mean = obs.mean(axis=0)
    likelihood_un = gaussian_pdf_2d(XX, YY, y_mean[0], y_mean[1], sigma_y)
    likelihood_2d = likelihood_un / (np.sum(likelihood_un) * dx * dy)

    # --- posteriors ∝ prior × likelihood ---
    post_A_un = prior_A_un * likelihood_un
    post_B_un = prior_B_un * likelihood_un
    post_C_un = prior_C_un * likelihood_un

    post_A = post_A_un / (np.sum(post_A_un) * dx * dy)
    post_B = post_B_un / (np.sum(post_B_un) * dx * dy)
    post_C = post_C_un / (np.sum(post_C_un) * dx * dy)

    labels        = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    priors_2d     = [prior_A, prior_B, prior_C]
    posteriors_2d = [post_A,  post_B,  post_C]

    for lbl, post in zip(labels, posteriors_2d):
        w = post * dx * dy
        mean1 = np.sum(XX * w)
        mean2 = np.sum(YY * w)
        var1  = np.sum(XX**2 * w) - mean1**2
        var2  = np.sum(YY**2 * w) - mean2**2
        print(f"Posterior {lbl}: mean=({mean1:.3f}, {mean2:.3f})  var=({var1:.3f}, {var2:.3f})")

    plot_densities_2d(
        XX, YY, priors_2d, labels, colors,
        r"$\pi(\theta|\lambda_\pi)$", plot_cfg, output_dir, f"{prefix}_prior.pdf",
    )
    plot_observations_2d(obs, plot_cfg, output_dir, f"{prefix}_likelihood.pdf", XX, YY)
    plot_densities_2d(
        XX, YY, posteriors_2d, labels, colors,
        r"$\tilde{\pi}(\boldsymbol{\theta}|x_{1:n})$", plot_cfg, output_dir, f"{prefix}_posterior.pdf",
        linestyle="--",
    )
    plot_prior_posterior_2d(
        XX, YY, priors_2d, posteriors_2d, colors, plot_cfg, output_dir, f"{prefix}_prior_posterior.pdf",
        labels=labels,
    )

    print(f"Saved 2D plots to {output_dir} with prefix '{prefix}'")


if __name__ == "__main__":
    # main()
    main_2d()

