import os
import numpy as np
import matplotlib.pyplot as plt
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


# ------------------------------------------------------------
# Density helpers
# ------------------------------------------------------------
def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def normalize_density(x, y):
    return y / np.trapz(y, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_from_grid(theta, density, n_samples, rng):
    dx = theta[1] - theta[0]
    probs = density * dx
    probs = probs / probs.sum()
    idx = rng.choice(len(theta), size=n_samples, replace=True, p=probs)
    return theta[idx]


# ------------------------------------------------------------
# Labels / styles
# ------------------------------------------------------------
LABELS = [r"$A$", r"$B$", r"$C$"]
LINESTYLES = ["-", "--", ":"]


# ------------------------------------------------------------
# Prior-only plot
# ------------------------------------------------------------
def plot_prior(theta, priors, colors, plot_cfg, output_dir, prefix):
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        )
    )
    fig.subplots_adjust(left=0.28, right=0.99, bottom=0.24, top=0.99)

    for p, color, ls, label in zip(priors, colors, LINESTYLES, LABELS):
        ax.plot(theta, p, linewidth=1.5, color=color, ls=ls, label=label)

    _style_ax(ax, xlabel=r"$\theta$", ylabel=r"$\pi(\theta)$")
    ax.legend(frameon=False, ncol=1, loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    _save_fig(fig, output_dir, f"{prefix}_prior.pdf", plot_cfg)


# ------------------------------------------------------------
# Posterior-only plot
# ------------------------------------------------------------
def plot_posterior(theta, posteriors, colors, plot_cfg, output_dir, prefix):
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        )
    )
    fig.subplots_adjust(left=0.28, right=0.99, bottom=0.24, top=0.99)

    for p, color, ls, label in zip(posteriors, colors, LINESTYLES, LABELS):
        ax.plot(theta, p, linewidth=1.5, color=color, ls=ls, label=label)

    _style_ax(ax, xlabel=r"$\theta$", ylabel=r"$\tilde{\pi}(\theta \mid x_{1:n})$")
    ax.legend(frameon=False, ncol=1, loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    _save_fig(fig, output_dir, f"{prefix}_posterior.pdf", plot_cfg)


# ------------------------------------------------------------
# Combined 1D prior/posterior plot
# analogous to plot_prior_posterior_2d(...)
# ------------------------------------------------------------
def plot_prior_posterior_1d(theta, priors, posteriors, colors, plot_cfg, output_dir, filename):
    from matplotlib.lines import Line2D

    n = len(priors)
    w = plot_cfg.plot.figure.size.width
    h = plot_cfg.plot.figure.size.height

    fig, axes = plt.subplots(
        1, n,
        figsize=(w * n * 0.85, h * 1.15),
        squeeze=False,
    )
    axes = axes[0]
    fig.subplots_adjust(wspace=0.35)

    xlim = (theta[0], theta[-1])
    ymax = max(max(np.max(p) for p in priors), max(np.max(p) for p in posteriors))

    for ax, prior, posterior, color, label in zip(axes, priors, posteriors, colors, LABELS):
        ax.plot(theta, prior, linewidth=1.5, color=color, ls="-")
        ax.plot(theta, posterior, linewidth=1.5, color=color, ls="--")

        ax.set_xlim(xlim)
        ax.set_ylim(0.0, ymax * 1.08)
        ax.set_title(label)

        ax.legend(
            handles=[
                Line2D([0], [0], color=color, lw=1.5, ls="-",  label=r"$\pi(\theta)$"),
                Line2D([0], [0], color=color, lw=1.5, ls="--", label=r"$\tilde{\pi}(\theta \mid x_{1:n})$"),
            ],
            frameon=False,
            loc="best",
        )

        _style_ax(
            ax,
            xlabel=r"$\theta$",
            ylabel="density" if ax is axes[0] else "",
        )
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    _save_fig(fig, output_dir, filename, plot_cfg)


# ------------------------------------------------------------
# Figure-18-style posterior simulation plot
# ------------------------------------------------------------
def simulate_quantity_of_interest(theta_draws, rng):
    """Simulate two quantities under two conditions, kept inside the log-scale plotting window."""
    n = len(theta_draws)

    eps_low  = rng.normal(loc=0.0, scale=0.16, size=n)
    eps_high = rng.normal(loc=0.0, scale=0.12, size=n)

    # Lower cloud: stronger dependence on theta
    s_low = sigmoid(-2.2 + 0.95 * theta_draws + eps_low)

    # Upper cloud: weaker dependence, high but not pressed against ceiling
    s_high = sigmoid(1.8 + 0.18 * theta_draws + eps_high)

    # Map to a positive range that fits well on log scale
    q_low = 0.3 + 79.7 * s_low
    q_high = 55.0 + 30.0 * s_high

    return q_low, q_high


def plot_prior_influence_figure18_style(
    theta,
    posteriors,
    colors,
    plot_cfg,
    output_dir,
    prefix,
    n_draws=600,
    seed=123,
):
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(
            plot_cfg.plot.figure.size.width * 2.8,
            plot_cfg.plot.figure.size.height * 1.15,
        ),
        squeeze=False,
    )
    axes = axes[0]
    fig.subplots_adjust(wspace=0.35)

    x_min, x_max = theta[0], theta[-1]

    for ax, posterior, color, label in zip(axes, posteriors, colors, LABELS):
        theta_draws = sample_from_grid(theta, posterior, n_draws, rng)
        q_low, q_high = simulate_quantity_of_interest(theta_draws, rng)

        ax.scatter(
            theta_draws,
            q_low,
            s=8,
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            rasterized=True,
        )

        ax.scatter(
            theta_draws,
            q_high,
            s=8,
            color="black",
            linewidths=0.0,
            alpha=0.9,
            rasterized=True,
        )

        ax.set_title(label)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.2, 120.0)
        ax.set_yscale("log")

        _style_ax(
            ax,
            xlabel=r"$\theta$",
            ylabel="Percent metabolized" if ax is axes[0] else "",
        )

    _save_fig(fig, output_dir, f"{prefix}_figure18_style.pdf", plot_cfg)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
@hydra.main(version_base="1.1", config_path="../configs/presentation/", config_name="slide_1")
def main(cfg: DictConfig) -> None:
    prefix = "slide_1_figure_18"
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    _apply_plot_rc(plot_cfg)
    colors = list(plot_cfg.plot.color_palette.colors)

    theta = np.linspace(-2.5, 6.5, 5000)

    # ---------------------------------------------------------------
    # Similar-looking priors with different tail behaviour
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

    # Likelihood in the tail
    y_obs, sigma_y = 3.5, 0.7
    likelihood_unnorm = gaussian_pdf(theta, y_obs, sigma_y)

    posterior_A = normalize_density(theta, prior_A * likelihood_unnorm)
    posterior_B = normalize_density(theta, prior_B * likelihood_unnorm)
    posterior_C = normalize_density(theta, prior_C * likelihood_unnorm)

    priors = [prior_A, prior_B, prior_C]
    posteriors = [posterior_A, posterior_B, posterior_C]

    plot_prior(theta, priors, colors, plot_cfg, output_dir, prefix)
    plot_posterior(theta, posteriors, colors, plot_cfg, output_dir, prefix)

    plot_prior_posterior_1d(
        theta,
        priors,
        posteriors,
        colors,
        plot_cfg,
        output_dir,
        f"{prefix}_prior_posterior.pdf",
    )

    plot_prior_influence_figure18_style(
        theta=theta,
        posteriors=posteriors,
        colors=colors,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        n_draws=700,
        seed=123,
    )

    print(f"Saved plots to {output_dir} with prefix '{prefix}'")


if __name__ == "__main__":
    main()