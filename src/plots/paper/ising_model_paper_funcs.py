import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from typing import Dict, List


def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{type1cm}",
    })


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
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-4, -4))
        ax.yaxis.set_major_formatter(fmt)

    ax.set_ylabel(plot_cfg.plot.param_latex_names[ylbl])

    if legend:
        ax.legend()

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
