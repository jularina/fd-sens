import os

import numpy as np
import matplotlib.pyplot as plt

from paper.plot_utils import apply_plot_rc, save_fig, make_figure


def plot_gaussian_copula_grid_pair(
    copula_grid_0,
    copula_grid_1,
    plot_cfg,
    output_dir: str,
    prefix: str,
    filename: str | None = None,
    xlabel: str = r"$\lambda_c$",
    ylabel: str = r"$\hat{\rho}_m^{\mathrm{FD}}(\tilde{\Pi}^{\lambda})$",
    label_0: str = r"$(G_0, \nu)$",
    label_1: str = r"$(T, \nu)$",
    mark_max_point: bool = True,
    mark_corner_point: bool = False,
    mark_x_values: list | None = None,
    mark_x_red_idx: int | None = None,
    show_grid_0: bool = True,
    logy: bool = False,
    ylim=None,
    show_ylabel: bool = True,
):
    """Plot two Gaussian-copula FD grids as two lines on a single axes."""
    try:
        apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_gaussian_copula_fd_grid.pdf"

    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = make_figure(plot_cfg)

    for i, (grid, label, line_color, point_color) in enumerate((
        (copula_grid_0, label_0, colors[0], colors[0]),
        (copula_grid_1, label_1, colors[1], colors[1]),
    )):
        if i == 0 and not show_grid_0:
            continue

        lambdas = np.asarray([x[0] for x in grid], dtype=float)
        values = np.asarray([x[1] for x in grid], dtype=float)
        order = np.argsort(lambdas)
        lambdas, values = lambdas[order], values[order]

        ax.plot(lambdas, values, linewidth=1.5, color=line_color, label=label, zorder=2)

        if mark_max_point:
            idx_star = int(np.argmax(values))
        elif mark_corner_point:
            # mark the endpoint (corner) with the largest value
            idx_star = 0 if values[0] >= values[-1] else len(values) - 1
        else:
            idx_star = None

        if idx_star is not None:
            ax.scatter(
                [lambdas[idx_star]], [values[idx_star]],
                s=30, color="red", zorder=5, marker="*", clip_on=False,
            )
            ax.axvline(
                lambdas[idx_star],
                linestyle=":", linewidth=1.0, color=point_color, alpha=0.8, zorder=1,
            )

        if mark_x_values is not None:
            x_vals_for_grid = mark_x_values[i] if isinstance(mark_x_values[0], (list, tuple)) else mark_x_values
            for j, x_val in enumerate(x_vals_for_grid):
                idx_x = int(np.argmin(np.abs(lambdas - x_val)))
                if j == mark_x_red_idx:
                    ax.scatter([lambdas[idx_x]], [values[idx_x]], s=30,
                               color="red", zorder=4, marker="*", clip_on=False)
                else:
                    ax.scatter([lambdas[idx_x]], [values[idx_x]], s=30,
                               color="black", zorder=4, marker="x", clip_on=False)

    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(ylabel, color="none")
        ax.tick_params(axis="y", labelcolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    if logy:
        ax.set_yscale("log")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(frameon=False, ncol=1, loc="best")

    try:
        save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
