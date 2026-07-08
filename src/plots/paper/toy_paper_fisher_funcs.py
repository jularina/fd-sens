from collections import defaultdict
from typing import List, Tuple, Dict, FrozenSet, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb, Normalize, ListedColormap, BoundaryNorm
import matplotlib.cm as cmx
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable
from scipy.special import logsumexp
import os
from scipy.stats import gaussian_kde
import time
from matplotlib.lines import Line2D
from scipy.stats import sem, t
import ot


# def plot_prior_densities_by_fd(
#     all_ksd_data: Dict[str, Dict],
#     cfg: DictConfig,
#     plot_cfg: DictConfig,
#     output_dir: str,
# ):
#     """
#     Plots all prior densities (from different distributions) in one figure,
#     using 4 discrete color bins based on KSD quantiles and a colorbar legend.
#     The colorbar shows a SINGLE representative KSD value (median within bin)
#     for each color, not an interval. Lower-KSD densities are plotted last (on top).
#     """
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.cm as cmx
#     from matplotlib.colors import to_rgb, ListedColormap, BoundaryNorm
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Setup figure
#     plt.rcParams.update({
#         "font.size": plot_cfg.plot.font.size,
#         "font.family": plot_cfg.plot.font.family,
#         "text.usetex": plot_cfg.plot.font.use_tex,
#         "text.latex.preamble": r"\usepackage{amsmath}",
#     })
#
#     fig, ax = plt.subplots(
#         figsize=(
#             plot_cfg.plot.figure.size.width,
#             plot_cfg.plot.figure.size.height,
#         ),
#         dpi=plot_cfg.plot.figure.dpi,
#     )
#
#     # Plot base prior
#     base_mu = cfg.data.base_prior.mu
#     base_sigma = cfg.data.base_prior.sigma
#     base_dist = Gaussian(mu=base_mu, sigma=base_sigma)
#     x = np.linspace(base_mu - 6 * base_sigma, base_mu + 6 * base_sigma, 300)
#     ax.plot(
#         x,
#         base_dist.pdf(x),
#         label="Base prior",
#         color="black",
#         linewidth=1,
#         linestyle="--"
#     )
#
#     # Collect all KSD values
#     all_ksd_values = []
#     for dist_data in all_ksd_data.values():
#         all_ksd_values.extend(dist_data["fd"].values())
#     all_ksd_values = np.array(all_ksd_values, dtype=float)
#
#     # Compute bin edges: min, Q1, Q2, Q3, max
#     q1, q2, q3 = np.quantile(all_ksd_values, [0.25, 0.5, 0.75])
#     vmin, vmax = float(np.min(all_ksd_values)), float(np.max(all_ksd_values))
#     edges = [vmin, q1, q2, q3, vmax]
#
#     # Use first 4 colors from palette (reversed if desired)
#     palette_colors = plot_cfg.plot.color_palette.colors[:4]
#     rgb_colors = [to_rgb(c) for c in palette_colors[::-1]]
#
#     # Helper for compact labels
#     def _fmt(v):
#         return f"{v:.3g}"  # 3 sig figs
#
#     # Build a list of items to plot, then sort by KSD (descending) so lowest KSD is on top
#     plotting_items = []
#     for dist_name, dist_data in all_ksd_data.items():
#         ksd_results = dist_data["fd"]
#         param_names = dist_data["param_names"]
#         distribution_cls = dist_data["distribution_cls"]
#
#         for param_tuple, ksd in ksd_results.items():
#             # Determine discrete color bin via edges
#             if ksd <= edges[1]:
#                 bin_idx = 0
#             elif ksd <= edges[2]:
#                 bin_idx = 1
#             elif ksd <= edges[3]:
#                 bin_idx = 2
#             else:
#                 bin_idx = 3
#
#             param_dict = dict(zip([p.replace("_0", "") for p in param_names], param_tuple))
#             try:
#                 dist = distribution_cls(**param_dict)
#                 pdf_vals = dist.pdf(x)
#                 plotting_items.append((float(ksd), pdf_vals, bin_idx, dist_name, param_dict))
#             except Exception as e:
#                 print(f"[WARN] Skipping {dist_name} with params {param_dict}: {e}")
#
#     # Sort by KSD descending so we draw high KSD first, low KSD last (on top)
#     plotting_items.sort(key=lambda t: t[0], reverse=True)
#
#     # Plot densities with 4-bin color coding
#     for ksd, pdf_vals, bin_idx, dist_name, param_dict in plotting_items:
#         ax.fill_between(
#             x,
#             pdf_vals,
#             color=rgb_colors[bin_idx],
#             alpha=0.8,
#             linewidth=0.7
#         )
#
#     # Axes formatting
#     ax.set_xlabel(plot_cfg.plot.param_latex_names.mu_0)
#     ax.set_ylabel(plot_cfg.plot.param_latex_names.priorsimple)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     # Create colorbar
#     cmap = ListedColormap(rgb_colors)
#     bounds = [0, 1, 2, 3, 4]
#     norm = BoundaryNorm(bounds, cmap.N)
#     sm = cmx.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#
#     # Compute a single representative value per bin: median of KSDs in that bin
#     bin_values = [[] for _ in range(4)]
#     for ksd, *_ in plotting_items:
#         if ksd <= edges[1]:
#             bin_values[0].append(ksd)
#         elif ksd <= edges[2]:
#             bin_values[1].append(ksd)
#         elif ksd <= edges[3]:
#             bin_values[2].append(ksd)
#         else:
#             bin_values[3].append(ksd)
#
#     single_labels = []
#     for i in range(4):
#         if len(bin_values[i]) > 0:
#             val = round(float(np.median(bin_values[i])), 2)
#         else:
#             # Fallback if a bin is empty: midpoint of its edges
#             val = 0.5 * (edges[i] + edges[i+1])
#         single_labels.append(_fmt(val))
#
#     cbar = fig.colorbar(sm, ax=ax, boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
#     cbar.set_label(plot_cfg.plot.param_latex_names.estimatedFDposteriorsShort)
#     cbar.ax.tick_params(labelsize=plot_cfg.plot.font.size)
#     cbar.set_ticklabels(single_labels)
#
#     if plot_cfg.plot.figure.tight_layout:
#         plt.tight_layout()
#
#     output_path = os.path.join(output_dir, "toy_gaussian_model_b.pdf")
#     fig.savefig(output_path, format="pdf", bbox_inches="tight")
#     plt.close(fig)


def plot_multivariate_priors_densities(all_params, all_ksds, output_dir, plot_cfg, true_theta=None, true_cov=None):
    """
    Plots joint prior densities (3 of them) with marginals using fixed 3-color scheme and KSD arrow bar.
    """
    os.makedirs(output_dir, exist_ok=True)

    sorted_keys = sorted(all_ksds, key=lambda k: all_ksds[k])
    palette_colors = plot_cfg.plot.color_palette.colors[:3]
    rgb_colors = [to_rgb(c) for c in palette_colors[::-1]]

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig = plt.figure(figsize=(3.0, 3.0))
    grid = plt.GridSpec(4, 4, hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(grid[1:, :-1])
    ax_x = fig.add_subplot(grid[0, :-1], sharex=ax_main)
    ax_y = fig.add_subplot(grid[1:, -1], sharey=ax_main)

    for i, key in enumerate(sorted_keys):
        mu = all_params[key]["mu"]
        cov = all_params[key]["cov"]
        color = rgb_colors[i]

        samples = np.random.multivariate_normal(mu, cov, size=2000)

        # Joint KDE
        sns.kdeplot(
            x=samples[:, 0],
            y=samples[:, 1],
            ax=ax_main,
            fill=True,
            levels=10,
            alpha=0.9,
            thresh=0.01,
            linewidths=0.7,
            color=color,
        )

        # Marginals
        sns.kdeplot(samples[:, 0], ax=ax_x, color=color, fill=True, alpha=0.6)
        sns.kdeplot(samples[:, 1], ax=ax_y, color=color, fill=True, alpha=0.6, vertical=True)

    # Plot true_theta reference
    if true_theta is not None:
        # Black lines
        ax_main.axvline(true_theta[0], color="k", linestyle="-", lw=1)
        ax_main.axhline(true_theta[1], color="k", linestyle="-", lw=1)
        ax_x.axvline(true_theta[0], color="k", linestyle="-", lw=1)
        ax_y.axhline(true_theta[1], color="k", linestyle="-", lw=1)

        # Overlay black marginal KDE lines
        true_samples = np.random.multivariate_normal(true_theta, true_cov, size=2000)
        sns.kdeplot(true_samples[:, 0], ax=ax_x, color="k", lw=1, fill=False)
        sns.kdeplot(true_samples[:, 1], ax=ax_y, color="k", lw=1, fill=False, vertical=True)

        sns.kdeplot(
            x=true_samples[:, 0],
            y=true_samples[:, 1],
            ax=ax_main,
            fill=False,
            levels=10,
            alpha=0.3,
            thresh=0.01,
            linewidths=0.7,
            color="black",
        )

    # Axis cleanup
    ax_x.axis("off")
    ax_y.axis("off")
    ax_main.set_xlabel("$\\mu_{01}$")
    ax_main.set_ylabel("$\\mu_{02}$")
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # Add custom arrow colorbar for KSD ordering
    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    cmap = ListedColormap(rgb_colors)
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)
    cb = plt.colorbar(cmx.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    cb.ax.tick_params(size=0, width=0, labelsize=0, left=False, right=False)
    cb.set_ticks([])

    # Arrow showing increasing KSD
    cb.ax.annotate(
        '',
        xy=(1.4, 0.7),
        xytext=(1.4, 0.3),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            lw=1,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=8
        ),
    )

    cb.set_label(plot_cfg.plot.param_latex_names.estimatedFDposteriorsShort)

    fig.tight_layout(rect=[0, 0, 0.9, 1.0])
    output_path = os.path.join(output_dir, "multivariate_joint_prior_plot.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_multi_line_plots(
    ksd_results: Dict[Tuple[float, ...], float],
    param_names: List[str],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    latex_param_names = plot_cfg.plot.param_latex_names
    colors = plot_cfg.plot.color_palette.colors

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    param_values = np.array(list(ksd_results.keys()))
    ksd_values = np.array(list(ksd_results.values()))

    if len(param_names) != 2:
        raise ValueError("This function currently supports exactly two parameters.")

    def make_multi_line_plot(
            fixed_idx: int,
            varying_idx: int,
            filename_prefix: str,
    ):
        fixed_param_name = param_names[fixed_idx]
        varying_param_name = param_names[varying_idx]
        fixed_param_latex = latex_param_names.get(fixed_param_name, fixed_param_name)
        varying_param_latex = latex_param_names.get(varying_param_name, varying_param_name)

        fixed_vals = np.unique(param_values[:, fixed_idx])

        fig, ax = plt.subplots(
            figsize=(
                plot_cfg.plot.figure.size.width,
                plot_cfg.plot.figure.size.height,
            ),
            dpi=plot_cfg.plot.figure.dpi,
        )

        for i, fixed_val in enumerate(fixed_vals):
            mask = param_values[:, fixed_idx] == fixed_val
            x = param_values[mask, varying_idx]
            y = ksd_values[mask]
            sorted_idx = np.argsort(x)
            x = x[sorted_idx]
            y = y[sorted_idx]

            # Take color directly from palette (cycle if needed)
            line_color = colors[i % len(colors)]

            ax.plot(
                x, y,
                marker='.',
                label=f"{fixed_param_latex} = {fixed_val:.0f}",
                color=line_color,
            )

            if getattr(plot_cfg.plot, "show_min_point",
                       False) and fixed_param_latex == "$\\sigma$" and fixed_val == 3.0:
                min_idx = np.argmin(y)
                ax.scatter(
                    x[min_idx], y[min_idx],
                    color="black",
                    zorder=5,
                    marker='x',
                    s=50,
                )

            if fixed_param_latex == "$\\sigma$" and fixed_val == 2.0:
                max_idx = np.argmax(y)
                ax.scatter(
                    x[max_idx], y[max_idx],
                    color="red",
                    zorder=6,
                    marker='*',
                    s=50,
                )

        ax.set_xlabel(varying_param_latex)
        ksd_latex = latex_param_names.get("estimatedFDposteriorsQuadraticForm")
        ylabel = f"log {ksd_latex}" if plot_cfg.plot.y_axis.log_scale else ksd_latex
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right')

        if plot_cfg.plot.y_axis.log_scale:
            ax.set_yscale("log")

        if plot_cfg.plot.figure.tight_layout:
            plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            "toy_gaussian_model_a.pdf"
        )
        fig.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close(fig)
        print(f"Saved combined KSD vs {varying_param_name} plot to: {save_path}")

    make_multi_line_plot(fixed_idx=1, varying_idx=0, filename_prefix="ksd_multiline")


def plot_single_param(
    ksd_results: Dict[float, float],
    param_name: str,
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names
    latex_param_name = plot_cfg.plot.param_latex_names.get(param_name)

    # Sort the keys for consistent plotting
    x_vals = np.array(sorted(ksd_results.keys()))
    y_vals = np.array([ksd_results[x] for x in x_vals])

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
    })

    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    ax.plot(x_vals, y_vals, marker='.', color=plot_cfg.plot.color_palette.colors[0], linewidth=2.5)

    ax.set_xlabel(latex_param_name)
    ksd_latex = latex_param_names.get("estimatedFDposteriorsQuadraticForm")
    ylabel = f"log {ksd_latex}" if plot_cfg.plot.y_axis.log_scale else ksd_latex
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    if plot_cfg.plot.y_axis.log_scale:
        ax.set_yscale("log")

    if getattr(plot_cfg.plot, "show_min_point", False):
        min_idx = np.argmin(y_vals)
        min_x = x_vals[min_idx]
        min_y = y_vals[min_idx]
        ax.scatter(
            min_x, min_y,
            color="black",
            zorder=5,
            marker='x',
            s=50,
        )

    if getattr(plot_cfg.plot, "show_max_point", False):
        max_idx = np.argmax(y_vals)
        max_x = x_vals[max_idx]
        max_y = y_vals[max_idx]
        ax.scatter(
            max_x, max_y,
            color="red",
            zorder=6,
            marker='*',
            s=50,
        )

    filename = f"ising-{param_name}.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig)

    print(f"Saved line plot to: {save_path}")


def plot_eta_surface(
    results: List[Tuple[Dict[str, float], np.ndarray, float]],
    corner_points: List[Dict[str, float]],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import LinearLocator
    os.makedirs(output_dir, exist_ok=True)
    latex_param_names = plot_cfg.plot.param_latex_names

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
    })

    def _key_from_params(p: Dict[str, float]) -> FrozenSet[Tuple[str, float]]:
        return frozenset({(k, float(f"{v:.8f}")) for k, v in p.items()})

    def _fmt(v: float) -> str:
        return f"{float(v):.3g}"

    def _get_mu_sigma_from_corner(cp: Dict[str, float]) -> Tuple[float, float]:
        def _first(cp, keys):
            for k in keys:
                if k in cp:
                    return cp[k]
            return None
        mu = _first(cp, ["mu_0", "mu0", "mu"])
        sg = _first(cp, ["sigma_0", "sigma0", "sigma", "std", "sd"])
        return mu, sg

    def _ensure_math(s: str) -> str:
        return s if "$" in s else f"${s}$"

    # labels from config
    try:
        mu_label = getattr(latex_param_names, "mu_0")
    except Exception:
        mu_label = r"\mu_0"
    try:
        sigma_label = getattr(latex_param_names, "sigma_0")
    except Exception:
        sigma_label = r"\sigma_0"
    mu_label = _ensure_math(mu_label)
    sigma_label = _ensure_math(sigma_label)

    # font sizes from config
    base_fs = int(plot_cfg.plot.font.size)
    corner_num_fs = max(7, int(base_fs * 1.25))
    legend_title_fs = max(7, int(base_fs * 1.05))
    legend_line_fs = max(7, int(base_fs * 1.00))
    corner_dot_size = max(10, int(base_fs * 1.4))  # <-- smaller black dots
    y_label_fs = max(base_fs + 2, int(base_fs * 1.25))  # <-- larger y-label

    # ---------- gather data ----------
    x, y, z = [], [], []
    coords_by_key: Dict[FrozenSet[Tuple[str, float]], Tuple[float, float, float]] = {}

    corner_points = [cp[0] for cp in corner_points]
    corner_keys: List[FrozenSet[Tuple[str, float]]] = [_key_from_params(cp) for cp in corner_points]
    for prior_params, eta_tilde, ksd_est in results:
        if len(eta_tilde) < 2:
            continue
        eta0, eta1 = float(eta_tilde[0]), float(eta_tilde[1])
        x.append(eta0)
        y.append(eta1)
        z_val = float(np.log10(ksd_est) if plot_cfg.plot.y_axis.log_scale else ksd_est)
        z.append(z_val)
        coords_by_key[_key_from_params(prior_params)] = (eta0, eta1, z_val)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    palette_colors = plot_cfg.plot.color_palette.colors[::-1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", palette_colors)

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(x, y, z, cmap=cmap, edgecolor="none", linewidth=0.2, antialiased=True)

    # ax.set_xlabel(r"$\lambda_0=\frac{\mu_0}{\sigma_0^2}$")
    # ax.set_ylabel(r"$\lambda_1=\frac{-0.5}{\sigma_0^2}$", fontsize=y_label_fs)
    ax.set_xlabel(r"$\lambda_0$")
    ax.set_ylabel(r"$\lambda_1$", fontsize=y_label_fs)

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    # ---------- max marker ----------
    max_idx = int(np.argmax(z))
    max_pt = (x[max_idx], y[max_idx], z[max_idx])
    ax.scatter(*max_pt, color="red", marker="*", s=50, zorder=8)

    # ---------- corners: numbers + straight trajectory + black dots ----------
    corner_coords_ordered: List[Tuple[float, float, float]] = []
    legend_lines: List[str] = []

    for idx, (cp, ck) in enumerate(zip(corner_points, corner_keys), start=1):
        if ck not in coords_by_key:
            continue
        cx, cy, cz = coords_by_key[ck]
        corner_coords_ordered.append((cx, cy, cz))

        # number at the corner (small z-lift)
        # z_lift = 0.02 * (ax.get_zlim()[1] - ax.get_zlim()[0])
        # ax.text(cx, cy, cz + z_lift, f"{idx}",
        #         fontsize=corner_num_fs, color="black",
        #         ha="center", va="bottom", zorder=12, weight="bold")
        ax.text(cx, cy, cz, f"{idx}",
                fontsize=corner_num_fs, color="black",
                ha="center", va="bottom", zorder=12, weight="bold")

        mu_val, sg_val = _get_mu_sigma_from_corner(cp)
        parts = []
        if mu_val is not None:
            parts.append(f"{mu_label}={_fmt(mu_val)}")
        if sg_val is not None:
            parts.append(f"{sigma_label}={_fmt(sg_val)}")
        legend_lines.append(f"{idx}: " + ", ".join(parts) if parts else f"{idx}: (corner)")

    # straight trajectory
    if len(corner_coords_ordered) >= 2:
        traj = np.array(corner_coords_ordered, dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color="black", linestyle="-", linewidth=1.1, alpha=0.7, zorder=9)

    # black points at corners (skip the red max)
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    tol_x = 1e-6 * max(1.0, (xlim[1] - xlim[0]))
    tol_y = 1e-6 * max(1.0, (ylim[1] - ylim[0]))
    tol_z = 1e-6 * max(1.0, (zlim[1] - zlim[0]))
    for (cx, cy, cz) in corner_coords_ordered:
        is_max_corner = (abs(cx - max_pt[0]) < tol_x) and (abs(cy - max_pt[1])
                                                           < tol_y) and (abs(cz - max_pt[2]) < tol_z)
        if not is_max_corner:
            ax.scatter(cx, cy, cz, color="black", s=corner_dot_size, zorder=11)

    # ---------- middle-right legend ----------
    # N = max(1, len(legend_lines))
    # height = min(0.60, 0.05 * N + 0.10)
    # bottom = 0.50 - height / 2.0
    # legend_ax = fig.add_axes([0.77, bottom, 0.21, height])
    # legend_ax.axis("off")
    # legend_ax.text(0.0, 1.02, "Corners", fontsize=legend_title_fs, fontweight="bold",
    #                ha="left", va="bottom")
    # ys = [0.5] if N == 1 else np.linspace(0.85, 0.10, N)
    # for yv, line in zip(ys, legend_lines):
    #     legend_ax.text(0.0, float(yv), line, fontsize=legend_line_fs, ha="left", va="center")

    # ---------- ticks: exactly 3 on x, y, z ----------
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.xaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_locator(LinearLocator(4))
    # ax.zaxis.set_major_locator(LinearLocator(4))

    # format tick labels with max 2 decimals (strip trailing zeros)
    def _fmt_two_decimals(x, pos):
        s = f"{x:.1f}".rstrip('0').rstrip('.')
        return s

    from matplotlib.ticker import LinearLocator, FuncFormatter
    _tickfmt = FuncFormatter(_fmt_two_decimals)
    ax.xaxis.set_major_formatter(_tickfmt)
    ax.yaxis.set_major_formatter(_tickfmt)
    ax.zaxis.set_major_formatter(_tickfmt)

    # Bring tick labels closer to the plot
    tick_pad = 2  # pixels; lower = closer
    ax.tick_params(axis='x', which='major', pad=tick_pad)
    ax.tick_params(axis='y', which='major', pad=tick_pad)
    ax.tick_params(axis='z', which='major', pad=tick_pad)

    # OPTIONAL (3D-specific): pull ticks further inward if needed
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo['tick']['outward_factor'] = 0.0  # reduce outward push
        a._axinfo['tick']['inward_factor'] = 0.2  # small inward pull

    # ---------- cosmetics ----------
    ax.view_init(elev=30, azim=50)
    ksd_qf_latex = getattr(latex_param_names, "estimatedFDposteriorsQuadraticForm")
    zmin, zmax = ax.get_zlim()
    xmid = np.max(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    ax.text(xmid, ymin - 0.05, zmax - 0.03, ksd_qf_latex,
            rotation=90, fontsize=base_fs, va="bottom", ha="left")

    # save
    filename = "eta_surface_from_corners.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved 3D eta surface with straight path, smaller black corner points, 3 ticks/axis, and legend to: {save_path}")


def plot_mu_sigma_contour(
    results: List[Tuple[Dict[str, float], np.ndarray, float]],
    corner_points: List[Dict[str, float]],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    """
    2D filled contour plot of the quadratic form over (mu, sigma) space.
    Non-elliptic contour shapes reveal the non-convexity of the objective
    in the original parametrisation.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.interpolate import griddata
    os.makedirs(output_dir, exist_ok=True)
    latex_param_names = plot_cfg.plot.param_latex_names

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
    })

    def _key_from_params(p: Dict[str, float]) -> FrozenSet[Tuple[str, float]]:
        return frozenset({(k, float(f"{v:.8f}")) for k, v in p.items()})

    def _get_mu_sigma(p: Dict[str, float]):
        def _first(d, keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None
        mu = _first(p, ["mu_0", "mu0", "mu"])
        sg = _first(p, ["sigma_0", "sigma0", "sigma", "std", "sd"])
        return mu, sg

    # ---------- gather data ----------
    x, y, z = [], [], []
    coords_by_key: Dict[FrozenSet[Tuple[str, float]], Tuple[float, float, float]] = {}

    corner_points_dicts = [cp[0] for cp in corner_points]
    corner_keys = [_key_from_params(cp) for cp in corner_points_dicts]

    for prior_params, _eta, ksd_est in results:
        mu_val, sg_val = _get_mu_sigma(prior_params)
        if mu_val is None or sg_val is None:
            continue
        mu_f, sg_f = float(mu_val), float(sg_val)
        z_val = float(np.log10(ksd_est) if plot_cfg.plot.y_axis.log_scale else ksd_est)
        x.append(mu_f)
        y.append(sg_f)
        z.append(z_val)
        coords_by_key[_key_from_params(prior_params)] = (mu_f, sg_f, z_val)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # interpolate onto a regular grid for smooth contours
    grid_res = 200
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

    palette_colors = plot_cfg.plot.color_palette.colors[::-1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", palette_colors)

    base_fs = int(plot_cfg.plot.font.size)
    corner_num_fs = max(7, int(base_fs * 1.25))
    corner_dot_size = max(10, int(base_fs * 1.4))

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    cf = ax.contourf(Xi, Yi, Zi, levels=20, cmap=cmap)
    ct = ax.contour(Xi, Yi, Zi, levels=20, colors="white", linewidths=0.4, alpha=0.5)
    cbar = fig.colorbar(cf, ax=ax, label=getattr(latex_param_names, "estimatedFDposteriorsQuadraticForm", "KSD"))
    from matplotlib.ticker import MaxNLocator
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()

    # axis labels
    mu_label = r"$\mu$"
    sigma_label = r"$\sigma$"

    ax.set_xlabel(mu_label if "$" in mu_label else f"${mu_label}$", fontsize=base_fs)
    ax.set_ylabel(sigma_label if "$" in sigma_label else f"${sigma_label}$", fontsize=base_fs)

    # ---------- max marker ----------
    max_idx = int(np.argmax(z))
    ax.scatter(x[max_idx], y[max_idx], color="red", marker="*", s=80,
               zorder=20, clip_on=False, label="max")

    # ---------- corners ----------
    corner_coords_ordered = []
    for idx, (cp, ck) in enumerate(zip(corner_points_dicts, corner_keys), start=1):
        if ck not in coords_by_key:
            continue
        cx, cy, _ = coords_by_key[ck]
        corner_coords_ordered.append((cx, cy))
        ax.text(cx, cy, f"{idx}", fontsize=corner_num_fs, color="black",
                ha="center", va="bottom", zorder=15, weight="bold", clip_on=False)
        is_max = (abs(cx - x[max_idx]) < 1e-9 and abs(cy - y[max_idx]) < 1e-9)
        if not is_max:
            ax.scatter(cx, cy, color="black", s=corner_dot_size, zorder=11, clip_on=False)

    if len(corner_coords_ordered) >= 2:
        traj = np.array(corner_coords_ordered, dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], color="black", linestyle="-",
                linewidth=1.1, alpha=0.7, zorder=9)

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    filename = "hyperparams_contour_from_corners.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2D mu/sigma contour plot to: {save_path}")


def plot_sdp_densities_and_logprior(
    basis_function,
    sdp_lambda_list: list[np.ndarray],
    radius_labels: list[float],
    estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-5, 5),
    resolution: int = 200,
) -> None:
    """
    Combined plot: top = densities, bottom = log prior.
    Single shared legend (same styling as plot_sdp_comparisons_multiple_radii).
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # --- Grid & features ---
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    Phi_x = basis_function.evaluate(x)  # (resolution, D)
    prior_density_true = prior_distribution.pdf(x).flatten()
    log_prior_true = prior_distribution.log_pdf(x).flatten()

    # --- Colors ---
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    # --- Names & labels ---
    names = plot_cfg.plot.param_latex_names
    fd_label = names.get("estimatedSensitivityMeasure")
    xlabel = names.get("theta")
    ylabel_density = names.get("nonparametric_prior", "Density")
    ylabel_logprior = names.get("log_prior", "log_prior")
    true_density_label = names.get("logbaseprior", "True Prior Density")
    true_logprior_label = names.get("logbaseprior")
    true_prior_label = names.get("baseprior")
    approx_sym = r"$\approx$"
    geq_sym = r"$\geq$"

    # --- Figure & Axes ---
    fig, (ax_density, ax_log) = plt.subplots(
        2, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
        sharex=True
    )

    # ===== Top panel: densities =====
    # SDP densities
    for i, (psi, r_label, ksd) in enumerate(zip(sdp_lambda_list, radius_labels, estimates)):
        f = (Phi_x @ psi).flatten()
        logZ = logsumexp(f) + np.log(dx)
        p_hat = np.exp(f - logZ)
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label} ({ksd:.1f})"
        ax_density.plot(
            x.flatten(),
            p_hat,
            label=label,
            linewidth=1.5,
            color=color,
        )

    ax_density.set_ylabel(ylabel_density)
    ax_density.grid(True, alpha=0.3)
    ax_density.spines["top"].set_visible(False)
    ax_density.spines["right"].set_visible(False)

    # True prior density
    ax_density.plot(
        x.flatten(),
        prior_density_true,
        label=true_prior_label,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    # ===== Bottom panel: log prior =====

    # SDP log prior approximations (centered)
    for i, (psi, r_label, _) in enumerate(zip(sdp_lambda_list, radius_labels, estimates)):
        f = (Phi_x @ psi).flatten()
        c = float(np.mean(log_prior_true - f))  # match mean
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label}"
        ax_log.plot(
            x.flatten(),
            f + c,
            label=label,
            linewidth=1.5,
            color=color,
        )

    ax_log.set_xlabel(xlabel)
    ax_log.set_ylabel(ylabel_logprior)
    ax_log.grid(True, alpha=0.3)
    ax_log.spines["top"].set_visible(False)
    ax_log.spines["right"].set_visible(False)

    # True log prior
    ax_log.plot(
        x.flatten(),
        log_prior_true,
        label=true_prior_label,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    # ===== Shared legend =====
    handles, labels = ax_density.get_legend_handles_labels()
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
    for t in leg.get_texts():
        t.set_wrap(True)
    leg.get_title().set_ha("right")
    leg._legend_box.align = "right"
    plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"] * 0.9)
    plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"] * 0.9)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout(rect=[0, 0, 0.95, 1])

    # ===== Save =====
    filename = "toy_univariate_gaussian_model_nonparametric.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


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
        color="steelblue",
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
        plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"] * 0.9)
        plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"] * 0.9)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout(rect=[0, 0, 0.95, 1])

    filename = "gaussian_1d_location_model_diff_radii.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sdp_posterior_comparison(
    basis_function,
    sdp_lambda_list: list,
    radius_labels: list,
    estimates: list,
    prior_distribution,
    model,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-5, 5),
    resolution: int = 200,
) -> None:
    """
    For each SDP candidate prior, compute the posterior on a θ grid and plot
    alongside the reference (conjugate Gaussian) posterior.

    The candidate prior log-density is f(θ) = Φ(θ)ᵀλ (unnormalized).
    The posterior is p(θ|data) ∝ exp(f(θ)) · L(data|θ), normalised numerically.
    The reference posterior is the exact conjugate Gaussian.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # Grid
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    x_flat = x.flatten()
    Phi_x = basis_function.evaluate(x)

    # log g(θ): reference prior log-density, shared across all candidate priors
    log_g = prior_distribution.log_pdf(x).flatten()

    # Likelihood parameters from model
    x_bar = float(np.squeeze(model.x_bar))
    n = int(model.observations_num)
    sigma_lik_sq = float(model.loss.var)
    loss_lr = float(model.loss_lr_init)

    # Log-likelihood on grid (Gaussian, constant terms dropped)
    log_lik = -0.5 * loss_lr * n * (x_flat - x_bar) ** 2 / sigma_lik_sq

    # Reference posterior: conjugate Gaussian N(mu_n, sigma_n^2)
    mu_0 = float(np.squeeze(prior_distribution.mu))
    sigma_0_sq = float(np.squeeze(prior_distribution.var))
    sigma_n_sq = 1.0 / (loss_lr * n / sigma_lik_sq + 1.0 / sigma_0_sq)
    mu_n = sigma_n_sq * (loss_lr * n * x_bar / sigma_lik_sq + mu_0 / sigma_0_sq)
    ref_posterior = (
        np.exp(-0.5 * (x_flat - mu_n) ** 2 / sigma_n_sq)
        / np.sqrt(2 * np.pi * sigma_n_sq)
    )

    # Colors & labels
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    names = plot_cfg.plot.param_latex_names
    fd_label = names.get("estimatedSensitivityMeasure")
    xlabel = names.get("theta")
    ylabel = names.get("nonparametric_prior", "Posterior density")
    ref_label = names.get("baseprior", r"$\Pi_\mathrm{ref}$ posterior")
    geq_sym = r"$\geq$"

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height / 2),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Posterior for each SDP candidate prior:
    # log p(θ|data) ∝ Φ(θ)ᵀλ + log g(θ) + log L(data|θ)
    for i, (psi, r_label, ksd) in enumerate(zip(sdp_lambda_list, radius_labels, estimates)):
        f = (Phi_x @ psi).flatten()
        log_post = f + log_g + log_lik
        log_Z = logsumexp(log_post) + np.log(dx)
        posterior = np.exp(log_post - log_Z)
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label} ({ksd:.1f})"
        ax.plot(x_flat, posterior, label=label, linewidth=1.5, color=color)

    # Reference posterior
    ax.plot(x_flat, ref_posterior, label=ref_label, linestyle="--", linewidth=1.5, color="black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title=fd_label,
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        frameon=False,
        labelspacing=0.4,
        handlelength=1.8,
        handletextpad=0.5,
        borderpad=0.4,
    )

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        fig.tight_layout(rect=[0, 0, 0.95, 1])

    filename = "toy_univariate_gaussian_model_posteriors.pdf"
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
        colors="steelblue",
        linewidths=0.5,
        linestyles="dashed",
        alpha=0.8,
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
            linewidths=1.0,
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


def plot_multivariate_joint_prior_densities_by_fd(results, output_dir, plot_cfg, true_theta=None, true_cov=None):
    """
    Plots joint KDE contours of multivariate priors, colored by FD magnitude (no fill).
    Overlays true density if provided. Uses color map based on config.
    Highlights the distribution with the largest KSD in red.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort results by KSD ascending (low to high)
    sorted_results = sorted(results, key=lambda x: x[2])
    ksds = [ksd for (_, _, ksd) in sorted_results]
    min_ksd, max_ksd = min(ksds), max(ksds)

    # Normalize KSDs
    norm = Normalize(vmin=min_ksd, vmax=max_ksd)

    # Custom colormap from config
    color_list = plot_cfg.plot.color_palette.colors
    cmap = LinearSegmentedColormap.from_list("ksd_cmap", color_list[::-1])

    # Prepare figure
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, ax = plt.subplots(figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height))

    N = 25  # Show top-N and bottom-N KSD priors only
    subset_results = sorted_results[:N] + sorted_results[-N:]

    # Identify the distribution with the maximum KSD
    max_ksd_entry = sorted_results[-1]
    max_ksd_mu = max_ksd_entry[0]["mu"]
    max_ksd_cov = max_ksd_entry[0]["cov"]

    # Plot all selected priors
    for param_dict, _, ksd_est in subset_results:
        mu = param_dict["mu"]
        cov = param_dict["cov"]

        try:
            samples = np.random.multivariate_normal(mu, cov, size=1000)
        except np.linalg.LinAlgError:
            print(f"[WARN] Skipping invalid covariance matrix: {cov}")
            continue

        # If this is the max-KSD distribution, highlight it in red
        is_max_ksd = np.allclose(mu, max_ksd_mu) and np.allclose(cov, max_ksd_cov)

        if is_max_ksd:
            color = "red"
            lw = 2.0
            alpha = 1.0
            levels = 3
            ax.plot(mu[0], mu[1], marker='*', markersize=5, color=color)
        else:
            color = cmap(norm(ksd_est))
            alpha = 0.4 + 0.6 * norm(ksd_est)
            lw = 0.5 + 1.0 * norm(ksd_est)
            levels = 1

        sns.kdeplot(
            x=samples[:, 0],
            y=samples[:, 1],
            ax=ax,
            fill=False,
            levels=levels,
            linewidths=lw,
            color=color,
            alpha=alpha,
        )

    # Overlay true density if provided
    if true_theta is not None and true_cov is not None:
        try:
            true_samples = np.random.multivariate_normal(true_theta, true_cov, size=2000)
            sns.kdeplot(
                x=true_samples[:, 0],
                y=true_samples[:, 1],
                ax=ax,
                fill=False,
                levels=3,
                linewidths=1.0,
                alpha=1.0,
                color="black",
            )
            ax.plot(true_theta[0], true_theta[1], "ko", markersize=5)
            ax.axvline(true_theta[0], color="k", linestyle="--", lw=1)
            ax.axhline(true_theta[1], color="k", linestyle="--", lw=1)
        except np.linalg.LinAlgError:
            print("[WARN] Skipping true density overlay due to invalid covariance.")

    # Labels and appearance
    ax.set_xlabel("$\\theta_1$")
    ax.set_ylabel("$\\theta_2$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(plot_cfg.plot.param_latex_names.estimatedFDposteriorsQuadraticForm)

    # Save
    output_path = os.path.join(output_dir, "toy_gaussian_model_multivariate.pdf")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_existing_methods_comparison_gaussians(
    output_dir: str,
    plot_cfg,
    mu_ref: np.ndarray,
    mu_cand_1: np.ndarray,
    Sigma_ref: np.ndarray,
    Sigma_cand_1: np.ndarray,
    mu_cand_2: np.ndarray = None,
    Sigma_cand_2: np.ndarray = None,
    filename: str = "comparison_existing_methods_gaussians.pdf",
    annotation_text: str = None,
    annotation_fontsize: int = 12,
) -> None:
    """
    Plot 2D Gaussian posterior contours:.
    """
    os.makedirs(output_dir, exist_ok=True)

    def gaussian_pdf_grid(mu: np.ndarray, cov: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Evaluate N(mu,cov) density on a meshgrid (X,Y).
        """
        pos = np.stack([X, Y], axis=-1)  # (H,W,2)
        d = mu.shape[0]
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        diff = pos - mu.reshape(1, 1, d)
        # quadratic form for each grid point
        qf = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det)
        return norm_const * np.exp(-0.5 * qf)

    # --- Plot styling (match your Overleaf style) ---
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
    )

    # Build a grid covering all three distributions
    # Use eigenvalues to set a reasonable extent (say ~3 std in principal directions)
    def extent_from_cov(cov: np.ndarray, k: float = 4.5):
        w, V = np.linalg.eigh(cov)
        r = k * np.sqrt(np.max(w))
        return r

    extents = [extent_from_cov(Sigma_ref), extent_from_cov(Sigma_cand_1)]
    if Sigma_cand_2 is not None:
        extents.append(extent_from_cov(Sigma_cand_2))
    r = max(extents)
    all_mus = [mu_ref, mu_cand_1]
    if mu_cand_2 is not None:
        all_mus.append(mu_cand_2)
    x_min = min(m[0] for m in all_mus) - r
    x_max = max(m[0] for m in all_mus) + r
    y_min = min(m[1] for m in all_mus) - r
    y_max = max(m[1] for m in all_mus) + r

    xs = np.linspace(x_min, x_max, 220)
    ys = np.linspace(y_min, y_max, 220)
    X, Y = np.meshgrid(xs, ys)

    Z_ref = gaussian_pdf_grid(mu_ref, Sigma_ref, X, Y)
    Z_1 = gaussian_pdf_grid(mu_cand_1, Sigma_cand_1, X, Y)
    Z_2 = gaussian_pdf_grid(mu_cand_2, Sigma_cand_2, X, Y) if mu_cand_2 is not None else None

    # Choose contour levels relative to each max so shapes are comparable
    def levels_from_Z(Z: np.ndarray):
        zmax = np.max(Z)
        return [0.05 * zmax, 0.15 * zmax, 0.35 * zmax]

    # Colors
    col_ref = plot_cfg.plot.color_palette.colors[0]
    col_1 = plot_cfg.plot.color_palette.colors[2]
    col_2 = plot_cfg.plot.color_palette.colors[1]

    # Reference posterior
    ax.contourf(
        X, Y, Z_ref,
        levels=[Z_ref.max() * 0.05, Z_ref.max()],
        colors=[col_ref],
        alpha=0.25
    )

    # Candidate 1
    ax.contourf(
        X, Y, Z_1,
        levels=[Z_1.max() * 0.05, Z_1.max()],
        colors=[col_1],
        alpha=0.25
    )

    # Candidate 2
    if Z_2 is not None:
        ax.contourf(
            X, Y, Z_2,
            levels=[Z_2.max() * 0.05, Z_2.max()],
            colors=[col_2],
            alpha=0.25
        )
    from scipy.stats import chi2

    prob_levels = [0.5, 0.95]
    chi_levels = chi2.ppf(prob_levels, df=2)

    def gaussian_density_levels(mu, cov, chi_levels):
        det = np.linalg.det(cov)
        norm = 1.0 / np.sqrt((2 * np.pi) ** 2 * det)
        return norm * np.exp(-0.5 * chi_levels)

    levels_ref = np.sort(gaussian_density_levels(mu_ref, Sigma_ref, chi_levels))
    levels_1 = np.sort(gaussian_density_levels(mu_cand_1, Sigma_cand_1, chi_levels))
    ax.contour(X, Y, Z_ref, levels=levels_ref, colors=[col_ref], linewidths=1.5)
    ax.contour(X, Y, Z_1, levels=levels_1, colors=[col_1], linewidths=1.2, linestyles="--")
    if Z_2 is not None:
        levels_2 = np.sort(gaussian_density_levels(mu_cand_2, Sigma_cand_2, chi_levels))
        ax.contour(X, Y, Z_2, levels=levels_2, colors=[col_2], linewidths=1.2, linestyles=":")

    # Mark the common mean
    ax.plot(mu_ref[0], mu_ref[1], marker="o", markersize=4.5, color="black")
    ax.axvline(mu_ref[0], color="k", linestyle="--", lw=0.8, alpha=0.6)
    ax.axhline(mu_ref[1], color="k", linestyle="--", lw=0.8, alpha=0.6)

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = [
        Line2D([0], [0], color=col_ref, lw=1.6, label=plot_cfg.plot.param_latex_names.referenceposterior),
        Line2D([0], [0], color=col_1, lw=1.2, ls="--", label=r"$\tilde{\Pi}^{\lambda_1}$"),
    ]
    if Z_2 is not None:
        handles.append(Line2D([0], [0], color=col_2, lw=1.2, ls=":", label=r"$\tilde{\Pi}^{\lambda_2}$"))
    ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.0, 1.08), loc="upper right")

    # Annotation box (the point of the figure)
    if annotation_text is not None:
        ax.text(
            0.02, 0.03, annotation_text,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=annotation_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.85),
        )

    fig.tight_layout()
    outpath = os.path.join(output_dir, filename)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {outpath}")


def fisher_divergence_gaussians_ref_expectation(
    mu_ref: np.ndarray,
    Sigma_ref: np.ndarray,
    mu_cand: np.ndarray,
    Sigma_cand: np.ndarray,
) -> float:
    """
    FD(P_ref || P_cand) = E_{X~P_ref} || s_ref(X) - s_cand(X) ||^2
    for Gaussians with score s(x) = -Sigma^{-1}(x-mu).
    """
    d = mu_ref.shape[0]
    Sref_inv = np.linalg.inv(Sigma_ref)
    Scand_inv = np.linalg.inv(Sigma_cand)

    A = Scand_inv - Sref_inv  # (d,d)
    # mean term simplifies nicely:
    # (A mu_ref + (Sref_inv mu_ref - Scand_inv mu_cand)) = Scand_inv (mu_ref - mu_cand)
    diff_mu = (mu_ref - mu_cand).reshape(d, 1)
    mean_term = float(diff_mu.T @ (Scand_inv.T @ Scand_inv) @ diff_mu)

    trace_term = float(np.trace(A @ Sigma_ref @ A.T))
    return trace_term + mean_term


def w2_gaussian(mu1: np.ndarray, Sigma1: np.ndarray, mu2: np.ndarray, Sigma2: np.ndarray) -> float:
    """
    2-Wasserstein distance between Gaussians:
    W2^2 = ||m1-m2||^2 + tr(S1 + S2 - 2*(S2^{1/2} S1 S2^{1/2})^{1/2})
    """
    diff = mu1 - mu2
    from scipy.linalg import sqrtm
    S2_sqrt = sqrtm(Sigma2)
    middle = S2_sqrt @ Sigma1 @ S2_sqrt
    middle_sqrt = sqrtm(middle)
    middle_sqrt = np.real_if_close(middle_sqrt, tol=1e5)
    w2_sq = float(diff @ diff + np.trace(Sigma1 + Sigma2 - 2.0 * middle_sqrt))

    return float(np.sqrt(max(w2_sq, 0.0)))


def estimate_w2_from_samples(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Estimate W2 distance between empirical distributions of X and Y.

    Parameters
    ----------
    X : np.ndarray, shape (m, d)
        Samples from first distribution.
    Y : np.ndarray, shape (n, d)
        Samples from second distribution.

    Returns
    -------
    float
        Estimated 2-Wasserstein distance.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    m = X.shape[0]
    n = Y.shape[0]

    a = np.ones(m) / m
    b = np.ones(n) / n

    # Squared Euclidean cost matrix
    M = ot.dist(X, Y, metric="euclidean") ** 2

    # Optimal transport cost = W2^2
    w2_sq = ot.emd2(a, b, M)

    return float(np.sqrt(max(w2_sq, 0.0)))

# def estimate_w2_from_samples(X: np.ndarray, Y: np.ndarray) -> float:
#     """
#     Estimate W2 distance between empirical distributions of X and Y.
#     Uses the exact sorted-samples formula in 1D, and generic OT otherwise.
#     """
#     X = np.asarray(X)
#     Y = np.asarray(Y)
#
#     if X.ndim == 2 and X.shape[1] == 1 and Y.ndim == 2 and Y.shape[1] == 1:
#         x_sorted = np.sort(X[:, 0])
#         y_sorted = np.sort(Y[:, 0])
#
#         if len(x_sorted) != len(y_sorted):
#             raise ValueError("1D shortcut assumes equal sample sizes.")
#
#         w2_sq = np.mean((x_sorted - y_sorted) ** 2)
#         return float(np.sqrt(max(w2_sq, 0.0)))
#
#     m = X.shape[0]
#     n = Y.shape[0]
#
#     a = np.ones(m) / m
#     b = np.ones(n) / n
#
#     M = ot.dist(X, Y, metric="euclidean") ** 2
#     w2_sq = ot.emd2(a, b, M)
#
#     return float(np.sqrt(max(w2_sq, 0.0)))


def score_gaussian(x: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    # x: (..., d)
    return -(x - mu) @ Sigma_inv.T  # consistent with row-vectors


def estimate_fd_from_ref_samples(
    X_ref: np.ndarray,
    mu_ref: np.ndarray,
    Sigma_ref: np.ndarray,
    mu_cand: np.ndarray,
    Sigma_cand: np.ndarray,
) -> float:
    Sref_inv = np.linalg.inv(Sigma_ref)
    Scand_inv = np.linalg.inv(Sigma_cand)
    s_ref = score_gaussian(X_ref, mu_ref, Sref_inv)
    s_cand = score_gaussian(X_ref, mu_cand, Scand_inv)
    diff = s_ref - s_cand
    return float(np.mean(np.sum(diff * diff, axis=1)))


def sample_gaussian(rng: np.random.Generator, mu: np.ndarray, Sigma: np.ndarray, m: int) -> np.ndarray:
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=m)


def cov_mle(X: np.ndarray) -> np.ndarray:
    # MLE covariance: divide by m (bias=True)
    return np.cov(X, rowvar=False, bias=True)


# -----------------------------
#   Distribution construction across d
# -----------------------------
def make_toeplitz_cov(d: int, rho: float = 0.35, diag: float = 1.0) -> np.ndarray:
    """
    SPD Toeplitz covariance with entries diag * rho^{|i-j|}.
    """
    idx = np.arange(d)
    C = rho ** np.abs(idx[:, None] - idx[None, :])
    return diag * C


def make_experiment_distributions(d: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Fix one reference and (i) one candidate for FD/mean,
    and (ii) two candidates for WIM.
    These are kept stable across d.
    """
    mu_ref = np.zeros(d)
    Sigma_ref = make_toeplitz_cov(d, rho=0.30, diag=1.0)

    # Candidate for FD + mean
    mu_cand = np.zeros(d)
    mu_cand[: min(3, d)] = np.array([0.4, -0.2, 0.3])[: min(3, d)]
    Sigma_cand = make_toeplitz_cov(d, rho=0.55, diag=1.2)

    # Two candidates for WIM
    mu_1 = np.zeros(d)
    mu_2 = np.zeros(d)
    mu_2[: min(3, d)] = np.array([0.8, -0.4, 0.0])[: min(3, d)]
    Sigma_1 = make_toeplitz_cov(d, rho=0.20, diag=0.9)
    Sigma_2 = make_toeplitz_cov(d, rho=0.60, diag=1.1)

    return {
        "ref": (mu_ref, Sigma_ref),
        "cand": (mu_cand, Sigma_cand),
        "wim1": (mu_1, Sigma_1),
        "wim2": (mu_2, Sigma_2),
    }


def kl_gaussian(p_mu: np.ndarray, p_Sigma: np.ndarray, q_mu: np.ndarray, q_Sigma: np.ndarray) -> float:
    """
    KL( N(p_mu,p_Sigma) || N(q_mu,q_Sigma) )
    """
    d = p_mu.shape[0]
    q_Sigma_inv = np.linalg.inv(q_Sigma)
    diff = (q_mu - p_mu).reshape(d, 1)

    sign_p, logdet_p = np.linalg.slogdet(p_Sigma)
    sign_q, logdet_q = np.linalg.slogdet(q_Sigma)
    if sign_p <= 0 or sign_q <= 0:
        raise ValueError("Covariance must be SPD for KL computation.")

    tr_term = float(np.trace(q_Sigma_inv @ p_Sigma))
    quad_term = float(diff.T @ q_Sigma_inv @ diff)
    return 0.5 * (tr_term + quad_term - d + (logdet_q - logdet_p))


def estimate_kl_from_ref_samples(
    X_ref: np.ndarray,
    mu_ref: np.ndarray,
    Sigma_ref: np.ndarray,
    mu_cand: np.ndarray,
    Sigma_cand: np.ndarray,
) -> float:
    """
    Monte Carlo estimator of KL(P_ref || P_cand) using samples from P_ref:
        KL = E_{X~P_ref}[ log p_ref(X) - log p_cand(X) ].
    """
    d = mu_ref.shape[0]
    Sref_inv = np.linalg.inv(Sigma_ref)
    Scand_inv = np.linalg.inv(Sigma_cand)

    sign_r, logdet_r = np.linalg.slogdet(Sigma_ref)
    sign_c, logdet_c = np.linalg.slogdet(Sigma_cand)
    if sign_r <= 0 or sign_c <= 0:
        raise ValueError("Covariance must be SPD for KL computation.")

    Xc_r = X_ref - mu_ref.reshape(1, d)
    Xc_c = X_ref - mu_cand.reshape(1, d)

    qf_r = np.einsum("bi,ij,bj->b", Xc_r, Sref_inv, Xc_r)
    qf_c = np.einsum("bi,ij,bj->b", Xc_c, Scand_inv, Xc_c)

    logp_r = -0.5 * (d * np.log(2 * np.pi) + logdet_r + qf_r)
    logp_c = -0.5 * (d * np.log(2 * np.pi) + logdet_c + qf_c)

    return float(np.mean(logp_r - logp_c))


def estimate_kl_from_ref_samples_kde(
    X_ref_eval: np.ndarray,
    X_ref_fit: np.ndarray,
    X_cand_fit: np.ndarray,
    bw_method: None,
    eps: float = 1e-12,
) -> float:
    """
    Monte Carlo estimator of KL(P_ref || P_cand) using samples from P_ref
    and KDE approximations for both densities.

    Parameters
    ----------
    X_ref_eval :
        Samples used to approximate the expectation, shape (m_eval, d).
        These should ideally be different from X_ref_fit to reduce bias.
    X_ref_fit :
        Samples used to fit KDE for the reference posterior, shape (m_ref, d).
    X_cand_fit :
        Samples used to fit KDE for the candidate posterior, shape (m_cand, d).
    bw_method :
        Bandwidth passed to scipy.stats.gaussian_kde.
        Examples: None, "scott", "silverman", or a float.
    eps :
        Small constant to avoid log(0).

    Returns
    -------
    float
        KDE-based estimator of KL(P_ref || P_cand).
    """
    if X_ref_eval.ndim != 2 or X_ref_fit.ndim != 2 or X_cand_fit.ndim != 2:
        raise ValueError("All inputs must have shape (n_samples, d).")

    d_ref_eval = X_ref_eval.shape[1]
    d_ref_fit = X_ref_fit.shape[1]
    d_cand_fit = X_cand_fit.shape[1]

    if not (d_ref_eval == d_ref_fit == d_cand_fit):
        raise ValueError("All sample sets must have the same dimension.")

    # scipy gaussian_kde expects shape (d, n_samples)
    kde_ref = gaussian_kde(X_ref_fit.T, bw_method=bw_method)
    kde_cand = gaussian_kde(X_cand_fit.T, bw_method=bw_method)

    p_ref = kde_ref(X_ref_eval.T)
    p_cand = kde_cand(X_ref_eval.T)

    logp_ref = np.log(np.maximum(p_ref, eps))
    logp_cand = np.log(np.maximum(p_cand, eps))

    return float(np.mean(logp_ref - logp_cand))


_METHOD_COLORS = {
    "wim":  "#450314",
    "kl":   "#7c397d",
    "mean": "#3F3FFF",
    "fd":   "#5b9bd5",
}


def _method_color(plot_cfg, method: str) -> str:
    """Map each method to a fixed color."""
    return _METHOD_COLORS[method]


def _alpha_for_dim(dims, d, alpha_min=0.25, alpha_max=0.95) -> float:
    """
    Fade within a method by varying alpha across dims.
    Here: earlier dims -> more opaque, later dims -> more transparent.
    """
    if len(dims) <= 1:
        return alpha_max
    i = dims.index(d)
    # i=0 -> alpha_max, i=end -> alpha_min
    return float(alpha_max - (alpha_max - alpha_min) * (i / (len(dims) - 1)))


def _rgba_with_alpha(color: str, alpha: float):
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)


def _linestyle_for_dim(dims, d):
    """
    Assign a deterministic linestyle to each dimension.
    """
    linestyles = ["-", "--", ":", "-."]

    idx = dims.index(d)

    return linestyles[idx % len(linestyles)]


def compute_global_ylim_error(results: Dict[str, Any], logy: bool = False) -> Tuple[float, float]:
    error_mean = results["error_mean"]
    error_ci = results["error_ci"]
    dims = results["dims"]
    ymin, ymax = np.inf, -np.inf
    for method in error_mean:
        for d in dims:
            y = np.array(error_mean[method][d], dtype=float)
            h = np.array(error_ci[method][d], dtype=float)
            lower, upper = y - h, y + h
            if logy:
                lower_pos = lower[lower > 0]
                if lower_pos.size > 0:
                    ymin = min(ymin, np.min(lower_pos))
            else:
                ymin = min(ymin, np.min(lower))
            ymax = max(ymax, np.max(upper))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        raise ValueError("Could not determine global y-limits.")
    if logy:
        ymin *= 0.95
        ymax *= 1.05
    else:
        yrange = ymax - ymin or max(abs(ymax), 1.0)
        ymin -= 0.05 * yrange
        ymax += 0.05 * yrange
    return ymin, ymax


def compute_global_ylim_time(results: Dict[str, Any], logy: bool = False) -> Tuple[float, float]:
    time_mean = results["time_mean"]
    time_ci = results["time_ci"]
    dims = results["dims"]
    ymin, ymax = np.inf, -np.inf
    for method in time_mean:
        for d in dims:
            y = np.array(time_mean[method][d], dtype=float)
            h = np.array(time_ci[method][d], dtype=float)
            lower, upper = y - h, y + h
            if logy:
                lower_pos = lower[lower > 0]
                if lower_pos.size > 0:
                    ymin = min(ymin, np.min(lower_pos))
            else:
                ymin = min(ymin, np.min(lower))
            ymax = max(ymax, np.max(upper))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        raise ValueError("Could not determine global y-limits.")
    if logy:
        ymin *= 0.95
        ymax *= 1.05
    else:
        yrange = ymax - ymin or max(abs(ymax), 1.0)
        ymin -= 0.05 * yrange
        ymax += 0.05 * yrange
    return ymin, ymax


def compute_gaussian_complexity_results(
    ms: List[int],
    dims: List[int],
    n_rep: int = 30,
    seed: int = 0,
    divergence: str = None,
) -> Dict[str, Any]:
    """
    Compute both finite-sample estimation errors and runtimes
    for FD / mean / WIM / KL in one shared Monte Carlo loop.
    """
    rng = np.random.default_rng(seed)

    methods = ["fd", "mean", "wim", "kl"]

    error_mean = {method: {d: [] for d in dims} for method in methods}
    error_ci = {method: {d: [] for d in dims} for method in methods}

    time_mean = {method: {d: [] for d in dims} for method in methods}
    time_ci = {method: {d: [] for d in dims} for method in methods}

    def mean_and_ci(
            x: np.ndarray,
            n_boot: int = 2000,
            alpha: float = 0.05,
            rng: np.random.Generator = None,
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for the mean.

        Returns
        -------
        mean : float
        ci_half_width : float
            Symmetric half-width so it can be plotted as mean ± ci.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        mean_x = float(np.mean(x))

        if n <= 1:
            return mean_x, 0.0

        if rng is None:
            rng = np.random.default_rng()

        boot_means = np.empty(n_boot)

        for b in range(n_boot):
            sample = x[rng.integers(0, n, size=n)]
            boot_means[b] = np.mean(sample)

        lower = np.quantile(boot_means, alpha / 2)
        upper = np.quantile(boot_means, 1 - alpha / 2)

        half_width = float(max(mean_x - lower, upper - mean_x))

        return mean_x, half_width

    for d in dims:
        dist = make_experiment_distributions(d)
        mu_ref, Sigma_ref = dist["ref"]
        mu_cand, Sigma_cand = dist["cand"]
        mu_1, Sigma_1 = dist["wim1"]
        mu_2, Sigma_2 = dist["wim2"]

        # exact targets
        fd_true = fisher_divergence_gaussians_ref_expectation(
            mu_ref, Sigma_ref, mu_cand, Sigma_cand
        )
        mu_cand_true = mu_cand
        w2_true = w2_gaussian(mu_ref, Sigma_ref, mu_cand, Sigma_cand)
        kl_true = kl_gaussian(mu_ref, Sigma_ref, mu_cand, Sigma_cand)

        for m in ms:
            print(f"Computing statistics for dim={d}, m={m}.")

            fd_err_rep = np.empty(n_rep, dtype=float)
            mean_err_rep = np.empty(n_rep, dtype=float)
            wim_err_rep = np.empty(n_rep, dtype=float)
            kl_err_rep = np.empty(n_rep, dtype=float)

            fd_time_rep = np.empty(n_rep, dtype=float)
            mean_time_rep = np.empty(n_rep, dtype=float)
            wim_time_rep = np.empty(n_rep, dtype=float)
            kl_time_rep = np.empty(n_rep, dtype=float)

            for r in range(n_rep):
                # shared samples
                X_ref = sample_gaussian(rng, mu_ref, Sigma_ref, m)
                X_cand = sample_gaussian(rng, mu_cand, Sigma_cand, m)
                # X1 = sample_gaussian(rng, mu_1, Sigma_1, m)
                # X2 = sample_gaussian(rng, mu_2, Sigma_2, m)

                # FD
                t0 = time.perf_counter()
                fd_hat = estimate_fd_from_ref_samples(
                    X_ref, mu_ref, Sigma_ref, mu_cand, Sigma_cand
                )
                fd_time_rep[r] = time.perf_counter() - t0
                fd_err_rep[r] = abs(fd_true - fd_hat)

                # KL
                t0 = time.perf_counter()
                # kl_hat = estimate_kl_from_ref_samples(
                #     X_ref, mu_ref, Sigma_ref, mu_cand, Sigma_cand
                # )
                n = X_ref.shape[0]
                perm = rng.permutation(n)
                n_train = n // 2
                X_ref_train = X_ref[perm[:n_train]]
                X_ref_test = X_ref[perm[n_train:]]
                kl_hat = estimate_kl_from_ref_samples_kde(
                    X_ref_eval=X_ref_test,
                    X_ref_fit=X_ref_train,
                    X_cand_fit=X_cand,
                    bw_method="scott",
                )
                kl_time_rep[r] = time.perf_counter() - t0
                kl_err_rep[r] = abs(kl_true - kl_hat)

                # mean
                t0 = time.perf_counter()
                mu_hat = np.mean(X_cand, axis=0)
                mean_time_rep[r] = time.perf_counter() - t0
                mean_err_rep[r] = float(np.linalg.norm(mu_cand_true - mu_hat, ord=2))

                # WIM
                t0 = time.perf_counter()
                w2_hat = estimate_w2_from_samples(X_ref, X_cand)
                wim_time_rep[r] = time.perf_counter() - t0
                wim_err_rep[r] = abs(w2_true - w2_hat)

            # store error summaries
            for method, arr in {
                "fd": fd_err_rep,
                "mean": mean_err_rep,
                "wim": wim_err_rep,
                "kl": kl_err_rep,
            }.items():
                m_, h_ = mean_and_ci(arr, rng=rng)
                error_mean[method][d].append(m_)
                error_ci[method][d].append(h_)

            # store time summaries
            for method, arr in {
                "fd": fd_time_rep,
                "mean": mean_time_rep,
                "wim": wim_time_rep,
                "kl": kl_time_rep,
            }.items():
                m_, h_ = mean_and_ci(arr)
                time_mean[method][d].append(m_)
                time_ci[method][d].append(h_)

    return {
        "ms": ms,
        "dims": dims,
        "error_mean": error_mean,
        "error_ci": error_ci,
        "time_mean": time_mean,
        "time_ci": time_ci,
    }


def plot_runtime_complexity_gaussians(
    output_dir: str,
    plot_cfg,
    ms: List[int],
    dims: List[int],
    n_rep: int = 10,
    seed: int = 0,
    logy: bool = False,
    results: Dict[str, Any] | None = None,
    divergence: str = None,
    ylim: Tuple[float, float] = None,
    xlim: Tuple[float, float] = None,
    show_ci: bool = False,
) -> Dict[str, Any]:
    """
    Plot runtime of estimating FD, mean, KL, and WIM.

    All figures are saved with identical axes size.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    if results is None:
        results = compute_gaussian_complexity_results(
            ms=ms,
            dims=dims,
            n_rep=n_rep,
            seed=seed,
        )

    time_mean = results["time_mean"]
    time_ci = results["time_ci"]
    methods = [divergence] if divergence is not None else ["fd", "mean", "kl", "wim"]

    def compute_global_ylim() -> Tuple[float, float]:
        ymin = np.inf
        ymax = -np.inf

        for method in methods:
            for d in dims:
                y = np.array(time_mean[method][d], dtype=float)
                h = np.array(time_ci[method][d], dtype=float)

                lower = y - h
                upper = y + h

                if logy:
                    lower_pos = lower[lower > 0]
                    if lower_pos.size > 0:
                        ymin = min(ymin, np.min(lower_pos))
                else:
                    ymin = min(ymin, np.min(lower))

                ymax = max(ymax, np.max(upper))

        if not np.isfinite(ymin) or not np.isfinite(ymax):
            raise ValueError("Could not determine global y-limits.")

        if logy:
            ymin *= 0.95
            ymax *= 1.05
        else:
            yrange = ymax - ymin
            if yrange <= 0:
                yrange = max(abs(ymax), 1.0)
            ymin -= 0.05 * yrange
            ymax += 0.05 * yrange

        return ymin, ymax

    ylim_global = ylim if ylim is not None else compute_global_ylim()

    def plot_method(method: str, ylim: Tuple[float, float], show_axes_labels: bool) -> None:
        fig, ax = _make_figure(plot_cfg)

        base = _method_color(plot_cfg, method)

        for d in dims:
            y = np.array(time_mean[method][d], dtype=float)
            h = np.array(time_ci[method][d], dtype=float)

            a_line = _alpha_for_dim(dims, d, alpha_min=0.3, alpha_max=1.0)
            line_c = _rgba_with_alpha(base, a_line)
            fill_c = _rgba_with_alpha(base, max(0.10, 0.55 * a_line))
            ls = _linestyle_for_dim(dims, d)

            lower = y - h
            upper = y + h
            if logy:
                lower = np.maximum(lower, 1e-12)

            ax.plot(
                ms,
                y,
                label=rf"$d_\Theta={d}$",
                color=line_c,
                linewidth=2.0,
                linestyle=ls,
            )
            if show_ci:
                ax.fill_between(ms, lower, upper, color=fill_c, linewidth=0)

        _apply_common_plot_style(
            ax,
            show_xlabel=show_axes_labels,
            show_ylabel=show_axes_labels,
            ylabel="Cost (sec.)",
            xlabel=r"$m$",
            logy=logy,
            ylim=ylim,
            xlim=xlim,
        )

        # if method == "wim":
        #     ax.legend(frameon=False, loc="upper left")

        outpath = os.path.join(output_dir, f"comparison_runtime_{method.lower()}.pdf")
        fig.savefig(outpath, format="pdf")
        plt.close(fig)
        print(f"[Saved] {outpath}")

    for method in methods:
        plot_method(
            method,
            ylim_global,
            show_axes_labels=(method == "fd"),
        )

    return results


def plot_param_nonparam_runtimes_gaussians(
    output_dir: str,
    plot_cfg,
    param_raw_times: Dict[str, Dict[str, Dict[str, float]]],
    nonparam_raw_times: Dict[str, Dict[str, Dict[str, float]]],
    dims: List[int],
    param_ms: List[int],
    nonparam_ms: List[int],
    logy: bool = False,
    ylim: Tuple[float, float] = None,
    xlim: Tuple[float, float] = None,
    ci_level: float = 0.95,
) -> None:
    """
    Plot two runtime figures (parametric FD and nonparametric eigenvalue) and save to
    output_dir/param_nonparam/.

    Both param_raw_times and nonparam_raw_times follow the same schema:
        {dim_str: {n_samples_str: {rep_str: elapsed_sec}}}
    Mean and CI are computed from replicate times internally.
    """
    out = os.path.join(output_dir, "param_nonparam")
    os.makedirs(out, exist_ok=True)

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    alpha = 1.0 - ci_level

    def _compute_mean_ci(raw_times, dims, ms):
        time_mean = {d: [] for d in dims}
        time_ci = {d: [] for d in dims}
        for d in dims:
            for m in ms:
                rep_dict = raw_times.get(str(d), {}).get(str(m), {})
                arr = np.array(list(rep_dict.values()), dtype=float)
                n = len(arr)
                mean_val = float(np.mean(arr)) if n > 0 else 0.0
                if n > 1:
                    se = sem(arr)
                    t_crit = t.ppf(1.0 - alpha / 2.0, df=n - 1)
                    ci_half = float(se * t_crit)
                else:
                    ci_half = 0.0
                time_mean[d].append(mean_val)
                time_ci[d].append(ci_half)
        return time_mean, time_ci

    param_tm, param_tc = _compute_mean_ci(param_raw_times, dims, param_ms)
    nonparam_tm, nonparam_tc = _compute_mean_ci(nonparam_raw_times, dims, nonparam_ms)

    # Compute shared ylim across both datasets if not provided
    if ylim is None:
        ymin, ymax = np.inf, -np.inf
        for d in dims:
            for mean_arr, ci_arr in [
                (np.array(param_tm[d], dtype=float), np.array(param_tc[d], dtype=float)),
                (np.array(nonparam_tm[d], dtype=float), np.array(nonparam_tc[d], dtype=float)),
            ]:
                lower = mean_arr - ci_arr
                upper = mean_arr + ci_arr
                if logy:
                    lower_pos = lower[lower > 0]
                    if lower_pos.size > 0:
                        ymin = min(ymin, float(np.min(lower_pos)))
                else:
                    ymin = min(ymin, float(np.min(lower)))
                ymax = max(ymax, float(np.max(upper)))
        if np.isfinite(ymin) and np.isfinite(ymax):
            if logy:
                ylim = (ymin * 0.95, ymax * 1.05)
            else:
                yrange = ymax - ymin or max(abs(ymax), 1.0)
                ylim = (ymin - 0.05 * yrange, ymax + 0.05 * yrange)

    color = plot_cfg.plot.color_palette.colors[0]

    def _save_runtime_fig(ms, time_mean_per_dim, time_ci_per_dim, filename):
        fig, ax = _make_figure(plot_cfg)
        for d in dims:
            y = np.array(time_mean_per_dim[d], dtype=float)
            h = np.array(time_ci_per_dim[d], dtype=float)
            a_line = _alpha_for_dim(dims, d, alpha_min=0.3, alpha_max=1.0)
            line_c = _rgba_with_alpha(color, a_line)
            ls = _linestyle_for_dim(dims, d)
            ax.plot(ms, y, label=rf"$d_\Theta={d}$", color=line_c, linewidth=2.0, linestyle=ls)
        _apply_common_plot_style(
            ax,
            show_xlabel=True,
            show_ylabel=True,
            ylabel="Cost (sec.)",
            xlabel=r"$m$",
            logy=logy,
            ylim=ylim,
            xlim=xlim,
        )
        ax.legend()
        outpath = os.path.join(out, filename)
        fig.savefig(outpath, format="pdf")
        plt.close(fig)
        print(f"[Saved] {outpath}")

    _save_runtime_fig(param_ms, param_tm, param_tc, "comparison_runtime_param.pdf")
    _save_runtime_fig(nonparam_ms, nonparam_tm, nonparam_tc, "comparison_runtime_nonparam.pdf")


def _apply_common_plot_style(
    ax,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    ylabel: str = "",
    xlabel: str = r"$m$",
    logy: bool = False,
    ylim: Tuple[float, float] | None = None,
    xlim: Tuple[float, float] | None = None,
) -> None:
    """
    Apply a common axis style while preserving identical axes size across figures.
    """
    # Always reserve space for labels, but only show text when requested.
    ax.set_xlabel(xlabel if show_xlabel else " ")
    ax.set_ylabel(ylabel if show_ylabel else " ")

    # Hide tick labels without changing layout geometry.
    # ax.tick_params(axis="x", labelbottom=show_xlabel)
    # ax.tick_params(axis="y", labelleft=show_ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if logy:
        ax.set_yscale("log")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x // 1000)}k" if x >= 10000 else str(int(x)))
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_figure(plot_cfg):
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        )
    )

    fig.subplots_adjust(
        left=0.28,
        right=0.99,
        bottom=0.24,
        top=0.99,
    )
    return fig, ax


def plot_finite_sample_complexity_gaussians(
    output_dir: str,
    plot_cfg,
    ms: List[int],
    dims: List[int],
    n_rep: int = 30,
    seed: int = 0,
    logy: bool = False,
    results: Dict[str, Any] | None = None,
    divergence: str = None,
    ylim: Tuple[float, float] = None,
    xlim: Tuple[float, float] = None,
    show_ci: bool = False,
) -> Dict[str, Any]:
    """
    Plot finite-sample error curves and return the computed results.

    All figures are saved with identical axes size.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    if results is None:
        results = compute_gaussian_complexity_results(
            ms=ms,
            dims=dims,
            n_rep=n_rep,
            seed=seed,
        )

    error_mean = results["error_mean"]
    error_ci = results["error_ci"]

    def compute_global_ylim(logy: bool = False) -> Tuple[float, float]:
        ymin = np.inf
        ymax = -np.inf

        for method in ["fd", "mean", "wim", "kl"]:
            for d in dims:
                y = np.array(error_mean[method][d], dtype=float)
                h = np.array(error_ci[method][d], dtype=float)

                lower = y - h
                upper = y + h

                if logy:
                    lower_pos = lower[lower > 0]
                    if lower_pos.size > 0:
                        ymin = min(ymin, np.min(lower_pos))
                else:
                    ymin = min(ymin, np.min(lower))

                ymax = max(ymax, np.max(upper))

        if not np.isfinite(ymin) or not np.isfinite(ymax):
            raise ValueError("Could not determine global y-limits.")

        if logy:
            ymin *= 0.95
            ymax *= 1.05
        else:
            yrange = ymax - ymin
            if yrange <= 0:
                yrange = max(abs(ymax), 1.0)
            ymin -= 0.05 * yrange
            ymax += 0.05 * yrange

        return ymin, ymax

    ylim_global = ylim if ylim is not None else compute_global_ylim(logy=logy)

    all_plot_specs = [
        ("fd",   r"$\left|\rho(\tilde{\Pi}^\lambda)-\hat{\rho}_m(\tilde{\Pi}^\lambda)\right|$",
         "comparison_measure_sample_complexity_fd.pdf",   True, True),
        ("mean", r"$\left\|\rho^{\mathrm{mean}}-\hat{\rho}^{\mathrm{mean}}_m\right\|_2$",
         "comparison_measure_sample_complexity_mean.pdf", False, False),
        ("wim",  r"$\left|\rho(\tilde{\Pi}^\lambda)-\hat{\rho}_m(\tilde{\Pi}^\lambda)\right|$",
         "comparison_measure_sample_complexity_wim.pdf",  False,  False),
        ("kl",   r"$\left|\rho^{\mathrm{KL}}-\hat{\rho}^{\mathrm{KL}}_m\right|$",
         "comparison_measure_sample_complexity_kl.pdf",   False, False),
    ]
    plot_specs = [(m, yl, f, xe, ye) for m, yl, f, xe, ye in all_plot_specs
                  if divergence is None or m == divergence]

    def plot_one(
        method: str,
        ylabel: str,
        filename: str,
        ylim: Tuple[float, float],
        show_xlabel: bool,
        show_ylabel: bool,
    ) -> None:
        fig, ax = _make_figure(plot_cfg)

        base = _method_color(plot_cfg, method)

        for d in dims:
            y = np.array(error_mean[method][d], dtype=float)
            h = np.array(error_ci[method][d], dtype=float)

            a_line = _alpha_for_dim(dims, d, alpha_min=0.5, alpha_max=1.0)
            line_c = _rgba_with_alpha(base, a_line)
            fill_c = _rgba_with_alpha(base, max(0.10, 0.55 * a_line))
            ls = _linestyle_for_dim(dims, d)

            lower = y - h
            upper = y + h
            if logy:
                lower = np.maximum(lower, 1e-12)

            ax.plot(
                ms,
                y,
                label=rf"$d_\Theta={d}$",
                color=line_c,
                linewidth=2.0,
                linestyle=ls,
            )
            if show_ci:
                ax.fill_between(ms, lower, upper, color=fill_c, linewidth=0)

        _apply_common_plot_style(
            ax,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            ylabel=ylabel,
            xlabel=r"$m$",
            logy=logy,
            ylim=ylim,
            xlim=xlim,
        )

        if method == "wim":
            ax.legend(frameon=False, loc="upper right")

        outpath = os.path.join(output_dir, filename)
        fig.savefig(outpath, format="pdf")
        plt.close(fig)
        print(f"[Saved] {outpath}")

    for method, ylabel, filename, show_xlabel, show_ylabel in plot_specs:
        plot_one(method=method, ylabel=ylabel, filename=filename,
                 ylim=ylim_global, show_xlabel=show_xlabel, show_ylabel=show_ylabel)

    return results


def plot_inverse_wishart_scale_ellipses_by_fd_one_subplot(results, output_dir, plot_cfg):
    """
    Plots 2D ellipses representing inverse Wishart scale matrices,
    colored by KSD value. Highlights the max-FD distribution in red.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort by KSD (ascending)
    sorted_results = sorted(results, key=lambda x: x[2])
    ksds = [ksd for (_, _, ksd) in sorted_results]
    min_ksd, max_ksd = min(ksds), max(ksds)

    # Normalize KSD values for colormap
    norm = Normalize(vmin=min_ksd, vmax=max_ksd)
    color_list = plot_cfg.plot.color_palette.colors
    cmap = LinearSegmentedColormap.from_list("ksd_cmap", color_list)

    # Prepare figure and style
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fig, ax = plt.subplots(figsize=(plot_cfg.plot.figure.size.width,
                                    plot_cfg.plot.figure.size.height))

    # Parameters for sampling
    step_middle = 5  # Pick every 4th from the middle

    # Slice sorted results
    subset_results = sorted_results[::step_middle] + [sorted_results[-1]]
    n_ellipses = len(subset_results)

    # Identify max-KSD distribution
    max_ksd_entry = sorted_results[-1]
    max_ksd_scale = max_ksd_entry[0]["scale"]

    # Helper to plot ellipses
    def plot_cov_ellipse(cov, pos, ax, nstd=1.5, **kwargs):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellip)

    # Layout for ellipses
    x_spacing = 3.0
    for idx, (param_dict, _, ksd) in enumerate(subset_results):
        scale = param_dict["scale"]
        pos = (idx * x_spacing, 0.0)  # Position ellipses along x-axis

        is_max_ksd = np.allclose(scale, max_ksd_scale)
        if is_max_ksd:
            color = "red"
            lw = 1.5
            alpha = 1.0
        else:
            color = cmap(norm(ksd))
            lw = 1.0
            alpha = 0.7

        try:
            plot_cov_ellipse(scale, pos, ax, edgecolor=color, lw=lw, alpha=alpha, facecolor='none')
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=3)
        except np.linalg.LinAlgError:
            print(f"[WARN] Skipping non-PD matrix: {scale}")

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim(-1, (n_ellipses - 1) * x_spacing + x_spacing)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("")
    ax.set_xticks([])

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Shrink main plot height to make space for colorbar
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + 0.12, box.width, box.height * 0.85])

    # Add horizontal colorbar below the plot
    cbar_height = 0.012
    spacing = 0.09  # vertical spacing between colorbars
    start_y = box.y0 - 0.047  # starting y-position for the first colorbar

    tick_locs = np.linspace(min_ksd+0.01, max_ksd-0.01, 3)
    tick_label_dict = {
        0: ["17", "17.3", "17.7"],
        1: ["17.5", "18.1", "18.4"],
        2: ["18.15", "18.5", "19.2"]
    }
    text_labels = {
        0: "$\\nu_0=5$",
        1: "$\\nu_0=10$",
        2: "$\\nu_0=15$"
    }

    for i in range(3):
        y_pos = start_y - i * (cbar_height + spacing)
        cbar_ax = fig.add_axes([box.x0, y_pos + 0.02, box.width, cbar_height])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_label_dict[i])
        if i == 1:
            cbar.set_label(plot_cfg.plot.param_latex_names.estimatedFDposteriorsShort, labelpad=25)

        fig.text(
            box.x0 - 0.05,  # X position (left of colorbar)
            0.95 * y_pos + cbar_height / 2,  # Y position (vertical center of bar)
            text_labels[i],
            ha='right', va='center',
            fontsize=plot_cfg.plot.font.size
        )

    # Save
    output_path = os.path.join(output_dir, "toy_inverse_wishart.pdf")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


class _TextOnlyHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        return None


def plot_runtime_parametric_nonparametric_with_ci(
    times_parametric: dict[int, dict[int, float]],
    times_nonparametric: dict[int, dict[int, dict[int, float]]],
    plot_cfg: Any,
    output_dir: str,
    ci_level: float = 0.68,
    filename: str = "runtime_parametric_nonparametric.pdf",
) -> None:
    """
    Single-axis plot:
      - Parametric KSD runtime (mean ± CI across runs)
      - Non-parametric KSD runtime (one line per basis size, mean ± CI)

    Parameters
    ----------
    times_parametric : dict
        {samples_num: {run_iter: time, ...}, ...}

    times_nonparametric : dict
        {samples_num: {basis_funcs_num: {run_iter: time, ...}, ...}, ...}

    ci_level : float
        Confidence interval level (default=0.95).
    """
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

    # Matplotlib rc from config
    plt.rcParams.update({
        "font.size": _deep_get(plot_cfg, "plot.font.size", 12),
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
    x_label = names.get("numPriorPosteriorSamples")
    y_label = names.get("runtimeSeconds", "Time (sec.)")

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

    # # --- Process parametric
    # sample_sizes_param, means_param, cis_param = [], [], []
    # for s in sorted(times_parametric.keys(), key=_to_int):
    #     vals = list(times_parametric[s].values())
    #     m, h = mean_ci(vals)
    #     sample_sizes_param.append(int(int(s)/1000))
    #     means_param.append(m)
    #     cis_param.append(h)

    # --- Process non-parametric
    sample_sizes = []
    by_basis = defaultdict(lambda: ([], [], []))
    for s in sorted(times_nonparametric.keys(), key=_to_int):
        sample_sizes.append(int(int(s) / 1000))
        for b, runs in times_nonparametric[s].items():
            vals = list(runs.values())
            m, h = mean_ci(vals)
            samples, means, cis = by_basis[b]
            samples.append(s)
            means.append(m)
            cis.append(h)

    # --- Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
    handles_ordered, labels_ordered = [], []

    # # Parametric
    # if sample_sizes_param:
    #     h_param = ax.plot(
    #         sample_sizes_param, means_param,
    #         marker=marker, markersize=ms, linewidth=lw,
    #         color=palette[0], label="Parametric",
    #     )[0]
    #     ax.fill_between(
    #         sample_sizes_param,
    #         np.array(means_param) - np.array(cis_param),
    #         np.array(means_param) + np.array(cis_param),
    #         color=palette[0], alpha=0.7
    #     )
    #     handles_ordered.append(h_param)
    #     labels_ordered.append("Parametric")

    # Non-parametric
    for i, (b, (samples, means, cis)) in enumerate(by_basis.items(), start=1):
        h_np = ax.plot(
            sample_sizes, means,
            marker=marker, markersize=ms, linewidth=lw,
            color=palette[i % len(palette)], label=rf"{b}"
        )[0]
        ax.fill_between(
            sample_sizes,
            np.array(means) - np.array(cis),
            np.array(means) + np.array(cis),
            color=palette[i % len(palette)], alpha=0.4
        )
        handles_ordered.append(h_np)
        labels_ordered.append(rf"K={b}")

    # Axes styling
    ax.set_xlabel(r"$m+l$ (×10³)")
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    # ax.set_xticklabels([])
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    # spacer = Line2D([], [], linestyle="none", label="")
    # handles_ordered.append(spacer)
    # labels_ordered.append("")
    legend_fs = float(_deep_get(plot_cfg, "plot.legend.fontsize",
                                plt.rcParams["font.size"] * 0.8))
    leg = ax.legend(
        handles_ordered, labels_ordered,
        loc="center",
        fontsize=legend_fs,
        bbox_to_anchor=(1.2, 0.5),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        # borderpad=0.4, handlelength=1.2, handletextpad=0.4, labelspacing=0.24,
    )
    leg._legend_box.align = "left"

    if tight:
        fig.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_runtime_nonparametric_with_ci(
    times_nonparametric: dict[int, dict[int, float]],
    plot_cfg: Any,
    output_dir: str,
    ci_level: float = 0.95,
    filename: str = "runtime_parametric_nonparametric.pdf",
) -> None:
    """
    Parameters
    ----------
    times_nonparametric : dict
        {samples_num: {run_iter: time, ...}, ...}

    ci_level : float
        Confidence interval level (default=0.95).
    """
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
        "font.size": _deep_get(plot_cfg, "plot.font.size", 12),
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
    tight = bool(_deep_get(plot_cfg, "plot.figure.tight_layout", True))

    names = _deep_get(plot_cfg, "plot.param_latex_names", {}) or {}
    x_label = names.get("K")
    y_label = names.get("runtimeSeconds", "Time (sec.)")

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

    # --- Process parametric
    basis_funcs_nums, means_param, cis_param = [], [], []
    for k in sorted(times_nonparametric.keys(), key=_to_int):
        vals = list(times_nonparametric[k].values())
        m, h = mean_ci(vals)
        basis_funcs_nums.append(k)
        means_param.append(m)
        cis_param.append(h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
    ax.plot(basis_funcs_nums, means_param,
            marker=marker, markersize=ms, linewidth=lw,
            color="#6C936C")
    ax.fill_between(
        basis_funcs_nums,
        np.array(means_param) - np.array(cis_param),
        np.array(means_param) + np.array(cis_param),
        color="#6C936C",
    )

    # Axes styling
    ax.set_xlabel(x_label)
    ax.set_xticks(basis_funcs_nums)
    # ax.set_xticklabels(basis_funcs_nums)
    ax.set_xticklabels([str(v) if i % 2 == 0 else ""
                        for i, v in enumerate(basis_funcs_nums)])
    ax.set_ylabel(y_label)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4), useMathText=True)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if tight:
        fig.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_runtime_nonparametric_diff_samples_num_with_ci(
    times_nonparametric: dict[int, dict[int, dict[int, float]]],
    plot_cfg: Any,
    output_dir: str,
    ci_level: float = 0.95,
    filename: str = "runtime_nonparametric_diff_samples_nums.pdf",
) -> None:
    """
    Single-axis plot of nonparametric optimisation runtime against the number of
    posterior/prior samples m, with one line per number of basis functions k.

    Parameters
    ----------
    times_nonparametric : dict
        {samples_num: {basis_funcs_num: {run_iter: time, ...}, ...}, ...}

    ci_level : float
        Confidence interval level (default=0.95).
    """
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
        "font.size": _deep_get(plot_cfg, "plot.font.size", 12),
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
    x_label = names.get("numPosteriorSamples", r"$m$")
    y_label = names.get("runtimeSeconds", "Time (sec.)")

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

    # --- Process: one series of (m, mean, ci) per basis size k
    samples_nums = sorted(times_nonparametric.keys(), key=_to_int)
    by_basis = defaultdict(lambda: ([], [], []))
    for m in samples_nums:
        for k, runs in times_nonparametric[m].items():
            vals = list(runs.values())
            mean, h = mean_ci(vals)
            xs, means, cis = by_basis[k]
            xs.append(int(m))
            means.append(mean)
            cis.append(h)

    # --- Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
    handles_ordered, labels_ordered = [], []

    for i, k in enumerate(sorted(by_basis.keys(), key=_to_int)):
        xs, means, cis = by_basis[k]
        color = palette[i % len(palette)]
        h_line = ax.plot(
            xs, means,
            marker=marker, markersize=ms, linewidth=lw,
            color=color, label=rf"K={k}",
        )[0]
        ax.fill_between(
            xs,
            np.array(means) - np.array(cis),
            np.array(means) + np.array(cis),
            color=color, alpha=0.4,
        )
        handles_ordered.append(h_line)
        labels_ordered.append(rf"K={k}")

    # Axes styling
    ax.set_xlabel(x_label)
    ax.set_xticks([int(m) for m in samples_nums])
    ax.set_xticklabels([int(m) for m in samples_nums])
    ax.set_ylabel(y_label)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4), useMathText=True)
    ax.grid(True, alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_fs = float(_deep_get(plot_cfg, "plot.legend.fontsize",
                                plt.rcParams["font.size"] * 0.8))
    ax.legend(
        handles_ordered, labels_ordered,
        loc="best",
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


def _save_fig(fig, output_dir: str, filename: str, plot_cfg):
    os.makedirs(output_dir, exist_ok=True)
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)


def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{type1cm}",
    })


def plot_gaussian_copula_grid(
    copula_grid,
    plot_cfg,
    output_dir: str,
    prefix: str,
    filename: str | None = None,
    xlabel: str = r"$\lambda_c$",
    ylabel: str = r"$\hat{\rho}_m^{\mathrm{FD}}(\tilde{\Pi}^{\lambda})$",
    mark_argmax: bool = True,
    logy: bool = False,
):
    """
    Plot the empirical FD objective over a 1D Gaussian-copula grid.

    Parameters
    ----------
    copula_grid :
        Iterable of pairs (lambda, value), e.g. output of
        evaluate_gaussian_copula_grid(...).
    plot_cfg :
        Plot configuration object used by _apply_plot_rc and _save_fig.
    output_dir : str
        Directory where the figure is saved.
    prefix : str
        Prefix used in the default filename.
    filename : str | None
        Optional filename. If None, a default name is used.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    mark_argmax : bool
        Whether to highlight the maximiser on the grid.
    """
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_gaussian_copula_fd_grid.pdf"

    if len(copula_grid) == 0:
        raise ValueError("copula_grid must contain at least one (lambda, value) pair.")

    lambdas = np.asarray([x[0] for x in copula_grid], dtype=float)
    values = np.asarray([x[1] for x in copula_grid], dtype=float)

    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    values = values[order]

    # palette from config (fallback to matplotlib)
    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(colors) < 2:
        reps = int(np.ceil(2 / max(len(colors), 1)))
        colors = (colors * reps)[:2] if len(colors) > 0 else ["C0", "C1"]

    line_color = colors[0]
    point_color = colors[1]

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(
        lambdas,
        values,
        linewidth=1.5,
        color=line_color,
        zorder=2,
    )

    if mark_argmax:
        idx_star = int(np.argmax(values))
        lambda_star = lambdas[idx_star]
        value_star = values[idx_star]

        ax.scatter(
            [lambda_star],
            [value_star],
            s=27,
            color="red",
            zorder=3,
            marker="*",
        )

        ax.axvline(
            lambda_star,
            linestyle=":",
            linewidth=1.0,
            color=point_color,
            alpha=0.8,
            zorder=1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    if logy:
        ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(handles, labels, frameon=False, ncol=1, loc="best")

    fig.tight_layout()

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


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
    """
    Plot two Gaussian-copula FD grids (e.g. idx_g0=0 and idx_g0=1) as two
    lines on a single axes.
    """
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_gaussian_copula_fd_grid.pdf"

    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = _make_figure(plot_cfg)

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
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_gaussian_copula_fd_decomposition(
    copula_grid,
    plot_cfg,
    output_dir: str,
    prefix: str,
    filename: str | None = None,
    xlabel: str = r"$\lambda_c$",
    label: str = r"$(G_0, \nu)$",
    logy: bool = False,
    ylim_fd=None,
    ylim_g=None,
):
    """
    Two-panel decomposition of the Gaussian copula FD sensitivity.

    Left panel : FD(λ) vs the structural approximation h(λ)·g(0),
                 where h(λ) = λ²/(1−λ²)² and g(0) is the score
                 factor evaluated at the data term's λ→0 limit.
    Right panel: Normalised score factor g(λ)/g(0), showing how much
                 the data term itself varies with λ relative to baseline.

    A flat right panel (g/g0 ≈ 1) means all growth comes from h(λ);
    a rising right panel means the posterior tails amplify sensitivity
    on top of the structural factor.
    """
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_copula_fd_decomposition.pdf"

    lambdas = np.asarray([x[0] for x in copula_grid], dtype=float)
    values = np.asarray([x[1] for x in copula_grid], dtype=float)
    order = np.argsort(lambdas)
    lambdas, values = lambdas[order], values[order]

    # Structural factor h(λ) = λ² / (1 − λ²)²
    lam2 = lambdas ** 2
    h = lam2 / (1.0 - lam2) ** 2

    # Score factor g(λ) = FD(λ) / h(λ), undefined at λ = 0
    valid = np.abs(lambdas) > 1e-6
    g = np.full_like(values, np.nan)
    g[valid] = values[valid] / h[valid]

    # g(0) approximation: g evaluated at the smallest valid |λ|
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return
    g0_idx = valid_idx[np.argmin(np.abs(lambdas[valid_idx]))]
    g0 = float(g[g0_idx])

    fd_structural = h * g0  # what FD would be if g were constant at g(0)
    g_norm = g / g0          # relative change of the score factor

    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    while len(colors) < 3:
        colors = colors * 2

    fig_w = plot_cfg.plot.figure.size.width
    fig_h = plot_cfg.plot.figure.size.height

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(fig_w * 2.0, fig_h),
        dpi=plot_cfg.plot.figure.dpi,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.95, wspace=0.38)

    # ── Left panel: FD(λ) and h(λ)·g(0) ──────────────────────────────────
    ax_l.plot(lambdas, values, linewidth=1.5, color=colors[0],
              label=r"$\mathrm{FD}(\lambda)$", zorder=3)
    ax_l.plot(lambdas, fd_structural, linewidth=1.5, color=colors[1],
              linestyle="--", label=r"$h(\lambda)\,g(0)$", zorder=2)
    ax_l.set_xlabel(xlabel)
    ax_l.set_ylabel(r"$\hat{\rho}_m^{\mathrm{FD}}$")
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.grid(axis="y", linestyle=":", alpha=0.35)
    ax_l.legend(title=label, frameon=False, loc="best")
    if logy:
        ax_l.set_yscale("log")
    if ylim_fd is not None:
        ax_l.set_ylim(ylim_fd)

    # ── Right panel: g(λ)/g(0) ────────────────────────────────────────────
    ax_r.plot(lambdas[valid], g_norm[valid], linewidth=1.5, color=colors[2],
              zorder=3)
    ax_r.axhline(1.0, linewidth=0.8, color="gray", linestyle="--", alpha=0.7,
                 zorder=1)
    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel(r"$g(\lambda)\,/\,g(0)$")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.grid(axis="y", linestyle=":", alpha=0.35)
    if ylim_g is not None:
        ax_r.set_ylim(ylim_g)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)


def plot_copula_all_pairs(
    grids_and_labels,
    plot_cfg,
    output_dir: str,
    filename: str,
    xlabel: str = r"$\theta$",
    ylabel: str = r"$\hat{\rho}_m^{\mathrm{FD}}(\tilde{\Pi}^{\theta})$",
    mark_max_point: bool = True,
    logy: bool = False,
    ylim=None,
):
    """
    Plot one FD-vs-parameter curve per pair on a single axes.

    Parameters
    ----------
    grids_and_labels : list of (grid, label)
        Each grid is a list of (theta, FD) pairs; label is the legend string.
    """
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = _make_figure(plot_cfg)

    for i, (grid, label) in enumerate(grids_and_labels):
        color = colors[i % len(colors)]
        lambdas = np.asarray([x[0] for x in grid], dtype=float)
        values = np.asarray([x[1] for x in grid], dtype=float)
        order = np.argsort(lambdas)
        lambdas, values = lambdas[order], values[order]

        ax.plot(lambdas, values, linewidth=1.5, color=color, label=label, zorder=2)

        if mark_max_point:
            idx_star = int(np.argmax(values))
            ax.scatter(
                [lambdas[idx_star]], [values[idx_star]],
                s=27, color="red", zorder=3, marker="*",
            )
            ax.axvline(
                lambdas[idx_star],
                linestyle=":", linewidth=1.0, color=color, alpha=0.8, zorder=1,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    if logy:
        ax.set_yscale("log")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(frameon=False, ncol=1, loc="best")

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_prior_neighbourhood_comparison(
    optimizer,
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-8, 8),
    resolution: int = 500,
    epsilon: float = 0.2,
    n_nonparam_samples: int = 40,
    seed: int = 42,
    mu_range: tuple = None,
    sigma_range: tuple = None,
    n_param_grid: int = 12,
) -> None:
    """
    Four separate figures comparing the flexibility of prior neighbourhoods.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    names = plot_cfg.plot.param_latex_names
    fd_label = names.get("estimatedSensitivityMeasure")
    xlabel = names.get("theta")
    ylabel = names.get("nonparametric_prior", "Density")
    xlabel = names.get("theta", r"$\theta$")
    ref_label = names.get("baseprior", r"$\Pi_\mathrm{ref}$")

    palette = ["#005500", "#007200", "#9EB8A0", "#ADEBC8"]
    col_band = palette[2]
    col_member = palette[1]

    # shared grid
    x = np.linspace(domain[0], domain[1], resolution)   # (R,)
    dx = float(x[1] - x[0])
    x_col = x[:, None]                                   # (R, 1) for basis eval

    pi_ref = prior_distribution.pdf(x_col).flatten()    # (R,)
    log_pi_ref = np.log(np.maximum(pi_ref, 1e-300))

    def _make_fig():
        return plt.subplots(
            figsize=(plot_cfg.plot.figure.size.width,
                     plot_cfg.plot.figure.size.height),
            dpi=plot_cfg.plot.figure.dpi,
        )

    def _finish(fig, ax, filename):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            handles=[Line2D([0], [0], color="black", linestyle="--", linewidth=1.5)],
            labels=[ref_label],
            frameon=False,
            fontsize=plt.rcParams["font.size"] * 0.85,
        )
        if getattr(plot_cfg.plot.figure, "tight_layout", False):
            fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # ------------------------------------------------------------------
    # Figure a: nonparametric (basis-function) neighbourhood
    # ------------------------------------------------------------------
    Phi_x = optimizer.basis_function.evaluate(x_col)   # (R, 1, K) for d=1

    rng = np.random.default_rng(seed)
    np_densities = []
    for _ in range(n_nonparam_samples):
        v = rng.standard_normal(optimizer.d)
        v_Ac_v = float(v @ optimizer.A_c @ v)
        if v_Ac_v < 1e-14:
            continue
        alpha = np.sqrt(max(optimizer.r - optimizer.c_c, 0.0) / v_Ac_v)
        lam = alpha * v
        log_p = (Phi_x @ lam).flatten() + log_pi_ref   # (R,)
        log_Z = float(logsumexp(log_p) + np.log(dx))
        np_densities.append(np.exp(log_p - log_Z))

    np_arr = np.array(np_densities)   # (n, R)
    lo_a, hi_a = np_arr.min(axis=0), np_arr.max(axis=0)

    fig_a, ax_a = _make_fig()
    ax_a.fill_between(x, lo_a, hi_a, color=col_band, alpha=0.3)
    for p in np_arr[:: max(1, len(np_densities) // 8)]:
        ax_a.plot(x, p, color=col_member, alpha=0.3, linewidth=0.7)
    ax_a.plot(x, pi_ref, color="black", linestyle="--", linewidth=1.5)
    _finish(fig_a, ax_a, "neighbourhood_nonparametric.pdf")

    # ------------------------------------------------------------------
    # Figure b: Huber epsilon-contamination (unimodal)
    # ------------------------------------------------------------------
    domain_mid = 0.5 * (domain[0] + domain[1])
    domain_half = 0.5 * (domain[1] - domain[0])

    contaminations = []
    # shifts across domain
    for mu_c in np.linspace(domain[0] + 0.15 * domain_half,
                            domain[1] - 0.15 * domain_half, 10):
        sig_c = domain_half * 0.12
        contaminations.append(
            np.exp(-0.5 * ((x - mu_c) / sig_c) ** 2) / (sig_c * np.sqrt(2 * np.pi))
        )
    # varying widths at the centre
    for w_frac in [0.06, 0.18, 0.40]:
        sig_c = domain_half * w_frac
        contaminations.append(
            np.exp(-0.5 * ((x - domain_mid) / sig_c) ** 2) / (sig_c * np.sqrt(2 * np.pi))
        )

    huber_list = []
    for q in contaminations:
        Z = float(np.trapz(q, x))
        if Z > 0:
            huber_list.append(q / Z)

    huber_arr = np.array(huber_list)
    lo_b, hi_b = huber_arr.min(axis=0), huber_arr.max(axis=0)

    fig_b, ax_b = _make_fig()
    ax_b.fill_between(x, lo_b, hi_b, color=col_band, alpha=0.3)
    for p in huber_arr[:: max(1, len(huber_list) // 8)]:
        ax_b.plot(x, p, color=col_member, alpha=0.3, linewidth=0.7)
    ax_b.plot(x, pi_ref, color="black", linestyle="--", linewidth=1.5)
    _finish(fig_b, ax_b, "neighbourhood_huber.pdf")

    # ------------------------------------------------------------------
    # Figure c: density-band neighbourhood
    # ------------------------------------------------------------------
    x_std = (x - domain_mid) / domain_half   # approx [-1, 1]

    def _band_member(w_raw):
        w = np.clip(w_raw, 1.0 - epsilon, 1.0 + epsilon)
        p = pi_ref * w
        Z = float(np.trapz(p, x))
        return p / Z if Z > 0 else pi_ref.copy()

    band_members = [
        _band_member(np.full_like(x, 1.0 + epsilon)),          # uniform inflate
        _band_member(np.full_like(x, 1.0 - epsilon)),          # uniform deflate
        _band_member(1.0 + epsilon * x_std),                    # right tilt
        _band_member(1.0 - epsilon * x_std),                    # left tilt
        _band_member(1.0 + epsilon * (2.0 * x_std**2 - 1.0)),  # bowl weight
        _band_member(1.0 - epsilon * (2.0 * x_std**2 - 1.0)),  # hat weight
        _band_member(1.0 + epsilon * np.sin(np.pi * x_std)),   # sine modulation
        _band_member(1.0 - epsilon * np.sin(np.pi * x_std)),   # sine modulation
    ]

    band_arr = np.array(band_members)
    lo_c, hi_c = band_arr.min(axis=0), band_arr.max(axis=0)

    fig_c, ax_c = _make_fig()
    ax_c.fill_between(x, lo_c, hi_c, color=col_band, alpha=0.3)
    for p in band_arr[::2]:
        ax_c.plot(x, p, color=col_member, alpha=0.3, linewidth=0.7)
    ax_c.plot(x, pi_ref, color="black", linestyle="--", linewidth=1.5)
    _finish(fig_c, ax_c, "neighbourhood_density_band.pdf")

    # ------------------------------------------------------------------
    # Figure d: parametric Gaussian neighbourhood
    # ------------------------------------------------------------------
    if mu_range is None:
        mu_range = (domain_mid - 0.4 * domain_half, domain_mid + 0.4 * domain_half)
    if sigma_range is None:
        sigma_range = (domain_half * 0.1, domain_half * 0.5)

    mu_vals = np.linspace(mu_range[0], mu_range[1], n_param_grid)
    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], n_param_grid)

    fig_d, ax_d = _make_fig()
    for mu_p in mu_vals:
        for sigma_p in sigma_vals:
            p = (np.exp(-0.5 * ((x - mu_p) / sigma_p) ** 2)
                 / (sigma_p * np.sqrt(2 * np.pi)))
            ax_d.plot(x, p, color=col_member, alpha=0.2, linewidth=0.7)
    ax_d.plot(x, pi_ref, color="black", linestyle="--", linewidth=1.5)
    _finish(fig_d, ax_d, "neighbourhood_parametric.pdf")


def plot_param_nonparam_skewness_comparison(
    worst_corner: dict,
    lambda_star: np.ndarray,
    basis_function,
    model,
    plot_cfg,
    output_dir: str = "outputs/param_nonparam",
    domain: tuple = (-15, 15),
    resolution: int = 400,
) -> None:
    """
    Compare worst-case parametric and nonparametric candidate priors found by optimisation.

    Left panel : reference prior + two worst-case candidate priors.
    Right panel : corresponding posteriors with posterior skewness gamma_1 in legend.

    Args:
        worst_corner: dict with keys 'mu', 'sigma' from parametric optimisation.
        lambda_star: coefficient vector from nonparametric SDP optimisation.
        basis_function: fitted basis function object (e.g. MaternBasisFunction).
        model: fitted Bayesian model providing likelihood and prior parameters.
        plot_cfg: plot configuration loaded from overleaf_plots_settings.yaml.
        output_dir: directory to save the PDF.
        domain: (lo, hi) range for the theta grid.
        resolution: number of grid points.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
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

    # Likelihood parameters
    x_bar = float(np.squeeze(model.x_bar))
    n = int(model.observations_num)
    sigma_lik_sq = float(model.loss.var)
    loss_lr = float(model.loss_lr_init)
    log_lik = -0.5 * loss_lr * n * (theta - x_bar) ** 2 / sigma_lik_sq

    def _posterior_from_log_unnorm(log_prior_un):
        log_post = log_prior_un + log_lik
        log_Z = logsumexp(log_post) + np.log(dx)
        return np.exp(log_post - log_Z)

    def _skewness(pdf):
        mu = np.sum(theta * pdf) * dx
        sigma = np.sqrt(np.sum((theta - mu) ** 2 * pdf) * dx)
        return np.sum(((theta - mu) / sigma) ** 3 * pdf) * dx

    # Reference posterior (conjugate)
    mu_0 = float(np.squeeze(prior_distribution.mu))
    sigma_0_sq = float(np.squeeze(prior_distribution.var))
    sigma_n_sq = 1.0 / (loss_lr * n / sigma_lik_sq + 1.0 / sigma_0_sq)
    mu_n = sigma_n_sq * (loss_lr * n * x_bar / sigma_lik_sq + mu_0 / sigma_0_sq)
    post_ref = np.exp(-0.5 * (theta - mu_n) ** 2 / sigma_n_sq) / np.sqrt(2 * np.pi * sigma_n_sq)

    # Parametric worst-case candidate
    mu_a, sigma_a = float(worst_corner["mu"]), float(worst_corner["sigma"])
    log_pi_a = -0.5 * ((theta - mu_a) / sigma_a) ** 2 - np.log(sigma_a * np.sqrt(2 * np.pi))
    pi_a = np.exp(log_pi_a)
    post_a = _posterior_from_log_unnorm(log_pi_a)
    skew_a = _skewness(post_a)

    # Nonparametric worst-case candidate
    Phi_x = basis_function.evaluate(x)
    f = (Phi_x @ lambda_star).flatten()
    log_pi_b_un = f + log_g
    log_pi_b = log_pi_b_un - (logsumexp(log_pi_b_un) + np.log(dx))
    pi_b = np.exp(log_pi_b)
    print(f"Nonparametric prior integral (should be 1.0): {np.sum(pi_b) * dx:.8f}")
    post_b = _posterior_from_log_unnorm(log_pi_b_un)
    skew_b = _skewness(post_b)

    # Colors & labels
    palette = ["#007200", "#ADEBC8"]
    col_ref = "black"
    col_a = palette[0]
    col_b = palette[1]

    names = plot_cfg.plot.param_latex_names
    xlabel = names.get("theta", r"$\theta$")
    ylabel_d = names.get("nonparametric_prior", "Density")

    fig_w = plot_cfg.plot.figure.size.width
    fig_h = plot_cfg.plot.figure.size.height
    fs = plot_cfg.plot.font.size

    fig, (ax_prior, ax_post) = plt.subplots(
        1, 2,
        figsize=(fig_w * 2, fig_h),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Left: priors
    ax_prior.plot(theta, g, color=col_ref, linestyle="--", linewidth=1.5, label=r"$\mathrm{Ref.}$")
    ax_prior.plot(theta, pi_a, color=col_a, linewidth=1.5, label=r"$\mathrm{Param.}$")
    ax_prior.plot(theta, pi_b, color=col_b, linewidth=1.5, label=r"$\mathrm{Ours}$")
    ax_prior.set_xlabel(xlabel)
    ax_prior.set_ylabel(r"$\mathrm{Prior}$")
    ax_prior.set_ylim(bottom=0)
    ax_prior.grid(True, alpha=0.3)
    ax_prior.spines["top"].set_visible(False)
    ax_prior.spines["right"].set_visible(False)
    ax_prior.legend(frameon=False, fontsize=fs * 0.9, labelspacing=0.4, handlelength=1.8, handletextpad=0.5)

    # Right: posteriors + skewness in legend
    label_a = rf"$\mathrm{{Skewness}}={0.0:.2f}$"
    label_b = rf"$\mathrm{{Skewness}}={skew_b:.2f}$"

    ax_post.plot(theta, post_ref, color=col_ref, linestyle="--", linewidth=1.5)
    ax_post.plot(theta, post_a, color=col_a, linewidth=1.5, label=label_a)
    ax_post.plot(theta, post_b, color=col_b, linewidth=1.5, label=label_b)
    ax_post.set_xlabel(xlabel)
    ax_post.set_ylabel(r"$\mathrm{Posterior}$")
    ax_post.set_ylim(bottom=0)
    ax_post.grid(True, alpha=0.3)
    ax_post.spines["top"].set_visible(False)
    ax_post.spines["right"].set_visible(False)
    ax_post.legend(frameon=False, fontsize=fs * 0.75, labelspacing=0.5, handlelength=1.8, handletextpad=0.5)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        fig.tight_layout()

    save_path = os.path.join(output_dir, "gaussian_1d_location_model_param_nonparam_skewness_comparison.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_param_nonparam_kurtosis_comparison(
    worst_corner: dict,
    lambda_star: np.ndarray,
    basis_function,
    model,
    plot_cfg,
    output_dir: str = "outputs/param_nonparam",
    domain: tuple = (-20, 20),
    resolution: int = 400,
    y_log: bool = False,
    legend: bool = False,
) -> None:
    """
    Compare worst-case parametric and nonparametric candidate priors. With
    y_log=True, the density axis is log-scaled so tail-weight differences
    (invisible on a linear density scale) stand out.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    theta = np.linspace(domain[0], domain[1], resolution)
    x = theta[:, None]
    dx = float(theta[1] - theta[0])
    floor = 1e-45

    # Reference prior
    prior_distribution = model.prior_init
    log_g = prior_distribution.log_pdf(x).flatten()
    g = np.exp(log_g)

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

    def _excess_kurtosis(pdf: np.ndarray) -> float:
        mu = np.sum(theta * pdf) * dx
        sigma = np.sqrt(np.sum((theta - mu) ** 2 * pdf) * dx)
        return float(np.sum(((theta - mu) / sigma) ** 4 * pdf) * dx - 3.0)

    # pi_a is an exact Gaussian by construction, whose excess kurtosis is
    # exactly 0 analytically; computing it via truncated-domain quadrature
    # is numerically unstable (4th moment is tail-sensitive, and the fixed
    # domain isn't guaranteed to cover +-few sigma_a around mu_a), so it's
    # hardcoded rather than estimated.
    kurt_a = 0.0
    kurt_b = _excess_kurtosis(pi_b)

    # Colors & labels
    palette = ["#007200", "#ADEBC8"]
    col_ref = "steelblue"
    col_a = palette[0]
    col_b = palette[1]

    names = plot_cfg.plot.param_latex_names
    xlabel = names.get("theta", r"$\theta$")
    ref_label = names.get("baseprior", r"$\mathrm{Ref.}$")
    nonparam_label = names.get("candidatepriornonparam", r"$\mathrm{Ours}$")

    fig_w = plot_cfg.plot.figure.size.width
    fig_h = plot_cfg.plot.figure.size.height
    fs = plot_cfg.plot.font.size

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plot_cfg.plot.figure.dpi)

    label_a = rf"$\mathrm{{Param.}}$ (kurt.={kurt_a:.2f})"
    label_b = rf"{nonparam_label} (kurt.={kurt_b:.2f})"

    ax.plot(theta, np.maximum(g, floor), color=col_ref, linestyle="--", linewidth=1.5, label=ref_label)
    ax.plot(theta, np.maximum(pi_a, floor), color=col_a, linewidth=1.5, label=label_a)
    ax.plot(theta, np.maximum(pi_b, floor), color=col_b, linewidth=1.5, label=label_b)
    if y_log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\pi$")
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=fs * 0.9, labelspacing=0.4, handlelength=1.8, handletextpad=0.5)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        fig.tight_layout()

    save_path = os.path.join(output_dir, "gaussian_1d_location_model_param_nonparam_kurtosis_comparison.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_param_nonparam_multimodality_comparison(
    worst_corner: dict,
    lambda_star: np.ndarray,
    basis_function,
    model,
    plot_cfg,
    output_dir: str = "outputs/param_nonparam",
    domain: tuple = (-10, 6),
    resolution: int = 400,
    legend: bool = False,
) -> None:
    """
    Compare worst-case parametric and nonparametric candidate priors, reporting
    the number of density modes (local maxima) in the legend. The Gaussian
    parametric family is unimodal by construction; the nonparametric
    basis-function tilt is not restricted to unimodal shapes.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
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

    def _count_modes(pdf: np.ndarray, rel_prominence: float = 0.05) -> int:
        """Count local maxima whose height is at least rel_prominence * max(pdf)."""
        is_local_max = (pdf[1:-1] > pdf[:-2]) & (pdf[1:-1] > pdf[2:])
        peak_idx = np.nonzero(is_local_max)[0] + 1
        if peak_idx.size == 0:
            return 1
        threshold = rel_prominence * pdf.max()
        return int(np.sum(pdf[peak_idx] >= threshold))

    modes_a = _count_modes(pi_a)
    modes_b = _count_modes(pi_b)

    # Colors & labels
    palette = ["#007200", "#ADEBC8"]
    col_ref = "steelblue"
    col_a = palette[0]
    col_b = palette[1]

    names = plot_cfg.plot.param_latex_names
    xlabel = names.get("theta", r"$\theta$")
    ref_label = names.get("baseprior", r"$\mathrm{Ref.}$")
    nonparam_label = names.get("candidatepriornonparam", r"$\mathrm{Ours}$")

    fig_w = plot_cfg.plot.figure.size.width
    fig_h = plot_cfg.plot.figure.size.height
    fs = plot_cfg.plot.font.size

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plot_cfg.plot.figure.dpi)

    label_a = rf"$\mathrm{{Param.}}$ (modes: {modes_a})"
    label_b = rf"{nonparam_label} (modes: {modes_b})"

    ax.plot(theta, g, color=col_ref, linestyle="--", linewidth=1.5, label=ref_label)
    ax.plot(theta, pi_a, color=col_a, linewidth=1.5, label=label_a)
    ax.plot(theta, pi_b, color=col_b, linewidth=1.5, label=label_b)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\pi$")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if legend:
        ax.legend(frameon=False, fontsize=fs * 0.9, labelspacing=0.4, handlelength=1.8, handletextpad=0.5)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        fig.tight_layout()

    save_path = os.path.join(output_dir, "gaussian_1d_location_model_param_nonparam_multimodality_comparison.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
