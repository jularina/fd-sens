from collections import defaultdict
from typing import List, Tuple, Dict, FrozenSet, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from matplotlib.colors import to_rgb, Normalize, ListedColormap, BoundaryNorm
import matplotlib.cm as cmx
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.cm import ScalarMappable
from scipy.special import logsumexp
import os
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from scipy.stats import sem, t

from src.distributions.gaussian import Gaussian


def plot_ksd_eta_surface(
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

    # ---------- helpers ----------
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

    # ---------- colormap & surface ----------
    palette_colors = plot_cfg.plot.color_palette.colors[::-1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", palette_colors)

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(x, y, z, cmap=cmap, edgecolor="none", linewidth=0.2, antialiased=True)

    ax.set_xlabel(r"$\gamma_0=\frac{\mu_0}{\sigma_0^2}$")
    ax.set_ylabel(r"$\gamma_1=\frac{-0.5}{\sigma_0^2}$", fontsize=y_label_fs)  # <-- larger y-label

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
        z_lift = 0.02 * (ax.get_zlim()[1] - ax.get_zlim()[0])
        ax.text(cx, cy, cz + z_lift, f"{idx}",
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
    N = max(1, len(legend_lines))
    height = min(0.60, 0.05 * N + 0.10)
    bottom = 0.50 - height / 2.0
    legend_ax = fig.add_axes([0.77, bottom, 0.21, height])
    legend_ax.axis("off")
    legend_ax.text(0.0, 1.02, "Corners", fontsize=legend_title_fs, fontweight="bold",
                   ha="left", va="bottom")
    ys = [0.5] if N == 1 else np.linspace(0.85, 0.10, N)
    for yv, line in zip(ys, legend_lines):
        legend_ax.text(0.0, float(yv), line, fontsize=legend_line_fs, ha="left", va="center")

    # ---------- ticks: exactly 3 on x, y, z ----------
    ax.xaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_locator(LinearLocator(4))

    # format tick labels with max 2 decimals (strip trailing zeros)
    def _fmt_two_decimals(x, pos):
        s = f"{x:.2f}".rstrip('0').rstrip('.')
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
    ksd_qf_latex = getattr(latex_param_names, "estimatedKSDposteriorsQuadraticForm", "KSD")
    zmin, zmax = ax.get_zlim()
    xmid = np.max(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    ax.text(xmid, ymin - 0.05, zmax - 0.03, ksd_qf_latex,
            rotation=90, fontsize=base_fs, va="bottom", ha="left")

    # save
    filename = "ksd_eta_surface_from_corners.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved 3D eta surface with straight path, smaller black corner points, 3 ticks/axis, and legend to: {save_path}")


def plot_ksd_single_param(
    ksd_results: Dict[float, float],
    param_name: str,
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names
    latex_param_name = plot_cfg.plot.param_latex_names.get(param_name, param_name)

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

    ax.plot(x_vals, y_vals, marker='.', color=plot_cfg.plot.color_palette.colors[0])

    ax.set_xlabel(latex_param_name)
    ksd_latex = latex_param_names.get("estimatedKSDposteriorsShort")
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

    filename = f"ksd_line_{param_name}.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig)

    print(f"Saved line plot to: {save_path}")


def plot_ksd_multi_line_plots(
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
        base_rgb = to_rgb(colors[0])  # Base color from palette
        base_hsv = colorsys.rgb_to_hsv(*base_rgb)

        # Compute average KSDs for each fixed value to sort and normalize for brightness
        avg_ksd_per_line = []
        for fixed_val in fixed_vals:
            mask = param_values[:, fixed_idx] == fixed_val
            y = ksd_values[mask]
            avg_ksd_per_line.append(np.mean(y))
        avg_ksd_per_line = np.array(avg_ksd_per_line)

        num_lines = len(fixed_vals)
        brightness_vals = np.linspace(0.2, 0.9, num_lines)

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

            # Adjust brightness of the base color
            shaded_rgb = colorsys.hsv_to_rgb(base_hsv[0], base_hsv[1], brightness_vals[i])

            ax.plot(
                x, y,
                marker='.',
                label=f"{fixed_param_latex} = {fixed_val:.0f}",
                color=shaded_rgb,
            )

            if getattr(plot_cfg.plot, "show_min_point", False) and fixed_param_latex == "$\\sigma_0$" and fixed_val == 2.0:
                min_idx = np.argmin(y)
                min_x = x[min_idx]
                min_y = y[min_idx]
                ax.scatter(
                    min_x, min_y,
                    color="black",
                    zorder=5,
                    marker='x',
                    s=50,
                )

            if getattr(plot_cfg.plot, "show_max_point", False) and fixed_param_latex == "$\\sigma_0$" and fixed_val == 2.0:
                max_idx = np.argmax(y)
                max_x = x[max_idx]
                max_y = y[max_idx]
                ax.scatter(
                    max_x, max_y,
                    color="red",
                    zorder=6,
                    marker='*',  # or '^', 's', etc.
                    s=50,
                )

        ax.set_xlabel(varying_param_latex)
        ksd_latex = latex_param_names.get("estimatedKSDposteriorsShort")
        ylabel = f"log {ksd_latex}" if plot_cfg.plot.y_axis.log_scale else ksd_latex
        ax.set_ylabel(ylabel)
        # title = f"{ksd_latex} vs {varying_param_latex}"
        # ax.set_title(title)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right')

        if plot_cfg.plot.y_axis.log_scale:
            ax.set_yscale("log")

        if plot_cfg.plot.figure.tight_layout:
            plt.tight_layout()

        save_path = os.path.join(output_dir, f"{filename_prefix}_{varying_param_name}_vs_{fixed_param_name}.pdf")
        fig.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close(fig)
        print(f"Saved combined KSD vs {varying_param_name} plot to: {save_path}")

    # Plot 1: Vary param1, lines for each fixed param2
    make_multi_line_plot(fixed_idx=1, varying_idx=0, filename_prefix="ksd_multiline")

    # Plot 2: Vary param2, lines for each fixed param1
    make_multi_line_plot(fixed_idx=0, varying_idx=1, filename_prefix="ksd_multiline")


def plot_gaussian_prior_densities_by_ksd(
    ksd_results: Dict[Tuple[float, ...], float],
    cfg: DictConfig,
    plot_cfg: DictConfig,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Base prior
    base_mu = cfg.data.base_prior.mu
    base_sigma = cfg.data.base_prior.sigma
    base_dist = Gaussian(mu=base_mu, sigma=base_sigma)

    # X-axis for PDF evaluation
    x = np.linspace(base_mu - 6 * base_sigma, base_mu + 6 * base_sigma, 300)

    # Apply plot config
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{type1cm}",
    })

    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Plot base prior
    ax.plot(
        x,
        base_dist.pdf(x),
        label="Base prior",
        color="black",
        linewidth=1,
        linestyle="--"
    )

    # KSD values and quantile bins
    ksd_vals = np.array(list(ksd_results.values()))
    sorted_keys = list(ksd_results.keys())
    quantiles = np.quantile(ksd_vals, [1/3, 2/3])

    # Use first 3 colors from color palette and reverse for KSD mapping (higher KSD → darker)
    palette_colors = plot_cfg.plot.color_palette.colors[:3]
    rgb_colors = [to_rgb(c) for c in palette_colors][::-1]  # reverse order

    # Assign color bins based on KSD
    color_bins = []
    for ksd in ksd_vals:
        if ksd <= quantiles[0]:
            color_bins.append(0)
        elif ksd <= quantiles[1]:
            color_bins.append(1)
        else:
            color_bins.append(2)

    # Plot densities with filled color
    for (key, bin_idx) in zip(sorted_keys, color_bins):
        mu_val, sigma_val = key
        dist = Gaussian(mu=mu_val, sigma=sigma_val)
        ax.fill_between(
            x,
            dist.pdf(x),
            color=rgb_colors[bin_idx],
            alpha=0.7,
            linewidth=0
        )

    # Axes labels and formatting
    ax.set_xlabel(plot_cfg.plot.param_latex_names.mu_0)
    ax.set_ylabel(plot_cfg.plot.param_latex_names.prior)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="upper right")

    # Create colorbar without ticks
    cmap = ListedColormap(rgb_colors)
    bounds = [0, 1, 2, 3]
    norm = BoundaryNorm(bounds, cmap.N)
    sm = cmx.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar with no ticks or tick lines
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.tick_params(
        size=0,
        width=0,
        length=0,
        labelsize=0,
        left=False,
        right=False
    )
    cbar.set_ticks([])  # Also ensure no major ticks
    cbar.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriorsShort)

    # Add subtle upward arrow inside the colorbar
    cbar.ax.annotate(
        '',
        xy=(1.4, 0.6),
        xytext=(1.4, 0.4),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            lw=1,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=8  # smaller arrow head
        ),
    )

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    output_path = os.path.join(output_dir, "prior_densities_by_ksd.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prior density plot with discrete color bar to: {output_path}")


def plot_prior_densities_by_ksd(
    all_ksd_data: Dict[str, Dict],
    cfg: DictConfig,
    plot_cfg: DictConfig,
    output_dir: str,
):
    """
    Plots all prior densities (from different distributions) in one figure,
    using 4 discrete color bins based on KSD quantiles and a colorbar legend.
    The colorbar shows a SINGLE representative KSD value (median within bin)
    for each color, not an interval. Lower-KSD densities are plotted last (on top).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
    from matplotlib.colors import to_rgb, ListedColormap, BoundaryNorm

    os.makedirs(output_dir, exist_ok=True)

    # Setup figure
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Plot base prior
    base_mu = cfg.data.base_prior.mu
    base_sigma = cfg.data.base_prior.sigma
    from src.distributions.gaussian import Gaussian
    base_dist = Gaussian(mu=base_mu, sigma=base_sigma)
    x = np.linspace(base_mu - 6 * base_sigma, base_mu + 6 * base_sigma, 300)
    ax.plot(
        x,
        base_dist.pdf(x),
        label="Base prior",
        color="black",
        linewidth=1,
        linestyle="--"
    )

    # Collect all KSD values
    all_ksd_values = []
    for dist_data in all_ksd_data.values():
        all_ksd_values.extend(dist_data["ksd"].values())
    all_ksd_values = np.array(all_ksd_values, dtype=float)

    # Compute bin edges: min, Q1, Q2, Q3, max
    q1, q2, q3 = np.quantile(all_ksd_values, [0.25, 0.5, 0.75])
    vmin, vmax = float(np.min(all_ksd_values)), float(np.max(all_ksd_values))
    edges = [vmin, q1, q2, q3, vmax]

    # Use first 4 colors from palette (reversed if desired)
    palette_colors = plot_cfg.plot.color_palette.colors[:4]
    rgb_colors = [to_rgb(c) for c in palette_colors[::-1]]

    # Helper for compact labels
    def _fmt(v):
        return f"{v:.3g}"  # 3 sig figs

    # Build a list of items to plot, then sort by KSD (descending) so lowest KSD is on top
    plotting_items = []
    for dist_name, dist_data in all_ksd_data.items():
        ksd_results = dist_data["ksd"]
        param_names = dist_data["param_names"]
        distribution_cls = dist_data["distribution_cls"]

        for param_tuple, ksd in ksd_results.items():
            # Determine discrete color bin via edges
            if ksd <= edges[1]:
                bin_idx = 0
            elif ksd <= edges[2]:
                bin_idx = 1
            elif ksd <= edges[3]:
                bin_idx = 2
            else:
                bin_idx = 3

            param_dict = dict(zip([p.replace("_0", "") for p in param_names], param_tuple))
            try:
                dist = distribution_cls(**param_dict)
                pdf_vals = dist.pdf(x)
                plotting_items.append((float(ksd), pdf_vals, bin_idx, dist_name, param_dict))
            except Exception as e:
                print(f"[WARN] Skipping {dist_name} with params {param_dict}: {e}")

    # Sort by KSD descending so we draw high KSD first, low KSD last (on top)
    plotting_items.sort(key=lambda t: t[0], reverse=True)

    # Plot densities with 4-bin color coding
    for ksd, pdf_vals, bin_idx, dist_name, param_dict in plotting_items:
        ax.fill_between(
            x,
            pdf_vals,
            color=rgb_colors[bin_idx],
            alpha=0.8,
            linewidth=0.7
        )

    # Axes formatting
    ax.set_xlabel(plot_cfg.plot.param_latex_names.mu_0)
    ax.set_ylabel(plot_cfg.plot.param_latex_names.prior)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create colorbar
    cmap = ListedColormap(rgb_colors)
    bounds = [0, 1, 2, 3, 4]
    norm = BoundaryNorm(bounds, cmap.N)
    sm = cmx.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Compute a single representative value per bin: median of KSDs in that bin
    bin_values = [[] for _ in range(4)]
    for ksd, *_ in plotting_items:
        if ksd <= edges[1]:
            bin_values[0].append(ksd)
        elif ksd <= edges[2]:
            bin_values[1].append(ksd)
        elif ksd <= edges[3]:
            bin_values[2].append(ksd)
        else:
            bin_values[3].append(ksd)

    single_labels = []
    for i in range(4):
        if len(bin_values[i]) > 0:
            val = round(float(np.median(bin_values[i])), 2)
        else:
            # Fallback if a bin is empty: midpoint of its edges
            val = 0.5 * (edges[i] + edges[i+1])
        single_labels.append(_fmt(val))

    cbar = fig.colorbar(sm, ax=ax, boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriorsShort)
    cbar.ax.tick_params(labelsize=plot_cfg.plot.font.size)
    cbar.set_ticklabels(single_labels)

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    output_path = os.path.join(output_dir, "prior_densities_by_ksd_ALL.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ksd_multi_line_plots_with_error_bands(
    ksd_results: Dict[Tuple[float, ...], List[float]],  # Updated to store lists of KSDs
    param_names: List[str],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names
    colors = plot_cfg.plot.color_palette.colors

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "figure.dpi": plot_cfg.plot.figure.dpi,
    })

    param_values = np.array(list(ksd_results.keys()))
    ksd_values = np.array(list(ksd_results.values()))

    if len(param_names) != 2:
        raise ValueError("This function currently supports exactly two parameters.")

    fixed_idx, varying_idx = 0, 1  # Fixed = observations_num, Varying = mu_0
    fixed_param_name = param_names[fixed_idx]
    varying_param_name = param_names[varying_idx]
    fixed_param_latex = latex_param_names.get(fixed_param_name, fixed_param_name)
    varying_param_latex = latex_param_names.get(varying_param_name, varying_param_name)

    fixed_vals = np.unique(param_values[:, fixed_idx])
    num_colors = len(colors)

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
        y_lists = ksd_values[mask]

        # Compute the median and ±3 std deviation for each varying parameter value
        medians = [np.median(y_list) for y_list in y_lists]
        lower_bounds = [np.median(y_list) - 3 * np.std(y_list) for y_list in y_lists]
        upper_bounds = [np.median(y_list) + 3 * np.std(y_list) for y_list in y_lists]

        sorted_idx = np.argsort(x)
        x = x[sorted_idx]
        medians = np.array(medians)[sorted_idx]
        lower_bounds = np.array(lower_bounds)[sorted_idx]
        upper_bounds = np.array(upper_bounds)[sorted_idx]

        color = colors[i % num_colors]
        ax.plot(
            x, medians,
            marker='.',
            label=f"{fixed_param_latex} = {int(fixed_val)}",
            color=color,
        )

        # Fill the shaded region for ±3 std deviations
        ax.fill_between(x, lower_bounds, upper_bounds, color=color, alpha=0.3)

        # Highlight minimum of the median curve
        min_idx = np.argmin(medians)
        ax.plot(
            x[min_idx],
            medians[min_idx],
            marker='*',
            markersize=8,
            color=color,
            label=None,
        )

    ylabel = latex_param_names.get("estimatedKSDposteriors", "KSD")
    if plot_cfg.plot.y_axis.log_scale:
        ax.set_yscale("log")
        ylabel = f"log {ylabel}"

    ax.set_xlabel(varying_param_latex)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    save_path = os.path.join(output_dir, f"ksd_multiline_error_bands_{varying_param_name}_vs_{fixed_param_name}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved KSD multiline plot with error bands to: {save_path}")


def plot_distribution_of_optimal_mu0(
    ksd_results: Dict[Tuple[float, float], List[float]],
    param_names: List[str],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os

    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names
    colors = list(plot_cfg.plot.color_palette.colors)

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "figure.dpi": plot_cfg.plot.figure.dpi,
    })

    # Group KSDs by obs_num
    obs_to_mu0_ksd = {}
    for (obs_num, mu_0), ksd_list in ksd_results.items():
        obs_to_mu0_ksd.setdefault(obs_num, {})[mu_0] = ksd_list

    # Flatten all optimal mu_0s into a dataframe
    data = []
    for obs_num, mu0_ksds in obs_to_mu0_ksd.items():
        num_repeats = len(next(iter(mu0_ksds.values())))
        for rep in range(num_repeats):
            min_ksd = float("inf")
            best_mu0 = None
            for mu_0, ksd_list in mu0_ksds.items():
                ksd = ksd_list[rep]
                if ksd < min_ksd:
                    min_ksd = ksd
                    best_mu0 = mu_0
            data.append({'mu_0': best_mu0, 'obs. num.': obs_num})

    df = pd.DataFrame(data)

    # Plot using hue to group by obs_num
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df,
        x='mu_0',
        hue='obs. num.',
        bins=20,
        kde=False,
        multiple="dodge",
        palette=colors,
        ax=ax,
    )

    ax.set_xlabel(latex_param_names.get("mu_0", "mu_0"))
    ax.set_ylabel("Count")
    ax.grid(True)
    ax.legend(title=latex_param_names.get("obs. num.", "obs. num."))

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    save_path = os.path.join(output_dir, "optimal_mu0_distribution_across_repeats.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined distribution plot of optimal mu_0s to: {save_path}")


def plot_multivariate_priors_densities_by_ksd(all_params, all_ksds, output_dir, plot_cfg, true_theta=None, true_cov=None):
    """
    Plots joint prior densities (3 of them) with marginals using fixed 3-color scheme and KSD arrow bar.
    """
    assert len(all_params) == 3, "Function currently assumes 3 density entries for coloring."

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

    cb.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriorsShort)

    fig.tight_layout(rect=[0, 0, 0.9, 1.0])
    output_path = os.path.join(output_dir, "multivariate_joint_prior_plot.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_multivariate_joint_prior_densities_by_ksd(results, output_dir, plot_cfg, true_theta=None, true_cov=None):
    """
    Plots joint KDE contours of multivariate priors, colored by KSD magnitude (no fill).
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

    N = 17  # Show top-N and bottom-N KSD priors only
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
            lw = 0.5 + 1.5 * norm(ksd_est)
            levels = 3

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
    ax.set_xlabel("$\\mu_{01}$")
    ax.set_ylabel("$\\mu_{02}$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriorsShort)

    # Save
    output_path = os.path.join(output_dir, "joint_prior_ksd_contours.pdf")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_inverse_wishart_scale_ellipses_by_ksd_one_subplot(results, output_dir, plot_cfg):
    """
    Plots 2D ellipses representing inverse Wishart scale matrices,
    colored by KSD value. Highlights the max-KSD distribution in red.
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
        0: ["0.61", "0.63", "0.65"],
        1: ["1.25", "1.4", "1.5"],
        2: ["2.9", "3.2", "3.5"]
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
            cbar.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriorsShort, labelpad=25)

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


def plot_sdp_comparisons_multiple_radii(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
    prior_distribution,
    plot_cfg: DictConfig,
    output_dir: str,
    domain: tuple = (-5, 5),
    resolution: int = 200,
) -> None:
    """
    Plot log prior and its SDP-relaxed approximations for multiple radius lower bounds.
    Saves a single-panel figure to PDF.

    Notes:
        - Only 1D plotting is supported (basis on x ∈ R).
        - prior_samples / posterior_samples are accepted for backwards-compat, but unused.
    """
    os.makedirs(output_dir, exist_ok=True)

    # LaTeX + fonts (align with your other plots)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # Figure
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width*1.3,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Grid on x and target log prior
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    phi_x = basis_function.evaluate(x)
    log_prior = prior_distribution.log_pdf(x).flatten()

    # Colors (cycle through palette for each radius curve)
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]  # matplotlib defaults fallback

    # Plot SDP approximations (shifted by best constant to match mean on grid)
    names = plot_cfg.plot.param_latex_names
    ksd_label = names.get("estimatedKSDposteriorsShort")
    approx_sym = r"$\approx$"
    geq_sym = r"$\geq$"

    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f_sdp = (phi_x @ psi).flatten()
        c = float(np.mean(log_prior - f_sdp))
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label} ({approx_sym} {ksd:.2f})"

        ax.plot(
            x.flatten(),
            f_sdp + c,
            label=label,
            linewidth=1.5,
            color=color,
        )

    # Plot base (true) log prior
    ax.plot(
        x.flatten(),
        log_prior,
        label=names.get("logbaseprior", "log Base Log Prior"),
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    # Labels and styling from latex name map
    ax.set_xlabel(names.get("mu_0", "theta"))
    ax.set_ylabel(names.get("log_prior", "log_prior"))

    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(
        title=ksd_label,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        labelspacing=0.4,  # tighter vertical space
        handlelength=1.8,
        handletextpad=0.5,
        borderpad=0.4,
    )
    for t in leg.get_texts():
        t.set_wrap(True)

    # Align legend title with labels
    leg.get_title().set_ha("right")
    leg._legend_box.align = "right"

    plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"] * 0.9)
    plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"] * 0.9)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    # Save and close
    filename = "toy_gaussian_model_nonparametric_optimisation.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved SDP log prior comparison plot: {save_path}")


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


def plot_sdp_densities_only(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-5, 5),
    resolution: int = 200,
    filename: str = "test",
) -> None:
    """
    Plot true prior density and normalized SDP densities (no samples).
    Matches styling (legend, colors, fonts, LaTeX) used in plot_sdp_comparisons_multiple_radii.
    Saves a single-panel figure to PDF.
    """
    os.makedirs(output_dir, exist_ok=True)

    # LaTeX + fonts (align with your other plots)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # Figure
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width*1.3,
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
    ksd_label = names.get("estimatedKSDposteriorsShort")
    xlabel = names.get("mu_0", "theta")
    ylabel = names.get("nonparametric_prior", "Density")
    true_label = names.get("logbaseprior", "True Prior Density")
    approx_sym = r"$\approx$"
    geq_sym = r"$\geq$"

    # Plot SDP densities (normalized on grid)
    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f = (Phi_x @ psi).flatten()
        logZ = logsumexp(f) + np.log(dx)
        p_hat = np.exp(f - logZ)
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label} ({approx_sym} {ksd:.2f})"
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

    # Axes labels & styling
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend styling identical to your other function
    leg = ax.legend(
        title=ksd_label,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
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
        plt.tight_layout()

    # Save and close
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved SDP densities plot: {save_path}")


def plot_sdp_densities_and_logprior(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
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
    ksd_label = names.get("estimatedKSDposteriorsShort")
    xlabel = names.get("mu_0", "theta")
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
    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
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
    for i, (psi, r_label, _) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
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
        title=ksd_label,
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
    filename = "toy_gaussian_model_nonparametric_optimisation_densities_logprior.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sdp_2d_densities(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = ((-5, 5), (-5, 5)),   # ((x_min, x_max), (y_min, y_max))
    resolution: int | tuple = 200,        # int or (nx, ny)
    contour_levels: int = 8,              # number of contour levels
) -> None:
    """
    2D plot:
      True prior contours (dark gray) + SDP density contours (palette).
    """
    os.makedirs(output_dir, exist_ok=True)

    def _f_grid(Phi_XY: np.ndarray, psi: np.ndarray, nx: int, ny: int) -> np.ndarray:
        fN2 = np.tensordot(Phi_XY, psi, axes=([-1], [0]))
        fN = fN2.sum(axis=1)
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
    ksd_label = names.get("estimatedKSDposteriorsShort")
    xlabel = names.get("mu_01", r"$\mu_{01}$")
    ylabel = names.get("mu_02", r"$\mu_{02}$")
    approx_sym = r"$\approx$"
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
    log_g_grid = prior_distribution.log_pdf(XY).reshape(ny, nx)

    # --- Figure ---
    fig, ax = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width*1.2,
                 plot_cfg.plot.figure.size.height*1.2),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # True prior contours (dark gray)
    ax.contour(
        X, Y, prior_density_true,
        levels=contour_levels,
        colors="grey",
        linewidths=1.5,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = []

    # SDP priors (contours in palette colors)
    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f = _f_grid(Phi_XY, psi, nx, ny)
        log_post = f + log_g_grid
        logZ = logsumexp(log_post) + np.log(dx * dy)
        p_hat = np.exp(log_post - logZ)

        color = sdp_palette[i % len(sdp_palette)]
        label = rf"r {geq_sym} {r_label} ({ksd:.1f})"
        ax.contour(
            X, Y, p_hat,
            levels=contour_levels,
            colors=color,
            linewidths=1.5,
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=1.8, label=label))

    legend_handles.append(Line2D([0], [0], color="dimgray", lw=1.5, label=r"$\Pi_{\mathrm{ref}}$"))
    fig.legend(
        handles=legend_handles,
        title=ksd_label,
        loc="center left",
        bbox_to_anchor=(0.96, 0.5),
        frameon=False,
    )

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout(rect=[0, 0, 0.95, 1])

    # Save
    filename = "toy_multivariate_gaussian_model_nonparametric_optimisation_densities_per_radius.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sdp_2d_densities_flexible(
    basis_functions,  # single basis or list aligned with psi_sdp_list
    psi_sdp_list,                # list of psi (each shape (B_i,))
    labels,                                  # radii OR basis sizes (one per psi)
    prior_distribution,
    plot_cfg,
    output_dir,
    domain: ((-5, 5), (-5, 5)),               # ((x_min, x_max), (y_min, y_max))
    resolution: 200,              # int or (nx, ny)
    contour_levels: int = 8,                          # number of contour levels
    ksd_estimates: list = None,            # optional list; same length as psi_sdp_list
    label_template: str = None,                       # e.g. r"r {geq} {label} ({approx} {ksd:.2f})" or r"B = {label}"
    legend_title: str = None,                         # title above legend
    true_contour_color: str = "grey",              # color for Π_ref contours
) -> None:
    """
    2D plot:
      True prior contours (dark gray) + SDP density contours (palette).
    Works with:
      - a single basis_function (reused for all psi), or
      - a list of basis_functions aligned with psi_sdp_list (e.g., different #basis per curve).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- helpers ----
    def _as_list(x, length: int):
        """Broadcast a single item to a list of given length, or validate list length."""
        if isinstance(x, (list, tuple)):
            if len(x) != length:
                raise ValueError(f"Expected {length} basis_functions, got {len(x)}.")
            return list(x)
        return [x] * length

    def _f_grid(Phi_XY: np.ndarray, psi: np.ndarray, nx: int, ny: int) -> np.ndarray:
        """
        Contract Phi_XY over basis (last axis) with psi, then sum over dimensions.
        - Phi_XY: (N, d, B_i) where B_i=#basis for curve i, d=#dims (2)
        - psi   : (B_i,)
        Returns f on grid as (ny, nx).
        """
        # (N, d, B) · (B,) -> (N, d) -> sum over d -> (N,) -> (ny, nx)
        fN2 = np.tensordot(Phi_XY, psi, axes=([-1], [0]))  # (N, d)
        fN = fN2.sum(axis=1)                                # (N,)
        return fN.reshape(ny, nx)

    # --- rcParams (LaTeX + font) ---
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # --- Colors (from config) ---
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        raise ValueError("plot_cfg.plot.color_palette.colors is empty.")
    sdp_palette = palette

    # --- Labels & names ---
    names = getattr(plot_cfg.plot, "param_latex_names", {})
    if legend_title is None:
        legend_title = names.get("estimatedKSDposteriorsShort")
    xlabel = names.get("mu_01", r"$\mu_{01}$")
    ylabel = names.get("mu_02", r"$\mu_{02}$")
    geq_sym = r"$\geq$"
    approx_sym = r"$\approx$"

    # Default legend entry format
    if label_template is None:
        label_template = (r"{label} ({ksd:.1f})" if ksd_estimates is not None else r"{label}")

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

    XY = np.column_stack([X.ravel(), Y.ravel()])  # (N, 2)
    N = XY.shape[0]

    # --- Basis list aligned with curves ---
    basis_list = _as_list(basis_functions, len(psi_sdp_list))

    # --- True prior (draw once) ---
    prior_density_true = prior_distribution.pdf(XY).reshape(ny, nx)

    # log g(θ): reference/base-measure log-density, shared across all candidate priors
    log_g_grid = prior_distribution.log_pdf(XY).reshape(ny, nx)

    # --- Figure ---
    fig, ax = plt.subplots(
        1, 1,
        figsize=(plot_cfg.plot.figure.size.width*1.2,
                 plot_cfg.plot.figure.size.height*1.2),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # True prior contours (dark gray)
    ax.contour(
        X, Y, prior_density_true,
        levels=contour_levels,
        colors=true_contour_color,
        linewidths=1,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend handles: Π_ref first
    legend_handles = []

    # --- SDP curves ---
    for i, (psi, label_val, bf) in enumerate(zip(psi_sdp_list, labels, basis_list)):
        # Evaluate the *matching* basis for this curve
        Phi_XY_i = bf.evaluate(XY)  # expected shape (N, 2, B_i)
        if Phi_XY_i.ndim != 3 or Phi_XY_i.shape[0] != N or Phi_XY_i.shape[1] != 2:
            raise ValueError(f"basis_functions[{i}].evaluate(XY) must return (N, 2, B), got {Phi_XY_i.shape}")

        if psi.ndim != 1 or psi.shape[0] != Phi_XY_i.shape[-1]:
            raise ValueError(
                f"psi_sdp_list[{i}] shape {psi.shape} must match #basis {Phi_XY_i.shape[-1]} for that curve."
            )

        f = _f_grid(Phi_XY_i, psi, nx, ny)
        log_post = f + log_g_grid
        logZ = logsumexp(log_post) + np.log(dx * dy)   # grid normalization
        p_hat = np.exp(log_post - logZ)

        color = sdp_palette[i % len(sdp_palette)]
        if ksd_estimates is not None:
            entry = label_template.format(label=label_val, ksd=ksd_estimates[i], geq=geq_sym, approx=approx_sym)
        else:
            entry = label_template.format(label=label_val, geq=geq_sym, approx=approx_sym)

        ax.contour(
            X, Y, p_hat,
            levels=contour_levels,
            colors=color,
            linewidths=1.5,
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=1.8, label=entry))

    legend_handles.append(Line2D([0], [0], color=true_contour_color, lw=1, label=r"$\Pi_{\mathrm{ref}}$"))
    fig.legend(
        handles=legend_handles,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(0.96, 0.5),
        frameon=False,
    )

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout(rect=[0, 0, 0.95, 1])

    # Save
    filename = "toy_multivariate_gaussian_model_nonparametric_optimisation_densities_per_basis_functions_num.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved 2D contours-only plot: {save_path}")


class _TextOnlyHandler(HandlerBase):
    """Legend handler that reserves space but draws no glyph (text-only row)."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        return [Rectangle((0, 0), 0, 0, transform=trans,
                          facecolor="none", edgecolor="none")]


def plot_runtime_parametric_nonparametric(
    times_list_parametric: list[tuple[int, float]],
    times_list_nonparametric: list[tuple[int, int, float]],
    plot_cfg: Any,
    output_dir: str,
) -> None:
    """
    Single-axis plot:
      - Parametric KSD runtime (one line)
      - Non-parametric KSD runtime (one line per basis size)

    Single legend box with the order:
      Parametric
      # of basis functions   (text-only "sub-title")
      5
      10
      ...
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
    lw = float(_deep_get(plot_cfg, "plot.line.width", 2.0))
    marker = _deep_get(plot_cfg, "plot.marker.style", "o")
    ms = float(_deep_get(plot_cfg, "plot.marker.size", 4.5))
    grid_alpha = float(_deep_get(plot_cfg, "plot.grid.alpha", 0.7))
    tight = bool(_deep_get(plot_cfg, "plot.figure.tight_layout", True))

    names = _deep_get(plot_cfg, "plot.param_latex_names", {}) or {}
    x_label = names.get("numPosteriorSamples", r"\# of Posterior Samples")
    y_label = names.get("runtimeSeconds", "Time (sec.)")
    legend_subtitle = names.get("legendBasisFunctions", r"Non-parametric: \# of basis functions")
    filename = _deep_get(plot_cfg, "plot.filenames.runtime_param_nonparam", "runtime_parametric_nonparametric.pdf")

    # Colors
    palette = list(getattr(_deep_get(plot_cfg, "plot.color_palette", {}), "colors", []))
    if not palette:
        palette = [f"C{i}" for i in range(10)]

    # Prepare data
    if times_list_parametric:
        sample_sizes_param, times_param = zip(*[(int(s), float(t)) for s, t in times_list_parametric])
    else:
        sample_sizes_param, times_param = [], []

    by_basis = defaultdict(list)
    for s, b, t in times_list_nonparametric:
        by_basis[int(b)].append((int(s), float(t)))
    for b in by_basis:
        by_basis[b].sort(key=lambda x: x[0])

    # Figure (single axis)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)

    # Plot lines
    handles_ordered, labels_ordered = [], []

    # 1) Parametric first
    if sample_sizes_param:
        h_param, = ax.plot(
            sample_sizes_param, times_param,
            marker=marker, markersize=ms, linewidth=lw,
            color=palette[0],
            label="Parametric",
        )
        handles_ordered.append(h_param)
        labels_ordered.append("Parametric")

    # 2) Insert a text-only "subtitle" row for the basis functions
    #    This shows as a label with no handle glyph.
    dummy_title = Line2D([], [], linestyle="none", label=legend_subtitle)
    handles_ordered.append(dummy_title)
    labels_ordered.append(legend_subtitle)

    # 3) Basis function curves
    for i, (b, pts) in enumerate(sorted(by_basis.items()), start=1):
        s_vals = [p[0] for p in pts]
        t_vals = [p[1] for p in pts]
        h_np, = ax.plot(
            s_vals, t_vals,
            marker=marker, markersize=ms, linewidth=lw,
            color=palette[i % len(palette)],
            label=rf"{b}",
        )
        handles_ordered.append(h_np)
        labels_ordered.append(rf"{b}")

    # Axes styling
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    spacer = Line2D([], [], linestyle="none", label="")  # blank row
    handles_ordered.append(spacer)
    labels_ordered.append("")
    legend_fs = float(_deep_get(plot_cfg, "plot.legend.fontsize",
                                plt.rcParams["font.size"] * 0.8))
    leg = ax.legend(
        handles_ordered, labels_ordered,
        loc="lower left",
        fontsize=legend_fs,
        bbox_to_anchor=(0.02, 0.5),
        frameon=True, fancybox=True, framealpha=0.95,
        borderpad=0.4, handlelength=1.2, handletextpad=0.4, labelspacing=0.24,
        handler_map={dummy_title: _TextOnlyHandler()},
    )
    leg._legend_box.align = "left"

    if tight:
        fig.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved runtime plot: {save_path}")


def plot_sdp_densities_by_basis_functions(
    basis_functions,
    basis_funcs_num_list: list[int],
    psi_sdp_list: list[np.ndarray],           # len matches basis_funcs_num_list
    ksd_estimates: list[float],               # len matches basis_funcs_num_list
    prior_distribution,                       # must have .pdf(x) or .log_pdf(x)
    plot_cfg: DictConfig,
    output_dir: str,
    domain: tuple = (-5, 5),
    resolution: int = 200,
) -> None:
    """
    Plot true prior density and normalized SDP densities for multiple basis sizes m.
    Saves a single-panel figure to PDF.

    basis_function_or_factory:
        - Either a callable m -> basis_function (with .evaluate(X) -> Phi(X)),
        - Or a list of basis functions aligned with psi_sdp_list (one per m).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Helpers ---
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

    # --- Matplotlib from config ---
    plt.rcParams.update({
        "font.size": _deep_get(plot_cfg, "plot.font.size", 12),
        "font.family": _deep_get(plot_cfg, "plot.font.family", "serif"),
        "text.usetex": bool(_deep_get(plot_cfg, "plot.font.use_tex", False)),
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig_w = float(_deep_get(plot_cfg, "plot.figure.size.width", 6.0)) * 1.3
    fig_h = float(_deep_get(plot_cfg, "plot.figure.size.height", 4.0))
    fig_dpi = int(_deep_get(plot_cfg, "plot.figure.dpi", 150))
    lw = float(_deep_get(plot_cfg, "plot.line.width", 1.5))

    # Labels & names
    names = _deep_get(plot_cfg, "plot.param_latex_names", {}) or {}
    xlabel = names.get("mu_0", "theta")
    ylabel = names.get("nonparametric_prior", "Density")
    true_label = names.get("baseprior", "True Prior Density")
    approx_sym = r"$\approx$"

    # Colors
    palette = list(getattr(_deep_get(plot_cfg, "plot.color_palette", {}), "colors", []))
    if not palette:
        palette = [f"C{i}" for i in range(10)]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)

    # Grid & true prior
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    if hasattr(prior_distribution, "pdf"):
        prior_density = prior_distribution.pdf(x).flatten()
    else:
        log_p = prior_distribution.log_pdf(x).flatten()
        logZ = logsumexp(log_p) + np.log(dx)
        prior_density = np.exp(log_p - logZ)

    # Plot each SDP density for its own m
    for i, (m, psi, ksd) in enumerate(zip(basis_funcs_num_list, psi_sdp_list, ksd_estimates)):
        basis_fn = basis_functions[i]
        Phi_x = basis_fn.evaluate(x)
        f = (Phi_x @ psi).flatten()
        logZ = logsumexp(f) + np.log(dx)
        p_hat = np.exp(f - logZ)

        color = palette[i % len(palette)]
        label = rf"$m={int(m)}$"
        ax.plot(x.flatten(), p_hat, label=label, linewidth=lw, color=color)

    ax.plot(x.flatten(), prior_density, label=true_label, linestyle="--", linewidth=lw, color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="best",
        frameon=False,
        labelspacing=0.4,
        handlelength=1.8,
        handletextpad=0.5,
        borderpad=0.4,
    )

    if bool(_deep_get(plot_cfg, "plot.figure.tight_layout", False)):
        plt.tight_layout()

    filename = _deep_get(
        plot_cfg, "plot.filenames.nonparametric_densities_by_basis",
        "toy_gaussian_model_nonparametric_optimisation_densities_by_basis.pdf",
    )
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved SDP densities-by-basis plot: {save_path}")


class _TextOnlyHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        return None


def plot_runtime_parametric_nonparametric_with_ci(
    times_parametric: dict[int, dict[int, float]],
    times_nonparametric: dict[int, dict[int, dict[int, float]]],
    plot_cfg: Any,
    output_dir: str,
    ci_level: float = 0.95,
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
    x_label = names.get("numPosteriorSamples", r"\# of Posterior Samples")
    y_label = names.get("runtimeSeconds", "Time (sec.)")
    legend_subtitle = names.get("legendBasisFunctions", r"Non-parametric: \# of basis functions")

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
    sample_sizes_param, means_param, cis_param = [], [], []
    for s in sorted(times_parametric.keys(), key=_to_int):
        vals = list(times_parametric[s].values())
        m, h = mean_ci(vals)
        sample_sizes_param.append(s)
        means_param.append(m)
        cis_param.append(h)

    # --- Process non-parametric
    by_basis = defaultdict(lambda: ([], [], []))
    for s in sorted(times_nonparametric.keys(), key=_to_int):
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

    # Parametric
    if sample_sizes_param:
        h_param = ax.plot(
            sample_sizes_param, means_param,
            marker=marker, markersize=ms, linewidth=lw,
            color=palette[0], label="Parametric",
        )[0]
        ax.fill_between(
            sample_sizes_param,
            np.array(means_param) - np.array(cis_param),
            np.array(means_param) + np.array(cis_param),
            color=palette[0], alpha=0.7
        )
        handles_ordered.append(h_param)
        labels_ordered.append("Parametric")

    # Subtitle
    dummy_title = Line2D([], [], linestyle="none", label=legend_subtitle)
    handles_ordered.append(dummy_title)
    labels_ordered.append(legend_subtitle)

    # Non-parametric
    for i, (b, (samples, means, cis)) in enumerate(sorted(by_basis.items()), start=1):
        h_np = ax.plot(
            samples, means,
            marker=marker, markersize=ms, linewidth=lw,
            color=palette[i % len(palette)], label=rf"{b}"
        )[0]
        ax.fill_between(
            samples,
            np.array(means) - np.array(cis),
            np.array(means) + np.array(cis),
            color=palette[i % len(palette)], alpha=0.7
        )
        handles_ordered.append(h_np)
        labels_ordered.append(rf"{b}")

    # Axes styling
    ax.set_xlabel(x_label)
    # ax.set_xticklabels([])
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    spacer = Line2D([], [], linestyle="none", label="")
    handles_ordered.append(spacer)
    labels_ordered.append("")
    legend_fs = float(_deep_get(plot_cfg, "plot.legend.fontsize",
                                plt.rcParams["font.size"] * 0.8))
    leg = ax.legend(
        handles_ordered, labels_ordered,
        loc="lower left",
        fontsize=legend_fs,
        bbox_to_anchor=(0.02, 0.5),
        frameon=True, fancybox=True, framealpha=0.95,
        borderpad=0.4, handlelength=1.2, handletextpad=0.4, labelspacing=0.24,
        handler_map={dummy_title: _TextOnlyHandler()},
    )
    leg._legend_box.align = "left"

    if tight:
        fig.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
