from typing import List, Tuple, Dict, FrozenSet, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from scipy.stats import gaussian_kde
import time
from matplotlib.lines import Line2D
import ot
from src.common.plots import apply_plot_rc, save_fig


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
    """Plot a filled 2D contour of the quadratic form over (mu, sigma) space."""
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
    ax.contour(Xi, Yi, Zi, levels=20, colors="white", linewidths=0.4, alpha=0.5)
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


def plot_multivariate_joint_prior_densities_by_fd(results, output_dir, plot_cfg, true_theta=None, true_cov=None):
    """Plot joint KDE contours of multivariate priors coloured by FD, highlighting the largest in red."""
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
    """Compute the closed-form Fisher divergence FD(P_ref || P_cand) between two Gaussians."""
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
    """Compute the closed-form 2-Wasserstein distance between two Gaussians."""
    diff = mu1 - mu2
    from scipy.linalg import sqrtm
    S2_sqrt = sqrtm(Sigma2)
    middle = S2_sqrt @ Sigma1 @ S2_sqrt
    middle_sqrt = sqrtm(middle)
    middle_sqrt = np.real_if_close(middle_sqrt, tol=1e5)
    w2_sq = float(diff @ diff + np.trace(Sigma1 + Sigma2 - 2.0 * middle_sqrt))

    return float(np.sqrt(max(w2_sq, 0.0)))


def estimate_w2_from_samples(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimate the 2-Wasserstein distance between the empirical distributions of X and Y."""
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
    """Return fixed reference, FD/mean candidate and two WIM candidate Gaussians for dimension d."""
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


def estimate_kl_from_ref_samples_kde(
    X_ref_eval: np.ndarray,
    X_ref_fit: np.ndarray,
    X_cand_fit: np.ndarray,
    bw_method: None,
    eps: float = 1e-12,
) -> float:
    """Estimate KL(P_ref || P_cand) by Monte Carlo over reference samples with KDE density estimates."""
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
    """Return an alpha that decreases linearly from alpha_max to alpha_min across dims."""
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
    """Compute finite-sample estimation errors and runtimes for the requested divergence(s)."""
    rng = np.random.default_rng(seed)

    methods = ["fd", "mean", "wim", "kl"] if divergence is None else [divergence]

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
        """Return the mean of x and the half-width of its bootstrap confidence interval."""
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

        # exact targets
        fd_true = fisher_divergence_gaussians_ref_expectation(
            mu_ref, Sigma_ref, mu_cand, Sigma_cand
        )
        mu_cand_true = mu_cand
        w2_true = w2_gaussian(mu_ref, Sigma_ref, mu_cand, Sigma_cand)
        kl_true = kl_gaussian(mu_ref, Sigma_ref, mu_cand, Sigma_cand)

        for m in ms:
            print(f"Computing statistics for dim={d}, m={m}.")

            err_rep = {method: np.empty(n_rep, dtype=float) for method in methods}
            time_rep = {method: np.empty(n_rep, dtype=float) for method in methods}

            for r in range(n_rep):
                # shared samples
                X_ref = sample_gaussian(rng, mu_ref, Sigma_ref, m)
                X_cand = sample_gaussian(rng, mu_cand, Sigma_cand, m)

                if "fd" in methods:
                    t0 = time.perf_counter()
                    fd_hat = estimate_fd_from_ref_samples(
                        X_ref, mu_ref, Sigma_ref, mu_cand, Sigma_cand
                    )
                    time_rep["fd"][r] = time.perf_counter() - t0
                    err_rep["fd"][r] = abs(fd_true - fd_hat)

                if "kl" in methods:
                    t0 = time.perf_counter()
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
                    time_rep["kl"][r] = time.perf_counter() - t0
                    err_rep["kl"][r] = abs(kl_true - kl_hat)

                if "mean" in methods:
                    t0 = time.perf_counter()
                    mu_hat = np.mean(X_cand, axis=0)
                    time_rep["mean"][r] = time.perf_counter() - t0
                    err_rep["mean"][r] = float(np.linalg.norm(mu_cand_true - mu_hat, ord=2))

                if "wim" in methods:
                    t0 = time.perf_counter()
                    w2_hat = estimate_w2_from_samples(X_ref, X_cand)
                    time_rep["wim"][r] = time.perf_counter() - t0
                    err_rep["wim"][r] = abs(w2_true - w2_hat)

            for method in methods:
                m_, h_ = mean_and_ci(err_rep[method], rng=rng)
                error_mean[method][d].append(m_)
                error_ci[method][d].append(h_)

                m_, h_ = mean_and_ci(time_rep[method])
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
    """Plot runtimes of estimating FD, mean, KL and WIM with identical axes sizes."""
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
    """Plot finite-sample error curves with identical axes sizes and return the computed results."""
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
        save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


