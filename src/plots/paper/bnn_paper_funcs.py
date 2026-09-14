import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, norm, t as student_t


def _alpha_fade_cmap(name: str, hex_color: str, alpha_low: float = 0.12, alpha_high: float = 1.0):
    """
    Single-hue colormap that fades low values out via alpha (mostly
    transparent) rather than blending toward white -- keeps the hue visible
    at low opacity instead of desaturating to an achromatic color.
    """
    r, g, b = to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(name, [(r, g, b, alpha_low), (r, g, b, alpha_high)])


# Single-hue dark-grey-blue palette for the weight-sensitivity heatmaps,
# using the active dark-grey-blue swatch from configs/plots/overleaf_plots_
# settings.yaml's color_palette ("#4d7298"): low sensitivity fades toward
# transparent rather than white, high sensitivity is fully opaque.
BNN_HEATMAP_CMAP = _alpha_fade_cmap("bnn_heatmap_greyblue_alpha", "#4d7298")

# Sequential grey-blue palette for the row/column marginal bars, built from
# the two active grey-blue swatches in that same color_palette: light-grey
# -blue ("#9dc3c2", low sensitivity) to dark-grey-blue ("#4d7298", high
# sensitivity) -- no white/disabled colors mixed in.
_BNN_BAR_COLORS = ["#9dc3c2", "#4d7298"]
BNN_BAR_CMAP = LinearSegmentedColormap.from_list("bnn_bar_greyblue", _BNN_BAR_COLORS)


def _deep_get(cfg, path, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def plot_bnn_weight_heatmaps(
    tensors: List[Dict[str, Any]],
    plot_cfg: Any,
    output_dir: str,
    filename: str = "bnn_weight_heatmaps.pdf",
    cmap: Any = BNN_HEATMAP_CMAP,
    value_label: str = r"$\widehat{S}_m^{\FD}(\widehat{\mathcal{Q}}_{r_j}^{j, K,l})$",
) -> None:
    """
    One heatmap per 2D weight tensor (output units x input units/features).
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": float(_deep_get(plot_cfg, "plot.font.size", 14)) * 1.2,
        "font.family": _deep_get(plot_cfg, "plot.font.family", "serif"),
        "text.usetex": bool(_deep_get(plot_cfg, "plot.font.use_tex", False)),
        "text.latex.preamble": r"\usepackage{amsmath}\providecommand{\FD}{\mathrm{FD}}",
    })
    fig_w = float(_deep_get(plot_cfg, "plot.figure.size.width", 6.0))
    fig_h = float(_deep_get(plot_cfg, "plot.figure.size.height", 4.0))
    fig_dpi = int(_deep_get(plot_cfg, "plot.figure.dpi", 150))

    n = len(tensors)
    fig = plt.figure(figsize=(n * fig_w * 1.0, fig_h * 1.3), dpi=fig_dpi)
    gs_root = fig.add_gridspec(1, 2, width_ratios=[n, 0.08], wspace=0.08)
    gs_outer = gs_root[0].subgridspec(1, n, wspace=0.2)

    matrices = [np.asarray(t["matrix"], dtype=float) for t in tensors]
    vmin = min(M.min() for M in matrices)
    vmax = max(M.max() for M in matrices)
    bar_norm = Normalize(vmin=vmin, vmax=vmax)

    im = None
    for i, t in enumerate(tensors):
        M = matrices[i]
        n_rows, n_cols = M.shape
        col_label = str(t.get("col_label", "column"))
        show_col_marginal = bool(t.get("show_col_marginal", True))
        gs = gs_outer[i].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.5)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_col = fig.add_subplot(gs[1, 0], sharex=ax_main) if show_col_marginal else None

        im = ax_main.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        ax_main.set_title(t.get("label", ""), fontsize=plt.rcParams["font.size"])
        ax_main.set_ylabel(t.get("row_label", "row"))
        ax_main.set_yticks(np.arange(-0.5, n_rows, 1))
        ax_main.set_yticklabels([])

        if show_col_marginal:
            # locator (the label ticks at cell centers, set further down).
            ax_main.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
            ax_main.tick_params(axis="x", which="major", bottom=False, top=False, labelbottom=False)
            ax_main.tick_params(axis="x", which="minor", bottom=True, top=False, labelbottom=False)
            col_means = M.mean(axis=0)
            ax_col.bar(np.arange(n_cols), col_means, color=BNN_BAR_CMAP(bar_norm(col_means)), alpha=1.0)
            ax_col.set_ylabel(r"avg. $\hat{S}_m^\mathrm{FD}(\widehat{\mathcal{Q}}_{r}^{K,l})$",
                              fontsize=plt.rcParams["font.size"]*0.9)
            if t.get("col_names"):
                ax_col.set_xticks(np.arange(n_cols))
                ax_col.set_xticklabels(t["col_names"], rotation=90, fontsize=plt.rcParams["font.size"] * 0.6)
                ax_col.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
                ax_col.tick_params(axis="x", which="minor", length=3)
            else:
                ax_col.set_xticks(np.arange(-0.5, n_cols, 1))
                ax_col.set_xticklabels([])
                ax_col.set_xlabel(col_label)
        else:
            ax_main.set_xticks(np.arange(-0.5, n_cols, 1))
            ax_main.set_xticklabels([])
            ax_main.set_xlabel(col_label)

        for ax in (ax_main, ax_col):
            if ax is not None:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

    cax = fig.add_subplot(gs_root[0, 1])
    fig.colorbar(im, cax=cax, label=value_label)
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)


def _layer_key(tensor_name: str) -> str:
    """'net.module.0.weight_prior' -> 'net.module.0' (groups weight+bias of one layer)."""
    return tensor_name.rsplit(".", 1)[0]


def plot_bnn_layer_sensitivity(
    group_results: Dict[str, Dict[str, Any]],
    plot_cfg: Any,
    output_dir: str,
    filename: str = "bnn_layer_sensitivity.pdf",
) -> None:
    """
    Sensitivity aggregated by *layer* (weight + bias of the same nn.Linear
    combined).
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": _deep_get(plot_cfg, "plot.font.size", 12),
        "font.family": _deep_get(plot_cfg, "plot.font.family", "serif"),
        "text.usetex": bool(_deep_get(plot_cfg, "plot.font.use_tex", False)),
    })
    fig_w = float(_deep_get(plot_cfg, "plot.figure.size.width", 6.0))
    fig_h = float(_deep_get(plot_cfg, "plot.figure.size.height", 4.0))
    fig_dpi = int(_deep_get(plot_cfg, "plot.figure.dpi", 150))
    palette = list(getattr(_deep_get(plot_cfg, "plot.color_palette", {}), "colors", []))
    if not palette:
        palette = [f"C{i}" for i in range(10)]

    layers: Dict[str, Dict[str, Any]] = {}
    for tensor_name, res in group_results.items():
        key = _layer_key(tensor_name)
        entry = layers.setdefault(key, {"sensitivity": []})
        entry["sensitivity"].append(np.asarray(res["sensitivity"]))

    layer_names = sorted(layers.keys(), key=lambda k: int(k.rsplit(".", 1)[-1]))
    short_names = [n.replace("net.module.", "L") for n in layer_names]

    sens_concat = [np.concatenate(layers[n]["sensitivity"]) for n in layer_names]
    means = [float(s.mean()) for s in sens_concat]
    stds = [float(s.std()) for s in sens_concat]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
    colors = [palette[i % len(palette)] for i in range(len(layer_names))]

    bars = ax.bar(short_names, means, yerr=stds, capsize=4, color=colors)
    ax.bar_label(bars, labels=[f"{m:.3g}" for m in means], padding=3, fontsize=plt.rcParams["font.size"] * 0.8)
    ax.set_ylabel(r"$\hat{S}_m^\mathrm{FD}(\widehat{\mathcal{Q}}_{r,K,l})$")
    ax.set_xlabel("BNN Layer")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)


def _plot_bnn_node_candidate_prior(
    ax,
    node: Dict[str, Any],
    palette: List[str],
    n_scales: float,
    resolution: int,
    y_log: bool,
    y_floor: float,
    show_legend: bool,
    show_title: bool = True,
    show_posterior_density: bool = False,
    xlabel: str = r"$\theta$",
) -> None:
    loc, scale = float(node["loc"]), float(node["scale"])
    df = node.get("df")
    x = np.linspace(loc - n_scales * scale, loc + n_scales * scale, resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])

    if df is not None:
        log_g = student_t.logpdf(x, df=df, loc=loc, scale=scale).flatten()
    else:
        log_g = norm.logpdf(x, loc=loc, scale=scale).flatten()
    ref_density = np.exp(log_g)

    Phi_x = node["basis"].evaluate(x)[:, 0, :]  # (resolution, K)
    f = Phi_x @ np.asarray(node["lambda_star"])
    log_cand = f + log_g
    logZ = logsumexp(log_cand) + np.log(dx)
    cand_density = np.exp(log_cand - logZ)

    if y_log:
        ref_density = np.maximum(ref_density, y_floor)
        cand_density = np.maximum(cand_density, y_floor)

    color = palette[0]
    ax.plot(x.flatten(), ref_density, linestyle="--", linewidth=1.5,
            color="black", label=r"$\Pi_{\mathrm{ref}}$")
    ax.plot(x.flatten(), cand_density, linewidth=1.5, color=color,
            label=r"$\Pi_K^{\lambda_m^\mathrm{sup}}$")

    if show_posterior_density:
        post_samples = np.asarray(node.get("posterior_samples"))
        if post_samples is not None and post_samples.size > 1 and np.std(post_samples) > 1e-12:
            try:
                kde = gaussian_kde(post_samples)
                kde_density = kde(x.flatten())
                if y_log:
                    kde_density = np.maximum(kde_density, y_floor)
                ax.fill_between(x.flatten(), kde_density, y_floor if y_log else 0.0,
                                alpha=0.25, color="gray", label=r"$\tilde{\Pi}_{\mathrm{ref}}$")
            except np.linalg.LinAlgError:
                pass

    if y_log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=y_floor)

    centers = np.asarray(getattr(node["basis"], "centers", []), dtype=float).reshape(-1)
    if centers.size:
        for c in (centers.min(), centers.max()):
            ax.axvline(c, color="black", linewidth=0.6, linestyle=":", alpha=0.5)

    title = node.get("label", "")
    subtitle_parts = []
    if node.get("sensitivity") is not None:
        subtitle_parts.append(r"$\hat{S}_m^\mathrm{FD}(\widehat{\mathcal{Q}}_{r_j,K,l})$"+f"={node['sensitivity']:.3f}")
    if subtitle_parts:
        title = f"{title.replace(".weight", "")}, {', '.join(subtitle_parts)}"
    if show_title:
        ax.set_title(title, fontsize=plt.rcParams["font.size"] * 0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\pi$" if not y_log else r"$\log\pi$")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=plt.rcParams["font.size"] * 0.8)


def plot_bnn_node_candidate_priors(
    nodes: List[Dict[str, Any]],
    plot_cfg: Any,
    output_dir: str,
    filename: str = "bnn_node_candidate_priors.pdf",
    n_scales: float = 6.0,
    resolution: int = 500,
    n_cols: int = 3,
    y_log: bool = False,
    y_floor: float = 1e-6,
    save_individual: bool = True,
    individual_dirname: str = "individual",
    show_posterior_density: bool = False,
    xlabel: str = r"$\theta$",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": _deep_get(plot_cfg, "plot.font.size", 12),
        "font.family": _deep_get(plot_cfg, "plot.font.family", "serif"),
        "text.usetex": bool(_deep_get(plot_cfg, "plot.font.use_tex", False)),
    })
    fig_w = float(_deep_get(plot_cfg, "plot.figure.size.width", 6.0))
    fig_h = float(_deep_get(plot_cfg, "plot.figure.size.height", 4.0))
    fig_dpi = int(_deep_get(plot_cfg, "plot.figure.dpi", 150))
    palette = list(getattr(_deep_get(plot_cfg, "plot.color_palette", {}), "colors", []))
    if not palette:
        palette = [f"C{i}" for i in range(10)]

    n = len(nodes)
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * fig_w, n_rows * fig_h),
        dpi=fig_dpi,
        squeeze=False,
    )

    for i, node in enumerate(nodes):
        ax = axes[i // n_cols][i % n_cols]
        _plot_bnn_node_candidate_prior(
            ax, node, palette, n_scales, resolution, y_log, y_floor, show_legend=False,
            show_posterior_density=show_posterior_density, xlabel=xlabel,
        )

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)

    if save_individual and n:
        indiv_dir = os.path.join(output_dir, individual_dirname)
        os.makedirs(indiv_dir, exist_ok=True)
        base, ext = os.path.splitext(filename)
        ext = ext or ".pdf"

        for i, node in enumerate(nodes):
            fig_i, ax_i = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
            _plot_bnn_node_candidate_prior(
                ax_i, node, palette, n_scales, resolution, y_log, y_floor, show_legend=False, show_title=False,
                show_posterior_density=show_posterior_density, xlabel=xlabel,
            )
            fig_i.tight_layout()
            label = str(node.get("label", f"node{i}"))
            safe_label = "".join(c if c.isalnum() else "_" for c in label)
            fig_i.savefig(os.path.join(indiv_dir, f"{base}_{i:02d}_{safe_label}{ext}"), bbox_inches="tight")
            plt.close(fig_i)
