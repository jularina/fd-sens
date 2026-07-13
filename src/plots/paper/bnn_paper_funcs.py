import os
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, norm


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
    filename: str = "bnn_boston_weight_heatmaps.pdf",
    cmap: str = "Greens",
    value_label: str = r"$r_j\,\omega_{\max,j}$",
) -> None:
    """
    One heatmap per 2D weight tensor (output units x input units/features),
    coloured by the actual per-node sensitivity r_j*omega_max, with marginal
    bar charts giving the row mean (right, per output unit) and column mean
    (bottom, per input unit/feature) -- so a *within-layer* tendency (a
    specific hot row/column, e.g. one input feature or one hidden unit that
    is consistently more sensitive than the rest) is visible directly, rather
    than only in a top-k list.

    tensors: list of dicts with keys:
        label: str                       -- panel title
        matrix: (n_rows, n_cols) ndarray -- value (e.g. sensitivity) per (row, col) node
        row_label, col_label: str        -- axis labels (e.g. "hidden unit", "feature")
        col_names: Optional[List[str]]   -- tick labels for columns (e.g. Boston features)
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

    n = len(tensors)
    fig = plt.figure(figsize=(n * fig_w * 1.4, fig_h * 1.6), dpi=fig_dpi)
    gs_outer = fig.add_gridspec(1, n, wspace=0.6)

    for i, t in enumerate(tensors):
        M = np.asarray(t["matrix"], dtype=float)
        n_rows, n_cols = M.shape

        gs = gs_outer[i].subgridspec(
            2, 2, width_ratios=[4, 1], height_ratios=[4, 1], wspace=0.08, hspace=0.08
        )
        ax_main = fig.add_subplot(gs[0, 0])
        ax_row = fig.add_subplot(gs[0, 1], sharey=ax_main)
        ax_col = fig.add_subplot(gs[1, 0], sharex=ax_main)

        im = ax_main.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest")
        ax_main.set_title(t.get("label", ""), fontsize=plt.rcParams["font.size"])
        ax_main.set_ylabel(t.get("row_label", "row"))
        ax_main.set_xticks([])
        if n_rows <= 8:
            # avoid matplotlib auto-generating fractional y-ticks (e.g. -0.4..0.4)
            # for a heatmap with very few rows (a single output unit, etc.)
            ax_main.set_yticks(np.arange(n_rows))
        fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04, label=value_label)

        row_means = M.mean(axis=1)
        ax_row.barh(np.arange(n_rows), row_means, color="#005500")
        ax_row.set_xlabel("row mean", fontsize=plt.rcParams["font.size"] * 0.8)
        plt.setp(ax_row.get_yticklabels(), visible=False)

        col_means = M.mean(axis=0)
        ax_col.bar(np.arange(n_cols), col_means, color="#005500")
        ax_col.set_ylabel("col.\nmean", fontsize=plt.rcParams["font.size"] * 0.8)
        ax_col.set_xlabel(t.get("col_label", "column"))
        if t.get("col_names"):
            ax_col.set_xticks(np.arange(n_cols))
            ax_col.set_xticklabels(t["col_names"], rotation=90, fontsize=plt.rcParams["font.size"] * 0.7)
        else:
            ax_col.set_xticks([])

        for ax in (ax_main, ax_row, ax_col):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)


def _layer_key(tensor_name: str) -> str:
    """'net.module.0.weight_prior' -> 'net.module.0' (groups weight+bias of one layer)."""
    return tensor_name.rsplit(".", 1)[0]


def plot_bnn_layer_sensitivity(
    group_results: Dict[str, Dict[str, Any]],
    plot_cfg: Any,
    output_dir: str,
    filename: str = "bnn_boston_layer_sensitivity.pdf",
) -> None:
    """
    Sensitivity aggregated by *layer* (weight + bias of the same nn.Linear
    combined), rather than by tensor, to read off the depth trend directly:
    mean and std of the per-node *actual* sensitivity (r_j*omega_max_j)
    within the layer. Deliberately does NOT show the total (summed)
    sensitivity per layer -- that quantity is dominated by how many nodes a
    layer has (since r_j = r/J is the same for every node), so it conflates
    layer width with per-weight fragility; averaging divides the width
    effect back out.
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

    ax.bar(short_names, means, yerr=stds, capsize=4, color=colors)
    ax.set_ylabel(r"$r_j\,\omega_{\max,j}$ (mean $\pm$ std)")
    ax.set_title("Mean sensitivity per layer")
    ax.set_xlabel("Layer")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_bnn_node_candidate_priors(
    nodes: List[Dict[str, Any]],
    plot_cfg: Any,
    output_dir: str,
    filename: str = "bnn_boston_node_candidate_priors.pdf",
    n_scales: float = 6.0,
    resolution: int = 500,
    n_cols: int = 3,
    y_log: bool = False,
    y_floor: float = 1e-6,
) -> None:
    """
    Small-multiples plot of the worst-case KEF candidate prior for a handful
    of individual scalar nodes (e.g. the most sensitive weights found by
    `compute_group_omega_max`), mirroring `toy_paper_fisher_funcs.plot_sdp_densities`
    but one panel per node instead of one panel per radius.

    Each panel shows: the reference prior N(loc, scale^2) (dashed steelblue),
    the worst-case candidate density pi_K^{lambda*} obtained from the
    per-node KEF coefficients `lambda_star` (solid, colored), and a KDE of
    that node's own posterior draws (grey fill) -- so one can see how far
    the local prior neighbourhood *can* move relative to where the posterior
    actually sits. Density is on a linear scale by default; pass `y_log=True`
    to switch to log-density (useful if the worst-case direction mainly
    reweights the tails, which is invisible on a linear scale).

    `nodes` is a list of dicts, each with keys:
        label: str            -- panel title, e.g. "L0.weight[194]"
        loc, scale: float      -- reference prior N(loc, scale^2)
        lambda_star: (K,)      -- worst-case KEF coefficients (see
                                   src.optimization.bnn_node_sensitivity.compute_node_lambda_star)
        basis: BaseBasisFunction -- fitted basis function (same object used to get lambda_star)
        posterior_samples: (m,) -- that node's own posterior draws
        sensitivity: float, optional -- the node's actual sensitivity r_j*omega_max, shown in the panel title
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

        loc, scale = float(node["loc"]), float(node["scale"])
        x = np.linspace(loc - n_scales * scale, loc + n_scales * scale, resolution)[:, None]
        dx = float(x[1, 0] - x[0, 0])

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

        color = palette[i % len(palette)]
        ax.plot(x.flatten(), ref_density, linestyle="--", linewidth=1.5,
                color="steelblue", label=r"$\Pi_{\mathrm{ref}}$")
        ax.plot(x.flatten(), cand_density, linewidth=1.5, color=color,
                label=r"$\Pi_K^{\lambda^\star}$")

        post_samples = np.asarray(node.get("posterior_samples"))
        if post_samples is not None and post_samples.size > 1 and np.std(post_samples) > 1e-12:
            try:
                kde = gaussian_kde(post_samples)
                kde_density = kde(x.flatten())
                if y_log:
                    kde_density = np.maximum(kde_density, y_floor)
                ax.fill_between(x.flatten(), kde_density, y_floor if y_log else 0.0,
                                 alpha=0.25, color="gray", label="Posterior (KDE)")
            except np.linalg.LinAlgError:
                pass

        if y_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=y_floor)

        # mark the outermost KEF centres: beyond them the candidate can only
        # extrapolate the fixed-centre basis, which is where it diverges most
        # visibly from the reference prior on this log scale.
        centers = np.asarray(getattr(node["basis"], "centers", []), dtype=float).reshape(-1)
        if centers.size:
            for c in (centers.min(), centers.max()):
                ax.axvline(c, color="black", linewidth=0.6, linestyle=":", alpha=0.5)

        title = node.get("label", "")
        subtitle_parts = []
        if node.get("sensitivity") is not None:
            subtitle_parts.append(rf"$r_j\omega_{{\max}}$={node['sensitivity']:.3f}")
        if subtitle_parts:
            title = f"{title}\n({', '.join(subtitle_parts)})"
        if node.get("display_radius") is not None:
            title = f"{title}\n" + rf"[$\Pi_K^{{\lambda^\star}}$ shown at $r={node['display_radius']:g}$]"
        ax.set_title(title, fontsize=plt.rcParams["font.size"] * 0.9)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("Density" if not y_log else "Density (log scale)")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(frameon=False, fontsize=plt.rcParams["font.size"] * 0.8)

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)
