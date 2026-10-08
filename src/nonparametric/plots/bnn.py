import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb


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

# Flat fill for the column marginal (percentage) bars: the active blue-mint
# swatch from that same color_palette ("#ADEBDC").
BNN_BAR_COLOR = "#ADEBDC"


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
    value_label: str = "Estimated per-parameter sensitivity",
) -> None:
    """
    One heatmap per 2D weight tensor (output units x input units/features).
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": float(_deep_get(plot_cfg, "plot.font.size", 14)) * 1.6,
        "font.family": _deep_get(plot_cfg, "plot.font.family", "serif"),
        "text.usetex": bool(_deep_get(plot_cfg, "plot.font.use_tex", False)),
        "text.latex.preamble": r"\usepackage{amsmath}\providecommand{\FD}{\mathrm{FD}}",
    })
    fig_w = float(_deep_get(plot_cfg, "plot.figure.size.width", 6.0))
    fig_h = float(_deep_get(plot_cfg, "plot.figure.size.height", 4.0))
    fig_dpi = int(_deep_get(plot_cfg, "plot.figure.dpi", 150))

    n = len(tensors)
    # Marginal-bar row is 1.5x the colorbar's original thickness; the
    # colorbar keeps that original thickness via a sub-gridspec (see below).
    marginal_ratio = 1.5
    fig = plt.figure(figsize=(n * fig_w * 1.0, fig_h * 1.4), dpi=fig_dpi)
    gs_outer = fig.add_gridspec(2, n, height_ratios=[4, marginal_ratio], hspace=0.5, wspace=0.3)

    matrices = [np.asarray(t["matrix"], dtype=float) for t in tensors]
    vmin = min(M.min() for M in matrices)
    vmax = max(M.max() for M in matrices)

    im = None
    no_marginal_idxs = []
    for i, t in enumerate(tensors):
        M = matrices[i]
        n_rows, n_cols = M.shape
        col_label = str(t.get("col_label", "column"))
        show_col_marginal = bool(t.get("show_col_marginal", True))
        ax_main = fig.add_subplot(gs_outer[0, i])
        ax_col = fig.add_subplot(gs_outer[1, i], sharex=ax_main) if show_col_marginal else None

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
            # Share (%) of the tensor's total sensitivity attributable to each
            # input column: 100 * sum_i S_ij / sum_{i,j'} S_ij'.
            col_sums = M.sum(axis=0)
            col_pct = 100.0 * col_sums / col_sums.sum()
            ax_col.bar(np.arange(n_cols), col_pct, color=BNN_BAR_COLOR, alpha=1.0)
            ax_col.set_ylabel(r"$\%$")
            if t.get("col_names"):
                ax_col.set_xticks(np.arange(n_cols))
                # Bold the name of the largest-share column. Baked into the
                # label string (\textbf under usetex, where fontweight is
                # ignored) since tick Text objects are regenerated at draw time.
                top_col = int(np.argmax(col_pct))
                usetex = plt.rcParams["text.usetex"]
                col_tick_names = [
                    rf"\textbf{{{name}}}" if (usetex and c == top_col) else str(name)
                    for c, name in enumerate(t["col_names"])
                ]
                col_tick_labels = ax_col.set_xticklabels(
                    col_tick_names, rotation=90, fontsize=plt.rcParams["font.size"] * 0.7
                )
                col_tick_labels[top_col].set_fontweight("bold")
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
            no_marginal_idxs.append(i)

        for ax in (ax_main, ax_col):
            if ax is not None:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

    if no_marginal_idxs:
        # One shared horizontal colorbar spanning all panels without their own
        # marginal bar (e.g. L2, L4), at the same row as the L0 marginal bar.
        gs_cbar = gs_outer[1, no_marginal_idxs[0]:no_marginal_idxs[-1] + 1].subgridspec(
            2, 1, height_ratios=[1, marginal_ratio - 1], hspace=0
        )
        cax = fig.add_subplot(gs_cbar[0])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(value_label, fontsize=plt.rcParams["font.size"] * 0.9)
        cax.spines["top"].set_visible(False)
        cax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)


