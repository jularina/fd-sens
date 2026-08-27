"""
Schematic ("potato diagram") figures illustrating the sieve construction of
Q_r^K (Eq. candidate_prior) and its Monte-Carlo approximation Q_r^{K,l}.

Figure 1 shows the nesting Q_r^{K_1} subset Q_r^{K_2} subset Q_r^{K_3} subset
... subset Q_r as concentric ('Russian doll') blobs sharing the same centre,
with a single black dot for Pi_ref shared by every set in the chain.

Figure 2 fixes K (not shown) and grows the number of Monte-Carlo samples l
used to estimate the FD constraint: nested rings around the true boundary
of Q_r^K -- representing the spread of the estimated constraint
\\hat{FD}_l(Pi_ref || Pi_K) across MC draws -- collapse onto the true
boundary as l grows.
"""
import os
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from src.utils.files_operations import load_plot_config

REPO_ROOT = Path(__file__).resolve().parents[4]


def _radial_profile(theta, harmonics, seed, mean=1.0):
    """Smooth, irregular ('potato') radius-vs-angle profile: a circle
    perturbed by a few low-frequency, randomly phased harmonics."""
    rng = np.random.default_rng(seed)
    wobble = np.ones_like(theta)
    for k, amp in harmonics:
        wobble += amp * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
    return mean * wobble / wobble.mean()


def _apply_rcparams(plot_cfg):
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })


def _shape_at(angle_rad, theta_grid, shape):
    """Linearly interpolate the (2*pi-periodic) radial profile at arbitrary angles."""
    theta_ext = np.concatenate([theta_grid, [theta_grid[0] + 2 * np.pi]])
    shape_ext = np.concatenate([shape, [shape[0]]])
    return np.interp(np.mod(angle_rad, 2 * np.pi), theta_ext, shape_ext)


def _closed(x, y):
    """Append the first vertex to the end so ax.plot draws a fully closed loop.

    theta runs over [0, 2*pi) with endpoint=False, so consecutive-point lines
    (unlike ax.fill, which closes polygons automatically) leave the last point
    unconnected to the first -- a small gap right at theta=0 (the positive
    x-axis, i.e. the right side of every one of these potato shapes)."""
    return np.append(x, x[0]), np.append(y, y[0])


def _blend_over_white(color, alpha):
    """Flatten (color, alpha) to an opaque RGB as it would look painted over a
    white background -- used so each nested layer's *own* alpha determines its
    look, instead of alpha-compositing on top of whatever was already painted
    underneath it (which would make a low-alpha inner layer look darker than
    intended, since it lets an already-dark base show through)."""
    r, g, b = mcolors.to_rgb(color)
    return (1 - alpha) + alpha * r, (1 - alpha) + alpha * g, (1 - alpha) + alpha * b


def _widehat_l_label(l, all_l_values, K_label="K"):
    """\\widehat{Q}_r^{K,l_i} with l_i's subscript rank taken from l's position
    among all_l_values sorted ascending (l_1 = fewest MC samples, etc.)."""
    rank = sorted(all_l_values).index(l) + 1
    return rf"\widehat{{\mathcal{{Q}}}}_r^{{{K_label},l_{{{rank}}}}}"


def plot_sieve_nested_neighbourhoods(
    plot_cfg,
    output_dir,
    K_labels=(r"\mathcal{Q}_r^{K_1}", r"\mathcal{Q}_r^{K_2}", r"\mathcal{Q}_r^{K_3}"),
    radius_scales=(0.32, 0.56, 0.80),
    outer_radius=1.0,
    outer_label=r"\mathcal{Q}_r",
    ref_label=r"\Pi_{\mathrm{ref}}",
    label_angles_deg=(160.0, 230.0, 10.0, 280.0),
    ref_label_angle_deg=245.0,
):
    """Single-panel 'Russian doll' diagram of the sieve
    Q_r^{K_1} subset ... subset Q_r^{K_last} subset Q_r.

    All boundaries share one fixed angular shape (only the radius scale
    differs), so the regions are concentric and containment is exact by
    construction; Pi_ref is drawn as a single black dot common to every set.
    Each label sits at its own angle, in the annular gap just outside its
    own boundary, so the labels never stack on top of one another.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = list(plot_cfg.plot.color_palette.colors)
    _apply_rcparams(plot_cfg)

    theta = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
    shape = _radial_profile(theta, harmonics=((1, 0.18), (2, 0.12), (3, 0.08), (5, 0.05)), seed=0, mean=1.0)

    scales = list(radius_scales) + [outer_radius]
    labels = list(K_labels) + [outer_label]
    fill_colors_by_rank = [colors[6], colors[7], colors[8], colors[9]]

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # draw largest-first so each smaller, more saturated region is layered on top,
    # leaving a visible annulus for every set in the chain
    order = sorted(range(len(scales)), key=lambda i: scales[i], reverse=True)
    for rank, i in enumerate(order):
        r = scales[i] * shape
        x, y = r * np.cos(theta), r * np.sin(theta)
        is_outer = (i == len(scales) - 1)
        ax.fill(x, y, color=fill_colors_by_rank[rank], alpha=0.45, linewidth=0)
        ax.plot(*_closed(x, y), color="black", linewidth=1.0, linestyle="--" if is_outer else "-")

    label_fontsize = plot_cfg.plot.font.size * 0.78
    for i, angle_deg in enumerate(label_angles_deg):
        angle_rad = np.deg2rad(angle_deg)
        own_r = scales[i] * _shape_at(angle_rad, theta, shape)
        if i == len(scales) - 1:
            label_r = own_r * 1.12  # just outside the outermost boundary
        else:
            next_r = scales[i + 1] * _shape_at(angle_rad, theta, shape)
            label_r = 0.5 * (own_r + next_r)  # in the gap to the next set out
        ax.text(
            label_r * np.cos(angle_rad), label_r * np.sin(angle_rad), f"${labels[i]}$",
            ha="center", va="center", fontsize=label_fontsize,
        )

    ax.scatter([0], [0], color="black", s=14, zorder=10)
    ref_angle_rad = np.deg2rad(ref_label_angle_deg)
    ref_r = 0.35 * scales[0] * _shape_at(ref_angle_rad, theta, shape)
    ax.text(
        ref_r * np.cos(ref_angle_rad), ref_r * np.sin(ref_angle_rad), f"${ref_label}$",
        ha="center", va="center", fontsize=label_fontsize, zorder=10,
    )

    # no aspect='equal': let the (otherwise circular) potato shapes stretch to
    # fill the configured width x height rectangle exactly, instead of a
    # square aspect ratio that a subsequent tight bbox crop would collapse to
    ax.set_xlim(-outer_radius * 1.3, outer_radius * 1.3)
    ax.set_ylim(-outer_radius * 1.3, outer_radius * 1.3)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    save_path = os.path.join(output_dir, "sieve_nested_neighbourhoods.pdf")
    fig.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"Saved nested sieve illustrative figure to: {save_path}")


def plot_mc_constraint_precision_over_l(
    plot_cfg,
    output_dir,
    l_values=(10, 40, 160),
    outer_radius=1.0,
    base_eps=0.85,
    true_label=r"\mathcal{Q}_r^{K}",
    label_angles_deg=(160.0, 230.0, 320.0, 280.0),
):
    """Single-panel diagram of the MC-approximated constraint set
    \\widehat{Q}_r^{K,l} shrinking onto the true Q_r^K as l grows.

    K is fixed (not shown); nested rings -- one per l, largest/lightest for
    the smallest l -- share the true boundary's angular shape (only the
    radius scale differs, so containment is exact by construction) and
    collapse onto the dashed true boundary as l -> infinity.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = list(plot_cfg.plot.color_palette.colors)
    _apply_rcparams(plot_cfg)

    theta = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
    shape = _radial_profile(theta, harmonics=((1, 0.22), (2, 0.14), (3, 0.09), (5, 0.05)), seed=0, mean=1.0)

    l_sorted = sorted(l_values, reverse=True)  # largest l -> smallest ring, i.e. ascending scale order
    l_ref = min(l_values)
    eps_values = [base_eps * np.sqrt(l_ref / l) for l in l_sorted]  # MC uncertainty shrinks like l^{-1/2}
    scales = [outer_radius] + [outer_radius * (1 + eps) for eps in eps_values]
    labels = [true_label] + [_widehat_l_label(l, l_values) for l in l_sorted]
    fill_colors_by_rank = [colors[6], colors[7], colors[8], colors[9]]

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # draw largest-first (loosest, l small) so each smaller, more precise
    # ring (larger l) is layered on top, leaving a visible annulus per l
    order = sorted(range(len(scales)), key=lambda i: scales[i], reverse=True)
    for rank, i in enumerate(order):
        r = scales[i] * shape
        x, y = r * np.cos(theta), r * np.sin(theta)
        is_true = (i == 0)
        ax.fill(x, y, color=fill_colors_by_rank[rank], alpha=0.45, linewidth=0)
        ax.plot(*_closed(x, y), color="black", linewidth=1.0, linestyle="--" if is_true else "-")

    label_fontsize = plot_cfg.plot.font.size * 0.66
    for i, angle_deg in enumerate(label_angles_deg[:len(scales)]):
        angle_rad = np.deg2rad(angle_deg)
        own_r = scales[i] * _shape_at(angle_rad, theta, shape)
        if i == len(scales) - 1:
            label_r = own_r * 1.12  # just outside the outermost (smallest-l) ring
        else:
            next_r = scales[i + 1] * _shape_at(angle_rad, theta, shape)
            label_r = 0.5 * (own_r + next_r)  # in the gap to the next ring out
        ax.text(
            label_r * np.cos(angle_rad), label_r * np.sin(angle_rad), f"${labels[i]}$",
            ha="center", va="center", fontsize=label_fontsize,
        )

    # no aspect='equal': let the (otherwise circular) rings stretch to fill
    # the configured width x height rectangle exactly, instead of a square
    # aspect ratio that a subsequent tight bbox crop would collapse to
    max_extent = max(scales) * shape.max() * 1.28
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    save_path = os.path.join(output_dir, "mc_constraint_precision_over_l.pdf")
    fig.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"Saved MC constraint precision (l) illustrative figure to: {save_path}")


def plot_sieve_and_mc_precision_combined(
    plot_cfg,
    output_dir,
    K_labels=(r"\mathcal{Q}_r^{K_1}", r"\mathcal{Q}_r^{K_2}", r"\mathcal{Q}_r^{K_3}"),
    radius_scales=(0.32, 0.58, 0.80),
    outer_radius=1.0,
    outer_label=r"\mathcal{Q}_r",
    ref_label=r"\Pi_{\mathrm{ref}}",
    K_label_angles_deg=(140.0, 140.0, 140.0, 140.0),
    K_label_text_inside_area=(0.8, 0.88, 0.9, 0.9),
    l_values=(40, 10),
    l_bundle_deviation=(0.35, 0.90),
    l_bundle_colors=("steelblue", "darkblue"),
    l_bundle_sizes=(7, 2),
    l_label_angles_deg=(10.0, 80.0),
    l_K_label="K_1",
):
    """Single combined diagram: the sieve chain
    Q_r^{K_1} subset Q_r^{K_2} subset Q_r^{K_3} subset Q_r, with a couple of
    MC-estimation bundles \\widehat{Q}_r^{K_1,l}: each is itself a sieve-like
    blob (same size and irregular-potato character as Q_r^{K_1}, via the same
    harmonics but its own random phases) that genuinely overlaps Q_r^{K_1} --
    poking out where its own bumps exceed the true boundary, dipping inside
    elsewhere -- rather than being simply nested inside/around it or offset
    away from it. l_bundle_deviation controls how much each bundle's shape
    departs from the true one: small (more MC samples, e.g. l_2) means it
    nearly coincides with -- so overlaps a lot with -- Q_r^{K_1}; large (fewer
    samples, e.g. l_1) means a more independently-shaped blob that overlaps it
    only partially. The two bundles are not nested inside one another either.
    Both stories (growing K, shrinking MC error) share one picture and one
    Pi_ref dot.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = list(plot_cfg.plot.color_palette.colors)
    _apply_rcparams(plot_cfg)

    theta = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
    shape = _radial_profile(theta, harmonics=((1, 0.18), (2, 0.12), (3, 0.08), (5, 0.05)), seed=0, mean=1.0)

    scales = list(radius_scales) + [outer_radius]
    labels = list(K_labels) + [outer_label]
    # Q_r^{K_1}, Q_r^{K_2}, Q_r^{K_3} all share one color (colors[9]) and are told
    # apart by alpha alone -- higher alpha for the smaller, innermost sets; Q_r
    # keeps its own distinct color (colors[6])
    fill_colors_by_rank = [colors[6], colors[9], colors[9], colors[9]]
    fill_alphas_by_rank = [0.45, 0.70, 0.55, 0.35]
    # pre-blended to opaque RGB per rank so each layer's own alpha sets its look,
    # rather than compositing on top of the (already painted) layers beneath it
    fill_render_colors_by_rank = [_blend_over_white(c, a) for c, a in zip(fill_colors_by_rank, fill_alphas_by_rank)]

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # sieve chain: draw largest-first so each smaller, more saturated region
    # is layered on top, leaving a visible annulus for every set in the chain
    order = sorted(range(len(scales)), key=lambda i: scales[i], reverse=True)
    for rank, i in enumerate(order):
        r = scales[i] * shape
        x, y = r * np.cos(theta), r * np.sin(theta)
        is_outer = (i == len(scales) - 1)
        ax.fill(x, y, color=fill_render_colors_by_rank[rank], linewidth=0)
        if not is_outer:
            ax.plot(*_closed(x, y), color="black", linewidth=1.0)
        if is_outer:
            outer_x, outer_y = x, y

    # tracks every plotted point (outer boundary + label anchors) so the axis
    # limits can be set tight to the actual content in x and y independently
    # -- true full-bleed, rather than one shared radial margin that leaves
    # slack wherever the (irregular) shape is narrower than its own peak
    xs_all = list(outer_x)
    ys_all = list(outer_y)

    label_fontsize = plot_cfg.plot.font.size * 0.6
    for i, angle_deg in enumerate(K_label_angles_deg):
        angle_rad = np.deg2rad(angle_deg)
        own_r = scales[i] * _shape_at(angle_rad, theta, shape)
        label_r = own_r * K_label_text_inside_area[i]  # inside its own neighbourhood, hugging the border from within
        lx, ly = label_r * np.cos(angle_rad), label_r * np.sin(angle_rad)
        # Q_r^{K_1}, Q_r^{K_2}, Q_r^{K_3} text in dark blue; Q_r itself uses its own
        # (fully opaque) area color instead of the faded alpha=0.45 fill
        text_color = colors[9] if i < len(scales) - 1 else fill_colors_by_rank[0]
        text_color = "black"
        ax.text(lx, ly, f"${labels[i]}$", ha="center", va="center",
                fontsize=label_fontsize, color=text_color, alpha=1.0)
        xs_all.append(lx)
        ys_all.append(ly)

    # MC-estimation bundles for K_1: each sample line in a bundle is its own
    # independent sieve-like shape (same harmonics as the true boundary, own
    # random phases via a fresh seed per line) blended with the true shape by
    # l_bundle_deviation[i] -- a small deviation nearly reproduces the true
    # boundary (lots of overlap); a large one is a genuinely different blob of
    # the same size that only partially overlaps Q_r^{K_1}.
    true_harmonics = ((1, 0.18), (2, 0.12), (3, 0.08), (5, 0.05))
    l_order = sorted(range(len(l_bundle_deviation)), key=lambda i: l_bundle_deviation[i], reverse=True)
    for i in l_order:
        for k in range(l_bundle_sizes[i]):
            own_shape = _radial_profile(theta, harmonics=true_harmonics, seed=1000 * i + k, mean=1.0)
            blended_shape = (1 - l_bundle_deviation[i]) * shape + l_bundle_deviation[i] * own_shape
            r = radius_scales[0] * blended_shape
            x, y = r * np.cos(theta), r * np.sin(theta)
            ax.plot(*_closed(x, y), color=l_bundle_colors[i], linewidth=0.4, alpha=0.75)

    l_label_fontsize = plot_cfg.plot.font.size * 0.58
    l_labels = [_widehat_l_label(l, l_values, K_label=l_K_label) for l in l_values]
    for i, angle_deg in enumerate(l_label_angles_deg):
        angle_rad = np.deg2rad(angle_deg)
        own_r = radius_scales[0] * _shape_at(angle_rad, theta, shape)
        label_r = own_r * (1 + 0.35 * l_bundle_deviation[i] + 0.05)  # just past this bundle's own spread
        lx, ly = label_r * np.cos(angle_rad), label_r * np.sin(angle_rad)
        ax.text(
            lx, ly, f"${l_labels[i]}$", ha="center", va="center", fontsize=l_label_fontsize, color=l_bundle_colors[i],
        )
        xs_all.append(lx)
        ys_all.append(ly)

    ax.scatter([0], [0], color="black", s=14, zorder=10)
    ax.text(
        -0.04 * outer_radius, -0.06 * outer_radius, f"${ref_label}$",
        ha="right", va="top", fontsize=label_fontsize, zorder=10,
    )

    # no aspect='equal': let the (otherwise circular) potato shapes stretch to
    # fill the configured width x height rectangle exactly. Limits are fit
    # tightly to the actual content bounding box (outer boundary + every
    # label anchor), independently in x and y, plus a small buffer for each
    # label's own text extent -- true full-bleed, like a tight_layout crop,
    # rather than one shared radial margin.
    xs_all, ys_all = np.array(xs_all), np.array(ys_all)
    x_pad = 0.04 * (xs_all.max() - xs_all.min())
    y_pad = 0.06 * (ys_all.max() - ys_all.min())
    ax.set_xlim(xs_all.min() - x_pad, xs_all.max() + x_pad)
    ax.set_ylim(ys_all.min() - y_pad, ys_all.max() + y_pad)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    save_path = os.path.join(output_dir, "sieve_and_mc.pdf")
    fig.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"Saved combined sieve + MC precision illustrative figure to: {save_path}")


if __name__ == "__main__":
    plot_config_path = os.path.join(REPO_ROOT, "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(REPO_ROOT, "outputs/paper/plots/illustrative/nonparam")
    plot_cfg = load_plot_config(plot_config_path)

    plot_sieve_nested_neighbourhoods(plot_cfg=plot_cfg, output_dir=output_dir)
    plot_mc_constraint_precision_over_l(plot_cfg=plot_cfg, output_dir=output_dir)
    plot_sieve_and_mc_precision_combined(plot_cfg=plot_cfg, output_dir=output_dir)
