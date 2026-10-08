import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from src.common.utils.files_operations import load_plot_config


# ------------------------------------------------------------
# Shared helpers (mirror slide_1.py style)
# ------------------------------------------------------------
def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size":           plot_cfg.plot.font.size,
        "font.family":         plot_cfg.plot.font.family,
        "text.usetex":         plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{type1cm}",
    })


def _make_figure_3d(plot_cfg):
    fig = plt.figure(figsize=(
        plot_cfg.plot.figure.size.width,
        plot_cfg.plot.figure.size.height,
    ))
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def _save_fig(fig, output_dir, filename, plot_cfg):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _style_ax_3d(ax, plot_cfg):
    ax.set_xlabel(r"$\lambda_0$", labelpad=4)
    ax.set_ylabel(r"$\lambda_1$", labelpad=4)
    ax.set_zlabel("")
    # Place z-label in 2D axes coords to guarantee visibility in PDFs
    ax.text2D(-0.08, 0.5, r"$\lambda_2$", transform=ax.transAxes,
              rotation=90, ha="center", va="center",
              fontsize=plot_cfg.plot.font.size)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgrey")
    ax.yaxis.pane.set_edgecolor("lightgrey")
    ax.zaxis.pane.set_edgecolor("lightgrey")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.tick_params(axis="both", labelsize=plot_cfg.plot.font.size - 2)
    ax.tick_params(axis="z", pad=-3)


# ------------------------------------------------------------
# Plot: hyperrectangle  C = [-1, 1]^3
# Drawn as wireframe edges + ghost faces so the z-axis label
# is never occluded.
# ------------------------------------------------------------
def plot_hyperrectangle(plot_cfg, color, output_dir):
    fig, ax = _make_figure_3d(plot_cfg)

    # 12 edges of the unit cube [-1,1]^3
    edges = [
        # bottom face
        ([-1,-1,-1],[ 1,-1,-1]), ([ 1,-1,-1],[ 1, 1,-1]),
        ([ 1, 1,-1],[-1, 1,-1]), ([-1, 1,-1],[-1,-1,-1]),
        # top face
        ([-1,-1, 1],[ 1,-1, 1]), ([ 1,-1, 1],[ 1, 1, 1]),
        ([ 1, 1, 1],[-1, 1, 1]), ([-1, 1, 1],[-1,-1, 1]),
        # vertical pillars
        ([-1,-1,-1],[-1,-1, 1]), ([ 1,-1,-1],[ 1,-1, 1]),
        ([ 1, 1,-1],[ 1, 1, 1]), ([-1, 1,-1],[-1, 1, 1]),
    ]
    for p0, p1 in edges:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=color, linewidth=1.2)

    # Ghost faces (very low alpha so z-axis label is never hidden)
    faces = [
        [[-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1]],
        [[-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]],
        [[-1,-1,-1],[ 1,-1,-1],[ 1,-1, 1],[-1,-1, 1]],
        [[-1, 1,-1],[ 1, 1,-1],[ 1, 1, 1],[-1, 1, 1]],
        [[-1,-1,-1],[-1, 1,-1],[-1, 1, 1],[-1,-1, 1]],
        [[ 1,-1,-1],[ 1, 1,-1],[ 1, 1, 1],[ 1,-1, 1]],
    ]
    poly = Poly3DCollection(faces, alpha=0.06, facecolor=color, edgecolor="none")
    ax.add_collection3d(poly)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.4, 1.4)
    ax.view_init(elev=22, azim=225)
    _style_ax_3d(ax, plot_cfg)
    _save_fig(fig, output_dir, "slide_constraints_hyperrectangle.pdf", plot_cfg)


# ------------------------------------------------------------
# Plot: L1-ball  C = {lambda : ||lambda||_1 <= 1}  (octahedron)
# ------------------------------------------------------------
def plot_l1_ball(plot_cfg, color, output_dir):
    fig, ax = _make_figure_3d(plot_cfg)

    v = {
        "px": [ 1,  0,  0], "nx": [-1,  0,  0],
        "py": [ 0,  1,  0], "ny": [ 0, -1,  0],
        "pz": [ 0,  0,  1], "nz": [ 0,  0, -1],
    }
    faces = [
        [v["px"], v["py"], v["pz"]],
        [v["px"], v["ny"], v["pz"]],
        [v["nx"], v["py"], v["pz"]],
        [v["nx"], v["ny"], v["pz"]],
        [v["px"], v["py"], v["nz"]],
        [v["px"], v["ny"], v["nz"]],
        [v["nx"], v["py"], v["nz"]],
        [v["nx"], v["ny"], v["nz"]],
    ]
    poly = Poly3DCollection(faces, alpha=0.18, facecolor=color, edgecolor=color,
                            linewidth=0.6)
    ax.add_collection3d(poly)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.view_init(elev=20, azim=225)
    _style_ax_3d(ax, plot_cfg)
    _save_fig(fig, output_dir, "slide_constraints_l1_ball.pdf", plot_cfg)


# ------------------------------------------------------------
# Plot: probability simplex  C = {lambda >= 0 : 1^T lambda = 1}
# ------------------------------------------------------------
def plot_simplex(plot_cfg, color, output_dir):
    fig, ax = _make_figure_3d(plot_cfg)

    face = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    poly = Poly3DCollection(face, alpha=0.35, facecolor=color, edgecolor=color,
                            linewidth=1.5)
    ax.add_collection3d(poly)

    edges = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    ax.plot(edges[:, 0], edges[:, 1], edges[:, 2], color=color, linewidth=1.5)

    for xs, ys, zs in [([0, 1.2], [0, 0], [0, 0]),
                        ([0, 0], [0, 1.2], [0, 0]),
                        ([0, 0], [0, 0], [0, 1.2])]:
        ax.plot(xs, ys, zs, color="black", linewidth=0.5, linestyle=":")

    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.set_zlim(-0.1, 1.2)
    ax.view_init(elev=22, azim=40)
    _style_ax_3d(ax, plot_cfg)
    _save_fig(fig, output_dir, "slide_constraints_simplex.pdf", plot_cfg)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_config_path = os.path.join(project_root, "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(project_root, "outputs/presentation/slide_constraints")

    plot_cfg = load_plot_config(plot_config_path)
    _apply_plot_rc(plot_cfg)
    colors = list(plot_cfg.plot.color_palette.colors)

    plot_hyperrectangle(plot_cfg, colors[0], output_dir)
    plot_l1_ball(plot_cfg, colors[1], output_dir)
    plot_simplex(plot_cfg, colors[2], output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
