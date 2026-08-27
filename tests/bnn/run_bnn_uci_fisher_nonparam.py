import os
import time
import warnings

import numpy as np
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import OmegaConf

from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.utils.files_operations import save_to_serializable_json, load_plot_config
from src.optimization.bnn_node_sensitivity import compute_group_omega_max, compute_node_lambda_star
from src.plots.paper.bnn_paper_funcs import (
    plot_bnn_node_candidate_priors,
    plot_bnn_layer_sensitivity,
    plot_bnn_weight_heatmaps,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _build_tensor_axis_meta(feature_names):
    """
    For each 2D weight tensor: what its rows/columns mean, and (if
    applicable) human-readable names for its columns -- used to detect
    within-layer tendencies (e.g. "is one input feature or one hidden unit
    consistently more sensitive than the rest of its layer?"). `feature_names`
    labels net.module.0's input axis and comes from the dataset's config
    (`data.feature_names`), so this works for any UCI dataset.
    """
    return {
        "net.module.0.weight_prior": {
            "row_label": "hidden unit (layer0 out)",
            "col_label": "input feature",
            "col_names": list(feature_names) if feature_names else None,
        },
        "net.module.2.weight_prior": {
            "row_label": "hidden unit (layer2 out)",
            "col_label": "hidden unit (layer0 out)",
            "col_names": None,
        },
        "net.module.4.weight_prior": {
            "row_label": "output",
            "col_label": "hidden unit (layer2 out)",
            "col_names": None,
        },
    }


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="bnn_boston_nonparam_studentt")
def run_bnn_uci_node_sensitivity(cfg) -> None:
    """
    Per-node nonparametric FD sensitivity analysis for the 3-layer BNN of
    Fortuin et al. (2022), posterior-sampled on a UCI regression dataset.
    """
    tensor_axis_meta = _build_tensor_axis_meta(cfg.data.get("feature_names"))
    tag = f"{cfg.data.get('dataset', 'uci')}_{cfg.data.get('reference_prior', 'gaussian')}"
    loader = instantiate(cfg.model, data_config=cfg.data)
    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    node_chunk_size = int(cfg.sensitivity.get("node_chunk_size", 1024))
    J = loader.total_nodes
    radius = float(cfg.sensitivity.radius)
    r_j = radius / J
    print(f"Total scalar nodes J = {J}. Uniform per-node radius r_j = r/J = {r_j:.6g}.")

    group_results = {}
    prior_samples_cache = {}
    start = time.perf_counter()
    for group_name in loader.param_groups:
        g = loader.groups[group_name]
        prior_samples = loader.sample_prior(group_name)
        prior_samples_cache[group_name] = prior_samples

        omega_max = compute_group_omega_max(
            posterior_samples=g["posterior"],
            loc=g["loc"],
            scale=g["scale"],
            prior_samples=prior_samples,
            basis_cls=basis_cls,
            basis_kwargs=basis_kwargs,
            node_chunk_size=node_chunk_size,
        )
        sensitivity = r_j * omega_max

        group_results[group_name] = {
            "loc": g["loc"],
            "scale": g["scale"],
            "n_nodes": g["n_nodes"],
            "shape": list(g["shape"]),
            "omega_max": omega_max,
            "sensitivity": sensitivity,
            "total_sensitivity": float(sensitivity.sum()),
            "mean_omega_max": float(omega_max.mean()),
            "max_omega_max": float(omega_max.max()),
        }
        print(
            f"{group_name}: n_nodes={g['n_nodes']}, loc={g['loc']:.4g}, scale={g['scale']:.4g}, "
            f"mean omega_max={omega_max.mean():.4f}, max omega_max={omega_max.max():.4f}, "
            f"total sensitivity={sensitivity.sum():.4f}"
        )
    elapsed = time.perf_counter() - start
    print(f"Per-node sensitivity computation time: {elapsed:.3f}s")

    total_sensitivity = float(sum(v["total_sensitivity"] for v in group_results.values()))
    print(f"Global FD sensitivity S^FD(Q_r) = {total_sensitivity:.4f} (r={radius}, J={J}).")

    top_k = int(cfg.sensitivity.get("top_k", 20))
    all_records = [
        (group_name, idx, float(sens), float(res["omega_max"][idx]))
        for group_name, res in group_results.items()
        for idx, sens in enumerate(res["sensitivity"])
    ]
    all_records.sort(key=lambda rec: rec[2], reverse=True)
    print(f"Top {top_k} most sensitive scalar nodes:")
    for group_name, idx, sens, omega in all_records[:top_k]:
        print(f"  {group_name}[{idx}]: sensitivity={sens:.6g}, omega_max={omega:.4f}")

    print("Within-layer tendencies (mean sensitivity r_j*omega_max by row/column):")
    row_col_summary = {}
    heatmap_tensors = []
    top_n_axis = 5
    for group_name, meta in tensor_axis_meta.items():
        res = group_results[group_name]
        shape = tuple(res["shape"])
        if len(shape) != 2:
            continue
        n_rows, n_cols = shape
        sensitivity_matrix = res["sensitivity"].reshape(n_rows, n_cols)
        row_means = sensitivity_matrix.mean(axis=1)
        col_means = sensitivity_matrix.mean(axis=0)
        col_names = meta.get("col_names")

        top_rows = np.argsort(-row_means)[:top_n_axis]
        top_cols = np.argsort(-col_means)[:top_n_axis]

        short_name = group_name.replace("net.module.", "L").replace("_prior", "").replace(".weight", "")
        row_desc = ", ".join(f"{r}({row_means[r]:.4g})" for r in top_rows)
        col_desc = ", ".join(
            f"{(col_names[c] if col_names else c)}({col_means[c]:.4g})" for c in top_cols
        )
        print(f"  {short_name}: top by {meta['row_label']}: {row_desc}")
        print(f"  {short_name}: top by {meta['col_label']}: {col_desc}")

        row_col_summary[group_name] = {
            "row_label": meta["row_label"],
            "col_label": meta["col_label"],
            "row_means": row_means,
            "col_means": col_means,
            "top_rows": [{"index": int(r), "mean_sensitivity": float(row_means[r])} for r in top_rows],
            "top_cols": [
                {
                    "index": int(c),
                    "name": (col_names[c] if col_names else None),
                    "mean_sensitivity": float(col_means[c]),
                }
                for c in top_cols
            ],
        }
        heatmap_tensors.append({
            "label": f"{short_name}",
            "matrix": sensitivity_matrix,
            "row_label": meta["row_label"],
            "col_label": meta["col_label"],
            "col_names": col_names,
        })

    def _hub_overlap(rows_group, cols_group):
        if rows_group not in row_col_summary or cols_group not in row_col_summary:
            return None
        rows = {r["index"] for r in row_col_summary[rows_group]["top_rows"]}
        cols = {c["index"] for c in row_col_summary[cols_group]["top_cols"]}
        return sorted(rows & cols)

    overlap_h0 = _hub_overlap("net.module.0.weight_prior", "net.module.2.weight_prior")
    if overlap_h0 is not None:
        print(
            f"  Layer0-output units in top-{top_n_axis} both as input encoders (L0 rows) "
            f"and as layer2 receivers (L2 cols): {overlap_h0 if overlap_h0 else 'none'}"
        )
    overlap_h2 = _hub_overlap("net.module.2.weight_prior", "net.module.4.weight_prior")
    if overlap_h2 is not None:
        print(
            f"  Layer2-output units in top-{top_n_axis} both as layer2 encoders (L2 rows) "
            f"and as output decoders (L4 cols): {overlap_h2 if overlap_h2 else 'none'}"
        )

    results_dir = os.path.join(get_original_cwd(), cfg.flags.results.output_dir)
    save_to_serializable_json(
        {
            "radius": radius,
            "J": J,
            "r_j": r_j,
            "total_sensitivity": total_sensitivity,
            "groups": {
                name: {k: v for k, v in res.items() if k not in ("omega_max", "sensitivity")}
                for name, res in group_results.items()
            },
            "top_k": [
                {"group": g, "index": i, "sensitivity": s, "omega_max": o}
                for g, i, s, o in all_records[:top_k]
            ],
            "row_col_summary": row_col_summary,
        },
        os.path.join(results_dir, f"bnn_node_sensitivity_{tag}.json"),
    )

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    plot_cfg = load_plot_config(plot_config_path)
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_bnn_layer_sensitivity(
        group_results=group_results,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"bnn_layer_sensitivity_{tag}.pdf",
    )
    if heatmap_tensors:
        plot_bnn_weight_heatmaps(
            tensors=heatmap_tensors,
            plot_cfg=plot_cfg,
            output_dir=output_dir,
            filename=f"bnn_weight_heatmaps_{tag}.pdf",
        )

    top_k_plot = int(cfg.sensitivity.get("top_k_plot", 6))
    display_radius = float(cfg.sensitivity.get("candidate_display_radius", r_j))
    candidate_nodes = []
    for group_name, idx, sens, omega in all_records[:top_k_plot]:
        g = loader.groups[group_name]
        lambda_star, omega_check, basis = compute_node_lambda_star(
            posterior_samples_col=g["posterior"][:, idx],
            loc=g["loc"],
            scale=g["scale"],
            prior_samples=prior_samples_cache[group_name],
            basis_cls=basis_cls,
            basis_kwargs=basis_kwargs,
            radius_j=display_radius,
        )
        short_name = group_name.replace("net.module.", "L").replace("_prior", "")
        candidate_nodes.append({
            "label": f"{short_name}[{idx}]",
            "loc": g["loc"],
            "scale": g["scale"],
            "df": g.get("df"),
            "lambda_star": lambda_star,
            "basis": basis,
            "posterior_samples": g["posterior"][:, idx],
            "omega_max": omega_check,
            "sensitivity": sens,
            "display_radius": display_radius,
        })

    plot_bnn_node_candidate_priors(
        nodes=candidate_nodes,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"bnn_node_candidate_priors_{tag}.pdf",
    )


if __name__ == "__main__":
    run_bnn_uci_node_sensitivity()
