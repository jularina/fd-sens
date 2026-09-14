import glob
import hashlib
import json
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import OmegaConf

from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.utils.files_operations import save_to_serializable_json, load_results_json, load_plot_config
from src.optimization.bnn_node_sensitivity import compute_group_omega_max, compute_node_lambda_star
from src.plots.paper.bnn_paper_funcs import (
    plot_bnn_node_candidate_priors,
    plot_bnn_layer_sensitivity,
    plot_bnn_weight_heatmaps,
)

warnings.filterwarnings("ignore", category=UserWarning)

# net.module.{0,2,4} are this BNN's 3 linear layers; each layer's mean
# sensitivity pools its weight_prior and bias_prior nodes together.
LAYER_GROUP_MAP: Dict[str, Tuple[str, str]] = {
    "Layer 0": ("net.module.0.weight_prior", "net.module.0.bias_prior"),
    "Layer 2": ("net.module.2.weight_prior", "net.module.2.bias_prior"),
    "Layer 4": ("net.module.4.weight_prior", "net.module.4.bias_prior"),
}

# Column order for the layer-sensitivity table matches the paper's appendix.
TABLE_DATASETS: Tuple[str, ...] = ("concrete", "yacht", "boston", "energy", "naval")
TABLE_DATASET_LABELS: Dict[str, str] = {
    "concrete": "Concrete", "yacht": "Yacht", "boston": "Boston", "energy": "Energy", "naval": "Naval",
}
TABLE_PRIORS: Tuple[str, ...] = ("gaussian", "studentt")
TABLE_PRIOR_LABELS: Dict[str, str] = {"gaussian": "Gaussian", "studentt": "Student-$t$"}

CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "paper", "real"))


def _project_root() -> str:
    """
    Repo root, resolved the same way whether this runs as a Hydra job (where
    get_original_cwd() is the authority) or as a plain function call outside
    any Hydra app context (e.g. looped over many configs in one process,
    where get_original_cwd() would raise).
    """
    try:
        return get_original_cwd()
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _config_tag(dataset: str, prior: str) -> str:
    return f"{dataset}_{prior}"


def _basis_config_hash(basis_type: str, basis_kwargs: Dict[str, Any]) -> str:
    """
    Short, stable fingerprint of the basis config (type + kwargs, e.g. nu,
    num_basis_functions, method) that actually determines the computed
    omega_max/sensitivity -- folded into the sensitivity-cache tag so that
    changing e.g. `nu` in a config invalidates the old cache instead of
    silently reusing it (the cache used to be keyed only on dataset+prior).
    """
    payload = json.dumps({"type": basis_type, "kwargs": basis_kwargs}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


# Full-fidelity cache of compute_bnn_group_sensitivities' output (per-node
# omega_max arrays + the prior/center-prior draws used to build each group's
# basis), as opposed to the lightweight per-group summary JSON saved under
# cfg.flags.results.output_dir (which drops the per-node arrays -- see
# _run_bnn_uci_node_sensitivity_core). Letting this be found and reloaded
# lets a rerun skip straight to plotting instead of repeating the slow
# per-node optimisation.
SENSITIVITY_CACHE_DIR = "data/bnn"
_CACHE_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def _sensitivity_cache_glob(tag: str) -> str:
    return os.path.join(_project_root(), SENSITIVITY_CACHE_DIR, f"node_sensitivity_cache_{tag}_*.json")


def _find_latest_sensitivity_cache(tag: str) -> str | None:
    matches = sorted(glob.glob(_sensitivity_cache_glob(tag)))
    return matches[-1] if matches else None


def _new_sensitivity_cache_path(tag: str) -> str:
    timestamp = datetime.now().strftime(_CACHE_TIMESTAMP_FMT)
    return os.path.join(
        _project_root(), SENSITIVITY_CACHE_DIR, f"node_sensitivity_cache_{tag}_{timestamp}.json"
    )


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
            "row_label": "out",
            "col_label": "input feature",
            "col_names": list(feature_names) if feature_names else None,
        },
        "net.module.2.weight_prior": {
            "row_label": "out",
            "col_label": "in",
            "col_names": None,
            "show_col_marginal": False,
        },
        "net.module.4.weight_prior": {
            "row_label": "output",
            "col_label": "in",
            "col_names": None,
            "show_col_marginal": False,
        },
    }


def compute_bnn_group_sensitivities(cfg) -> Tuple[Any, Dict[str, Dict[str, Any]], float, int, float, Dict, Dict]:
    """
    Core per-group FD sensitivity computation shared by
    run_bnn_uci_node_sensitivity (single dataset/prior, with plots) and
    run_bnn_uci_layer_sensitivity_table (all datasets/priors, aggregated into
    a summary table): builds the loader, draws the (FD-estimation,
    centre-selection) prior samples for every param group, and computes each
    group's per-node omega_max/sensitivity.

    If a cached run for this dataset/prior tag already exists under
    data/bnn/ (see SENSITIVITY_CACHE_DIR), the slow per-node optimisation is
    skipped and its full per-node results (omega_max + the prior/center
    draws used to build each group's basis) are loaded instead -- this is
    what lets a rerun go straight to plotting. Otherwise the sensitivities
    are computed as before and a new, timestamped cache file is written.

    Returns (loader, group_results, radius, J, r_j, prior_samples_cache,
    center_prior_samples_cache).
    """
    loader = instantiate(cfg.model, data_config=cfg.data)
    J = loader.total_nodes
    r_j = float(cfg.sensitivity.r_j)
    radius = r_j * J
    base_tag = _config_tag(cfg.data.get("dataset", "uci"), cfg.data.get("reference_prior", "gaussian"))

    # basis_cls/basis_kwargs (e.g. nu, num_basis_functions) determine the
    # computed omega_max, so they must be resolved *before* the cache check
    # and folded into its key -- otherwise changing e.g. nu and rerunning
    # would silently hit the old cache and skip recomputation entirely.
    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    tag = f"{base_tag}_b{_basis_config_hash(cfg.optimize.nonparametric.basis_funcs_type, basis_kwargs)}"

    cache_path = _find_latest_sensitivity_cache(tag)
    if cache_path is not None:
        print(f"Found existing sensitivity cache at {cache_path}, skipping computation.")
        cached = load_results_json(cache_path)
        group_results = {}
        prior_samples_cache = {}
        center_prior_samples_cache = {}
        for group_name, res in cached["groups"].items():
            omega_max = np.asarray(res["omega_max"], dtype=float)
            sensitivity = r_j * omega_max
            group_results[group_name] = {
                "loc": res["loc"],
                "scale": res["scale"],
                "n_nodes": res["n_nodes"],
                "shape": res["shape"],
                "omega_max": omega_max,
                "sensitivity": sensitivity,
                "total_sensitivity": float(sensitivity.sum()),
                "mean_omega_max": float(omega_max.mean()),
                "max_omega_max": float(omega_max.max()),
            }
            prior_samples_cache[group_name] = np.asarray(res["prior_samples"], dtype=float)
            center_prior_samples_cache[group_name] = np.asarray(res["center_prior_samples"], dtype=float)
        return loader, group_results, radius, J, r_j, prior_samples_cache, center_prior_samples_cache

    node_chunk_size = int(cfg.sensitivity.get("node_chunk_size", 1024))
    center_samples_num = int(cfg.data.get("center_prior_samples_num", 5000))
    print(f"Total scalar nodes J = {J}. Uniform per-node radius r_j = {r_j:.6g} (global r = r_j*J = {radius:.6g}).")

    group_results = {}
    prior_samples_cache = {}
    center_prior_samples_cache = {}
    start = time.perf_counter()
    for group_name in loader.param_groups:
        group_start = time.perf_counter()
        g = loader.groups[group_name]
        prior_samples = loader.sample_prior(group_name)
        prior_samples_cache[group_name] = prior_samples

        # Fresh, independent prior draw used only to pick basis centres via
        # kmeans -- never the same samples used above to estimate the FD
        # constraint matrix A_c (mirrors the toy model's centers_pool_samples
        # pattern in run_gaussian_priors_nonparametric_diff_radii).
        center_prior_samples = loader.sample_prior(group_name, n_samples=center_samples_num)
        center_prior_samples_cache[group_name] = center_prior_samples

        omega_max = compute_group_omega_max(
            posterior_samples=g["posterior"],
            loc=g["loc"],
            scale=g["scale"],
            prior_samples=prior_samples,
            basis_cls=basis_cls,
            basis_kwargs=basis_kwargs,
            node_chunk_size=node_chunk_size,
            center_prior_samples=center_prior_samples,
        )
        sensitivity = r_j * omega_max
        group_elapsed = time.perf_counter() - group_start

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
        print(f"  {group_name}: optimisation time={group_elapsed:.3f}s")
    elapsed = time.perf_counter() - start
    print(f"Per-node sensitivity computation time: {elapsed:.3f}s")

    new_cache_path = _new_sensitivity_cache_path(tag)
    save_to_serializable_json(
        {
            "radius": radius,
            "J": J,
            "r_j": r_j,
            "groups": {
                name: {
                    "loc": res["loc"],
                    "scale": res["scale"],
                    "n_nodes": res["n_nodes"],
                    "shape": res["shape"],
                    "omega_max": res["omega_max"],
                    "prior_samples": prior_samples_cache[name],
                    "center_prior_samples": center_prior_samples_cache[name],
                }
                for name, res in group_results.items()
            },
        },
        new_cache_path,
    )
    print(f"Saved sensitivity cache to {new_cache_path}")

    return loader, group_results, radius, J, r_j, prior_samples_cache, center_prior_samples_cache


def _run_bnn_uci_node_sensitivity_core(cfg) -> None:
    """
    Core of run_bnn_uci_node_sensitivity, factored out as a plain function so
    it can also be called directly (outside a Hydra job context) when looping
    over many configs, e.g. from run_bnn_uci_all_datasets_sensitivity. Uses
    _project_root() instead of get_original_cwd() so it works either way.
    """
    core_start = time.perf_counter()
    tensor_axis_meta = _build_tensor_axis_meta(cfg.data.get("feature_names"))
    tag = f"{cfg.data.get('dataset', 'uci')}_{cfg.data.get('reference_prior', 'gaussian')}"
    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)

    start = time.perf_counter()
    loader, group_results, radius, J, r_j, prior_samples_cache, center_prior_samples_cache = (
        compute_bnn_group_sensitivities(cfg)
    )
    total =  time.perf_counter() - start
    print(f"Total optimisation time: {total:.3f}s")

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
            "show_col_marginal": meta.get("show_col_marginal", True),
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

    results_dir = os.path.join(_project_root(), cfg.flags.results.output_dir)
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

    plot_config_path = os.path.join(_project_root(), "configs/plots/overleaf_plots_settings.yaml")
    plot_cfg = load_plot_config(plot_config_path)
    output_dir = os.path.join(_project_root(), cfg.flags.plots.output_dir)
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
    print("A_c diagnostics for top nodes (checks whether a large omega_max is genuine "
          "or an artifact of a near-singular A_c -- i.e. a direction the prior samples "
          "barely explore):")
    for group_name, idx, sens, omega in all_records[:top_k_plot]:
        g = loader.groups[group_name]
        lambda_star, omega_check, basis, diag = compute_node_lambda_star(
            posterior_samples_col=g["posterior"][:, idx],
            loc=g["loc"],
            scale=g["scale"],
            prior_samples=prior_samples_cache[group_name],
            basis_cls=basis_cls,
            basis_kwargs=basis_kwargs,
            radius_j=display_radius,
            center_prior_samples=center_prior_samples_cache[group_name],
        )
        short_name = group_name.replace("net.module.", "L").replace("_prior", "")
        print(
            f"  {short_name}[{idx}]: sensitivity={sens:.6g}, omega_max={omega_check:.4g}, "
            f"A_c min_eig={diag['ac_min_eig']:.3e}, max_eig={diag['ac_max_eig']:.3e}, "
            f"cond={diag['ac_cond']:.3e}, rank_kept={diag['ac_rank_kept']}/{diag['ac_rank_total']}, "
            f"truncated={diag['ac_truncated']}"
        )
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

    # Linear-scale density plots can hide tail-only divergence (the exact
    # region a large omega_max -- driven by a near-singular A_c direction the
    # prior barely explores -- would show up in); re-render the top-3 most
    # sensitive nodes on a log y-axis to check whether they diverge there.
    top3_log_dir = os.path.join(output_dir, "top3_log_scale")
    plot_bnn_node_candidate_priors(
        nodes=candidate_nodes[:3],
        plot_cfg=plot_cfg,
        output_dir=top3_log_dir,
        filename=f"bnn_node_candidate_priors_{tag}_top3_logscale.pdf",
        y_log=True,
        save_individual=False,
    )

    core_elapsed = time.perf_counter() - core_start
    print(f"Total run_bnn_uci_node_sensitivity time: {core_elapsed:.3f}s")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="bnn_boston_nonparam_gaussian")
def run_bnn_uci_node_sensitivity(cfg) -> None:
    """
    Per-node nonparametric FD sensitivity analysis for the 3-layer BNN of
    Fortuin et al. (2022), posterior-sampled on a UCI regression dataset.
    """
    _run_bnn_uci_node_sensitivity_core(cfg)


def run_bnn_uci_all_datasets_sensitivity() -> None:
    """
    Runs _run_bnn_uci_node_sensitivity_core for all 5 UCI datasets under both
    reference-prior families (10 configs: configs/paper/real/
    bnn_{dataset}_nonparam_{gaussian,studentt}.yaml), each saving its own
    bnn_node_sensitivity_{dataset}_{prior}.json (via the same
    save_to_serializable_json call as a single-config run). This is the slow,
    run-once-per-change step; run_bnn_uci_layer_sensitivity_table only reads
    the JSONs it leaves behind, so it stays cheap however often it's re-run.
    """
    for dataset in TABLE_DATASETS:
        for prior in TABLE_PRIORS:
            config_path = os.path.join(CONFIGS_DIR, f"bnn_{dataset}_nonparam_{prior}.yaml")
            cfg = OmegaConf.load(config_path)
            print(f"=== {TABLE_DATASET_LABELS[dataset]} / {TABLE_PRIOR_LABELS[prior]} ===")
            _run_bnn_uci_node_sensitivity_core(cfg)


def _layer_mean_sensitivities(group_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Per-layer mean sensitivity: pools a layer's weight_prior and bias_prior
    nodes together (mean = combined total_sensitivity / combined n_nodes,
    equivalent to the mean over the concatenated per-node sensitivity arrays
    since every node in the network shares the same r_j).
    """
    layer_means = {}
    for layer_label, (weight_group, bias_group) in LAYER_GROUP_MAP.items():
        total_sensitivity = (
            group_results[weight_group]["total_sensitivity"] + group_results[bias_group]["total_sensitivity"]
        )
        n_nodes = group_results[weight_group]["n_nodes"] + group_results[bias_group]["n_nodes"]
        layer_means[layer_label] = total_sensitivity / n_nodes
    return layer_means


def _render_layer_sensitivity_latex(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """
    Renders `results[dataset][prior][layer] -> mean sensitivity` as the
    layer-by-dataset-by-reference-prior LaTeX table used in the paper's
    appendix. Within each (dataset, prior) column, the layer with the
    highest mean sensitivity is bolded.
    """
    layer_labels = list(LAYER_GROUP_MAP.keys())

    def _cell(dataset: str, prior: str, layer: str) -> str:
        value = results[dataset][prior][layer]
        top_layer = max(layer_labels, key=lambda l: results[dataset][prior][l])
        text = f"{value:.2f}"
        return rf"\textbf{{{text}}}" if layer == top_layer else text

    dataset_header = "\n& ".join(
        rf"\multicolumn{{2}}{{c{'' if dataset == TABLE_DATASETS[-1] else '|'}}}"
        rf"{{\textbf{{{TABLE_DATASET_LABELS[dataset]}}}}}"
        for dataset in TABLE_DATASETS
    )
    prior_header = "\n& ".join(
        f"{TABLE_PRIOR_LABELS[TABLE_PRIORS[0]]} & {TABLE_PRIOR_LABELS[TABLE_PRIORS[1]]}" for _ in TABLE_DATASETS
    )

    row_lines = []
    for layer in layer_labels:
        cells = "\n& ".join(
            f"{_cell(dataset, TABLE_PRIORS[0], layer)} & {_cell(dataset, TABLE_PRIORS[1], layer)}"
            for dataset in TABLE_DATASETS
        )
        row_lines.append(f"{layer}\n& {cells} \\\\")

    col_spec = "lcc" + "|cc" * (len(TABLE_DATASETS) - 1)
    rows = "\n\n".join(row_lines)

    return rf"""\begin{{table*}}[t]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{{col_spec}}}
\toprule
& {dataset_header} \\
\textbf{{Layer}}
& {prior_header} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
}}
\caption{{\textit{{BNN for UCI dataset.}} Layer-wise mean $\widehat{{S}}_m^{{\FD}}(\widehat{{\mathcal{{Q}}}}{{r,K,l}})$ for the UCI datasets under two $\trueprior$.}}
\label{{apptab:uci-sensitivity-per-layer}}
\end{{table*}}"""


def run_bnn_uci_layer_sensitivity_table(save_path: str = None) -> str:
    """
    Reads the per-dataset node_sensitivity_cache_{dataset}_{prior}_*.json
    files under data/bnn/ (see SENSITIVITY_CACHE_DIR / compute_bnn_group_
    sensitivities) -- the newest cache per dataset/prior tag is used. These
    are written the first time compute_bnn_group_sensitivities runs for a
    given config (e.g. via run_bnn_uci_node_sensitivity or
    run_bnn_uci_all_datasets_sensitivity), so this function does no
    sensitivity computation itself, only aggregation: it's cheap and safe to
    re-run as long as every dataset/prior has been computed at least once.

    Aggregates each dataset/prior's per-group omega_max into a mean
    sensitivity per network layer (pooling each layer's weight_prior and
    bias_prior nodes, sensitivity = r_j * omega_max), and renders the result
    as the LaTeX table used in the paper's appendix
    (apptab:uci-sensitivity-per-layer).
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in TABLE_DATASETS:
        results[dataset] = {}
        for prior in TABLE_PRIORS:
            tag = _config_tag(dataset, prior)
            cache_path = _find_latest_sensitivity_cache(tag)
            if cache_path is None:
                raise FileNotFoundError(
                    f"No sensitivity cache found for tag '{tag}' under "
                    f"{os.path.join(_project_root(), SENSITIVITY_CACHE_DIR)}/. Run "
                    f"run_bnn_uci_all_datasets_sensitivity() (or run_bnn_uci_node_sensitivity for "
                    f"bnn_{dataset}_nonparam_{prior}.yaml) first."
                )
            cached = load_results_json(cache_path)
            r_j = float(cached["r_j"])
            group_results = {
                group_name: {
                    "total_sensitivity": r_j * float(np.sum(res["omega_max"])),
                    "n_nodes": int(res["n_nodes"]),
                }
                for group_name, res in cached["groups"].items()
            }
            results[dataset][prior] = _layer_mean_sensitivities(group_results)

    latex = _render_layer_sensitivity_latex(results)
    print(latex)

    if save_path:
        with open(save_path, "w") as f:
            f.write(latex)
        print(f"Saved LaTeX table to: {save_path}")

    return latex


if __name__ == "__main__":
    run_bnn_uci_node_sensitivity()
    # run_bnn_uci_all_datasets_sensitivity()
    # run_bnn_uci_layer_sensitivity_table(save_path="outputs/paper/results/bnn/uci_layer_sensitivity_table.tex")
