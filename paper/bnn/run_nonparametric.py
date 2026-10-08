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

from src.nonparametric.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.common.utils.files_operations import save_to_serializable_json, load_results_json, load_plot_config
from src.nonparametric.node_sensitivity import compute_group_omega_max
from src.nonparametric.plots.bnn import plot_bnn_weight_heatmaps

warnings.filterwarnings("ignore", category=UserWarning)

# net.module.{0,2,4} are the 3 linear layers; each layer pools its weight and bias prior nodes.
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
    """Return the repo root, whether or not running inside a Hydra job."""
    try:
        return get_original_cwd()
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _config_tag(dataset: str, prior: str) -> str:
    return f"{dataset}_{prior}"


def _basis_config_hash(basis_type: str, basis_kwargs: Dict[str, Any]) -> str:
    """Return a short stable hash of the basis config, used to key the sensitivity cache."""
    payload = json.dumps({"type": basis_type, "kwargs": basis_kwargs}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


# Full cache of compute_bnn_group_sensitivities' output, so reruns can skip straight to plotting.
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
    """Describe each 2D weight tensor's row/column meanings and column names for tendency analysis."""
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
            "row_label": "out",
            "col_label": "in",
            "col_names": None,
            "show_col_marginal": False,
        },
    }


def compute_bnn_group_sensitivities(
    cfg, use_cache: bool = True,
) -> Tuple[Any, Dict[str, Dict[str, Any]], float, int, float, Dict, Dict]:
    """Compute (or load cached) per-node FD sensitivities for every BNN parameter group."""
    loader = instantiate(cfg.model, data_config=cfg.data)
    J = loader.total_nodes
    r_j = float(cfg.sensitivity.r_j)
    radius = r_j * J
    base_tag = _config_tag(cfg.data.get("dataset", "uci"), cfg.data.get("reference_prior", "gaussian"))

    # Basis settings change omega_max, so resolve them before the cache check and fold them into its key.
    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    tag = f"{base_tag}_b{_basis_config_hash(cfg.optimize.nonparametric.basis_funcs_type, basis_kwargs)}"

    cache_path = _find_latest_sensitivity_cache(tag) if use_cache else None
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

        # Fresh prior draw used only to pick basis centres, independent of the samples used for A_c.
        center_prior_samples = loader.sample_prior(group_name, n_samples=center_samples_num)
        center_prior_samples_cache[group_name] = center_prior_samples

        omega_max = compute_group_omega_max(
            posterior_samples=g["posterior"],
            loc=g["loc"],
            scale=g["scale"],
            prior_samples=prior_samples,
            basis_cls=basis_cls,
            basis_kwargs=basis_kwargs,
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

    if use_cache:
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


def _run_bnn_uci_node_sensitivity_core(cfg, use_cache: bool = True) -> None:
    """Run per-node BNN sensitivity analysis for one config, callable outside a Hydra job."""
    core_start = time.perf_counter()
    tensor_axis_meta = _build_tensor_axis_meta(cfg.data.get("feature_names"))
    tag = f"{cfg.data.get('dataset', 'uci')}_{cfg.data.get('reference_prior', 'gaussian')}"

    start = time.perf_counter()
    _, group_results, radius, J, r_j, _, _ = compute_bnn_group_sensitivities(cfg, use_cache=use_cache)
    total = time.perf_counter() - start
    print(f"Total optimisation time: {total:.3f}s")

    total_sensitivity = float(sum(v["total_sensitivity"] for v in group_results.values()))
    print(f"Global FD sensitivity S^FD(Q_r) = {total_sensitivity:.4f} (r={radius}, J={J}).")

    heatmap_tensors = []
    for group_name, meta in tensor_axis_meta.items():
        res = group_results[group_name]
        shape = tuple(res["shape"])
        if len(shape) != 2:
            continue
        n_rows, n_cols = shape
        sensitivity_matrix = res["sensitivity"].reshape(n_rows, n_cols)
        short_name = group_name.replace("net.module.", "L").replace("_prior", "").replace(".weight", "")
        heatmap_tensors.append({
            "label": f"{short_name}",
            # Divided by the per-node radius r_j, so the heatmap is radius independent.
            "matrix": sensitivity_matrix / r_j,
            "row_label": meta["row_label"],
            "col_label": meta["col_label"],
            "col_names": meta.get("col_names"),
            "show_col_marginal": meta.get("show_col_marginal", True),
        })

    plot_config_path = os.path.join(_project_root(), "configs/plots/overleaf_plots_settings.yaml")
    plot_cfg = load_plot_config(plot_config_path)
    output_dir = os.path.join(_project_root(), cfg.flags.plots.output_dir)
    plot_bnn_weight_heatmaps(
        tensors=heatmap_tensors,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"bnn_weight_heatmaps_{tag}.pdf",
        value_label=r"Estimated per-parameter sensitivity / $r_j$",
    )

    core_elapsed = time.perf_counter() - core_start
    print(f"Total run_bnn_uci_node_sensitivity time: {core_elapsed:.3f}s")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="bnn_boston_nonparam_gaussian")
def run_bnn_uci_node_sensitivity(cfg) -> None:
    """Per-node nonparametric FD sensitivity analysis for the Fortuin et al. (2022) BNN on a UCI dataset."""
    _run_bnn_uci_node_sensitivity_core(cfg)


def run_bnn_uci_all_datasets_sensitivity() -> None:
    """Run per-node sensitivity for all 5 UCI datasets under both reference-prior families."""
    for dataset in TABLE_DATASETS:
        for prior in TABLE_PRIORS:
            config_path = os.path.join(CONFIGS_DIR, f"bnn_{dataset}_nonparam_{prior}.yaml")
            cfg = OmegaConf.load(config_path)
            print(f"=== {TABLE_DATASET_LABELS[dataset]} / {TABLE_PRIOR_LABELS[prior]} ===")
            _run_bnn_uci_node_sensitivity_core(cfg)


def _layer_mean_sensitivities(group_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Return each layer's mean sensitivity, pooling its weight and bias nodes."""
    layer_means = {}
    for layer_label, (weight_group, bias_group) in LAYER_GROUP_MAP.items():
        total_sensitivity = (
            group_results[weight_group]["total_sensitivity"] + group_results[bias_group]["total_sensitivity"]
        )
        n_nodes = group_results[weight_group]["n_nodes"] + group_results[bias_group]["n_nodes"]
        layer_means[layer_label] = total_sensitivity / n_nodes
    return layer_means


def _render_layer_sensitivity_latex(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Render per-layer mean sensitivities as the appendix LaTeX table, bolding each column's maximum."""
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
    """Aggregate cached per-node sensitivities into a per-layer LaTeX table for the paper's appendix."""
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
