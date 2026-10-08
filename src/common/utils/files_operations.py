from typing import Dict, Any
import yaml
import numpy as np
from omegaconf import OmegaConf
import os
import json


def load_plot_config(path: str):
    with open(path, "r") as f:
        plot_cfg = yaml.safe_load(f)
    return OmegaConf.create(plot_cfg)


def load_numpy_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Expected numpy array in {path}")
    return arr


def _to_serialisable(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    return obj


def save_to_serializable_json(results: Dict[str, Any], path: str) -> None:
    results_ser = _to_serialisable(results)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_ser, f, indent=2)


def load_results_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        results = json.load(f)
    return results


def convert_dim_keys_to_int(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert nested dimension keys like "5" back to integers in a loaded results dict.
    """
    for top_key in ["error_mean", "error_ci", "time_mean", "time_ci"]:
        if top_key not in results:
            continue

        for method, method_dict in results[top_key].items():
            if isinstance(method_dict, dict):
                results[top_key][method] = {
                    int(k): v for k, v in method_dict.items()
                }

    return results
