import importlib
import pathlib
import re

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.nonparametric.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.nonparametric.sensitivity import nonparametric_prior_sensitivity
from src.parametric.sensitivity import prior_sensitivity

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT / "configs")))
def test_targets_and_basis_names_resolve(path):
    text = path.read_text()
    for target in re.findall(r"^\s*_target_:\s*([\w.]+)", text, flags=re.M):
        module, _, attr = target.rpartition(".")
        assert hasattr(importlib.import_module(module), attr), target
    for basis in re.findall(r"^\s*(?:basis_funcs_type|basis):\s*(\w+)", text, flags=re.M):
        assert basis in BASIS_FUNCTIONS_REGISTRY, basis


def _compose(name):
    with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "configs" / "examples")):
        return compose(config_name=name)


def test_parametric_example_config_runs():
    cfg = _compose("gaussian_location")
    np.random.seed(0)
    model = instantiate(cfg.model, data_config=cfg.data)
    result = prior_sensitivity(model, OmegaConf.to_container(cfg.sensitivity.natural_box, resolve=True))
    assert result.sensitivity > 0.0
    assert result.fd_min == pytest.approx(0.0, abs=1e-8)


def test_nonparametric_example_config_runs():
    cfg = _compose("gaussian_location_nonparam")
    np.random.seed(0)
    model = instantiate(cfg.model, data_config=cfg.data)
    result = nonparametric_prior_sensitivity(
        model, radius=cfg.nonparametric.radius, basis=cfg.nonparametric.basis,
        basis_kwargs={**OmegaConf.to_container(cfg.nonparametric.basis_kwargs, resolve=True), "num_basis_functions": 20},
    )
    assert result.sensitivity > 0.0
