import importlib
import pathlib
import re

import pytest

from src.nonparametric.basis_functions import BASIS_FUNCTIONS_REGISTRY

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
