import json

import numpy as np
import pytest

from interpretation.plots import (
    plot_component_shares, plot_ecdf, plot_kde, plot_quantiles, save_sensitivity_result, summarise_quantiles,
)
from src.parametric.sensitivity import prior_sensitivity

from tests.test_parametric import BOX


@pytest.fixture
def draws():
    rng = np.random.default_rng(0)
    return {"reference": rng.normal(size=(500, 2)), "candidate": rng.normal(0.5, 1.0, size=(500, 2))}


def test_quantile_table_has_one_row_per_fit_and_variable(draws):
    table = summarise_quantiles(draws, variables=["a", "b"])
    assert list(table["fit"]) == ["reference", "reference", "candidate", "candidate"]
    assert table.loc[0, "50%"] == pytest.approx(np.median(draws["reference"][:, 0]))


def test_draw_plots_write_files(draws, tmp_path):
    plot_quantiles(draws, tmp_path, variables=["a", "b"])
    plot_kde(draws, tmp_path)
    plot_ecdf(draws, tmp_path)
    for name in ("quantiles.csv", "quantiles.png", "kde.png", "ecdf.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_draws_with_mismatched_columns_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        plot_kde({"reference": np.zeros((10, 2)), "candidate": np.zeros((10, 3))}, tmp_path)


def test_result_json_and_component_shares(two_independent_locations, tmp_path):
    box = {name: BOX["theta"] for name in ("a", "b")}
    result = prior_sensitivity(two_independent_locations, box, independent=True)
    save_sensitivity_result(result, tmp_path)
    saved = json.loads((tmp_path / "sensitivity_result.json").read_text())
    assert saved["sensitivity"] == pytest.approx(result.sensitivity)
    assert set(saved["components"]) == {"a", "b"}
    plot_component_shares(result, tmp_path)
    assert (tmp_path / "component_shares.png").stat().st_size > 0


def test_component_shares_need_components(gaussian_location, tmp_path):
    result = prior_sensitivity(gaussian_location[0], BOX)
    with pytest.raises(ValueError):
        plot_component_shares(result, tmp_path)
