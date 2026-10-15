import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.nonparametric.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.common.utils.files_operations import load_plot_config, save_to_serializable_json
from src.nonparametric.node_sensitivity import compute_group_omega_max, compute_node_lambda_star
from paper.plot_utils import apply_plot_rc, save_fig
from paper.posteriordb.plots_parametric import plot_posterior_predictive_with_data
from paper.posteriordb.run_parametric import (
    _PIT_EPS,
    _to_z_space,
    _fd_z_posterior_gaussian_in_z,
    _z_box_from_cfg,
    _z_box_tag,
    _run_stage,
    _load_corner_draws_json,
    _stack_corner_chains,
    _ar_posterior_predictive,
    _summarise_bands,
    _summarise_samples,
    print_predictive_variance_decomposition,
    x_years,
    y,
    y_centered,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Kilpisjarvi's prior has 7 scalar blocks: alpha, beta1..beta5 ~ N(0, 5^2), sigma ~ HalfCauchy(gamma).
LATEX_NAMES = {
    "alpha": r"$\alpha$",
    "beta1": r"$\beta_1$",
    "beta2": r"$\beta_2$",
    "beta3": r"$\beta_3$",
    "beta4": r"$\beta_4$",
    "beta5": r"$\beta_5$",
    "sigma": r"$\sigma$",
}
COMPONENT_ORDER = ["alpha", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma"]

# Shared z-space neighbourhood: N(mu_z, sigma_z^2), mu_z in Z_MU_RANGE, sigma_z in Z_SIGMA_RANGE.
Z_MU_RANGE = (-0.4, 0.4)
Z_SIGMA_RANGE = (0.8, 1.25)

NONPARAM_Z_STAN_DIR = "outputs/paper/results/kilpisjarvi/nonparam/stan"


def _box_corners(mu_z_range, sigma_z_range):
    return [(mu, sig) for mu in mu_z_range for sig in sigma_z_range]


def compute_parametric_sensitivity(loader, mu_z_range=Z_MU_RANGE, sigma_z_range=Z_SIGMA_RANGE) -> dict:
    """Per component, return the posterior-based FD_z sup over the shared Gaussian-in-z box."""
    corners = _box_corners(mu_z_range, sigma_z_range)
    fd_z_sup = {}
    for group_name in loader.param_groups:
        g = loader.groups[group_name]
        posterior = np.asarray(g["posterior"], dtype=float)
        for local_idx, node_name in enumerate(g["node_names"]):
            z_post = _to_z_space(g["prior_dist"], posterior[:, local_idx])
            z_mean, z_meansq = float(np.mean(z_post)), float(np.mean(z_post ** 2))
            fd_z_sup[node_name] = max(
                _fd_z_posterior_gaussian_in_z(mu_z, sig_z, z_mean, z_meansq) for mu_z, sig_z in corners
            )
    return fd_z_sup


def _fd_z_prior_gaussian_in_z(mu_z_cand: float, sigma_z_cand: float) -> float:
    """Exact prior-based FD_z for a Gaussian-in-z candidate against an N(0, 1) reference."""
    a = 1.0 / sigma_z_cand ** 2 - 1.0
    b = -mu_z_cand / sigma_z_cand ** 2
    return float(a ** 2 + b ** 2)


def compute_shared_radius(mu_z_range=Z_MU_RANGE, sigma_z_range=Z_SIGMA_RANGE) -> float:
    """Return the radius r_j bounding the FD of every prior in the z-box, shared by all components."""
    return max(_fd_z_prior_gaussian_in_z(mu_z, sig_z) for mu_z, sig_z in _box_corners(mu_z_range, sigma_z_range))


def compute_nonparametric_sensitivity(
    loader,
    basis_cls,
    basis_kwargs: dict,
    r_j: float,
    center_samples_num: int = 5000,
) -> dict:
    """Compute per-component FDsens+ sensitivities in z-space and return them with the optimisation time."""
    sensitivity = {}
    optimisation_time = 0.0
    for group_name in loader.param_groups:
        g = loader.groups[group_name]
        prior_dist = g["prior_dist"]
        posterior_z = _to_z_space(prior_dist, g["posterior"])
        prior_samples_z = _to_z_space(prior_dist, loader.sample_prior(group_name))
        center_prior_samples_z = _to_z_space(
            prior_dist, loader.sample_prior(group_name, n_samples=center_samples_num)
        )

        start = time.perf_counter()
        omega_max = compute_group_omega_max(
            posterior_samples=posterior_z,
            loc=0.0,
            scale=1.0,
            prior_samples=prior_samples_z,
            basis_cls=basis_cls,
            basis_kwargs=basis_kwargs,
            center_prior_samples=center_prior_samples_z,
        )
        optimisation_time += time.perf_counter() - start

        for node_name, omega in zip(g["node_names"], omega_max):
            sensitivity[node_name] = r_j * float(omega)
    return sensitivity, optimisation_time


def compute_worst_case_kef(loader, basis_cls, basis_kwargs: dict, r_j: float, center_samples_num: int = 5000) -> dict:
    """Per component, return the worst-case KEF (basis, lambda_star) at the shared radius r_j in z-space."""
    records = {}
    for group_name in loader.param_groups:
        g = loader.groups[group_name]
        prior_dist = g["prior_dist"]
        posterior_z = _to_z_space(prior_dist, g["posterior"])
        prior_samples_z = _to_z_space(prior_dist, loader.sample_prior(group_name))
        center_prior_samples_z = _to_z_space(
            prior_dist, loader.sample_prior(group_name, n_samples=center_samples_num)
        )
        for local_idx, node_name in enumerate(g["node_names"]):
            lam_star, omega_max, basis, _ = compute_node_lambda_star(
                posterior_samples_col=posterior_z[:, local_idx],
                loc=0.0,
                scale=1.0,
                prior_samples=prior_samples_z,
                basis_cls=basis_cls,
                basis_kwargs=basis_kwargs,
                radius_j=r_j,
                center_prior_samples=center_prior_samples_z,
            )
            records[node_name] = {"group": group_name, "lambda_star": lam_star, "basis": basis, "omega_max": omega_max}
    return records


def _matern72_closed_form(r: np.ndarray, lengthscale: float, variance: float) -> np.ndarray:
    """Matern nu=7/2 kernel in closed form, as in the Stan model."""
    x = np.sqrt(7.0) * np.abs(r) / lengthscale
    return variance * (1.0 + x + 0.4 * x ** 2 + x ** 3 / 15.0) * np.exp(-x)


def _export_kef_stan_data(records: dict, loader, y_full: np.ndarray, K: int, r_j: float, path: str) -> None:
    """Write Stan data for stan/kilpisjarvi_ark_kef.stan, checking its closed-form kernel against the basis."""
    centres, lambdas, ells, variances = [], [], [], []
    for name in COMPONENT_ORDER:
        basis = records[name]["basis"]
        if not np.isclose(basis.nu, 3.5):
            raise ValueError(f"Stan model hard-codes Matern nu=3.5; got nu={basis.nu} for {name}.")
        c = np.asarray(basis.centers, dtype=float).reshape(-1)
        z_check = np.linspace(-4.0, 4.0, 101)
        phi_python = basis.evaluate(z_check.reshape(-1, 1))[:, 0, :]
        phi_stan = _matern72_closed_form(z_check[:, None] - c[None, :], basis.lengthscale, basis.variance)
        if not np.allclose(phi_python, phi_stan, rtol=1e-6, atol=1e-10):
            raise ValueError(f"Closed-form Matern-7/2 does not match basis.evaluate for {name}.")
        centres.append(c)
        lambdas.append(np.asarray(records[name]["lambda_star"], dtype=float).reshape(-1))
        ells.append(float(basis.lengthscale))
        variances.append(float(basis.variance))

    M = {len(c) for c in centres}
    if len(M) != 1:
        raise ValueError(f"Components have different numbers of centres: {sorted(M)}.")

    ref = {name: loader.groups[records[name]["group"]]["prior_dist"] for name in COMPONENT_ORDER}
    y_full = np.asarray(y_full, dtype=float).reshape(-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_to_serializable_json(
        {
            "K": int(K),
            "T": int(len(y_full)),
            "y": y_full,
            "M": int(M.pop()),
            "mu_ref": [float(ref[n].mu) for n in COMPONENT_ORDER[:-1]],
            "sigma_ref": [float(ref[n].sigma) for n in COMPONENT_ORDER[:-1]],
            "gamma_sigma": float(ref["sigma"].gamma),
            "pit_eps": _PIT_EPS,
            "centres": np.stack(centres),
            "lambda": np.stack(lambdas),
            "ell": ells,
            "var_k": variances,
            "r_j": float(r_j),
        },
        path,
    )
    print(f"Saved Stan data for the worst-case KEF prior (r_j={r_j:g}) to {path}")


def _percentages(values: dict) -> dict:
    total = sum(values.values())
    return {name: v / total * 100.0 for name, v in values.items()}


def _draw_component_sensitivity_stack(ax, percentages: dict, title: str) -> None:
    """Draw a single stacked bar of component contributions, coloured by rank and labelled above 4%."""
    ranked = sorted(COMPONENT_ORDER, key=lambda k: percentages[k])
    n = len(ranked)
    low = np.array(to_rgb("#4d7298"))
    high = low + 0.75 * (np.array([1.0, 1.0, 1.0]) - low)
    color_map = {k: tuple(low + (i / max(n - 1, 1)) * (high - low)) for i, k in enumerate(ranked)}

    left = 0.0
    min_label_pct = 4.0
    for k in COMPONENT_ORDER:
        pct = percentages[k]
        ax.barh(0, pct, left=left, color=color_map[k], alpha=0.5, edgecolor="white", linewidth=0.5, height=1.0)
        if pct >= min_label_pct:
            ax.text(left + pct / 2, 0, f"{LATEX_NAMES.get(k, k)} {pct:.1f}%", ha="center", va="center",
                    rotation=0, fontsize=8, color="black")
        left += pct

    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0%", "100%"])
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel(title, rotation=0, ha="right", va="center", labelpad=4,
                  fontsize=plt.rcParams["font.size"] * 0.85)


def plot_component_sensitivity_bar_param_vs_nonparam(
    plot_cfg,
    output_dir: str,
    percentages_param: dict,
    percentages_nonparam: dict,
    filename: str,
) -> None:
    """Two stacked bars, FDsens above FDsens+, with 0%/100% tick labels only under the bottom bar."""
    apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(plot_cfg.plot.figure.size.width * 1.2, plot_cfg.plot.figure.size.height * 0.6),
        dpi=plot_cfg.plot.figure.dpi,
    )
    _draw_component_sensitivity_stack(axes[0], percentages_param, r"\texttt{FDsens}")
    _draw_component_sensitivity_stack(axes[1], percentages_nonparam, r"\texttt{FDsens+}")
    axes[0].tick_params(axis="x", labelbottom=False)
    fig.tight_layout(h_pad=0.0, pad=0.3)
    save_fig(fig, output_dir, filename, plot_cfg)
    plt.close(fig)
    print(f"Saved plot to {os.path.join(output_dir, filename)}")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi_nonparam")
def main(cfg: DictConfig) -> None:
    if not _run_stage(cfg, "sensitivity"):
        return
    loader = instantiate(cfg.model, data_config=cfg.data)
    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)

    start = time.perf_counter()
    param_sensitivity = compute_parametric_sensitivity(loader)
    param_time = time.perf_counter() - start

    r_j = compute_shared_radius()
    print(f"Shared radius r_j (prior-based FD_z sup over the parametric box): {r_j:.4f}")

    nonparam_sensitivity, nonparam_time = compute_nonparametric_sensitivity(
        loader, basis_cls, basis_kwargs, r_j,
        center_samples_num=int(cfg.data.get("center_prior_samples_num", 5000)),
    )
    print(f"Optimisation time: FDsens {param_time:.3f}s, FDsens+ {nonparam_time:.3f}s.")

    percentages_param = _percentages(param_sensitivity)
    percentages_nonparam = _percentages(nonparam_sensitivity)
    print("Per-component share of the total FD sensitivity (%):")
    for name in COMPONENT_ORDER:
        print(f"  {name}: FDsens={percentages_param[name]:.1f}%, FDsens+={percentages_nonparam[name]:.1f}%")

    plot_cfg = load_plot_config(os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml"))
    plot_component_sensitivity_bar_param_vs_nonparam(
        plot_cfg=plot_cfg,
        output_dir=os.path.join(get_original_cwd(), cfg.flags.plots.output_dir),
        percentages_param=percentages_param,
        percentages_nonparam=percentages_nonparam,
        filename="kilpisjarvi_param_vs_nonparam_sensitivity_percentages_both_in_z_scale.pdf",
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi_nonparam")
def export_worst_case_stan_data(cfg: DictConfig) -> None:
    """Find the worst-case KEF prior at the radius of the predictive z-box and export its Stan data."""
    if not _run_stage(cfg, "optimise"):
        return
    loader = instantiate(cfg.model, data_config=cfg.data)
    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    K = sum(1 for name in COMPONENT_ORDER if name.startswith("beta"))

    mu_z_range, sigma_z_range = _z_box_from_cfg(cfg)
    r_j = compute_shared_radius(mu_z_range, sigma_z_range)
    print(f"Predictive z-box: mu_z in {mu_z_range}, sigma_z in {sigma_z_range}; shared radius r_j = {r_j:.4f}")

    records = compute_worst_case_kef(
        loader, basis_cls, basis_kwargs, r_j,
        center_samples_num=int(cfg.data.get("center_prior_samples_num", 5000)),
    )
    _export_kef_stan_data(
        records, loader, y_centered, K, r_j,
        os.path.join(get_original_cwd(), NONPARAM_Z_STAN_DIR,
                     f"stan_data_kef_z{_z_box_tag(mu_z_range, sigma_z_range)}.json"),
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi_nonparam")
def plot_posterior_predictive(cfg: DictConfig) -> None:
    """Plot the posterior predictive under the reference and the worst-case KEF prior (Stan draws)."""
    if not _run_stage(cfg, "plot_predictive"):
        return
    loader = instantiate(cfg.model, data_config=cfg.data)
    K = sum(1 for name in COMPONENT_ORDER if name.startswith("beta"))
    plot_cfg = load_plot_config(os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml"))
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.predictive_output_dir)
    prefix = cfg.playground.get("output_prefix", "kilpisjarvi_nonparam")

    mu_z_range, sigma_z_range = _z_box_from_cfg(cfg)
    draws_path = os.path.join(
        get_original_cwd(), NONPARAM_Z_STAN_DIR, f"draws_kef_z{_z_box_tag(mu_z_range, sigma_z_range)}.json"
    )
    print(f"Worst-case KEF posterior draws: {draws_path}")
    worst_samples = _stack_corner_chains(_load_corner_draws_json(draws_path), K=K, warmup=0, max_draws=None)
    ref_samples = loader.model.posterior_samples_init

    _summarise_samples(ref_samples, K, "reference posterior")
    _summarise_samples(worst_samples, K, "worst-case KEF posterior")
    print_predictive_variance_decomposition(y_centered, ref_samples, K, name="reference posterior")
    print_predictive_variance_decomposition(y_centered, worst_samples, K, name="worst-case KEF posterior")

    mode, seed = "one_step", 27
    ref_mean, ref_lo, ref_hi = _summarise_bands(
        _ar_posterior_predictive(y_full=y_centered, samples=ref_samples, K=K, mode=mode, seed=seed)
    )
    worst_mean, worst_lo, worst_hi = _summarise_bands(
        _ar_posterior_predictive(y_full=y_centered, samples=worst_samples, K=K, mode=mode, seed=seed)
    )

    y_mean_offset = float(np.mean(y))
    x_pred_years = x_years[K:]
    all_values = np.concatenate([y, ref_lo + y_mean_offset, ref_hi + y_mean_offset,
                                 worst_lo + y_mean_offset, worst_hi + y_mean_offset])
    global_ylim = (float(all_values.min()), float(all_values.max()))

    plot_posterior_predictive_with_data(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        x_years=x_years,
        y_uncentered=y,
        x_pred_years=x_pred_years,
        pred_mean=ref_mean + y_mean_offset,
        pred_lo=ref_lo + y_mean_offset,
        pred_hi=ref_hi + y_mean_offset,
        pred_label=r"$\tilde{x}_{\mathrm{ref}} \pm 95\%$ CI",
        filename=f"{prefix}_posterior_predictive_ref.pdf",
        pred_color="#7c397d",
        ylim=global_ylim,
    )
    plot_posterior_predictive_with_data(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        x_years=x_years,
        y_uncentered=y,
        x_pred_years=x_pred_years,
        pred_mean=worst_mean + y_mean_offset,
        pred_lo=worst_lo + y_mean_offset,
        pred_hi=worst_hi + y_mean_offset,
        pred_label=r"$\tilde{x} \pm 95\%$ CI",
        filename="kilpisjarvi_posterior_predictive_worst_case.pdf",
        pred_color="#5b9bd5",
        ylim=global_ylim,
        show_ylabel=False,
    )
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
    export_worst_case_stan_data()
    plot_posterior_predictive()
