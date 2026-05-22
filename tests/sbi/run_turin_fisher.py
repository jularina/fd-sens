import os
import warnings
from typing import List
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time
import json

from src.utils.files_operations import load_plot_config
from src.utils.files_operations import deepcopy_cfg
from src.utils.choosers import pick_optimizer
from src.discrepancies.posterior_fisher import PosteriorFDParametric as PosteriorFDBase
from src.plots.paper.sbi_paper_funcs import *
from src.optimization.corner_points_fisher import (
    OptimizationCornerPointsCompositePrior
)
from src.plots.paper.toy_paper_fisher_funcs import plot_runtime_nonparametric_with_ci, plot_gaussian_copula_grid_pair

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


def _filter_components(cfg: DictConfig, keep_names: List[str]) -> DictConfig:
    new_cfg = deepcopy_cfg(cfg)
    comps = new_cfg.ksd.optimize.prior.Composite.components
    new_cfg.ksd.optimize.prior.Composite.components = [c for c in comps if c.get("name") in keep_names]
    return new_cfg


def _eval_corners_with_cfg(ksd_est, cfg_like: DictConfig) -> Tuple:
    optimizer = pick_optimizer(cfg_like, ksd_est)
    qf_corners, worst_corner_dict = optimizer.evaluate_all_prior_corners()
    rows = []
    for corners in qf_corners:
        rows.append({
            "prior_corner": list(corners[0]) if not isinstance(corners[0], dict) else corners[0],
            "value": float(corners[2]),
        })
    return rows, worst_corner_dict


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="sbi_nle_turin")
def main_old(cfg: DictConfig) -> None:
    # model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    # estimator_posterior = PosteriorFDNonParametric(model=model)
    # print("FD: ", estimator_posterior.estimate_fisher_prior_only())
    # estimator_prior = PriorFDNonParametric(model=model)
    # psi_sdp_list, fd_estimates_list = [], []
    # basis_funcs_num_list = [8, 12, 16, 20, 24]
    # radii_list = [10, 15, 20]
    # nonparam_metrics, nonparam_eta_stars = defaultdict(dict), defaultdict(dict)
    #
    # for radius in radii_list:
    #     for basis_funcs_num in basis_funcs_num_list:
    #         cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
    #         optimizer = OptimisationNonparametricBase(
    #             estimator_posterior,
    #             estimator_prior,
    #             cfg.ksd.optimize.prior.nonparametric,
    #             radius=radius
    #         )
    #         start = time.perf_counter()
    #         result_sdp = optimizer.optimize_through_sdp_relaxation()
    #         elapsed = time.perf_counter() - start
    #         print(f"SDP primal relaxation time: {elapsed}")
    #
    #         start = time.perf_counter()
    #         result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
    #         elapsed = time.perf_counter() - start
    #         print(f"SDP dual time: {elapsed}")
    #
    #         nonparam_metrics[basis_funcs_num][radius] = result_sdp["primal_value"]
    #         nonparam_eta_stars[basis_funcs_num][radius] = result_sdp["eta_star"]
    #         psi_sdp_list.append(result_sdp["eta_star"])
    #         fd_estimates_list.append(result_sdp["primal_value"])
    #         print(f"Radius: {radius}, basis funcs num: {basis_funcs_num}, objective estimate: {result_sdp["primal_value"]}.")

    # Plots
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    # plot_ksd_heatmap(data_dict=nonparam_metrics, plot_cfg=plot_cfg, output_dir=output_dir, log_scale=False)

    # Optimisation times
    # times_nonparametric = defaultdict(dict)
    # steps = 10
    # basis_funcs_num_list = [8, 16, 24, 32, 40, 48]
    #
    # for step in range(steps):
    #     print(f"************* Parametric running step {step} *************.")
    #     for basis_funcs_num in basis_funcs_num_list:
    #         print(f"K={basis_funcs_num}*.")
    #         cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
    #         model = instantiate(cfg.model, data_config=cfg.data)
    #         estimator_prior = PriorFDNonParametric(model=model)
    #         estimator_posterior = PosteriorFDNonParametric(model=model)
    #         optimizer = OptimisationNonparametricBase(
    #             estimator_posterior,
    #             estimator_prior,
    #             cfg.ksd.optimize.prior.nonparametric,
    #             radius=10
    #         )
    #         start = time.perf_counter()
    #         result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
    #         elapsed = time.perf_counter() - start
    #         times_nonparametric[basis_funcs_num][step] = elapsed
    #
    data_path = os.path.join(get_original_cwd(), "data/sbi/runtimes/")
    # with open(data_path + "nonparametric_optimisation_times_diff_basis_funcs_nums.json", "w") as f:
    #     json.dump(times_nonparametric, f, indent=4)

    with open(data_path + "nonparametric_optimisation_times_diff_basis_funcs_nums.json", "r") as f:
        nonparametric_optimisation_times = json.load(f)

    plot_runtime_nonparametric_with_ci(
        nonparametric_optimisation_times,
        plot_cfg,
        output_dir,
        filename="runtime_nonparametric_qcqp_sbi.pdf"
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="sbi_nle_turin")
def main(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "sbi")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    print("=== FD for PosteriorDB model ===")
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher for prior: {fisher_estimator.estimate_fisher_prior_only():.4f}")

    print("Starting optimisation of all parameters at once.")
    start = time.perf_counter()
    optimizer = OptimizationCornerPointsCompositePrior(fisher_estimator,
                                                       cfg.fd.optimize.prior.Composite,
                                                       cfg.fd.optimize.loss.GaussianLogLikelihoodWithGivenGrads,
                                                       )
    qf_corners, eta_star = optimizer.evaluate_all_prior_corners()
    elapsed = time.perf_counter() - start
    print(f"Corner eta^star is {eta_star}.")
    print(f"Time for optimisation of all parameters at once: {elapsed:.3f} sec.")

    print("Starting black-box optimisation.")
    start = time.perf_counter()
    bb = optimizer.black_box_optimize_prior_box_global(
        seed=0,
        maxiter=150,
        popsize=20,
        workers=1,
        updating="deferred",
    )
    elapsed = time.perf_counter() - start
    print("Black-box sup:", bb.val_sup)
    print("Black-box inf:", bb.val_inf)
    print("Black-box S_hat:", bb.S_hat)
    print(f"Time for black-box optimisation of all parameters at once: {elapsed:.3f} sec.")

    print("Starting per component optimisation.")
    names = ["theta_1", "theta_2", "theta_3", "theta_4"]
    start = time.perf_counter()
    sup_res, eta_sup_blocks = optimizer.evaluate_all_prior_corners_per_component(component_names=names)
    eta_inf_blocks, values_inf = optimizer.minimize_prior_per_component_qp(names)
    print("Per-component argmax corners:")
    for k in names:
        print(k, eta_sup_blocks[k], sup_res[k][0][1])

    print("Per-component infimum:")
    for n in names:
        print(n, eta_inf_blocks[n], values_inf[n])
    elapsed = time.perf_counter() - start
    print(f"Time for per-component optimisation: {elapsed/len(names):.3f} sec.")

    print("Starting Gaussian copula black-box optimisation.")
    start = time.perf_counter()
    copula_res = optimizer.black_box_optimize_gaussian_copula(
        lambda_range=(-0.5, 0.5),
        seed=0,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        polish=True,
        workers=1,
        updating="deferred",
    )
    elapsed = time.perf_counter() - start

    print(f"Copula lambda_sup: {copula_res.lambda_sup}")
    print(f"Copula val_sup: {copula_res.val_sup}")
    print(f"Copula lambda_inf: {copula_res.lambda_inf}")
    print(f"Copula val_inf: {copula_res.val_inf}")
    print(f"Copula S_hat: {copula_res.S_hat}")
    print(f"Copula nfev_sup: {copula_res.nfev_sup}")
    print(f"Copula nfev_inf: {copula_res.nfev_inf}")
    print(f"Time for Gaussian copula optimisation: {elapsed:.3f} sec.")

    print("Starting Gaussian copula grid evaluation.")
    start = time.perf_counter()
    copula_grid_g0, lambda_star_g0, val_star_g0 = optimizer.evaluate_gaussian_copula_grid_and_argmax(
        lambda_range=(-0.2, 0.0),
        n_grid=200,
        idx_g0=0,
        idx_nu=2,
    )
    copula_grid_T, lambda_star_T, val_star_T = optimizer.evaluate_gaussian_copula_grid_and_argmax(
        lambda_range=(0.0, 0.2),
        n_grid=200,
        idx_g0=1,
        idx_nu=2,
    )
    elapsed = time.perf_counter() - start
    print(f"Grid lambda^star (g0): {lambda_star_g0}, FD={val_star_g0}")
    print(f"Grid lambda^star (T):  {lambda_star_T}, FD={val_star_T}")
    print(f"Time for Gaussian copula grid evaluation: {elapsed:.3f} sec.")

    print("Starting Gaussian copula grid evaluation (narrow range).")
    start = time.perf_counter()
    copula_grid_g0_narrow, lambda_star_g0_narrow, val_star_g0_narrow = optimizer.evaluate_gaussian_copula_grid_and_argmax(
        lambda_range=(-0.105, 0.01),
        n_grid=200,
        idx_g0=0,
        idx_nu=2,
    )
    copula_grid_T_narrow, lambda_star_T_narrow, val_star_T_narrow = optimizer.evaluate_gaussian_copula_grid_and_argmax(
        lambda_range=(-0.01, 0.105),
        n_grid=200,
        idx_g0=1,
        idx_nu=2,
    )
    elapsed = time.perf_counter() - start
    print(f"Grid lambda^star narrow (g0): {lambda_star_g0_narrow}, FD={val_star_g0_narrow}")
    print(f"Grid lambda^star narrow (T):  {lambda_star_T_narrow}, FD={val_star_T_narrow}")
    print(f"Time for narrow Gaussian copula grid evaluation: {elapsed:.3f} sec.")

    all_values = (
        [x[1] for x in copula_grid_g0]
        + [x[1] for x in copula_grid_T]
        + [x[1] for x in copula_grid_g0_narrow]
        + [x[1] for x in copula_grid_T_narrow]
    )
    global_ylim = (min(all_values), max(all_values))

    plot_gaussian_copula_grid_pair(
        copula_grid_0=copula_grid_g0,
        copula_grid_1=copula_grid_T,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        logy=True,
        ylim=global_ylim,
    )
    marked_x_values = [[], [-0.018, 0.06, 0.11, 0.12]]  # [values for g0, values for T]
    plot_gaussian_copula_grid_pair(
        copula_grid_0=copula_grid_g0,
        copula_grid_1=copula_grid_T,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        filename=f"{prefix}_gaussian_copula_fd_grid_marked.pdf",
        logy=True,
        ylim=global_ylim,
        mark_max_point=False,
        show_grid_0=False,
        mark_x_values=marked_x_values,
        mark_x_red_idx=1,  # index in marked_x_values[1] that gets red star; others are black crosses
    )
    # plot_gaussian_copula_grid_pair(
    #     copula_grid_0=copula_grid_g0_narrow,
    #     copula_grid_1=copula_grid_T_narrow,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     prefix=prefix,
    #     filename=f"{prefix}_gaussian_copula_fd_grid_narrow.pdf",
    #     logy=True,
    #     mark_max_point=False,
    #     mark_corner_point=True,
    #     ylim=global_ylim,
    #     show_ylabel=False,
    # )
    #
    # for lam in [-0.25, -0.5, -0.75, -0.95]:
    #     diag = fisher_estimator.diagnose_gaussian_copula_l2(lam, eps=0.0)
    #     print(f"\nDiagnostics for lambda={lam}")
    #     for k, v in diag.items():
    #         print(f"{k}: {v}")


if __name__ == "__main__":
    # main_old()
    main()
