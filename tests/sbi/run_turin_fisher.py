import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time

from src.utils.files_operations import load_plot_config
from src.discrepancies.posterior_fisher import PosteriorFDParametric as PosteriorFDBase
from src.plots.paper.sbi_paper_funcs import *
from src.optimization.corner_points_fisher import (
    OptimizationCornerPointsCompositePrior
)
from src.plots.paper.toy_paper_fisher_funcs import (
    plot_gaussian_copula_grid_pair,
    plot_gaussian_copula_fd_decomposition,
)

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="sbi_nle_turin")
def main(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "sbi")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    optimizer = OptimizationCornerPointsCompositePrior(fisher_estimator,
                                                       cfg.fd.optimize.prior.Composite,
                                                       cfg.fd.optimize.loss.GaussianLogLikelihoodWithGivenGrads,
                                                       )

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
        n_grid=1000,
        idx_g0=0,
        idx_nu=2,
        apply_z_transform=False,
    )
    copula_grid_T, lambda_star_T, val_star_T = optimizer.evaluate_gaussian_copula_grid_and_argmax(
        lambda_range=(-0.0, 0.2),
        n_grid=1000,
        idx_g0=1,
        idx_nu=2,
        apply_z_transform=False,
    )
    elapsed = time.perf_counter() - start
    print(f"Grid lambda^star (g0): {lambda_star_g0}, FD={val_star_g0}")
    print(f"Grid lambda^star (T):  {lambda_star_T}, FD={val_star_T}")
    print(f"Time for Gaussian copula grid evaluation: {elapsed:.3f} sec.")

    all_values = (
        [x[1] for x in copula_grid_g0]
        + [x[1] for x in copula_grid_T]
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
    marked_x_values = [[], [0.06, 0.11, 0.12]]  # [values for g0, values for T]
    plot_gaussian_copula_grid_pair(
        copula_grid_0=copula_grid_g0,
        copula_grid_1=copula_grid_T,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        filename=f"{prefix}_gaussian_copula_fd_grid_marked.pdf",
        logy=False,
        ylim=global_ylim,
        mark_max_point=False,
        show_grid_0=False,
        mark_x_values=marked_x_values,
        mark_x_red_idx=0,  # index in marked_x_values[1] that gets red star; others are black crosses
    )

    plot_gaussian_copula_fd_decomposition(
        copula_grid=copula_grid_g0,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        filename=f"{prefix}_copula_fd_decomposition_g0.pdf",
        label=r"$(G_0, \nu)$",
    )
    plot_gaussian_copula_fd_decomposition(
        copula_grid=copula_grid_T,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        filename=f"{prefix}_copula_fd_decomposition_T.pdf",
        label=r"$(T, \nu)$",
    )


if __name__ == "__main__":
    main()
