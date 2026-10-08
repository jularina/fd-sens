import numpy as np
import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time

from src.parametric.fisher import PosteriorFDParametric
from src.common.utils.files_operations import load_plot_config
from src.parametric.plots.posteriordb import plot_complexity_bar
from src.parametric.corner_points import OptimizationCornerPointsCompositePrior

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")

TIMING_DIR = "data/ark"


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_posteriordb")
def main(cfg: DictConfig) -> None:
    """
    Times the three optimisation routines (full convex QF, per-component QF, black-box dual annealing)
    and saves the timings used by compare_complexities.
    """
    n_runs = cfg.playground.get("n_timing_runs", 500)
    n_runs_bb = cfg.playground.get("n_timing_runs_bb", 100)
    timing_output_dir = os.path.join(get_original_cwd(), TIMING_DIR)
    os.makedirs(timing_output_dir, exist_ok=True)

    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDParametric(model=model)
    print(f"Initial Fisher for prior: {fisher_estimator.estimate_fisher_prior_only():.4f}")

    optimizer = OptimizationCornerPointsCompositePrior(
        fisher_estimator,
        cfg.fd.optimize.prior.Composite,
        cfg.fd.optimize.loss.GaussianARLogLikelihood,
    )
    names = ["alpha", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma"]

    print(f"Starting optimisation of all parameters at once ({n_runs} runs).")
    times_full = np.empty(n_runs)
    for i in range(n_runs):
        start = time.perf_counter()
        optimizer.evaluate_all_prior_corners()
        optimizer.minimize_prior_full_qp()
        times_full[i] = time.perf_counter() - start
        print(f"  Run {i + 1}/{n_runs}: {times_full[i]:.3f} sec.")
    np.savez(os.path.join(timing_output_dir, "timing_qf_full.npz"), times=times_full)

    print(f"Starting per component optimisation ({n_runs} runs).")
    times_decomp = np.empty(n_runs)
    for i in range(n_runs):
        start = time.perf_counter()
        optimizer.evaluate_all_prior_corners_per_component(component_names=names)
        optimizer.minimize_prior_per_component_qp(names)
        times_decomp[i] = time.perf_counter() - start
        print(f"  Run {i + 1}/{n_runs}: {times_decomp[i]:.3f} sec.")
    np.savez(os.path.join(timing_output_dir, "timing_qf_decomp.npz"), times=times_decomp)

    print(f"Starting black-box optimisation ({n_runs_bb} runs).")
    times_bb = np.empty(n_runs_bb)
    for i in range(n_runs_bb):
        start = time.perf_counter()
        optimizer.black_box_optimize_prior_box_global(
            method="dual_annealing",
            seed=i,
            maxiter=150,
            n_restarts=5,
        )
        times_bb[i] = time.perf_counter() - start
        print(f"  Run {i + 1}/{n_runs_bb}: {times_bb[i]:.3f} sec.")
    np.savez(os.path.join(timing_output_dir, "timing_bb.npz"), times=times_bb)
    print(f"Saved timing results to {timing_output_dir}")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_posteriordb")
def compare_complexities(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    timing_dir = os.path.join(get_original_cwd(), TIMING_DIR)
    qf_full_time_sec = np.load(os.path.join(timing_dir, "timing_qf_full.npz"))["times"]
    qf_decomp_time_sec = np.load(os.path.join(timing_dir, "timing_qf_decomp.npz"))["times"]
    black_box_time_sec = np.load(os.path.join(timing_dir, "timing_bb.npz"))["times"]

    plot_complexity_bar(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        filename="ark_computational_cost.pdf",
        use_log10=True,
        qf_full_time_sec=qf_full_time_sec,
        qf_decomp_time_sec=qf_decomp_time_sec,
        black_box_time_sec=black_box_time_sec,
    )


if __name__ == "__main__":
    # main()
    compare_complexities()
