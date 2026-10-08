import time
import warnings
import os
import hydra
import numpy as np
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from src.common.utils.files_operations import load_plot_config
from src.parametric.plots.ising import *
from src.parametric.fisher import PosteriorFDParametric
from src.common.losses.ising.ising_gradients import IsingGradients

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")

# Reference learning rates from Matsubara et al., Syring & Hong and Lyddon et al. (paper, Section 5.2)
BETA_REFS = {
    "pseudolikelihood": {"matsubara": 0.6, "syring": 0.279, "lyddon": 0.635},
    "dfd": {"matsubara": 0.12, "syring": 0.018, "lyddon": 0.017854},
}
LOSS_TO_FILE_NAME = {"pseudolikelihood": "PseudoBayes", "dfd": "FDBayes"}
SAMPLES_DIR = "/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis"


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model")
def main(cfg: DictConfig, dnum=1000, pnum=5000, epsilon=0.4) -> None:
    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")

    for loss, beta_refs in BETA_REFS.items():
        for method, beta_ref in beta_refs.items():
            cfg.data.loss_lr_init = beta_ref
            cfg.data.posterior_samples_path = f"{SAMPLES_DIR}/{LOSS_TO_FILE_NAME[loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{loss}_posteriors_samples_{method}.npy"
            cfg.data.pseudoliklelhood_grads_path = f"{SAMPLES_DIR}/{LOSS_TO_FILE_NAME[loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{loss}_grads_{method}.npy"
            model = instantiate(cfg.model, data_config=cfg.data)
            start_time = time.time()
            fisher_estimator = PosteriorFDParametric(model=model)
            print(f"[{loss}/{method}] Initial Fisher: {fisher_estimator.estimate_fisher_lr_only():.4f}")
            print(f"Time: {time.time() - start_time}")

            results = {}
            left = beta_ref - epsilon if beta_ref - epsilon > 0 else 0.01
            right = beta_ref + epsilon
            grid = np.sort(np.concatenate([np.linspace(left, right, 999), [beta_ref]]))
            for lr in grid:
                model.set_lr_parameter(lr)
                fisher_estimator = PosteriorFDParametric(model=model)
                fisher = fisher_estimator.estimate_fisher_lr_only()
                results[lr] = fisher
                print(f"Lr: {lr}, FD: {fisher:.4f}")

            arr = np.array(list(results.items()))
            np.save(
                data_path + f"{LOSS_TO_FILE_NAME[loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_data_{loss}_lr_optimisation_{method}.npy", arr)


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model")
def create_combined_plots(cfg: DictConfig, dnum=1000, pnum=5000):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir, f"{dnum}")
    plot_cfg = load_plot_config(plot_config_path)
    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    methods = ["matsubara", "syring", "lyddon"]
    method_labels = {
        "matsubara": "Matsubara et.al.",
        "syring": "Syring et.al.",
        "lyddon": "Lyddon et.al.",
    }

    observations_path = f"{SAMPLES_DIR}/PseudoBayes_size=6_theta=5.0_dnum=1000_pnum=2000_data.npy"
    X_obs = np.load(observations_path)
    ising_grads = IsingGradients(size=6)

    for grad_loss in ["pseudolikelihood", "dfd"]:
        samples_by_method = {}
        for method in methods:
            path = f"{SAMPLES_DIR}/{LOSS_TO_FILE_NAME[grad_loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{grad_loss}_posteriors_samples_{method}.npy"
            if os.path.exists(path):
                samples_by_method[method] = np.load(path)
        plot_loss_gradient_vs_theta(
            X=X_obs,
            ising_grads=ising_grads,
            loss=grad_loss,
            samples_by_method=samples_by_method,
            method_labels=method_labels,
            theta_min=3.5,
            theta_max=6.0,
            n_theta=1000,
            plot_cfg=plot_cfg,
            output_dir=output_dir,
            filename=f"ising-loss-gradient-{grad_loss}-{dnum}.pdf",
        )
        plot_loss_gradient_times_density(
            X=X_obs,
            ising_grads=ising_grads,
            loss=grad_loss,
            samples_by_method=samples_by_method,
            method_labels=method_labels,
            theta_min=3.5,
            theta_max=6.0,
            n_theta=1000,
            plot_cfg=plot_cfg,
            output_dir=output_dir,
            filename=f"ising-loss-gradient-times-density-{grad_loss}-{dnum}.pdf",
        )

    for loss, beta_refs in BETA_REFS.items():
        lr_grids = [
            np.load(os.path.join(
                data_path,
                f"{LOSS_TO_FILE_NAME[loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_data_{loss}_lr_optimisation_{method}.npy"
            ))
            for method in methods
        ]
        plot_lr_vs_method_multi(
            lr_grids=lr_grids,
            methods=[method_labels[m] for m in methods],
            beta_refs=[beta_refs[m] for m in methods],
            plot_cfg=plot_cfg,
            output_dir=output_dir,
            filename=f"ising-lr-comparison-{loss}-{dnum}.pdf",
            xlabel=r"$\lambda_L$",
            legend=False,
            ylbl="estimatedFDposteriorsQuadraticForm",
            logy=False,
            loss=loss,
            ylim=None,
            lr_bars=None
        )


if __name__ == "__main__":
    # main()
    create_combined_plots()
