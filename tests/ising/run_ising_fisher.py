import warnings
import os
import hydra
import numpy as np
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from src.utils.files_operations import load_plot_config
from src.plots.paper.ising_model_paper_funcs import *
from src.discrepancies.posterior_fisher import PosteriorFDParametric as PosteriorFDBase
from src.losses.ising.ising_gradients import IsingGradients

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model_smaller_dataset")
def main(cfg: DictConfig, dnum=100, pnum=5000, epsilon=0.4) -> None:
    loss_configs = {
        "pseudolikelihood": {"matsubara": 0.6, "syring": 0.279, "lyddon": 0.635},
        "dfd": {"matsubara": 0.12, "syring": 0.018, "lyddon": 0.017854},
        "ksd": {"matsubara": 10.14, "syring": 1.06},
    }
    loss_configs = {
        "pseudolikelihood": {"matsubara": 0.7273, "syring": 0.486, "lyddon": 0.668050},
        "dfd": {"matsubara": 1.5271, "syring": 0.023, "lyddon": 0.031323},
        "ksd": {"matsubara": 53, "syring": 2.685},
    }
    loss_to_file_name = {"pseudolikelihood": "PseudoBayes", "dfd": "FDBayes", "ksd": "KSDBayes"}
    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    base = "/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis"

    for loss, beta_refs in loss_configs.items():
        methods = ["matsubara", "syring"] if loss == "ksd" else ["matsubara", "syring", "lyddon"]
        for method in methods:
            beta_ref = beta_refs[method]
            cfg.data.loss_lr_init = beta_ref
            cfg.data.posterior_samples_path = f"{base}/{loss_to_file_name[loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{loss}_posteriors_samples_{method}.npy"
            cfg.data.pseudoliklelhood_grads_path = f"{base}/{loss_to_file_name[loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{loss}_grads_{method}.npy"
            model = instantiate(cfg.model, data_config=cfg.data)
            fisher_estimator = PosteriorFDBase(model=model)
            print(f"[{loss}/{method}] Initial Fisher: {fisher_estimator.estimate_fisher_lr_only():.4f}")

            results = {}
            left = beta_ref - epsilon if beta_ref - epsilon > 0 else 0.01
            right = beta_ref + epsilon
            grid = np.sort(np.concatenate([np.linspace(left, right, 999), [beta_ref]]))
            for lr in grid:
                model.set_lr_parameter(lr)
                fisher_estimator = PosteriorFDBase(model=model)
                fisher = fisher_estimator.estimate_fisher_lr_only()
                results[lr] = fisher
                print(f"Lr: {lr}, FD: {fisher:.4f}")

            arr = np.array(list(results.items()))
            np.save(
                data_path + f"{loss_to_file_name[loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_data_{loss}_lr_optimisation_{method}.npy", arr)


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model")
def create_combined_plots(cfg: DictConfig, dnum=1000, pnum=5000):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir, f"{dnum}")
    plot_cfg = load_plot_config(plot_config_path)
    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    losses = ["pseudolikelihood", "dfd", "ksd"]
    methods = ["matsubara", "syring", "lyddon"]
    loss_to_file_name = {"pseudolikelihood": "PseudoBayes", "dfd": "FDBayes", "ksd": "KSDBayes"}
    beta_refs_by_method = {
        "matsubara": [0.6, 0.12, 10.14],
        "syring": [0.279, 0.018, 1.06],
        "lyddon": [0.635, 0.017854, 0.00000001],
    }
    # beta_refs_by_method = {
    #     "matsubara": [0.7273, 1.5271, 53],
    #     "syring": [0.486, 0.023, 2.685],
    #     "lyddon": [0.668050, 0.031323, 0.00000001],
    # }
    method_labels = {
        "matsubara": "Matsubara et.al.",
        "syring": "Syring et.al.",
        "lyddon": "Lyddon et.al.",
    }

    base = "/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis"
    observations_path = f"{base}/PseudoBayes_size=6_theta=5.0_dnum=1000_pnum=2000_data.npy"
    X_obs = np.load(observations_path)
    ising_grads = IsingGradients(size=6)

    for grad_loss in ["pseudolikelihood", "dfd"]:
        samples_by_method = {}
        for method in methods:
            path = f"{base}/{loss_to_file_name[grad_loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{grad_loss}_posteriors_samples_{method}.npy"
            if os.path.exists(path):
                samples_by_method[method] = np.load(path)
        plot_loss_gradient_vs_theta(
            X=X_obs,
            ising_grads=ising_grads,
            loss=grad_loss,
            samples_by_method=samples_by_method,
            method_labels=method_labels,
            theta_min=3.5,
            theta_max=8.0,
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
            theta_max=5.5,
            n_theta=1000,
            plot_cfg=plot_cfg,
            output_dir=output_dir,
            filename=f"ising-loss-gradient-times-density-{grad_loss}-{dnum}.pdf",
        )

    all_grids = {}
    for loss_idx, loss in enumerate(losses):
        all_grids[loss] = {}
        for method in methods:
            if loss == "ksd" and method_labels[method] == "Lyddon et.al.":
                continue
            arr = np.load(
                os.path.join(
                    data_path,
                    f"{loss_to_file_name[loss]}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_data_{loss}_lr_optimisation_{method}.npy"
                )
            )
            all_grids[loss][method] = arr

    all_values = []
    for loss_idx, loss in enumerate(losses):
        for method in methods:
            if loss == "ksd" and method_labels[method] == "Lyddon et.al.":
                continue
            arr = all_grids[loss][method]
            xs = np.array(arr[:, 0], dtype=float)
            ys = np.array(arr[:, 1], dtype=float)
            beta_ref = beta_refs_by_method[method][loss_idx]
            left = max(beta_ref - 0.05, 0.01)
            right = beta_ref + 0.05
            mask = (xs >= left) & (xs <= right)
            all_values.extend(ys[mask][ys[mask] > 0].tolist())

    global_ylim = (min(all_values), 5.0)

    for loss_idx, loss in enumerate(losses):
        lr_grids = []
        beta_refs = []
        labels = []

        for method in methods:
            if loss == "ksd" and method_labels[method] == "Lyddon et.al.":
                continue
            lr_grids.append(all_grids[loss][method])
            beta_refs.append(beta_refs_by_method[method][loss_idx])
            labels.append(method_labels[method])

        plot_lr_vs_method_multi(
            lr_grids=lr_grids,
            methods=labels,
            beta_refs=beta_refs,
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model")
def create_posterior_histogram_plots(cfg: DictConfig, dnum=1000, pnum=5000) -> None:
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir, f"{dnum}")
    plot_cfg = load_plot_config(plot_config_path)

    base = "/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis/"
    loss_to_file_name = {"pseudolikelihood": "PseudoBayes", "dfd": "FDBayes", "ksd": "KSDBayes"}
    loss_labels = {
        "pseudolikelihood": r"$L^\mathrm{PL}$",
        "dfd": r"$L^\mathrm{DFD}$",
        "ksd": r"$L^\mathrm{KSD}$",
    }
    method_labels = {
        "matsubara": "Matsubara et.al.",
        "syring": "Syring et.al.",
        "lyddon": "Lyddon et.al.",
    }

    samples_by_loss = {}
    for loss, prefix in loss_to_file_name.items():
        samples_by_loss[loss] = {}
        for method in method_labels:
            if loss == "ksd" and method_labels[method] == "Lyddon et.al.":
                continue
            path = f"{base}{prefix}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{loss}_posteriors_samples_{method}_worst_case_lr.npy"
            if os.path.exists(path):
                samples_by_loss[loss][method] = np.load(path)

    ref_samples_by_loss = {}
    for loss, prefix in loss_to_file_name.items():
        ref_samples_by_loss[loss] = {}
        for method in method_labels:
            if loss == "ksd" and method_labels[method] == "Lyddon et.al.":
                continue
            path = f"{base}{prefix}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{loss}_posteriors_samples_{method}.npy"
            if not os.path.exists(path):
                loss_typo = loss.replace("pseudolikelihood", "pseudoliklelhood")
                path = f"{base}{prefix}_size=6_theta=5.0_dnum={dnum}_pnum={pnum}_{loss_typo}_posteriors_samples_{method}.npy"
            if os.path.exists(path):
                ref_samples_by_loss[loss][method] = np.load(path)

    # plot_posterior_histograms(
    #     samples_by_loss=samples_by_loss,
    #     loss_labels=loss_labels,
    #     method_labels=method_labels,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     dnum=dnum,
    #     ref_samples_by_loss=None
    # )

    plot_posterior_per_combination(
        samples_by_loss=samples_by_loss,
        ref_samples_by_loss=ref_samples_by_loss,
        loss_labels=loss_labels,
        method_labels=method_labels,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        dnum=dnum,
    )


if __name__ == "__main__":
    # main()
    create_combined_plots()
    # create_posterior_histogram_plots()
