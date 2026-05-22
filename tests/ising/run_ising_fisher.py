import warnings
import os
import hydra
import numpy as np
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from src.utils.files_operations import load_plot_config
from src.plots.paper.ising_model_paper_funcs import *
from src.discrepancies.posterior_fisher import PosteriorFDParametric as PosteriorFDBase

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model")
def main(cfg: DictConfig) -> None:
    loss = cfg.data.loss_type
    beta_refs = cfg.data.beta_refs
    loss_to_file_name = {"pseudolikelihood": "PseudoBayes", "dfd": "FDBayes", "ksd": "KSDBayes"}

    # Matsubara
    cfg.data.loss_lr_init = beta_refs["matsubara"]
    cfg.data.posterior_samples_path = f"/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis/{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_{loss}_posteriors_samples_matsubara.npy"
    cfg.data.pseudoliklelhood_grads_path = f"/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis/{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_{loss}_grads_matsubara.npy"
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher_lr_only():.4f}")

    results = {}
    left = beta_refs["matsubara"]-0.05 if beta_refs["matsubara"]-0.05 > 0 else 0.01
    right = beta_refs["matsubara"] + 0.05
    for lr in np.linspace(left, right, 1000):
        model.set_lr_parameter(lr)
        fisher_estimator = PosteriorFDBase(model=model)
        fisher = fisher_estimator.estimate_fisher_lr_only()
        results[lr] = fisher
        print(f"Lr: {lr}, FD: {fisher:.4f}")

    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    arr = np.array(list(results.items()))
    np.save(
        data_path + f"{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_data_{loss}_lr_optimisation_matsubara.npy", arr)

    # Syring
    cfg.data.loss_lr_init = beta_refs["syring"]
    cfg.data.posterior_samples_path = f"/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis/{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_{loss}_posteriors_samples_syring.npy"
    cfg.data.pseudoliklelhood_grads_path = f"/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis/{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_{loss}_grads_syring.npy"
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher_lr_only():.4f}")

    results = {}
    left = beta_refs["syring"]-0.05 if beta_refs["syring"]-0.05 > 0 else 0.01
    right = beta_refs["syring"] + 0.05
    for lr in np.linspace(left, right, 1000):
        model.set_lr_parameter(lr)
        fisher_estimator = PosteriorFDBase(model=model)
        fisher = fisher_estimator.estimate_fisher_lr_only()
        results[lr] = fisher
        print(f"Lr: {lr}, FD: {fisher:.4f}")

    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    arr = np.array(list(results.items()))
    np.save(
        data_path + f"{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_data_{loss}_lr_optimisation_syring.npy", arr)

    # Lyddon
    cfg.data.loss_lr_init = beta_refs["lyddon"]
    cfg.data.posterior_samples_path = f"/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis/{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_{loss}_posteriors_samples_lyddon.npy"
    cfg.data.pseudoliklelhood_grads_path = f"/Users/arinaodv/Desktop/folder/study_phd/code/Discrete-Fisher-Bayes/Ising/samplesForKSDSensitivityAnalysis/{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_{loss}_grads_lyddon.npy"
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher_lr_only():.4f}")

    results = {}
    left = beta_refs["lyddon"]-0.05 if beta_refs["lyddon"]-0.05 > 0 else 0.01
    right = beta_refs["lyddon"] + 0.05
    for lr in np.linspace(left, right, 1000):
        model.set_lr_parameter(lr)
        fisher_estimator = PosteriorFDBase(model=model)
        fisher = fisher_estimator.estimate_fisher_lr_only()
        results[lr] = fisher
        print(f"Lr: {lr}, FD: {fisher:.4f}")

    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    arr = np.array(list(results.items()))
    np.save(
        data_path + f"{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_data_{loss}_lr_optimisation_lyddon.npy", arr)


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model")
def create_combined_plots(cfg: DictConfig):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    # lr_combinations_m = np.load(data_path + "PseudoBayes_size=6_theta=5.0_dnum=1000_pnum=2000_data_pseudolikelihood_lr_optimisation_matsubara.npy")
    # lr_combinations_s = np.load(data_path + "PseudoBayes_size=6_theta=5.0_dnum=1000_pnum=2000_data_pseudolikelihood_lr_optimisation_syring.npy")
    # lr_combinations_l = np.load(
    #     data_path + "PseudoBayes_size=6_theta=5.0_dnum=1000_pnum=2000_data_pseudolikelihood_lr_optimisation_lyddon.npy")
    #
    # plot_lr_vs_ksd_multi(
    #     lr_grids=[lr_combinations_m, lr_combinations_s, lr_combinations_l],
    #     ds=["Matsubara et.al.", "Syring et.al.", "Lyddon et.al."],
    #     beta_refs=[0.6, 0.52, 0.64],
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     filename="ising-lr-comparison.pdf",
    #     xlabel=r"$\beta$",
    #     ylbl="estimatedFDposteriorsQuadraticForm",
    # )

    losses = ["pseudolikelihood", "dfd", "ksd"]
    methods = ["matsubara", "syring", "lyddon"]
    loss_to_file_name = {"pseudolikelihood": "PseudoBayes", "dfd": "FDBayes", "ksd": "KSDBayes"}
    beta_refs_by_method = {
        "matsubara": [0.6, 0.02, 0.39],
        "syring": [0.52, 0.34, 0.33],
        "lyddon": [0.64, 0.015, 2.34],
    }
    method_labels = {
        "matsubara": "Matsubara et.al.",
        "syring": "Syring et.al.",
        "lyddon": "Lyddon et.al.",
    }

    # for method in ["matsubara", "syring", "lyddon"]:
    #     lr_grids = []
    #     for loss in losses:
    #         arr = np.load(
    #             os.path.join(
    #                 data_path,
    #                 f"{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_data_{loss}_lr_optimisation_{method}.npy"
    #             )
    #         )
    #         lr_grids.append(arr)
    #     plot_lr_vs_loss_multi(
    #         lr_grids=lr_grids,
    #         losses=[r"$L^\mathrm{PL}$", r"$L^\mathrm{DFD}$", r"$L^\mathrm{KSD}$"],
    #         beta_refs=beta_refs_by_method[method],
    #         plot_cfg=plot_cfg,
    #         output_dir=output_dir,
    #         filename=f"ising-lr-comparison-{method}.pdf",
    #         xlabel=r"$\beta$",
    #         legend=False,
    #         ylbl="estimatedFDposteriorsQuadraticForm",
    #         logy=True,
    #         method=method,
    #     )

    loss_labels = {
        "pseudolikelihood": r"$L^\mathrm{PL}$",
        "dfd": r"$L^\mathrm{DFD}$",
        "ksd": r"$L^\mathrm{KSD}$",
    }

    # Pre-load all grids and compute global y limits across all losses and methods
    all_grids = {}
    for loss_idx, loss in enumerate(losses):
        all_grids[loss] = {}
        for method in methods:
            arr = np.load(
                os.path.join(
                    data_path,
                    f"{loss_to_file_name[loss]}_size=6_theta=5.0_dnum=1000_pnum=2000_data_{loss}_lr_optimisation_{method}.npy"
                )
            )
            all_grids[loss][method] = arr

    all_values = []
    for loss_idx, loss in enumerate(losses):
        for method in methods:
            arr = all_grids[loss][method]
            xs = np.array(arr[:, 0], dtype=float)
            ys = np.array(arr[:, 1], dtype=float)
            beta_ref = beta_refs_by_method[method][loss_idx]
            left = max(beta_ref - 0.05, 0.01)
            right = beta_ref + 0.05
            mask = (xs >= left) & (xs <= right)
            all_values.extend(ys[mask].tolist())

    global_ylim = (min(all_values), max(all_values))

    for loss_idx, loss in enumerate(losses):
        lr_grids = []
        beta_refs = []
        labels = []

        for method in methods:
            lr_grids.append(all_grids[loss][method])
            beta_refs.append(beta_refs_by_method[method][loss_idx])
            labels.append(method_labels[method])

        plot_lr_vs_method_multi(
            lr_grids=lr_grids,
            methods=labels,
            beta_refs=beta_refs,
            plot_cfg=plot_cfg,
            output_dir=output_dir,
            filename=f"ising-lr-comparison-{loss}.pdf",
            xlabel=r"$\lambda_L$",
            legend=False,
            ylbl="estimatedFDposteriorsQuadraticForm",
            logy=True,
            loss=loss,
            ylim=global_ylim,
            lr_bars=None
        )


if __name__ == "__main__":
    # main()
    create_combined_plots()
