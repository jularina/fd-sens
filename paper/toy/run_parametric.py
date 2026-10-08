from src.parametric.corner_points import *
from src.common.utils.files_operations import *
from src.common.utils.distributions import DISTRIBUTION_MAP
from src.common.bayesian_model.base import BayesianModel
from src.parametric.plots.toy import *
from src.parametric.fisher import PosteriorFDParametric

import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd

warnings.filterwarnings("ignore", category=UserWarning)


def density_plot_across_multivariate_prior_parameter_sets(
    cfg,
    model,
    qf_priors_all_combinations,
):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_multivariate_joint_prior_densities_by_fd(
        results=qf_priors_all_combinations,
        output_dir=output_dir,
        plot_cfg=plot_cfg,
        true_theta=cfg.data.base_prior.mu,
        true_cov=cfg.data.base_prior.cov
    )


def plots_across_gaussian_prior_parameters_ranges(cfg, model: BayesianModel):
    """Compute the Fisher divergence over the Gaussian prior hyperparameter grid and plot it."""
    results = {}
    box_cfg = cfg.fd.optimize.prior.Gaussian.parameters_box_range
    distribution_cls = DISTRIBUTION_MAP["Gaussian"]
    param_names = list(box_cfg.ranges.keys())
    param_ranges = [
        np.round(np.linspace(*box_cfg.ranges[name], num=box_cfg.nums[name]), 2)
        for name in param_names
    ]
    for values in np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_names)):
        prior_params = dict(zip(param_names, values))
        model.set_candidate_prior_parameters(prior_params, distribution_cls=distribution_cls)
        estimator = PosteriorFDParametric(model=model)
        fisher = estimator.estimate_fisher_prior_only()
        results[tuple(values)] = fisher
        print(f"Prior: {prior_params}. Fisher Divergence: {fisher:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_multi_line_plots(results, param_names, plot_cfg, output_dir)


def plots_across_gaussian_parameters_ranges_etas_quadratic_form(cfg, eta_results, corner_points):
    """Plot the FD quadratic-form surface over the natural parameters (etas)."""
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_eta_surface(eta_results, corner_points, plot_cfg, output_dir)


def plots_across_gaussian_parameters_ranges_mu_sigma_quadratic_form(cfg, prior_combinations, prior_corners):
    """Plot the FD quadratic-form surface over the (mu, sigma) parametrisation."""
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_mu_sigma_contour(prior_combinations, prior_corners, plot_cfg, output_dir)


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian")
def run_gaussian_priors(cfg, save_samples: bool = False) -> None:
    """Compute the Fisher divergence and run the corner-point search for univariate Gaussian priors."""
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)

    fisher_estimator = PosteriorFDParametric(model=model)
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher_prior_only():.4f}")

    optimizer = OptimizationCornerPointsUnivariateGaussian(
        fisher_estimator,
        cfg.fd.optimize.prior.Gaussian,
        cfg.fd.optimize.loss.GaussianLogLikelihood
    )
    prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
    prior_combinations = optimizer.evaluate_all_prior_combinations()

    plots_across_gaussian_prior_parameters_ranges(cfg, model)
    plots_across_gaussian_parameters_ranges_etas_quadratic_form(cfg, prior_combinations, prior_corners)
    plots_across_gaussian_parameters_ranges_mu_sigma_quadratic_form(cfg, prior_combinations, prior_corners)


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors(cfg, save_samples: bool = False) -> None:
    """Compute the Fisher divergence and run the corner-point search for multivariate Gaussian priors."""
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)

    fisher_estimator = PosteriorFDParametric(model=model)
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher_prior_only():.4f}")

    optimizer = OptimizationCornerPointsMultivariateGaussian(
        fisher_estimator, cfg.fd.optimize.prior.MultivariateGaussian,
        cfg.fd.optimize.loss.MultivariateGaussianLogLikelihood)
    qf_priors_all_combinations = optimizer.evaluate_all_prior_combinations()

    density_plot_across_multivariate_prior_parameter_sets(
        cfg, model, qf_priors_all_combinations=qf_priors_all_combinations)


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
def comparison_plot_existing_methods(cfg):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    mu_ref = np.array([3.06293078, 3.05897246])
    mu_cand_1 = np.array([3.06293078, 3.05897246])
    mu_cand_2 = np.array([3.06293078, 3.05897246])
    Sigma_ref = np.array([[7.95761567e-03, 2.11077339e-05],
                          [2.11077339e-05, 7.95761567e-03]])
    Sigma_1 = np.array([[2.00e-02, 4.00e-03],
                        [4.00e-03, 6.00e-03]])
    Sigma_2 = np.array([[5.00e-03, -3.00e-03],
                        [-3.00e-03, 2.00e-02]])

    plot_existing_methods_comparison_gaussians(
        output_dir=output_dir,
        plot_cfg=plot_cfg,
        mu_ref=mu_ref,
        mu_cand_1=mu_cand_1,
        mu_cand_2=mu_cand_2,
        Sigma_ref=Sigma_ref,
        Sigma_cand_1=Sigma_1,
        Sigma_cand_2=Sigma_2,
        filename="comparison_same_mean_diff_cov.pdf",
        annotation_fontsize=10,
        annotation_text=(
            r"$\rho^{\mathrm{mean}}(\tilde{\Pi}^{\lambda_j})=0$" "\n"
            r"$\rho^{\mathrm{FD}}(\tilde{\Pi}^{\lambda_j})>0$"
        ),
    )
    mu_ref = np.array([3.06293078, 3.05897246])
    mu_cand_1 = np.array([3.4, 2.5])
    mu_cand_2 = np.array([3.5, 2.7])
    Sigma_ref = np.array([[7.95761567e-03, 2.11077339e-05],
                          [2.11077339e-05, 7.95761567e-03]])
    Sigma_1 = np.array([[7.95761567e-03, 2.11077339e-05],
                        [2.11077339e-05, 7.95761567e-03]])
    Sigma_2 = np.array([[7.95761567e-03, 2.11077339e-05],
                        [2.11077339e-05, 7.95761567e-03]])

    plot_existing_methods_comparison_gaussians(
        output_dir=output_dir,
        plot_cfg=plot_cfg,
        mu_ref=mu_ref,
        mu_cand_1=mu_cand_1,
        mu_cand_2=mu_cand_2,
        Sigma_ref=Sigma_ref,
        Sigma_cand_1=Sigma_1,
        Sigma_cand_2=Sigma_2,
        filename="comparison_same_cov_diff_mean.pdf",
        annotation_text=(
            r"$\rho^{\mathrm{cov}}(\tilde{\Pi}^{\lambda_j})=0$" "\n"
            r"$\rho^{\mathrm{FD}}(\tilde{\Pi}^{\lambda_j})>0$"
        ),
    )

    comparison_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian/comparison/")

    combined_results_path = os.path.join(comparison_dir, "finite_sample_results.json")
    if os.path.exists(combined_results_path):
        combined_results = load_results_json(combined_results_path)
        combined_results = convert_dim_keys_to_int(combined_results)
        error_ylim = compute_global_ylim_error(combined_results, logy=True)
        time_ylim = compute_global_ylim_time(combined_results, logy=True)
    else:
        error_ylim = None
        time_ylim = None

    divergence_configs = {
        "fd": {
            "ms": list(range(500, 10001, 500)),
            "dims": [5, 25, 100],
        },
        "mean": {
            "ms": list(range(500, 10001, 500)),
            "dims": [5, 25, 100],
        },
        "wim": {
            "ms": list(range(1000, 5001, 500)),
            "dims": [5, 25, 100],
        },
        "kl": {
            "ms": list(range(1000, 5001, 500)),
            "dims": [5, 25, 100],
        },
    }

    all_ms = [m for cfg in divergence_configs.values() for m in cfg["ms"]]
    xlim = (min(all_ms), max(all_ms)+1000)

    for divergence, config in divergence_configs.items():
        ms = config["ms"]
        dims = config["dims"]
        results_path = os.path.join(comparison_dir, f"finite_sample_results_{divergence}.json")

        if os.path.exists(results_path):
            results = load_results_json(results_path)
            results = convert_dim_keys_to_int(results)
        else:
            results = compute_gaussian_complexity_results(
                ms=ms,
                dims=dims,
                n_rep=500,
                seed=27,
                divergence=divergence,
            )
            save_to_serializable_json(results, results_path)

        plot_finite_sample_complexity_gaussians(
            output_dir=output_dir,
            plot_cfg=plot_cfg,
            divergence=divergence,
            ms=ms,
            dims=dims,
            logy=True,
            results=results,
            ylim=error_ylim,
            xlim=xlim,
        )

        plot_runtime_complexity_gaussians(
            output_dir=output_dir,
            plot_cfg=plot_cfg,
            divergence=divergence,
            ms=ms,
            dims=dims,
            logy=True,
            results=results,
            ylim=time_ylim,
            xlim=xlim,
        )


if __name__ == "__main__":
    # run_gaussian_priors()
    # run_multivariate_gaussian_priors()
    comparison_plot_existing_methods()
