from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.optimization.corner_points_fisher import *
from src.utils.files_operations import *
from src.utils.distributions import DISTRIBUTION_MAP
from src.bayesian_model.base import BayesianModel
from src.plots.paper.toy_paper_fisher_funcs import *
from src.discrepancies.prior_fisher import PriorFDNonParametric
from src.discrepancies.posterior_fisher import PosteriorFDBase, PosteriorFDParametric, PosteriorFDNonParametric

import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
import time
import json

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
    """
    Recalculates Fisher along all the possible hyperparameters combination across the ranges
    """
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
    """
    Recalculates FD along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
    """
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_eta_surface(eta_results, corner_points, plot_cfg, output_dir)


def plots_across_gaussian_parameters_ranges_mu_sigma_quadratic_form(cfg, prior_combinations, prior_corners):
    """
    Same grid as plots_across_gaussian_parameters_ranges_etas_quadratic_form but
    visualises the (non-convex) quadratic form surface over the original
    (mu, sigma) parametrisation instead of the natural parameters.
    """
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_mu_sigma_contour(prior_combinations, prior_corners, plot_cfg, output_dir)


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian")
def run_gaussian_priors(cfg, save_samples: bool = True) -> None:
    """
    Main function to compute Fisher divergence and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
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
def run_multivariate_gaussian_priors(cfg, save_samples: bool = True) -> None:
    """
    Main function to compute Fisher and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    # Nonparametric optimisation
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    sdp_psi_list, sdp_fd_estimates_list, radius_labels = [], [], []
    sdp_psi_list, sdp_fd_estimates_list = [], []

    for radius in [0.5, 1.0, 5.0, 10.0]:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius=radius
        )
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        elapsed = time.perf_counter() - start
        print(f"SDP primal relaxation time: {elapsed}")

        start = time.perf_counter()
        result_lagrange_dual = optimizer.optimize_through_dual_1d_lambda()
        elapsed = time.perf_counter() - start
        print(f"Lagrange dual time: {elapsed}")

        start = time.perf_counter()
        result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
        elapsed = time.perf_counter() - start
        print(f"SDP dual time: {elapsed}")

        sdp_psi_list.append(result_sdp["eta_star"])
        sdp_fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_sdp_densities_and_logprior(
        basis_function=optimizer.basis_function,
        psi_sdp_list=sdp_psi_list,
        radius_labels=radius_labels,
        estimates=sdp_fd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-5, 12),
        resolution=300
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = True) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    # Nonparametric optimisation
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    psi_sdp_list, fd_estimates_list, radius_labels = [], [], []

    for radius in [0.5, 1.0, 5.0, 10.0]:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius=radius
        )
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        elapsed = time.perf_counter() - start
        print(f"SDP relaxation time: {elapsed}")

        psi_sdp_list.append(result_sdp["eta_star"])
        fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_sdp_2d_densities(
        basis_function=optimizer.basis_function,
        psi_sdp_list=psi_sdp_list,
        radius_labels=radius_labels,
        ksd_estimates=fd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-10, 15), (-10, 15)),
        resolution=300
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_nonparametric_basis_funcs_nums(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    posterior_samples = np.load(output_dir + "/posterior_samples.npy")
    prior_samples = np.load(output_dir + "/prior_samples.npy")

    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")
    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []

    # Nonparametric optimization
    estimator_prior = PriorFDNonParametric(samples=prior_samples, model=model, candidate_type="prior")
    estimator_posterior = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")

    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []
    basis_funcs_nums = [5, 10, 15, 20]
    basis_list = []

    for basis_funcs_num in basis_funcs_nums:
        cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius_lower_bound=5.0
        )
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        elapsed = time.perf_counter() - start
        print(f"SDP time: {elapsed}")

        start = time.perf_counter()
        result_tr = optimizer.optimize_through_qcqp_trust_region()
        elapsed = time.perf_counter() - start
        print(f"TR time: {elapsed}")

        psi_sdp_list.append(result_sdp["psi_opt"])
        ksd_estimates_list.append(result_sdp["est"])
        basis_list.append(optimizer.basis_function)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_sdp_2d_densities_flexible(
        basis_functions=basis_list,
        psi_sdp_list=psi_sdp_list,
        labels=basis_funcs_nums,
        ksd_estimates=ksd_estimates_list,
        label_template=r"K = {label} ({approx} {ksd:.2f})",
        legend_title=plot_cfg.plot.param_latex_names.get("estimatedFDposteriorsShort", ""),
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-10, 15), (-10, 15)),
        resolution=300,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_priors_nonparametric_diff_samples_num(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    times_list_parametric, times_list_nonparametric = [], []
    samples_nums_list = [int(x) for x in np.linspace(1000, 10000, 10)]
    basis_funcs_num_list = [int(x) for x in np.linspace(5, 15, 3)]
    times_parametric, times_nonparametric = defaultdict(dict),  defaultdict(lambda: defaultdict(dict))
    steps = 200

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            cfg.data.posterior_samples_num = sample_nums
            model = instantiate(cfg.model, data_config=cfg.data)
            start = time.perf_counter()
            fisher_estimator = PosteriorFDBase(model=model)
            optimizer = OptimizationCornerPointsUnivariateGaussian(
                fisher_estimator,
                cfg.ksd.optimize.prior.Gaussian,
                cfg.ksd.optimize.loss.GaussianLogLikelihood
            )
            prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
            elapsed = time.perf_counter() - start
            largest_fd = prior_corners[0][2]
            times_list_parametric.append((sample_nums, elapsed))
            times_parametric[sample_nums][step] = elapsed
            print(f"***Parametric*** Samples: {sample_nums}, Initial FD: {largest_fd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/univariate_gaussian/runtimes/")
    os.makedirs(data_path, exist_ok=True)
    with open(data_path + "parametric_qcqp_optimisation_times.json", "w") as f:
        json.dump(times_parametric, f, indent=4)

    samples_nums_list = [int(x) for x in np.linspace(500, 10000, 10)]
    for step in range(steps):
        print(f"Non-parametric QCQP running step {step}.")
        for sample_nums in samples_nums_list:
            for basis_funcs_num in basis_funcs_num_list:
                cfg.data.posterior_samples_num = sample_nums
                cfg.data.prior_samples_num = sample_nums
                cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
                model = instantiate(cfg.model, data_config=cfg.data)
                start = time.perf_counter()
                estimator_prior = PriorFDNonParametric(model=model)
                estimator_posterior = PosteriorFDNonParametric(model=model)
                optimizer = OptimisationNonparametricBase(
                    estimator_posterior,
                    estimator_prior,
                    cfg.ksd.optimize.prior.nonparametric,
                    radius=2.0
                )
                result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
                elapsed = time.perf_counter() - start
                largest_fd = result_sdp_dual["dual_value"]
                times_list_nonparametric.append((sample_nums*2, basis_funcs_num, elapsed))
                times_nonparametric[sample_nums*2][basis_funcs_num][step] = elapsed
                print(
                    f"***Non-parametric*** Samples: {sample_nums*2}, Basis Functions num: {basis_funcs_num}, Initial FD: {largest_fd:.4f}, Time: {elapsed:.3f} sec")

    with open(data_path + "nonparametric_qcqp_optimisation_times.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_diff_samples_num(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    times_list_parametric, times_list_nonparametric = [], []
    samples_nums_list = [int(x) for x in np.linspace(1000, 10000, 10)]
    basis_funcs_num_list = [int(x) for x in np.linspace(5, 15, 3)]
    times_parametric, times_nonparametric = defaultdict(dict),  defaultdict(lambda: defaultdict(dict))
    steps = 200

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            cfg.data.posterior_samples_num = sample_nums
            model = instantiate(cfg.model, data_config=cfg.data)
            start = time.perf_counter()
            fisher_estimator = PosteriorFDBase(model=model)
            optimizer = OptimizationCornerPointsMultivariateGaussian(
                fisher_estimator, cfg.ksd.optimize.prior.MultivariateGaussian,
                cfg.ksd.optimize.loss.MultivariateGaussianLogLikelihood)
            prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
            elapsed = time.perf_counter() - start
            largest_fd = prior_corners[0][2]
            times_list_parametric.append((sample_nums, elapsed))
            times_parametric[sample_nums][step] = elapsed
            print(f"***Parametric*** Samples: {sample_nums}, Initial FD: {largest_fd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/")
    os.makedirs(data_path, exist_ok=True)
    with open(data_path + "parametric_qcqp_optimisation_times.json", "w") as f:
        json.dump(times_parametric, f, indent=4)

    samples_nums_list = [int(x) for x in np.linspace(500, 10000, 10)]
    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            for basis_funcs_num in basis_funcs_num_list:
                cfg.data.posterior_samples_num = sample_nums
                cfg.data.prior_samples_num = sample_nums
                cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
                model = instantiate(cfg.model, data_config=cfg.data)
                start = time.perf_counter()
                estimator_prior = PriorFDNonParametric(model=model)
                estimator_posterior = PosteriorFDNonParametric(model=model)
                optimizer = OptimisationNonparametricBase(
                    estimator_posterior,
                    estimator_prior,
                    cfg.ksd.optimize.prior.nonparametric,
                    radius=2.0
                )
                result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
                elapsed = time.perf_counter() - start
                largest_ksd = result_sdp_dual["dual_value"]
                times_list_nonparametric.append((sample_nums*2, basis_funcs_num, elapsed))
                times_nonparametric[sample_nums*2][basis_funcs_num][step] = elapsed
                print(
                    f"***Non-parametric*** Samples: {sample_nums*2}, Basis Functions num: {basis_funcs_num}, Initial FD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    with open(data_path + "nonparametric_qcqp_optimisation_times.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_diff_basis_funcs_num(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    times_list_parametric, times_list_nonparametric = [], []
    basis_funcs_num_list = [int(x) for x in np.linspace(5, 25, 21)]
    times_nonparametric = defaultdict(dict)
    steps = 100

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for basis_funcs_num in basis_funcs_num_list:
            cfg.data.posterior_samples_num = 1000
            cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
            model = instantiate(cfg.model, data_config=cfg.data)
            estimator_prior = PriorFDNonParametric(model=model)
            estimator_posterior = PosteriorFDNonParametric(model=model)
            optimizer = OptimisationNonparametricBase(
                estimator_posterior,
                estimator_prior,
                cfg.ksd.optimize.prior.nonparametric,
                radius=2.0
            )
            start = time.perf_counter()
            result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
            elapsed = time.perf_counter() - start
            largest_ksd = result_sdp_dual["dual_value"]
            times_list_nonparametric.append((basis_funcs_num, elapsed))
            times_nonparametric[basis_funcs_num][step] = elapsed
            print(
                f"***Non-parametric*** Basis Functions num: {basis_funcs_num}, Initial FD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/")
    with open(data_path + "nonparametric_optimisation_times_diff_basis_funcs_nums.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


if __name__ == "__main__":
    # run_gaussian_priors()
    # run_multivariate_gaussian_priors()
    comparison_plot_existing_methods()
    # run_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_basis_funcs_nums()
    # run_gaussian_priors_nonparametric_diff_samples_num()
    # run_multivariate_gaussian_priors_diff_samples_num()
    # run_multivariate_gaussian_priors_diff_basis_funcs_num()
