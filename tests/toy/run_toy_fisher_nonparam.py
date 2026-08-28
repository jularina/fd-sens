from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.distributions.gaussian import Gaussian, MultivariateGaussian
import numpy as np
from src.optimization.qcqp import ParametricQCQPBase
from src.optimization.corner_points_fisher import *
from src.utils.files_operations import *
from src.plots.paper.toy_paper_fisher_funcs import *
from src.discrepancies.prior_fisher import PriorFDParametric, PriorFDNonParametric
from src.discrepancies.posterior_fisher import PosteriorFDParametric, PosteriorFDNonParametric

import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
import time

warnings.filterwarnings("ignore", category=UserWarning)


def _json_keys_to_int(obj):
    """Recursively cast dict keys serialised as strings by json back to int."""
    if isinstance(obj, dict):
        return {int(k): _json_keys_to_int(v) for k, v in obj.items()}
    return obj


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian")
def run_gaussian_priors_qcqp(cfg) -> None:
    """
    Compute Fisher divergence and optimize parametrically with QCQP.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    # Prior
    prior_fd = PriorFDParametric(model=model)
    print(f"FD from score differences: {prior_fd.estimate_fisher_prior_only():.4f}")
    A_c, b_c, c_c = prior_fd.compute_fisher_quadratic_form_prior_only()
    eta = model.prior_candidate.natural_parameters()
    print(f"FD from quadratic form: {eta @ A_c @ eta + b_c @ eta + c_c:.4f}")

    # Posterior
    posterior_fd = PosteriorFDParametric(model)
    # Prior-only
    A, b, c = posterior_fd.compute_fisher_quadratic_form_prior_only()
    eta = model.prior_candidate.natural_parameters()
    # print("Prior-only: score diff:", posterior_fd.estimate_fisher_prior_only())
    # print("Prior-only: quadratic :", eta @ A @ eta + b @ eta + c)
    # LR-only
    A_lr, b_lr, c_lr = posterior_fd.compute_fisher_quadratic_form_lr_only()
    beta = model.loss_lr
    # print("LR-only: score diff:", posterior_fd.estimate_fisher_lr_only())
    # print("LR-only: quadratic :", A_lr * beta ** 2 + b_lr * beta + c_lr)

    # Radius choice
    eta_ref = model.prior_init.natural_parameters()
    min_r = eta_ref @ A_c @ eta_ref + eta_ref @ b_c + c_c

    # QCQP Optimisation
    solver = ParametricQCQPBase(posterior_fd, prior_fd)
    solution = solver.solve_generalized_eigenvalue(r=1, check_kernel_condition=True)
    print("lambda_star:", solution.lambda_star)
    print("eta_star:", solution.eta_star)
    print("constraint x^T A_c x:", solution.achieved_constraint)
    print("objective  x^T A x  :", solution.achieved_objective)

    print("SDP lambda t dual")
    sdp_lambda_t_dual_solution = solver.solve_dual_sdp_lambda_t(radius=1)
    print("lambda_star:", sdp_lambda_t_dual_solution.lambda_star)
    print("eta_star:", sdp_lambda_t_dual_solution.eta_star)
    print("dual_value:", sdp_lambda_t_dual_solution.dual_value)
    print("objective at eta_star:", sdp_lambda_t_dual_solution.primal_value)
    print("constraint at eta_star:", sdp_lambda_t_dual_solution.constraint_value, "(should be <= r)")

    print("Lagrange dual")
    lagrange_dual_solution = solver.solve_dual_1d_lambda(radius=1.0)
    print("eta_star:", lagrange_dual_solution.eta_star)
    print("lambda_star:", lagrange_dual_solution.lambda_star)
    print("dual_value:", lagrange_dual_solution.dual_value)
    print("objective at eta_star:", lagrange_dual_solution.primal_value)
    print("constraint at eta_star:", lagrange_dual_solution.constraint_value, "(should be <= r)")

    print("SDP relaxation")
    sdp_dual_solution = solver.solve_primal_sdp_relaxation(radius=1)
    print("lambda_star:", sdp_dual_solution.lambda_star)
    print("eta_star:", sdp_dual_solution.eta_star)
    print("dual_value:", sdp_dual_solution.dual_value)
    print("objective at eta_star:", sdp_dual_solution.primal_value)
    print("constraint at eta_star:", sdp_dual_solution.constraint_value, "(should be <= r)")


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior,
        estimator_prior,
        cfg.optimize.nonparametric,
        radius=5.0
    )
    start = time.perf_counter()
    result_sdp = optimizer.optimize_through_sdp_relaxation()
    elapsed = time.perf_counter() - start
    print(f"SDP primal relaxation time: {elapsed}")
    print(f"SDP primal value:           {result_sdp['primal_value']:.4f}")
    print(f"SDP constraint value:       {result_sdp['constraint_value']:.4f}")

    start = time.perf_counter()
    result_eig = optimizer.optimize_through_generalized_eigenvalue()
    elapsed = time.perf_counter() - start
    print(f"Eigenvalue time:            {elapsed}")
    print(f"Eigenvalue omega_star:      {result_eig['omega_star']:.4f}")
    print(f"Eigenvalue primal value:    {result_eig['primal_value']:.4f}")
    print(f"Eigenvalue theoretical:     {result_eig['theoretical_value']:.4f}")
    print(f"Eigenvalue constraint:      {result_eig['constraint_value']:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_prior_neighbourhood_comparison(
        optimizer=optimizer,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-15, 15),
        resolution=500,
        epsilon=0.2,
        n_nonparam_samples=40,
        mu_range=(-2.0, 6.0),
        sigma_range=(2.0, 8.0),
        n_param_grid=12,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    sdp_lambda_list, sdp_fd_estimates_list, radius_labels = [], [], []
    sdp_lambda_list, sdp_fd_estimates_list = [], []

    for radius in [0.5, 1.0, 5.0, 10.0]:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.optimize.nonparametric,
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

        sdp_lambda_list.append(result_sdp["lambda_star"])
        sdp_fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    # plot_sdp_densities_and_logprior(
    #     basis_function=optimizer.basis_function,
    #     sdp_lambda_list=sdp_lambda_list,
    #     radius_labels=radius_labels,
    #     estimates=sdp_fd_estimates_list,
    #     prior_distribution=model.prior_init,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     domain=(-10, 12),
    #     resolution=500
    # )

    mu_n, sigma_n2 = model.compute_posterior_params()
    posterior_dist = Gaussian(mu=mu_n, sigma=np.sqrt(sigma_n2))
    plot_sdp_densities(
        basis_function=optimizer.basis_function,
        sdp_lambda_list=sdp_lambda_list,
        radius_labels=radius_labels,
        estimates=sdp_fd_estimates_list,
        prior_distribution=model.prior_init,
        posterior_distribution=posterior_dist,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-10, 12),
        resolution=500,
        show_legend=False,
    )

    # plot_sdp_posterior_comparison(
    #     basis_function=optimizer.basis_function,
    #     sdp_lambda_list=sdp_lambda_list,
    #     radius_labels=radius_labels,
    #     estimates=sdp_fd_estimates_list,
    #     prior_distribution=model.prior_init,
    #     model=model,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     domain=(1, 4),
    #     resolution=500,
    # )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="multivariate_gaussian_nonparam")
def run_multivariate_gaussian_priors_nonparametric(cfg, save_samples: bool = False) -> None:
    """
    Compute FD and run nonparametric SDP optimisation for the bivariate Gaussian model.
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    radius = cfg.optimize.nonparametric.get("radius", 5.0)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior,
        estimator_prior,
        cfg.optimize.nonparametric,
        radius=radius,
    )
    start = time.perf_counter()
    result_sdp = optimizer.optimize_through_sdp_relaxation()
    elapsed = time.perf_counter() - start
    print(f"SDP primal relaxation time: {elapsed:.3f}s")
    print(f"Nonparametric FD (primal value): {result_sdp['primal_value']:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_sdp_2d_densities(
        basis_function=optimizer.basis_function,
        psi_sdp_list=[result_sdp["lambda_star"]],
        radius_labels=[radius],
        ksd_estimates=[result_sdp["primal_value"]],
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-20, 20), (-20, 20)),
        resolution=300,
        show_legend=True,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="multivariate_gaussian_nonparam")
def run_multivariate_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = False) -> None:
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
    sdp_lambda_list, fd_estimates_list, radius_labels = [], [], []

    for radius in [0.5, 1.0, 5.0, 10.0]:  # [0.5, 1.0, 5.0, 15.0]
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.optimize.nonparametric,
            radius=radius
        )
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        elapsed = time.perf_counter() - start
        print(f"SDP relaxation time: {elapsed}")

        sdp_lambda_list.append(result_sdp["lambda_star"])
        fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    posterior_dist = MultivariateGaussian(mu=model.mu_n, cov=model.Sigma_n)
    plot_sdp_2d_densities(
        basis_function=optimizer.basis_function,
        psi_sdp_list=sdp_lambda_list,
        radius_labels=radius_labels,
        ksd_estimates=fd_estimates_list,
        prior_distribution=model.prior_init,
        posterior_distribution=posterior_dist,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-6, 10), (-5, 10)),
        resolution=500,
        contour_levels=5,
        show_legend=False,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian_param_nonparam")
def run_param_nonparam_comparison_skewness(cfg) -> None:
    """
    Run parametric and nonparametric optimisation for the univariate Gaussian model
    and produce a comparison plot of the worst-case candidate priors and posteriors.
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    # Parametric optimisation
    posterior_fd = PosteriorFDParametric(model=model)
    param_optimizer = OptimizationCornerPointsUnivariateGaussian(
        posterior_fd,
        cfg.fd.optimize.prior.Gaussian,
        cfg.fd.optimize.loss.GaussianLogLikelihood,
    )
    prior_corners, worst_corner = param_optimizer.evaluate_all_prior_corners()
    print(f"Worst parametric prior: mu={worst_corner['mu']:.4f}, sigma={worst_corner['sigma']:.4f}")
    print(f"Worst parametric FD: {prior_corners[0][2]:.4f}")

    # Nonparametric optimisation
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    radius = cfg.optimize.nonparametric.get("radius", 5.0)
    nonparam_optimizer = OptimisationNonparametricBase(
        estimator_posterior,
        estimator_prior,
        cfg.optimize.nonparametric,
        radius=radius,
    )
    result_sdp = nonparam_optimizer.optimize_through_sdp_relaxation()
    print(f"Nonparametric FD (primal value): {result_sdp['primal_value']:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_param_nonparam_skewness_comparison(
        worst_corner=worst_corner,
        lambda_star=result_sdp["lambda_star"],
        basis_function=nonparam_optimizer.basis_function,
        model=model,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian_param_nonparam_kurtosis")
def run_param_nonparam_comparison_kurtosis(cfg) -> None:
    """
    Weak-data univariate Gaussian location experiment illustrating sensitivity
    to tail behaviour. The Gaussian parametric neighbourhood can only shift or
    rescale the prior and so always has zero excess kurtosis; the nonparametric
    KEF candidate, with basis centres spread over a wide interval covering both
    the posterior region and the tails, can develop heavier tails and positive
    excess kurtosis.
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    # Parametric optimisation
    posterior_fd = PosteriorFDParametric(model=model)
    param_optimizer = OptimizationCornerPointsUnivariateGaussian(
        posterior_fd,
        cfg.fd.optimize.prior.Gaussian,
        cfg.fd.optimize.loss.GaussianLogLikelihood,
    )
    prior_corners, worst_corner = param_optimizer.evaluate_all_prior_corners()
    print(f"Worst parametric prior: mu={worst_corner['mu']:.4f}, sigma={worst_corner['sigma']:.4f}")
    print(f"Worst parametric FD: {prior_corners[0][2]:.4f}")

    # Nonparametric optimisation
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    radius = cfg.optimize.nonparametric.get("radius", 5.0)
    nonparam_optimizer = OptimisationNonparametricBase(
        estimator_posterior,
        estimator_prior,
        cfg.optimize.nonparametric,
        radius=radius,
    )
    result_sdp = nonparam_optimizer.optimize_through_sdp_relaxation()
    print(f"Nonparametric FD (primal value): {result_sdp['primal_value']:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_param_nonparam_kurtosis_comparison(
        worst_corner=worst_corner,
        lambda_star=result_sdp["lambda_star"],
        basis_function=nonparam_optimizer.basis_function,
        model=model,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-20, 20),
        resolution=500,
        y_log=True,
        legend=False,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian_param_nonparam_multimodal")
def run_param_nonparam_multimodality_comparison(cfg) -> None:
    """
    Same as run_param_nonparam_comparison, but using a config with a larger
    nonparametric radius and more/smaller-lengthscale basis functions so the
    worst-case nonparametric candidate's multimodality is actually visible
    (with the default config it optimises to a unimodal solution).
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    # Parametric optimisation
    posterior_fd = PosteriorFDParametric(model=model)
    param_optimizer = OptimizationCornerPointsUnivariateGaussian(
        posterior_fd,
        cfg.fd.optimize.prior.Gaussian,
        cfg.fd.optimize.loss.GaussianLogLikelihood,
    )
    prior_corners, worst_corner = param_optimizer.evaluate_all_prior_corners()
    print(f"Worst parametric prior: mu={worst_corner['mu']:.4f}, sigma={worst_corner['sigma']:.4f}")
    print(f"Worst parametric FD: {prior_corners[0][2]:.4f}")

    # Nonparametric optimisation
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    radius = cfg.optimize.nonparametric.get("radius")
    nonparam_optimizer = OptimisationNonparametricBase(
        estimator_posterior,
        estimator_prior,
        cfg.optimize.nonparametric,
        radius=radius,
    )
    result_sdp = nonparam_optimizer.optimize_through_sdp_relaxation()
    print(f"Nonparametric FD (primal value): {result_sdp['primal_value']:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_param_nonparam_multimodality_comparison(
        worst_corner=worst_corner,
        lambda_star=result_sdp["lambda_star"],
        basis_function=nonparam_optimizer.basis_function,
        model=model,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian_nonparam")
def run_multivariate_gaussian_diff_basis_funcs_num_runtimes(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.
    """
    runtimes_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/nonparam")
    runtimes_path = os.path.join(runtimes_dir, "nonparametric_optimisation_times_diff_basis_funcs_nums.json")

    if os.path.exists(runtimes_path):
        print(f"Found existing runtimes at {runtimes_path}, skipping computation.")
        times_list = _json_keys_to_int(load_results_json(runtimes_path))
    else:
        model = instantiate(cfg.model, data_config=cfg.data)
        output_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian")

        if save_samples:
            os.makedirs(output_dir, exist_ok=True)
            np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
            np.save(output_dir + "/observations.npy", model.observations)
            np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

        estimator_prior = PriorFDNonParametric(model=model)
        estimator_posterior = PosteriorFDNonParametric(model=model)
        times_list = defaultdict(dict)
        basis_funcs_num = [int(x) for x in np.linspace(5, 31, 14)]
        iters = 500

        for k in basis_funcs_num:
            for step in range(iters):
                print(f"Basis funcs = {k}, step={step}.")
                cfg.optimize.nonparametric.basis_funcs_kwargs["num_basis_functions"] = k
                optimizer = OptimisationNonparametricBase(
                    estimator_posterior,
                    estimator_prior,
                    cfg.optimize.nonparametric,
                    radius=5.0
                )
                start = time.perf_counter()
                _ = optimizer.optimize_through_generalized_eigenvalue()
                elapsed = time.perf_counter() - start
                times_list[k][step] = elapsed

        save_to_serializable_json(times_list, runtimes_path)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_runtime_nonparametric_with_ci(
        times_list,
        plot_cfg,
        output_dir,
        filename="gaussian_2d_location_model_runtime_diff_basis_funcs_nums.pdf",
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian_nonparam")
def run_multivariate_gaussian_diff_samples_num_runtimes(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and time nonparametric optimisation for a grid of
    posterior/prior sample sizes m, one line per number of basis functions k.
    """
    runtimes_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/nonparam")
    runtimes_path = os.path.join(runtimes_dir, "nonparametric_optimisation_times_diff_samples_nums.json")

    if os.path.exists(runtimes_path):
        print(f"Found existing runtimes at {runtimes_path}, skipping computation.")
        times_list = _json_keys_to_int(load_results_json(runtimes_path))
    else:
        samples_num_list = [int(x) for x in np.linspace(1000, 5000, 5)]
        basis_funcs_num = [10, 15, 25, 30]
        iters = 500
        times_list = defaultdict(lambda: defaultdict(dict))

        cfg.data.posterior_samples_path = None
        cfg.data.prior_samples_path = None
        prior_samples_num = cfg.data.prior_samples_num

        for m in samples_num_list:
            cfg.data.posterior_samples_num = m
            for k in basis_funcs_num:
                cfg.optimize.nonparametric.basis_funcs_kwargs["num_basis_functions"] = k
                for step in range(iters):
                    print(f"Samples num = {m}, basis funcs = {k}, step={step}.")
                    model = instantiate(cfg.model, data_config=cfg.data)
                    model.posterior_samples_init = model.sample_posterior(n_samples=m)
                    model.prior_samples_init = model.sample_from_base_prior(n_samples=prior_samples_num)
                    model.m = m
                    model.m_prior = prior_samples_num
                    estimator_prior = PriorFDNonParametric(model=model)
                    estimator_posterior = PosteriorFDNonParametric(model=model)
                    optimizer = OptimisationNonparametricBase(
                        estimator_posterior,
                        estimator_prior,
                        cfg.optimize.nonparametric,
                        radius=5.0
                    )
                    start = time.perf_counter()
                    _ = optimizer.optimize_through_generalized_eigenvalue()
                    elapsed = time.perf_counter() - start
                    times_list[m][k][step] = elapsed

        save_to_serializable_json(times_list, runtimes_path)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_runtime_nonparametric_diff_samples_num_with_ci(
        times_list,
        plot_cfg,
        output_dir,
        filename="gaussian_2d_location_model_runtime_diff_samples_nums.pdf",
    )


if __name__ == "__main__":
    run_param_nonparam_comparison_skewness()
    # run_param_nonparam_comparison_kurtosis()
    # run_param_nonparam_multimodality_comparison()
    # run_gaussian_priors_nonparametric()
    # run_multivariate_gaussian_priors_nonparametric()
    # run_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_diff_radii()

    # run_multivariate_gaussian_diff_basis_funcs_num_runtimes()
    # run_multivariate_gaussian_diff_samples_num_runtimes()
