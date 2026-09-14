from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.distributions.gaussian import Gaussian, MultivariateGaussian
import numpy as np
from src.optimization.qcqp import ParametricQCQPBase
from src.optimization.corner_points_fisher import *
from src.utils.files_operations import *
from src.plots.paper.toy_paper_fisher_funcs import *
from src.discrepancies.prior_fisher import PriorFDParametric, PriorFDNonParametric
from src.discrepancies.posterior_fisher import PosteriorFDParametric, PosteriorFDNonParametric
from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.basis_functions.basis_functions import rbf_gaussian_gram_closed_form
from scipy.linalg import eigh as scipy_eigh

import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import OmegaConf
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

    Basis centres are selected via the "random" method (i.i.d. subsample) from
    a fresh, independent draw of the reference prior -- never from the
    prior/posterior samples used to estimate the Fisher divergence -- using
    the same fixed seed (27) as run_gaussian_priors_nonparametric_diff_center_methods,
    so the r=10.0 result here matches that function's "random (fresh prior draw)" case.
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

    # Fresh, independent draw from the reference prior, used only as the pool
    # to select centres from -- never the samples that feed the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(model.prior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["method"] = "random"
    basis_function = basis_cls(**basis_kwargs)

    for radius in [0.5, 1.0, 5.0, 10.0]:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.optimize.nonparametric,
            radius=radius,
            basis_function=basis_function,
        )
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_generalized_eigenvalue()
        elapsed = time.perf_counter() - start
        print(f"Generalised eigenvalue time: {elapsed}")

        sdp_lambda_list.append(result_sdp["lambda_star"])
        sdp_fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)
        print(f"Radius: {radius}, sensitivity: {result_sdp['primal_value']}")

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
        domain=(-6, 12),
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_center_methods(cfg, save_samples: bool = False) -> None:
    """
    Compare basis-function centre-selection strategies at a fixed radius, each
    as its own plot with the SDP worst-case candidate density, the true prior,
    and the underlying sample points (rug of dots) and resulting basis centres
    (rug of x's) along the x-axis.

    The Fisher-divergence estimate always uses the saved/loaded prior and
    posterior samples. Centres are never chosen from those same samples;
    prior-based centres are instead drawn from a fresh, independent i.i.d.
    draw of the reference prior (same size as the saved prior samples), from
    which a subset is then selected via the chosen method:
      (a) Halton quantile-mapped centres from the fresh reference-prior draw.
      (b) K-means centres from the same fresh reference-prior draw.
      (c) Halton quantile-mapped centres from the posterior samples.
      (d) Random (i.i.d. subsample) centres from the fresh reference-prior draw.
      (e) Random (i.i.d. subsample) centres from the posterior samples.
    """
    radius = 10.0
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    plots_output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    original_prior_samples = model.prior_samples_init

    # FD estimation always uses the saved/loaded prior and posterior samples.
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)

    # Fresh, independent draw from the reference prior, used only as the pool
    # to select centres from -- never the samples that feed the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(original_prior_samples))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    base_basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    base_basis_kwargs["num_basis_functions"] = 30
    base_basis_kwargs["nu"] = 5.0

    def _run_and_plot(method_label, filename, optimizer):
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        print(f"[{method_label}] Nonparametric FD (primal value): {result_sdp['primal_value']:.4f}")
        plot_sdp_density_with_centers(
            basis_function=optimizer.basis_function,
            lambda_star=result_sdp["lambda_star"],
            estimate=result_sdp["primal_value"],
            prior_distribution=model.prior_init,
            plot_cfg=plot_cfg,
            output_dir=plots_output_dir,
            filename=filename,
            domain=(-10, 12),
            resolution=500,
        )
        return result_sdp

    # (a) Halton, centres from a fresh reference-prior draw
    basis_kwargs = dict(base_basis_kwargs)
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["method"] = "halton"
    basis_function_halton = basis_cls(**basis_kwargs)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
        basis_function=basis_function_halton,
    )
    _run_and_plot(
        method_label="Halton (fresh prior draw)",
        filename="gaussian_1d_location_model_centers_halton_prior_fresh.pdf",
        optimizer=optimizer,
    )

    # (b) K-means, centres from the same fresh reference-prior draw
    basis_kwargs = dict(base_basis_kwargs)
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["method"] = "kmeans"
    basis_function_kmeans = basis_cls(**basis_kwargs)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
        basis_function=basis_function_kmeans,
    )
    _run_and_plot(
        method_label="K-means (fresh prior draw)",
        filename="gaussian_1d_location_model_centers_kmeans_prior_fresh.pdf",
        optimizer=optimizer,
    )

    # (c) Halton, centres selected from the posterior samples
    basis_kwargs = dict(base_basis_kwargs)
    basis_kwargs["prior_samples"] = None
    basis_kwargs["posterior_samples"] = estimator_posterior.samples
    basis_kwargs["estimation_samples_source"] = "posterior"
    basis_kwargs["method"] = "halton"
    basis_function_posterior = basis_cls(**basis_kwargs)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
        basis_function=basis_function_posterior,
    )
    _run_and_plot(
        method_label="Halton (posterior)",
        filename="gaussian_1d_location_model_centers_halton_posterior.pdf",
        optimizer=optimizer,
    )

    # (d) Random, centres from the same fresh reference-prior draw
    basis_kwargs = dict(base_basis_kwargs)
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["method"] = "random"
    basis_function_random_prior = basis_cls(**basis_kwargs)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
        basis_function=basis_function_random_prior,
    )
    _run_and_plot(
        method_label="Random (fresh prior draw)",
        filename="gaussian_1d_location_model_centers_random_prior_fresh.pdf",
        optimizer=optimizer,
    )

    # (e) Random, centres selected from the posterior samples
    basis_kwargs = dict(base_basis_kwargs)
    basis_kwargs["prior_samples"] = None
    basis_kwargs["posterior_samples"] = estimator_posterior.samples
    basis_kwargs["estimation_samples_source"] = "posterior"
    basis_kwargs["method"] = "random"
    basis_function_random_posterior = basis_cls(**basis_kwargs)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
        basis_function=basis_function_random_posterior,
    )
    _run_and_plot(
        method_label="Random (posterior)",
        filename="gaussian_1d_location_model_centers_random_posterior.pdf",
        optimizer=optimizer,
    )

    # Combined figure: K-means (fresh prior), Random (fresh prior), and
    # Random (posterior) densities overlaid in one panel, with each method's
    # basis centres shown in its own colour-matched rug strip underneath --
    # one such combined figure per number of basis functions K, so the effect
    # of K on the centre-selection comparison is also visible.
    def _compute(basis_function):
        optimizer = OptimisationNonparametricBase(
            estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
            basis_function=basis_function,
        )
        return optimizer.optimize_through_sdp_relaxation()

    for K in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        k_basis_kwargs = dict(base_basis_kwargs)
        k_basis_kwargs["num_basis_functions"] = K

        kmeans_kwargs = dict(k_basis_kwargs, prior_samples=centers_pool_samples, posterior_samples=None,
                              estimation_samples_source="prior", method="kmeans")
        basis_function_kmeans_k = basis_cls(**kmeans_kwargs)
        result_kmeans_k = _compute(basis_function_kmeans_k)

        random_prior_kwargs = dict(k_basis_kwargs, prior_samples=centers_pool_samples, posterior_samples=None,
                                    estimation_samples_source="prior", method="random")
        basis_function_random_prior_k = basis_cls(**random_prior_kwargs)
        result_random_prior_k = _compute(basis_function_random_prior_k)

        random_posterior_kwargs = dict(k_basis_kwargs, prior_samples=None, posterior_samples=estimator_posterior.samples,
                                        estimation_samples_source="posterior", method="random")
        basis_function_random_posterior_k = basis_cls(**random_posterior_kwargs)
        result_random_posterior_k = _compute(basis_function_random_posterior_k)

        print(
            f"[K={K}] K-means (fresh prior): {result_kmeans_k['primal_value']:.4f}, "
            f"Random (fresh prior): {result_random_prior_k['primal_value']:.4f}, "
            f"Random (posterior): {result_random_posterior_k['primal_value']:.4f}"
        )

        plot_sdp_density_with_centers_combined(
            basis_functions=[basis_function_kmeans_k, basis_function_random_prior_k, basis_function_random_posterior_k],
            lambda_star_list=[
                result_kmeans_k["lambda_star"],
                result_random_prior_k["lambda_star"],
                result_random_posterior_k["lambda_star"],
            ],
            estimates=[
                result_kmeans_k["primal_value"],
                result_random_prior_k["primal_value"],
                result_random_posterior_k["primal_value"],
            ],
            labels=["K-means (fresh prior)", "Random (fresh prior)", "Random (posterior)"],
            prior_distribution=model.prior_init,
            plot_cfg=plot_cfg,
            output_dir=plots_output_dir,
            filename=f"gaussian_1d_location_model_centers_combined_K{K}.pdf",
            domain=(-10, 12),
            resolution=500,
        )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_kernels(cfg, save_samples: bool = False) -> None:
    """
    Compare basis-function kernel choices at a fixed radius, each as its own
    plot with the SDP worst-case candidate density, the true prior, and the
    resulting basis centres (rug of dots) along the x-axis:
      (a) Matern kernel, nu=1.5
      (b) Matern kernel, nu=3.0
      (c) Matern kernel, nu=7.0
      (d) RBF kernel

    Plus one combined plot overlaying all three Matern nu's worst-case
    candidate densities in a single panel, with one legend entry per nu.

    All kernels select their centres via k-means (chosen for its guaranteed
    minimum inter-centre separation, which keeps the constraint Gram matrix
    A_c well-conditioned -- plain i.i.d. random selection can draw
    near-duplicate centres and make A_c numerically singular) from a single
    fresh, independent i.i.d. draw of the reference prior -- never the
    samples used to estimate the Fisher divergence.
    """
    radius = 10.0
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    plots_output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    original_prior_samples = model.prior_samples_init

    # FD estimation always uses the saved/loaded prior and posterior samples.
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)

    # Fresh, independent draw from the reference prior, used only as the pool
    # to select centres from -- never the samples that feed the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(original_prior_samples))

    num_basis_functions = 30

    def _run_and_plot(method_label, filename, basis_function, show_centers=True, show_yaxis=True):
        optimizer = OptimisationNonparametricBase(
            estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
            basis_function=basis_function,
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        print(f"[{method_label}] Nonparametric FD (primal value): {result_sdp['primal_value']:.4f}")
        plot_sdp_density_with_centers(
            basis_function=optimizer.basis_function,
            lambda_star=result_sdp["lambda_star"],
            estimate=result_sdp["primal_value"],
            prior_distribution=model.prior_init,
            plot_cfg=plot_cfg,
            output_dir=plots_output_dir,
            filename=filename,
            domain=(-10, 12),
            resolution=500,
            show_centers=show_centers,
            show_yaxis=show_yaxis,
        )
        return result_sdp

    # (a)-(c) Matern, varying smoothness nu (nu must be > 1 for C^1 basis functions)
    matern_cls = BASIS_FUNCTIONS_REGISTRY["MaternBasisFunction"]
    matern_basis_functions, matern_lambda_list, matern_nu_labels, matern_estimates = [], [], [], []
    for nu in [1.5, 3.0, 7.0]:
        basis_function = matern_cls(
            posterior_samples=None,
            prior_samples=centers_pool_samples,
            num_basis_functions=num_basis_functions,
            method="kmeans",
            nu=nu,
            estimation_samples_source="prior",
        )
        nu_tag = str(nu).replace(".", "p")
        result_sdp = _run_and_plot(
            method_label=f"Matern (nu={nu})",
            filename=f"gaussian_1d_location_model_kernel_matern_nu_{nu_tag}.pdf",
            basis_function=basis_function,
        )
        matern_basis_functions.append(basis_function)
        matern_lambda_list.append(result_sdp["lambda_star"])
        matern_nu_labels.append(nu)
        matern_estimates.append(result_sdp["primal_value"])

    # Combined plot: all Matern nu's overlaid in one panel, one legend entry per nu.
    plot_sdp_matern_nu_comparison(
        basis_functions=matern_basis_functions,
        sdp_lambda_list=matern_lambda_list,
        nu_labels=matern_nu_labels,
        estimates=matern_estimates,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=plots_output_dir,
        filename="gaussian_1d_location_model_kernel_matern_diff_nu.pdf",
        domain=(-10, 12),
        resolution=500,
    )

    # (d) RBF
    rbf_cls = BASIS_FUNCTIONS_REGISTRY["RBFBasisFunction"]
    basis_function = rbf_cls(
        posterior_samples=None,
        prior_samples=centers_pool_samples,
        num_basis_functions=num_basis_functions,
        method="kmeans",
        estimation_samples_source="prior",
    )
    _run_and_plot(
        method_label="RBF",
        filename="gaussian_1d_location_model_kernel_rbf.pdf",
        basis_function=basis_function,
        show_centers=False,
        show_yaxis=False,
    )


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

    Basis centres are selected via the "random" method (i.i.d. subsample), K=30,
    nu=5, from a fresh, independent draw of the reference prior -- never from the
    prior/posterior samples used to estimate the Fisher divergence -- using the
    same fixed seed (27) as the univariate counterpart.

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

    # Fresh, independent draw from the reference prior, used only as the pool
    # to select centres from -- never the samples that feed the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(model.prior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["num_basis_functions"] = 30
    basis_kwargs["nu"] = 5
    basis_kwargs["method"] = "random"
    basis_function = basis_cls(**basis_kwargs)

    for radius in [0.5, 1.0, 5.0, 10.0]:  # [0.5, 1.0, 5.0, 15.0]
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.optimize.nonparametric,
            radius=radius,
            basis_function=basis_function,
        )
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_generalized_eigenvalue()
        elapsed = time.perf_counter() - start
        print(f"Generalised eigenvalue time: {elapsed}")

        sdp_lambda_list.append(result_sdp["lambda_star"])
        fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)
        print(f"Radius: {radius}, sensitivity: {result_sdp['primal_value']}")

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
        domain=((-5, 9), (-4, 9)),
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

    The nonparametric setup mirrors run_gaussian_priors_nonparametric_diff_radii at
    r=10.0: basis centres are selected via the "random" method (i.i.d. subsample)
    from a fresh, independent draw of the reference prior -- never from the
    prior/posterior samples used to estimate the Fisher divergence -- using the
    same fixed seed (27), and optimisation is done via the generalised eigenvalue
    solver.
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
    radius = 10.0

    # Fresh, independent draw from the reference prior, used only as the pool
    # to select centres from -- never the samples that feed the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(model.prior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["method"] = "random"
    basis_function = basis_cls(**basis_kwargs)

    nonparam_optimizer = OptimisationNonparametricBase(
        estimator_posterior,
        estimator_prior,
        cfg.optimize.nonparametric,
        radius=radius,
        basis_function=basis_function,
    )
    result_sdp = nonparam_optimizer.optimize_through_generalized_eigenvalue()
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
            config_name="univariate_gaussian_param_nonparam")
def run_param_nonparam_comparison_skewness_matched_radius(cfg) -> None:
    """
    Same skewness comparison as run_param_nonparam_comparison_skewness, but the
    nonparametric worst-case prior is computed with the exact same setup as
    run_gaussian_priors_nonparametric_diff_radii's r=10.0 case: basis centres
    selected via the "random" method (i.i.d. subsample) from a fresh,
    independent draw of the reference prior (same fixed seed 27), never from
    the prior/posterior samples used to estimate the Fisher divergence, at a
    fixed radius r=10.0, solved via the generalised eigenvalue method.

    The parametric side is unrelated: its worst case is just whatever (mu,
    sigma) in the box maximises posterior sensitivity (same as
    run_param_nonparam_comparison_skewness), shown alongside the nonparametric
    r=10.0 worst case for comparison -- the two are not matched to a common FD
    budget here.

    For the nonparametric worst-case prior to actually come out identical to
    run_gaussian_priors_nonparametric_diff_radii's r=10.0 case, the model must
    also be instantiated with the exact same data as that function's config
    (univariate_gaussian_nonparam) -- not this function's own config
    (univariate_gaussian_param_nonparam), which uses a different, weaker
    observations_num (5 vs 100), unseeded freshly-drawn prior/posterior
    samples (1000, vs 5000 loaded from the same saved files each run), and
    different basis hyperparameters (num_basis_functions=10, nu=2.0 vs 30,
    5.0). Those data/basis settings are overridden below to match.
    """
    # Match run_gaussian_priors_nonparametric_diff_radii's data exactly (same
    # saved observations/prior/posterior samples, same counts), so both
    # functions build the same model and the nonparametric worst case is
    # reproducible across them.
    cfg.data.observations_path = "data/univariate_gaussian/observations.npy"
    cfg.data.posterior_samples_path = "data/univariate_gaussian/posterior_samples.npy"
    cfg.data.prior_samples_path = "data/univariate_gaussian/prior_samples.npy"
    cfg.data.posterior_samples_num = 5000
    cfg.data.prior_samples_num = 5000
    model = instantiate(cfg.model, data_config=cfg.data)

    # Parametric optimisation: worst-case Gaussian prior within the (mu, sigma) box
    # (worst case = maximises posterior sensitivity, the box's natural objective).
    posterior_fd = PosteriorFDParametric(model=model)
    param_optimizer = OptimizationCornerPointsUnivariateGaussian(
        posterior_fd,
        cfg.fd.optimize.prior.Gaussian,
        cfg.fd.optimize.loss.GaussianLogLikelihood,
    )
    _, worst_corner = param_optimizer.evaluate_all_prior_corners()
    print(f"Worst parametric prior: mu={worst_corner['mu']:.4f}, sigma={worst_corner['sigma']:.4f}")

    radius = 10.0

    # Nonparametric optimisation -- same setup as run_gaussian_priors_nonparametric_diff_radii
    # at r=10.0.
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)

    # Fresh, independent draw from the reference prior, used only as the pool
    # to select centres from -- never the samples that feed the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(model.prior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    # Match run_gaussian_priors_nonparametric_diff_radii's basis hyperparameters
    # (this config's own defaults are num_basis_functions=10, nu=2.0).
    basis_kwargs["num_basis_functions"] = 30
    basis_kwargs["nu"] = 5.0
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["method"] = "random"
    basis_function = basis_cls(**basis_kwargs)

    nonparam_optimizer = OptimisationNonparametricBase(
        estimator_posterior,
        estimator_prior,
        cfg.optimize.nonparametric,
        radius=radius,
        basis_function=basis_function,
    )
    result_sdp = nonparam_optimizer.optimize_through_generalized_eigenvalue()
    print(f"Nonparametric FD (primal value): {result_sdp['primal_value']:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_param_nonparam_prior_and_stats(
        worst_corner=worst_corner,
        lambda_star=result_sdp["lambda_star"],
        basis_function=nonparam_optimizer.basis_function,
        model=model,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="gaussian_1d_location_model_param_nonparam_prior_stats_matched_radius.pdf",
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
    np.random.seed(27)
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
        resolution=2000,
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
    Main function to compute FD and time nonparametric optimisation for a grid of
    basis-function counts K=[50, 100, 200], one line per total number of
    prior+posterior samples m+l (500, 5000, 10000; m=l=(m+l)/2 each).
    """
    runtimes_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/nonparam")
    runtimes_path = os.path.join(
        runtimes_dir, "nonparametric_optimisation_times_diff_basis_funcs_K50_100_200_400_mplusl_500_1000_5000_10000.json"
    )

    if os.path.exists(runtimes_path):
        print(f"Found existing runtimes at {runtimes_path}, skipping computation.")
        times_list = _json_keys_to_int(load_results_json(runtimes_path))
    else:
        output_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian")

        if save_samples:
            model = instantiate(cfg.model, data_config=cfg.data)
            os.makedirs(output_dir, exist_ok=True)
            np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
            np.save(output_dir + "/observations.npy", model.observations)
            np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

        total_samples_list = [500, 1000, 5000, 10000]
        basis_funcs_num = [50, 100, 200, 400]
        iters = 500
        times_list = defaultdict(lambda: defaultdict(dict))

        cfg.data.posterior_samples_path = None
        cfg.data.prior_samples_path = None

        for total_m in total_samples_list:
            m = total_m // 2
            cfg.data.posterior_samples_num = m
            cfg.data.prior_samples_num = m
            for k in basis_funcs_num:
                cfg.optimize.nonparametric.basis_funcs_kwargs["num_basis_functions"] = k
                for step in range(iters):
                    print(f"Total samples (m+l) = {total_m}, basis funcs = {k}, step={step}.")
                    model = instantiate(cfg.model, data_config=cfg.data)
                    model.posterior_samples_init = model.sample_posterior(n_samples=m)
                    model.prior_samples_init = model.sample_from_base_prior(n_samples=m)
                    model.m = m
                    model.m_prior = m
                    estimator_prior = PriorFDNonParametric(model=model)
                    estimator_posterior = PosteriorFDNonParametric(model=model)
                    start = time.perf_counter()
                    optimizer = OptimisationNonparametricBase(
                        estimator_posterior,
                        estimator_prior,
                        cfg.optimize.nonparametric,
                        radius=5.0
                    )
                    _ = optimizer.optimize_through_generalized_eigenvalue()
                    elapsed = time.perf_counter() - start
                    times_list[total_m][k][step] = elapsed

        save_to_serializable_json(times_list, runtimes_path)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_runtime_nonparametric_diff_basis_funcs_num_diff_samples_with_ci(
        times_list,
        plot_cfg,
        output_dir,
        filename="gaussian_2d_location_model_runtime_diff_basis_funcs_nums_diff_samples.pdf",
    )


def _gamma_max_generalized_eig(A: np.ndarray, A_c: np.ndarray, nugget: float = 1e-10) -> float:
    """Largest generalised eigenvalue of A v = gamma A_c v, with a PD-safety nugget on A_c."""
    A = 0.5 * (A + A.T)
    A_c = 0.5 * (A_c + A_c.T)
    min_eig = float(np.linalg.eigvalsh(A_c).min())
    if min_eig < nugget:
        A_c = A_c + (nugget - min_eig) * np.eye(A_c.shape[0])
    return float(scipy_eigh(A, A_c, eigvals_only=True)[-1])


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_closed_form_convergence(
    cfg, radius: float = 5.0, n_repeats: int = 1000,
) -> None:
    """
    Validates the closed-form FDsens+ sensitivity for the RBF kernel with a
    Gaussian reference prior/posterior against its Monte-Carlo (plug-in)
    estimate, and plots the absolute error between the two as a function of
    the total number of prior+posterior samples m+l (each point draws
    m=l=l samples, so m+l=2l), averaged (with +/- 1 sem band) over
    independent resamples at each sample size. The error is expected to
    shrink as m+l grows.

    Since prior and posterior are both exactly Gaussian for this conjugate
    toy model, S^FD(Q_r^K) = r * gamma_max can be computed exactly from
    A, A_c via rbf_gaussian_gram_closed_form, with A_c built from
    (mu_ref, Sigma_ref) and A from (mu_post, Sigma_post). The RBF centres
    (a fixed grid spanning +/- 3 prior std) and lengthscale (the inter-centre
    spacing) are held fixed across all repeats/sample sizes, so the only
    source of discrepancy from the closed-form value is Monte-Carlo sample
    noise in the plug-in A, A_c.

    The sensitivity error is also decomposed into the part coming from
    estimating the objective A alone (posterior samples; A_c held at its
    closed-form value) and the part coming from estimating the constraint
    A_c alone (prior samples; A held at its closed-form value); these are
    computed and cached alongside the full plug-in error (both A and A_c
    estimated), though only the full plug-in error is plotted in the
    combined figure.

    Two further plots show the matrix-level relative Frobenius-norm
    estimation error directly, independent of the downstream generalised
    eigenproblem: ||A - Ahat||_F / ||A||_F (vs m) and
    ||A_c - Achat||_F / ||A_c||_F (vs l).

    Per-repeat errors are cached to a JSON file at
    data/univariate_gaussian/closed_form_convergence/closed_form_convergence_errors.json
    and reused on subsequent runs if present -- delete the file to force a
    recompute (e.g. after changing sample_sizes, n_repeats, radius or basis).
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    mu_ref = float(model.prior_init.mu)
    sigma_ref = float(model.prior_init.sigma)
    mu_post, sigma_n2_post = model.compute_posterior_params()
    mu_post = float(np.asarray(mu_post).item())
    sigma_n2_post = float(np.asarray(sigma_n2_post).item())

    num_basis_functions = 30
    span = 3.0
    centers = np.linspace(
        mu_ref - span * sigma_ref, mu_ref + span * sigma_ref, num_basis_functions
    ).reshape(-1, 1)
    lengthscale = float(centers[1, 0] - centers[0, 0])

    fixed_centers_cls = BASIS_FUNCTIONS_REGISTRY["FixedCentersRBFBasisFunction"]
    basis_function = fixed_centers_cls(
        loc=0.0,
        scale=1.0,
        center_multiples=centers.flatten(),
        lengthscale=lengthscale,
    )

    # Closed-form A_c (theta ~ prior) and A (theta ~ posterior).
    A_c_closed = rbf_gaussian_gram_closed_form(
        centers, lengthscale, mu=np.array([mu_ref]), Sigma=np.array([[sigma_ref ** 2]]),
    )
    A_closed = rbf_gaussian_gram_closed_form(
        centers, lengthscale, mu=np.array([mu_post]), Sigma=np.array([[sigma_n2_post]]),
    )
    gamma_max_closed = _gamma_max_generalized_eig(A_closed, A_c_closed)
    S_closed = radius * gamma_max_closed
    A_closed_norm = float(np.linalg.norm(A_closed, ord="fro"))
    A_c_closed_norm = float(np.linalg.norm(A_c_closed, ord="fro"))
    print(f"Closed-form gamma_max: {gamma_max_closed:.6f}, S_closed: {S_closed:.6f}")

    sample_size_start, sample_size_stop, sample_size_step = 1000, 15000, 1000
    sample_sizes = list(np.linspace(
        sample_size_start, sample_size_stop,
        num=int((sample_size_stop - sample_size_start) / sample_size_step) + 1,
        dtype=int,
    ))

    errors_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian/closed_form_convergence")
    errors_path = os.path.join(errors_dir, "closed_form_convergence_errors.json")

    if os.path.exists(errors_path):
        print(f"Found existing errors at {errors_path}, skipping computation.")
        errors_by_l = load_results_json(errors_path)
    else:
        errors_by_l = {}
        for l in sample_sizes:
            errors_full, errors_obj, errors_constr = [], [], []
            errors_A, errors_Ac = [], []
            for _ in range(n_repeats):
                model.prior_samples_init = model.sample_from_base_prior(n_samples=l)
                model.posterior_samples_init = model.sample_posterior(n_samples=l)
                model.m = l
                model.m_prior = l

                estimator_prior = PriorFDNonParametric(model=model)
                estimator_posterior = PosteriorFDNonParametric(model=model)

                A_c_hat, _, _ = estimator_prior.compute_non_parametric_fisher_quadratic_form_prior_only(basis_function)
                A_hat, _, _ = estimator_posterior.compute_non_parametric_fisher_quadratic_form_prior_only(basis_function)

                # Full plug-in: both A and A_c estimated from samples.
                S_hat_full = radius * _gamma_max_generalized_eig(A_hat, A_c_hat)
                # Objective-only: A estimated (posterior samples), A_c at its closed-form value.
                S_hat_obj = radius * _gamma_max_generalized_eig(A_hat, A_c_closed)
                # Constraint-only: A_c estimated (prior samples), A at its closed-form value.
                S_hat_constr = radius * _gamma_max_generalized_eig(A_closed, A_c_hat)

                errors_full.append(abs(S_hat_full - S_closed))
                errors_obj.append(abs(S_hat_obj - S_closed))
                errors_constr.append(abs(S_hat_constr - S_closed))
                # Matrix-level relative Frobenius-norm error of each plug-in estimator
                # against its closed-form target, independent of the downstream
                # generalised eigenproblem: ||A - Ahat||_F / ||A||_F.
                errors_A.append(float(np.linalg.norm(A_hat - A_closed, ord="fro")) / A_closed_norm)
                errors_Ac.append(float(np.linalg.norm(A_c_hat - A_c_closed, ord="fro")) / A_c_closed_norm)

            errors_by_l[str(l)] = {
                "full": errors_full,
                "objective": errors_obj,
                "constraint": errors_constr,
                "A_matrix": errors_A,
                "Ac_matrix": errors_Ac,
            }
            print(
                f"l={l}: full={np.mean(errors_full):.4e}, "
                f"objective={np.mean(errors_obj):.4e}, constraint={np.mean(errors_constr):.4e}, "
                f"||A-Ahat||_F/||A||_F={np.mean(errors_A):.4e}, "
                f"||Ac-Achat||_F/||Ac||_F={np.mean(errors_Ac):.4e}"
            )

        save_to_serializable_json(errors_by_l, errors_path)

    def _mean_sem(values: list) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))

    def _series_for(errors_dict: dict, x_values: list, keys_labels: list[tuple[str, str]]) -> dict:
        series = {}
        for label, key in keys_labels:
            means, sems = zip(*(_mean_sem(errors_dict[str(x)][key]) for x in x_values))
            series[label] = (list(means), list(sems))
        return series

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    # Each point uses m=l=l prior/posterior samples, so the total amount of
    # data behind the full plug-in estimate is m+l=2l; plot against that
    # total rather than the per-source count l (cache lookups still key on
    # the per-source l via sample_sizes).
    total_sizes = [int(2 * l) for l in sample_sizes]
    plot_closed_form_sensitivity_error(
        sample_sizes=total_sizes,
        series=_series_for(errors_by_l, sample_sizes, [
            (r"Full plug-in ($\widehat A$, $\widehat A_c$)", "full"),
        ]),
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="gaussian_1d_location_model_closed_form_convergence.pdf",
        xlabel=r"$m + l$",
    )

    plot_closed_form_sensitivity_error(
        sample_sizes=sample_sizes,
        series=_series_for(errors_by_l, sample_sizes, [(r"$\|A - \widehat{A}\|_F$", "A_matrix")]),
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="gaussian_1d_location_model_closed_form_A_error.pdf",
        ylabel=r"$\|A - \widehat{A}\|_F / \|A\|_F$",
        xlabel=r"$m$",
    )

    plot_closed_form_sensitivity_error(
        sample_sizes=sample_sizes,
        series=_series_for(errors_by_l, sample_sizes, [(r"$\|A_c - \widehat{A}_c\|_F$", "Ac_matrix")]),
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="gaussian_1d_location_model_closed_form_Ac_error.pdf",
        ylabel=r"$\|A_c - \widehat{A}_c\|_F / \|A_c\|_F$",
        xlabel=r"$l$",
    )

if __name__ == "__main__":
    # run_param_nonparam_comparison_skewness()
    # run_param_nonparam_comparison_skewness_matched_radius()
    # run_param_nonparam_comparison_kurtosis()
    # run_param_nonparam_multimodality_comparison()
    # run_gaussian_priors_nonparametric()
    # run_multivariate_gaussian_priors_nonparametric()
    # run_gaussian_priors_nonparametric_diff_radii()
    # run_gaussian_priors_nonparametric_diff_center_methods()
    # run_gaussian_priors_nonparametric_diff_kernels()
    # run_multivariate_gaussian_priors_nonparametric_diff_radii()
    run_multivariate_gaussian_diff_basis_funcs_num_runtimes()
    # run_gaussian_priors_nonparametric_closed_form_convergence()
