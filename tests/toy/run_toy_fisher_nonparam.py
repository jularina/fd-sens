from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.distributions.gaussian import MultivariateGaussian
from src.optimization.qcqp import ParametricQCQPBase
from src.optimization.corner_points_fisher import *
from src.utils.files_operations import *
from src.plots.paper.toy_paper_fisher_funcs import *
from src.discrepancies.prior_fisher import PriorFDParametric, PriorFDNonParametric
from src.discrepancies.posterior_fisher import PosteriorFDParametric, PosteriorFDNonParametric

import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import open_dict
import time
import json

warnings.filterwarnings("ignore", category=UserWarning)


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

    plot_sdp_densities_and_logprior(
        basis_function=optimizer.basis_function,
        sdp_lambda_list=sdp_lambda_list,
        radius_labels=radius_labels,
        estimates=sdp_fd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-10, 12),
        resolution=500
    )

    plot_sdp_densities(
        basis_function=optimizer.basis_function,
        sdp_lambda_list=sdp_lambda_list,
        radius_labels=radius_labels,
        estimates=sdp_fd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-10, 12),
        resolution=500
    )

    plot_sdp_posterior_comparison(
        basis_function=optimizer.basis_function,
        sdp_lambda_list=sdp_lambda_list,
        radius_labels=radius_labels,
        estimates=sdp_fd_estimates_list,
        prior_distribution=model.prior_init,
        model=model,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(1, 4),
        resolution=500,
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
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="multivariate_gaussian_nonparam")
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
    sdp_lambda_list, fd_estimates_list, radius_labels = [], [], []

    for radius in [0.5, 15.0]:  # [0.5, 1.0, 5.0, 15.0]
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
        domain=((-10, 10), (-10, 10)),
        resolution=500,
        contour_levels=10
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="multivariate_gaussian_nonparam")
def run_multivariate_gaussian_nonparam_runtimes(cfg) -> None:
    """
    Runtime benchmark: nonparametric eigenvalue optimisation across
    dims=(5, 25, 100) and n_samples=1000..10000 (step 1000), 100 reps each.
    Results saved to data/nonparam/multivariate_gaussian/runtimes/eigenvalue_runtimes.json.

    If eigenvalue_runtimes.json already exists, skips the benchmark and instead
    plots parametric vs nonparametric runtime figures, saving to the param_nonparam
    subfolder of cfg.flags.plots.output_dir.
    """
    dims = [2, 5, 7]
    radius = float(cfg.optimize.nonparametric.get("radius"))
    data_path = os.path.join(get_original_cwd(), "data/nonparam/multivariate_gaussian/runtimes/")
    os.makedirs(data_path, exist_ok=True)

    eigenvalue_path = os.path.join(data_path, "eigenvalue_runtimes.json")
    param_path = os.path.join(data_path, "parametric_optimisation_for_comparison.json")

    # --- Parametric benchmark ---
    if not os.path.exists(param_path):
        print("Parametric benchmark not found — running parametric optimisation.")
        sample_nums = list(range(1000, 5001, 1000))
        n_reps = 10
        param_times = defaultdict(lambda: defaultdict(dict))

        for dim in dims:
            print(f"\n=== Parametric dim={dim} ===")
            with open_dict(cfg):
                cfg.data.base_prior.mu = np.zeros(dim).tolist()
                cfg.data.base_prior.cov = (np.eye(dim) * 4.0).tolist()
                cfg.data.candidate_prior.mu = np.zeros(dim).tolist()
                cfg.data.candidate_prior.cov = (np.eye(dim) * 4.0).tolist()
                cfg.data.true_dgp.mu = (np.ones(dim) * 2.0).tolist()
                cfg.data.true_dgp.cov = (np.eye(dim) * 4.0).tolist()
                cfg.data.loss.mu = (np.ones(dim) * 2.0).tolist()
                cfg.data.loss.cov = (np.eye(dim) * 4.0).tolist()
                opt_pbr = cfg.optimize.parametric.prior.MultivariateGaussian.parameters_box_range
                opt_pbr.ranges.mu = {str(i): [-4.0, 5.0] for i in range(dim)}
                opt_pbr.nums.mu = {str(i): 2 for i in range(dim)}
                opt_pbr.ranges.cov = {f"{i}_{i}": [2.0, 3.0] for i in range(dim)}
                opt_pbr.nums.cov = {f"{i}_{i}": 2 for i in range(dim)}

            for n_samples in sample_nums:
                with open_dict(cfg):
                    cfg.data.posterior_samples_num = n_samples
                    cfg.data.prior_samples_num = n_samples

                rep_times = []
                for rep in range(n_reps):
                    model = instantiate(cfg.model, data_config=cfg.data)
                    fisher_estimator = PosteriorFDParametric(model=model)
                    optimizer = OptimizationCornerPointsMultivariateGaussian(
                        fisher_estimator, cfg.optimize.parametric.prior.MultivariateGaussian,
                        cfg.optimize.parametric.loss.MultivariateGaussianLogLikelihood)
                    start = time.perf_counter()
                    qf_priors_all_corners = optimizer.evaluate_all_prior_corners()
                    elapsed = time.perf_counter() - start

                    param_times[str(dim)][str(n_samples)][str(rep)] = elapsed
                    rep_times.append(elapsed)

                    print(
                        f"*** Parametric dim={dim:3d}, n_samples={n_samples:5d}, rep={rep}:  mean={np.mean(rep_times):.4f}s  std={np.std(rep_times):.4f} ***")

        with open(param_path, "w") as f:
            json.dump(param_times, f, indent=2)
        print(f"\nSaved parametric runtimes to {param_path}")
    else:
        print(f"Found {param_path}, loading parametric runtimes.")

    # --- Nonparametric benchmark ---
    if not os.path.exists(eigenvalue_path):
        times = defaultdict(lambda: defaultdict(dict))
        sample_nums = list(range(1000, 10001, 1000))
        n_reps = 10
        for dim in dims:
            print(f"\n=== Nonparametric dim={dim} ===")
            with open_dict(cfg):
                cfg.data.base_prior.mu = np.zeros(dim).tolist()
                cfg.data.base_prior.cov = (np.eye(dim) * 4.0).tolist()
                cfg.data.candidate_prior.mu = np.zeros(dim).tolist()
                cfg.data.candidate_prior.cov = (np.eye(dim) * 4.0).tolist()
                cfg.data.true_dgp.mu = (np.ones(dim) * 2.0).tolist()
                cfg.data.true_dgp.cov = (np.eye(dim) * 4.0).tolist()
                cfg.data.loss.mu = (np.ones(dim) * 2.0).tolist()
                cfg.data.loss.cov = (np.eye(dim) * 4.0).tolist()

            for n_samples in sample_nums:
                with open_dict(cfg):
                    cfg.data.posterior_samples_num = n_samples
                    cfg.data.prior_samples_num = n_samples

                rep_times = []
                for rep in range(n_reps):
                    model = instantiate(cfg.model, data_config=cfg.data)
                    estimator_prior = PriorFDNonParametric(model=model)
                    estimator_posterior = PosteriorFDNonParametric(model=model)
                    optimizer = OptimisationNonparametricBase(
                        estimator_posterior,
                        estimator_prior,
                        cfg.optimize.nonparametric,
                        radius=radius,
                    )
                    start = time.perf_counter()
                    optimizer.optimize_through_generalized_eigenvalue()
                    elapsed = time.perf_counter() - start

                    times[str(dim)][str(n_samples)][str(rep)] = elapsed
                    rep_times.append(elapsed)

                    print(
                        f"********* dim={dim:3d}, n_samples={n_samples:5d}, rep={rep}: mean={np.mean(rep_times):.4f}s  std={np.std(rep_times):.4f}*********")

        with open(eigenvalue_path, "w") as f:
            json.dump(times, f, indent=2)
        print(f"\nSaved nonparametric runtimes to {eigenvalue_path}")
    else:
        print(f"Found {eigenvalue_path}, loading nonparametric runtimes.")

    # --- Plot ---
    with open(eigenvalue_path) as f:
        nonparam_raw_times = json.load(f)
    with open(param_path) as f:
        param_raw_times = json.load(f)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    nonparam_ms = sorted(int(k) for k in next(iter(nonparam_raw_times.values())).keys())
    param_ms = sorted(int(k) for k in next(iter(param_raw_times.values())).keys())

    plot_param_nonparam_runtimes_gaussians(
        output_dir=output_dir,
        plot_cfg=plot_cfg,
        param_raw_times=param_raw_times,
        nonparam_raw_times=nonparam_raw_times,
        dims=dims,
        param_ms=param_ms,
        nonparam_ms=nonparam_ms,
        logy=True,
        xlim=(1000, 10500)
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian")
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
            fisher_estimator = PosteriorFDParametric(model=model)
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
def run_priors_optimisation_runtimes(cfg, dim: str = "multivariate"):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    data_path = os.path.join(get_original_cwd(), f"data/{dim}_gaussian/runtimes/")
    with open(data_path + "parametric_optimisation_times.json", "r") as f:
        parametric_optimisation_times = json.load(f)

    with open(data_path + "nonparametric_optimisation_times.json", "r") as f:
        nonparametric_optimisation_times = json.load(f)

    plot_runtime_parametric_nonparametric_with_ci(
        parametric_optimisation_times,
        nonparametric_optimisation_times,
        plot_cfg,
        output_dir,
        filename=f"runtime_parametric_nonparametric_{dim}.pdf"
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
def run_priors_optimisation_runtimes(cfg):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/")

    with open(data_path + "nonparametric_qcqp_optimisation_times.json", "r") as f:
        nonparametric_optimisation_times = json.load(f)
    with open(data_path + "parametric_qcqp_optimisation_times.json", "r") as f:
        parametric_optimisation_times = json.load(f)
    plot_runtime_parametric_nonparametric_with_ci(
        parametric_optimisation_times,
        nonparametric_optimisation_times,
        plot_cfg,
        output_dir,
        filename="runtime_parametric_nonparametric_qcqp_multivariate.pdf"
    )

    # with open(data_path + "nonparametric_qcqp_optimisation_times.json", "r") as f:
    #     nonparametric_optimisation_times = json.load(f)
    # plot_runtime_nonparametric_with_ci(
    #     nonparametric_optimisation_times,
    #     plot_cfg,
    #     output_dir,
    #     filename="runtime_parametric_nonparametric_qcqp_univariate.pdf"
    # )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian_param_nonparam")
def run_param_nonparam_comparison(cfg) -> None:
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

    plot_param_nonparam_comparison_optimised(
        worst_corner=worst_corner,
        lambda_star=result_sdp["lambda_star"],
        basis_function=nonparam_optimizer.basis_function,
        model=model,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    # run_param_nonparam_comparison()
    # run_gaussian_priors_nonparametric()
    # run_multivariate_gaussian_priors_nonparametric()
    # run_multivariate_gaussian_nonparam_runtimes()

    # run_gaussian_priors_nonparametric_diff_radii()
    run_multivariate_gaussian_priors_nonparametric_diff_radii()
    # run_gaussian_priors_nonparametric_diff_samples_num()
    # run_multivariate_gaussian_priors_diff_basis_funcs_num()
    # run_priors_optimisation_runtimes()
