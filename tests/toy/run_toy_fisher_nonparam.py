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
import copy

warnings.filterwarnings("ignore", category=UserWarning)


def _closed_form_conjugate_gaussian_M(mu_ref, Sigma_ref, x_bar, Sigma_over_n) -> float:
    """
    Closed form M = ||l(.;x_1:n)||_{L^inf(Pi_ref)} / Z_ref for the conjugate
    Gaussian location model, with likelihood ~ N(x_bar, Sigma/n) and reference
    prior ~ N(mu_ref, Sigma_ref):

        M = sqrt(|Sigma/n + Sigma_ref| / |Sigma/n|)
            * exp(0.5 * (x_bar - mu_ref)^T (Sigma/n + Sigma_ref)^{-1} (x_bar - mu_ref))

    so that S^FD(Q_r) = M * r (Thm. exact-fd-sensitivity).
    """
    mu_ref = np.atleast_1d(np.asarray(mu_ref, dtype=float))
    x_bar = np.atleast_1d(np.asarray(x_bar, dtype=float))
    Sigma_ref = np.atleast_2d(np.asarray(Sigma_ref, dtype=float))
    Sigma_over_n = np.atleast_2d(np.asarray(Sigma_over_n, dtype=float))

    combined = Sigma_over_n + Sigma_ref
    diff = x_bar - mu_ref
    log_det_ratio = np.linalg.slogdet(combined)[1] - np.linalg.slogdet(Sigma_over_n)[1]
    quad_form = float(diff @ np.linalg.solve(combined, diff))
    return float(np.exp(0.5 * log_det_ratio + 0.5 * quad_form))


def _k_dependent_basis_settings(
    samples: np.ndarray, K: int, n_mc_samples: int, span: float = 3.0, safe_fraction: float = 0.2,
) -> tuple[int, float]:
    """
    (K_eff, lengthscale) for an isotropic RBF/Matern sieve whose bandwidth
    shrinks as the number of basis functions grows: lengthscale =
    domain_scale / K_eff^(1/d), with K_eff = min(K, safe_fraction *
    n_mc_samples) *also* used as the actual number of basis functions built
    (not just to compute the bandwidth).

    Without this, MaternBasisFunction(Multidim)'s default bandwidth is
    estimated once from `samples` independent of K (median pairwise sample
    distance, or sample covariance), so raising num_basis_functions in a
    config just adds near-duplicate centres at a fixed, too-wide bandwidth
    and the sieve sensitivity barely grows (verified empirically: it
    plateaus after only a few dozen centres).

    K_eff is capped at safe_fraction * n_mc_samples (n_mc_samples = number of
    Monte-Carlo prior/posterior samples used to estimate A, A_c) because
    pushing the bandwidth past what those samples can resolve makes A, A_c
    ill-conditioned and the generalised eigenvalue blow up by orders of
    magnitude (verified empirically: stable and monotonically increasing up
    to K ~ 0.2x n_mc_samples, then degrades and eventually explodes).
    Crucially, K_eff must also cap the *number of centres actually built*:
    capping only the bandwidth while still placing all K (possibly far more
    than K_eff) centres just recreates the same ill-conditioning from
    near-duplicate centres crammed inside a bandwidth sized for fewer of them
    (verified empirically). So requesting more basis functions than
    safe_fraction * n_mc_samples silently gets fewer than requested; that
    ceiling is a property of the finite sample size, not of this schedule,
    and can only be raised by increasing prior/posterior_samples_num.
    """
    samples = np.asarray(samples, dtype=float)
    _, d = samples.shape
    domain_scale = span * float(np.sqrt(np.max(np.var(samples, axis=0))))
    K_eff = min(K, max(1, int(safe_fraction * n_mc_samples)))
    if K > K_eff:
        print(
            f"WARNING: num_basis_functions K={K} exceeds {safe_fraction:.0%} of the "
            f"Monte-Carlo sample size (n={n_mc_samples}) used to estimate A, A_c. "
            f"Using only K_eff={K_eff} basis functions (with a correspondingly "
            "smaller bandwidth) to avoid an ill-conditioned generalised "
            "eigenvalue problem; increase prior/posterior_samples_num to "
            "safely benefit from a larger K."
        )
    lengthscale = float(domain_scale / (K_eff ** (1.0 / d)))
    return K_eff, lengthscale


def _anisotropic_precision_from_lengthscale(samples: np.ndarray, lengthscale: float) -> np.ndarray:
    """
    Precision matrix combining the samples' covariance *shape* (so
    orientation/anisotropy, e.g. Sigma_ref's off-diagonal correlation, is
    preserved like MaternBasisFunctionMultidim's default
    _estimate_precision_from_samples) with an overall *scale* set by
    `lengthscale` (so it still shrinks with K per
    _k_dependent_basis_settings): normalises inv(cov(samples)) to unit
    determinant (encodes only shape) then divides by lengthscale**2 (sets
    the scale), so det(precision) == det(I / lengthscale**2) in an isotropic
    basis of the same dimension.
    """
    samples = np.asarray(samples, dtype=float)
    d = samples.shape[1]
    cov = np.cov(samples, rowvar=False)
    P0 = np.linalg.inv(cov)
    P0 = 0.5 * (P0 + P0.T)
    P0 /= np.linalg.det(P0) ** (1.0 / d)
    return P0 / lengthscale ** 2


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
    result_eig = optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)
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

    # Lengthscale shrinks with K so increasing num_basis_functions in the
    # config actually grows the achievable sensitivity, instead of the
    # class's default (K-independent, sample-based) bandwidth causing it to
    # plateau after a few dozen centres -- see _k_dependent_basis_settings.
    n_mc_samples = min(len(model.prior_samples_init), len(model.posterior_samples_init))
    basis_kwargs["num_basis_functions"], basis_kwargs["lengthscale"] = _k_dependent_basis_settings(
        centers_pool_samples, basis_kwargs["num_basis_functions"], n_mc_samples,
    )
    basis_function = basis_cls(**basis_kwargs)

    M_closed = _closed_form_conjugate_gaussian_M(
        mu_ref=model.prior_init.mu,
        Sigma_ref=model.prior_init.var,
        x_bar=model.x_bar,
        Sigma_over_n=model.loss.var / model.observations_num,
    )
    print(f"Closed-form M (conjugate Gaussian location model): {M_closed:.4f}")

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
        print(
            f"Radius: {radius}, sensitivity: {result_sdp['primal_value']}, closed-form S^FD = M*r: {M_closed * radius:.4f}")

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
        domain=(1, 5),
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
    draw of the reference prior (same size as the saved prior samples), and
    posterior-based centres from a fresh, independent draw of the conjugate
    posterior (same size as the saved posterior samples), from which a subset
    is then selected via the chosen method:
      (a) Halton quantile-mapped centres from the fresh reference-prior draw.
      (b) K-means centres from the same fresh reference-prior draw.
      (c) Halton quantile-mapped centres from the fresh posterior draw.
      (d) Random (i.i.d. subsample) centres from the fresh reference-prior draw.
      (e) Random (i.i.d. subsample) centres from the fresh posterior draw.
      (f) K-means centres from a fresh draw of the normalised likelihood
          N(theta_hat, sigma^2 / (n * lr)), i.e. centres placed around the
          maximum-likelihood estimate theta_hat = x_bar.
    """
    radius = 5.0
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
    # Likewise, a fresh, independent draw from the (conjugate) posterior, used
    # only as the pool for posterior-based centres and their lengthscale --
    # never the posterior samples that feed the FD estimate.
    centers_pool_posterior_samples = model.sample_posterior(n_samples=len(model.posterior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    base_basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    base_basis_kwargs["num_basis_functions"] = 30
    base_basis_kwargs["nu"] = 5.0

    # Lengthscale shrinks with K (per basis function's own centre count) so
    # increasing num_basis_functions actually grows the achievable
    # sensitivity instead of plateauing -- see _k_dependent_basis_settings.
    n_mc_samples = min(len(model.prior_samples_init), len(model.posterior_samples_init))

    def _apply_k_schedule(kwargs, samples_for_scale):
        kwargs["num_basis_functions"], kwargs["lengthscale"] = _k_dependent_basis_settings(
            samples_for_scale, kwargs["num_basis_functions"], n_mc_samples,
        )
        return kwargs

    # Solved as a generalised eigenvalue problem on A_c's well-conditioned
    # subspace (b = b_c = 0 here, g = pi_ref) -- not the SDP relaxation.
    def _run_and_plot(method_label, filename, optimizer):
        result_sdp = optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)
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
    basis_kwargs = _apply_k_schedule(basis_kwargs, centers_pool_samples)
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
    basis_kwargs = _apply_k_schedule(basis_kwargs, centers_pool_samples)
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
    basis_kwargs["posterior_samples"] = centers_pool_posterior_samples
    basis_kwargs["estimation_samples_source"] = "posterior"
    basis_kwargs["method"] = "halton"
    # Not scheduled: posterior samples are far more concentrated than the
    # fresh prior draw, so the same K-dependent formula produces a
    # disproportionately narrow (unstable) bandwidth here -- verified
    # empirically (already inflated at K=30, SDP relaxation unbounded by
    # K=70 in the loop below). Left on the class's own auto-estimated
    # (K-independent) bandwidth.
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
    basis_kwargs = _apply_k_schedule(basis_kwargs, centers_pool_samples)
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
    basis_kwargs["posterior_samples"] = centers_pool_posterior_samples
    basis_kwargs["estimation_samples_source"] = "posterior"
    basis_kwargs["method"] = "random"
    # Not scheduled: see the "Halton (posterior)" case above.
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

    # (f) K-means, centres from a fresh draw of the normalised likelihood.
    # For the Gaussian location model L(theta)^lr is proportional to
    # N(theta; x_bar, sigma^2 / (n * lr)), so the pool concentrates around the
    # MLE theta_hat = x_bar -- independent of both the prior and posterior
    # samples used for the FD estimate.
    theta_hat = float(np.asarray(model.x_bar).reshape(-1)[0])
    likelihood_sd = float(np.sqrt(model.loss.var / (model.observations_num * model.loss_lr_init)))
    likelihood_pool_samples = np.random.default_rng(27).normal(
        theta_hat, likelihood_sd, size=(len(original_prior_samples), 1),
    )
    print(f"[Likelihood max] theta_hat={theta_hat:.4f}, likelihood sd={likelihood_sd:.4f}")
    basis_kwargs = dict(base_basis_kwargs)
    basis_kwargs["prior_samples"] = None
    basis_kwargs["posterior_samples"] = likelihood_pool_samples
    basis_kwargs["estimation_samples_source"] = "posterior"
    basis_kwargs["method"] = "kmeans"
    # Not scheduled: the likelihood pool is as concentrated as the posterior
    # samples -- see the "Halton (posterior)" case above.
    basis_function_likelihood = basis_cls(**basis_kwargs)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
        basis_function=basis_function_likelihood,
    )
    _run_and_plot(
        method_label="K-means (likelihood maximum)",
        filename="gaussian_1d_location_model_centers_kmeans_likelihood_max.pdf",
        optimizer=optimizer,
    )

    # Combined figure: K-means (fresh prior), Random (fresh prior), Random
    # (posterior) and K-means (likelihood maximum) densities overlaid in one panel, with each method's
    # basis centres shown in its own colour-matched rug strip underneath --
    # one such combined figure per number of basis functions K, so the effect
    # of K on the centre-selection comparison is also visible.
    def _compute(basis_function):
        optimizer = OptimisationNonparametricBase(
            estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
            basis_function=basis_function,
        )
        return optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)

    for K in [60, 70, 80, 90, 100]:
        k_basis_kwargs = dict(base_basis_kwargs)
        k_basis_kwargs["num_basis_functions"] = K

        kmeans_kwargs = dict(k_basis_kwargs, prior_samples=centers_pool_samples, posterior_samples=None,
                             estimation_samples_source="prior", method="kmeans")
        kmeans_kwargs = _apply_k_schedule(kmeans_kwargs, centers_pool_samples)
        basis_function_kmeans_k = basis_cls(**kmeans_kwargs)
        result_kmeans_k = _compute(basis_function_kmeans_k)

        random_prior_kwargs = dict(k_basis_kwargs, prior_samples=centers_pool_samples, posterior_samples=None,
                                   estimation_samples_source="prior", method="random")
        random_prior_kwargs = _apply_k_schedule(random_prior_kwargs, centers_pool_samples)
        basis_function_random_prior_k = basis_cls(**random_prior_kwargs)
        result_random_prior_k = _compute(basis_function_random_prior_k)

        # Not scheduled: see the "Halton (posterior)" case above.
        random_posterior_kwargs = dict(k_basis_kwargs, prior_samples=None, posterior_samples=centers_pool_posterior_samples,
                                       estimation_samples_source="posterior", method="random")
        basis_function_random_posterior_k = basis_cls(**random_posterior_kwargs)
        result_random_posterior_k = _compute(basis_function_random_posterior_k)

        # Not scheduled: see the "K-means (likelihood maximum)" case above.
        likelihood_kwargs = dict(k_basis_kwargs, prior_samples=None, posterior_samples=likelihood_pool_samples,
                                 estimation_samples_source="posterior", method="kmeans")
        basis_function_likelihood_k = basis_cls(**likelihood_kwargs)
        result_likelihood_k = _compute(basis_function_likelihood_k)

        print(
            f"[K={K}] K-means (fresh prior): {result_kmeans_k['primal_value']:.4f}, "
            f"Random (fresh prior): {result_random_prior_k['primal_value']:.4f}, "
            f"Random (posterior): {result_random_posterior_k['primal_value']:.4f}, "
            f"K-means (likelihood maximum): {result_likelihood_k['primal_value']:.4f}"
        )

        plot_sdp_density_with_centers_combined(
            basis_functions=[
                basis_function_kmeans_k,
                basis_function_random_prior_k,
                basis_function_random_posterior_k,
                basis_function_likelihood_k,
            ],
            lambda_star_list=[
                result_kmeans_k["lambda_star"],
                result_random_prior_k["lambda_star"],
                result_random_posterior_k["lambda_star"],
                result_likelihood_k["lambda_star"],
            ],
            estimates=[
                result_kmeans_k["primal_value"],
                result_random_prior_k["primal_value"],
                result_random_posterior_k["primal_value"],
                result_likelihood_k["primal_value"],
            ],
            labels=["K-means (fresh prior)", "Random (fresh prior)", "Random (posterior)",
                    "K-means (likelihood maximum)"],
            prior_distribution=model.prior_init,
            plot_cfg=plot_cfg,
            output_dir=plots_output_dir,
            filename=f"gaussian_1d_location_model_centers_combined_K{K}.pdf",
            domain=(-6, 12),
            resolution=500,
        )

    # Recommended centre selection (K=100): draw K_max >> K candidates from the
    # mixture alpha * posterior + (1 - alpha) * prior -- each component a fresh
    # draw, separate from the FD-estimation samples -- then select a
    # well-spread subset of K via k-means. Compared, for several alphas,
    # against candidates from the prior only, the posterior only, and the
    # (in practice unknown) oracle likelihood around its maximiser theta_hat.
    # All use k-means, so only the candidate pool differs.
    #
    # Unlike the figures above, A and A_c are estimated here from larger fresh
    # prior/posterior draws (n_rec_mc each, separate from every centre pool):
    # with 1000 prior samples, narrow bases clustered near theta_hat get too
    # few prior samples under each bump, so A_c is underestimated and the
    # sensitivity inflated far above the population value. Verified against
    # exact quadrature: still ~10% too high at 50k, within ~4% at 200k, and
    # within ~1% at 500k.
    K = 100
    K_max = 10 * K
    mixture_alphas = [0.25, 0.5, 0.75]
    rec_basis_kwargs = dict(base_basis_kwargs, num_basis_functions=K, method="kmeans")

    def _mixture_pool(alpha):
        n_post = int(round(alpha * K_max))
        return np.concatenate([
            centers_pool_posterior_samples[:n_post],
            centers_pool_samples[:K_max - n_post],
        ], axis=0)

    rec_methods = []
    for alpha in mixture_alphas:
        pool = _mixture_pool(alpha)
        kwargs = dict(rec_basis_kwargs, prior_samples=pool, posterior_samples=None, estimation_samples_source="prior")
        kwargs = _apply_k_schedule(kwargs, pool)
        rec_methods.append((rf"Mixture $\alpha={alpha:g}$", basis_cls(**kwargs)))

    prior_only_pool = centers_pool_samples[:K_max]
    kwargs = dict(rec_basis_kwargs, prior_samples=prior_only_pool, posterior_samples=None,
                  estimation_samples_source="prior")
    kwargs = _apply_k_schedule(kwargs, prior_only_pool)
    rec_methods.append(("Prior only", basis_cls(**kwargs)))

    # Not scheduled: see the "Halton (posterior)" case above.
    kwargs = dict(rec_basis_kwargs, prior_samples=None, posterior_samples=centers_pool_posterior_samples[:K_max],
                  estimation_samples_source="posterior")
    rec_methods.append(("Posterior only", basis_cls(**kwargs)))

    # Not scheduled: see the "K-means (likelihood maximum)" case above.
    kwargs = dict(rec_basis_kwargs, prior_samples=None, posterior_samples=likelihood_pool_samples[:K_max],
                  estimation_samples_source="posterior")
    rec_methods.append(("Oracle likelihood", basis_cls(**kwargs)))

    n_rec_mc = 50000
    rec_model = copy.copy(model)
    np.random.seed(2027)
    rec_model.prior_samples_init = model.sample_from_base_prior(n_samples=n_rec_mc)
    rec_model.posterior_samples_init = model.sample_posterior(n_samples=n_rec_mc)
    rec_estimator_prior = PriorFDNonParametric(model=rec_model)
    rec_estimator_posterior = PosteriorFDNonParametric(model=rec_model)

    def _compute_rec(basis_function):
        optimizer = OptimisationNonparametricBase(
            rec_estimator_posterior, rec_estimator_prior, cfg.optimize.nonparametric, radius=radius,
            basis_function=basis_function,
        )
        return optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)

    rec_results = [_compute_rec(basis_function) for _, basis_function in rec_methods]
    for (label, _), result in zip(rec_methods, rec_results):
        print(f"[K={K}, recommended] {label}: {result['primal_value']:.4f}")

    # Population ceiling over *all* smooth perturbations (b = b_c = 0):
    # sup_f E_post|f'|^2 / E_prior|f'|^2 = sup_theta p_post / p_prior, which is
    # proportional to the likelihood and so attained at theta_hat.
    mu_n, sigma_n2 = model.compute_posterior_params()
    theta_grid = np.linspace(-20, 25, 200001)[:, None]
    log_ratio = (
        -0.5 * (theta_grid[:, 0] - float(np.ravel(mu_n)[0])) ** 2 / float(sigma_n2)
        - 0.5 * np.log(2 * np.pi * float(sigma_n2))
        - np.asarray(model.prior_init.log_pdf(theta_grid)).reshape(-1)
    )
    ceiling = radius * float(np.exp(log_ratio.max()))
    ceiling_at = float(theta_grid[np.argmax(log_ratio), 0])
    print(f"[K={K}, recommended] Population ceiling r * sup p_post/p_prior: {ceiling:.4f} at theta={ceiling_at:.4f}")

    plot_sdp_density_with_centers_combined(
        basis_functions=[basis_function for _, basis_function in rec_methods],
        lambda_star_list=[result["lambda_star"] for result in rec_results],
        estimates=[result["primal_value"] for result in rec_results],
        labels=[label for label, _ in rec_methods],
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=plots_output_dir,
        filename=f"gaussian_1d_location_model_centers_recommended_K{K}.pdf",
        domain=(-6, 12),
        resolution=500,
        colors=["#1b4f72", "#2e86c1", "#85c1e9", "#6C936C", "#d68910", "#922b21"],
        legend_labels=True,
        upper_bound=ceiling,
        upper_bound_at=ceiling_at,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_kernels(cfg, save_samples: bool = False) -> None:
    """
    Compare basis-function kernel choices at a fixed radius, each as its own
    plot with the SDP worst-case candidate density, the true prior, and the
    resulting basis centres (rug of dots) along the x-axis:
      (a) Matern kernel, nu=2.5
      (b) Matern kernel, nu=5.5
      (c) Matern kernel, nu=7.5
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
    radius = 5.0
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

    num_basis_functions = 100

    # Lengthscale shrinks with K (per basis function's own centre count) so
    # increasing num_basis_functions actually grows the achievable
    # sensitivity instead of plateauing -- see _k_dependent_basis_settings.
    n_mc_samples = min(len(model.prior_samples_init), len(model.posterior_samples_init))
    num_basis_functions, kernel_lengthscale = _k_dependent_basis_settings(
        centers_pool_samples, num_basis_functions, n_mc_samples,
    )

    def _run_and_plot(method_label, filename, basis_function, show_centers=True, show_yaxis=True):
        optimizer = OptimisationNonparametricBase(
            estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
            basis_function=basis_function,
        )
        result_sdp = optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)
        print(f"[{method_label}] Nonparametric FD (primal value): {result_sdp['primal_value']:.4f}")
        plot_sdp_density_with_centers(
            basis_function=optimizer.basis_function,
            lambda_star=result_sdp["lambda_star"],
            estimate=result_sdp["primal_value"],
            prior_distribution=model.prior_init,
            plot_cfg=plot_cfg,
            output_dir=plots_output_dir,
            filename=filename,
            domain=(-8, 12),
            resolution=500,
            show_centers=show_centers,
            show_yaxis=show_yaxis,
        )
        return result_sdp

    # (a)-(c) Matern, varying smoothness nu (nu must be > 1 for C^1 basis functions)
    matern_cls = BASIS_FUNCTIONS_REGISTRY["MaternBasisFunction"]
    matern_basis_functions, matern_lambda_list, matern_nu_labels, matern_estimates = [], [], [], []
    for nu in [2.5, 5.5, 7.5]:
        basis_function = matern_cls(
            posterior_samples=None,
            prior_samples=centers_pool_samples,
            num_basis_functions=num_basis_functions,
            lengthscale=kernel_lengthscale,
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
        lengthscale=kernel_lengthscale,
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
    result_sdp = optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)
    elapsed = time.perf_counter() - start
    print(f"Generalised eigenvalue time: {elapsed:.3f}s")
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
    basis_kwargs["nu"] = 5
    basis_kwargs["method"] = "random"

    # Precision (metric="full") shrinks (isotropically) with K so increasing
    # num_basis_functions in the config actually grows the achievable
    # sensitivity, instead of the class's default (K-independent,
    # covariance-based) bandwidth causing it to plateau -- see
    # _k_dependent_basis_settings. Previously num_basis_functions was
    # hardcoded to 30 here, so editing the config's value had no effect at
    # all; it now comes from the config like every other basis_funcs_kwargs
    # entry.
    n_mc_samples = min(len(model.prior_samples_init), len(model.posterior_samples_init))
    basis_kwargs["num_basis_functions"], ell = _k_dependent_basis_settings(
        centers_pool_samples, basis_kwargs["num_basis_functions"], n_mc_samples,
    )
    basis_kwargs["precision"] = _anisotropic_precision_from_lengthscale(centers_pool_samples, ell)
    basis_function = basis_cls(**basis_kwargs)

    M_closed = _closed_form_conjugate_gaussian_M(
        mu_ref=model.prior_init.mu,
        Sigma_ref=model.prior_init.cov,
        x_bar=model.x_bar,
        Sigma_over_n=model.loss.cov / model.observations_num,
    )
    print(f"Closed-form M (conjugate Gaussian location model): {M_closed:.4f}")

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
        print(
            f"Radius: {radius}, sensitivity: {result_sdp['primal_value']}, closed-form S^FD = M*r: {M_closed * radius:.4f}")

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
            config_name="multivariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_sensitivity_vs_K(
    cfg, radius: float = 5.0, x_log_scale: bool = False,
) -> None:
    """
    Validates that the closed-form RBF/Gaussian sieve sensitivity
    S^FD(Q_r^K) = r * gamma_max(A, A_c) converges to the exact
    (unrestricted) closed-form sensitivity S^FD(Q_r) = M*r
    (Thm. exact-fd-sensitivity) as the number of RBF basis functions K
    grows, for the conjugate Gaussian location model (univariate or
    multivariate -- dimension-agnostic, see below).

    A and A_c are computed analytically via rbf_gaussian_gram_closed_form
    (exact expectations under the Gaussian reference prior/posterior)
    rather than estimated by Monte Carlo from samples, so this isolates the
    *sieve approximation* (bias) question from finite-sample estimation
    noise -- the latter is instead studied (at a single fixed K) by
    run_gaussian_priors_nonparametric_closed_form_convergence.

    RBF centres sit on a growing d-dimensional square grid centred at
    mu_ref (K = n_side^d basis functions per grid side n_side), spanning
    +/- `span` prior standard deviations per axis (using the largest prior
    variance across dimensions so the grid covers all axes), with
    lengthscale fixed to the grid spacing -- so the lengthscale shrinks as
    the grid gets denser, as required for the sieve to be consistent.
    A naive Matern/RBF sieve whose bandwidth is instead estimated once from
    the sample pool (independent of K) does *not* converge this way: it
    plateaus well below the true value once the fixed-bandwidth basis
    functions become near-collinear (verified empirically before adding
    this function).

    Dimension-agnostic: works for both the univariate (Gaussian prior) and
    multivariate (MultivariateGaussian prior) configs -- d is inferred from
    the model's prior, and the cache directory / plot filename are tagged
    "univariate_gaussian"/"gaussian_1d_location_model" or
    "multivariate_gaussian"/"gaussian_2d_location_model" accordingly, so the
    two configs' cached results and plots never collide.
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    mu_ref = np.atleast_1d(np.asarray(model.prior_init.mu, dtype=float))
    cov_ref = getattr(model.prior_init, "cov", model.prior_init.var)
    Sigma_ref = np.atleast_2d(np.asarray(cov_ref, dtype=float))
    mu_post, cov_post = model.compute_posterior_params()
    mu_post = np.atleast_1d(np.asarray(mu_post, dtype=float))
    Sigma_post = np.atleast_2d(np.asarray(cov_post, dtype=float))
    d = mu_ref.shape[0]
    dim_tag = "univariate_gaussian" if d == 1 else "multivariate_gaussian"
    plot_tag = "gaussian_1d_location_model" if d == 1 else "gaussian_2d_location_model"

    # Univariate converges to the true value fast enough (in K) that the
    # grid can be pushed out until the sieve estimate actually reaches it;
    # multivariate converges far slower in K (see the docstring), so a
    # comparably large n_side would mean K = n_side^2 basis functions --
    # infeasible -- so it keeps the original, more modest grid.
    if d == 1:
        grid_sides = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200,
                      250, 300, 400, 500, 700, 1000, 1500, 2000]
    else:
        grid_sides = [3, 4, 5, 7, 10, 14, 20, 28, 40, 56, 64, 72, 80]

    results_dir = os.path.join(get_original_cwd(), f"data/{dim_tag}/sensitivity_vs_K")
    grid_tag = "_".join(str(n) for n in grid_sides)
    results_path = os.path.join(results_dir, f"sensitivity_vs_K_r{radius:g}_grid_{grid_tag}.json")

    if os.path.exists(results_path):
        print(f"Found existing results at {results_path}, skipping computation.")
        cached = load_results_json(results_path)
        true_sensitivity = cached["true_sensitivity"]
        basis_funcs_nums = cached["basis_funcs_nums"]
        estimates = cached["estimates"]
        print(f"True sensitivity at r={radius}: {true_sensitivity:.4f}")
        for K, S_hat in zip(basis_funcs_nums, estimates):
            print(f"K={K}, sensitivity: {S_hat:.4f}")
    else:
        loss_cov = getattr(model.loss, "cov", model.loss.var)
        M_closed = _closed_form_conjugate_gaussian_M(
            mu_ref=mu_ref,
            Sigma_ref=Sigma_ref,
            x_bar=model.x_bar,
            Sigma_over_n=np.atleast_2d(np.asarray(loss_cov, dtype=float)) / model.observations_num,
        )
        true_sensitivity = M_closed * radius
        print(f"Closed-form M: {M_closed:.4f}, true sensitivity at r={radius}: {true_sensitivity:.4f}")

        span = 3.0
        half_width = span * float(np.sqrt(np.max(np.diag(Sigma_ref))))

        basis_funcs_nums, estimates = [], []
        for n_side in grid_sides:
            axis = np.linspace(-half_width, half_width, n_side)
            mesh = np.meshgrid(*([axis] * d))
            centers = np.stack([m.ravel() for m in mesh], axis=1) + mu_ref
            lengthscale = float(axis[1] - axis[0])

            A_c_closed = rbf_gaussian_gram_closed_form(centers, lengthscale, mu=mu_ref, Sigma=Sigma_ref)
            A_closed = rbf_gaussian_gram_closed_form(centers, lengthscale, mu=mu_post, Sigma=Sigma_post)
            gamma_max = _gamma_max_generalized_eig(A_closed, A_c_closed)
            S_hat = radius * gamma_max

            K = n_side ** d
            basis_funcs_nums.append(K)
            estimates.append(S_hat)
            print(f"K={K} (grid n_side={n_side}, d={d}), lengthscale={lengthscale:.4f}, sensitivity: {S_hat:.4f}")

        save_to_serializable_json(
            {
                "true_sensitivity": true_sensitivity,
                "basis_funcs_nums": basis_funcs_nums,
                "estimates": estimates,
            },
            results_path,
        )

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_sensitivity_vs_basis_funcs_num(
        basis_funcs_nums=basis_funcs_nums,
        estimates=estimates,
        true_value=true_sensitivity,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"{plot_tag}_sensitivity_vs_K.pdf",
        x_log_scale=x_log_scale,
    )


def run_gaussian_priors_nonparametric_sensitivity_vs_K_combined(
    grid_sides_univariate: list = None,
    grid_sides_multivariate: list = None,
    radius: float = 5.0,
    x_log_scale: bool = True,
) -> None:
    """
    Overlay the univariate and multivariate sensitivity-vs-K results (each
    produced beforehand by run_gaussian_priors_nonparametric_sensitivity_vs_K
    for its own config) in one figure: univariate on the left y-axis,
    multivariate on the right -- see plot_sensitivity_vs_basis_funcs_num_dual.
    This reads each config's cached JSON directly and does not recompute or
    instantiate any model; raises FileNotFoundError if either config's
    result for the given grid_sides/radius hasn't been computed yet.

    Plain (non-hydra) function, run from the repo root -- unlike the other
    functions in this file, it needs no cfg (it never instantiates a model),
    so it isn't decorated with @hydra.main.
    """
    if grid_sides_univariate is None:
        grid_sides_univariate = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200,
                                 250, 300, 400, 500, 700, 1000, 1500, 2000]
    if grid_sides_multivariate is None:
        grid_sides_multivariate = [3, 4, 5, 7, 10, 14, 20, 28, 40, 56, 64, 72, 80]

    def _grid_tag(grid_sides):
        return "_".join(str(n) for n in grid_sides)

    uni_path = os.path.join(
        "data/univariate_gaussian/sensitivity_vs_K",
        f"sensitivity_vs_K_r{radius:g}_grid_{_grid_tag(grid_sides_univariate)}.json",
    )
    multi_path = os.path.join(
        "data/multivariate_gaussian/sensitivity_vs_K",
        f"sensitivity_vs_K_r{radius:g}_grid_{_grid_tag(grid_sides_multivariate)}.json",
    )
    for path in (uni_path, multi_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run run_gaussian_priors_nonparametric_sensitivity_vs_K() "
                "for that config (with matching grid_sides/radius) first."
            )

    uni = load_results_json(uni_path)
    multi = load_results_json(multi_path)

    plot_cfg = load_plot_config("configs/plots/overleaf_plots_settings.yaml")
    output_dir = "outputs/paper/plots/fisher/combined"

    plot_sensitivity_vs_basis_funcs_num_dual(
        basis_funcs_nums_left=uni["basis_funcs_nums"],
        estimates_left=uni["estimates"],
        true_value_left=uni["true_sensitivity"],
        basis_funcs_nums_right=multi["basis_funcs_nums"],
        estimates_right=multi["estimates"],
        true_value_right=multi["true_sensitivity"],
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        x_log_scale=x_log_scale,
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
    result_sdp = nonparam_optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)
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
    result_sdp = nonparam_optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_closed_form_convergence(
    cfg, radius: float = 5.0, n_repeats: int = 1000,
) -> None:
    """
    Validates the closed-form FDsens+ sensitivity for the RBF kernel with a
    Gaussian reference prior/posterior against its Monte-Carlo (plug-in)
    estimate, and plots the absolute error between the two as a function of
    the number of posterior samples m = number of prior samples l, averaged
    (with +/- 1 sem band) over independent resamples at each sample size.
    The error is expected to shrink as m = l grows.

    Since prior and posterior are both exactly Gaussian for this conjugate
    toy model, S^FD(Q_r^K) = r * gamma_max can be computed exactly from A,
    A_c via rbf_gaussian_gram_closed_form, with A_c built from (mu_ref,
    Sigma_ref) and A from (mu_post, Sigma_post). The RBF centres (a square
    grid spanning +/- 3 prior std per axis, centred at mu_ref, with K =
    n_side^d basis functions, n_side chosen from the config's
    num_basis_functions -- see below) and lengthscale (the inter-centre
    spacing) are held fixed across all repeats/sample sizes, so the only
    source of discrepancy from the closed-form value is Monte-Carlo sample
    noise in the plug-in A, A_c. The Monte-Carlo side uses
    FixedCentersRBFBasisFunctionMultidim, whose joint isotropic kernel
    matches rbf_gaussian_gram_closed_form's exactly.

    Dimension-agnostic: works for both the univariate (Gaussian prior) and
    multivariate (MultivariateGaussian prior) configs -- d is inferred from
    the model's prior, and the cache directory / plot filenames are tagged
    "univariate_gaussian"/"gaussian_1d_location_model" or
    "multivariate_gaussian"/"gaussian_2d_location_model" accordingly, so the
    two configs' cached results and plots never collide.

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
    data/{univariate,multivariate}_gaussian/closed_form_convergence/closed_form_convergence_errors_K{K}.json
    (K in the filename so a different config's num_basis_functions -- hence a
    different actual K -- never silently reuses another K's stale cache) and
    reused on subsequent runs if present -- delete the file to force a
    recompute (e.g. after changing sample_sizes, n_repeats, or radius).
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    # Dimension-agnostic: works for both the univariate model (Gaussian prior,
    # exposing scalar .mu/.var) and the multivariate one (MultivariateGaussian
    # prior, exposing vector .mu/matrix .cov).
    mu_ref = np.atleast_1d(np.asarray(model.prior_init.mu, dtype=float))
    cov_ref = getattr(model.prior_init, "cov", model.prior_init.var)
    Sigma_ref = np.atleast_2d(np.asarray(cov_ref, dtype=float))
    mu_post, cov_post = model.compute_posterior_params()
    mu_post = np.atleast_1d(np.asarray(mu_post, dtype=float))
    Sigma_post = np.atleast_2d(np.asarray(cov_post, dtype=float))
    d = mu_ref.shape[0]
    dim_tag = "univariate_gaussian" if d == 1 else "multivariate_gaussian"
    plot_tag = "gaussian_1d_location_model" if d == 1 else "gaussian_2d_location_model"

    # K from the config, laid out as a square grid: the closest n_side^d to
    # the configured num_basis_functions (e.g. 1000 -> n_side=32 -> K=1024).
    num_basis_functions_cfg = cfg.optimize.nonparametric.basis_funcs_kwargs.num_basis_functions
    n_side = int(round(num_basis_functions_cfg ** (1.0 / d)))
    span = 3.0
    half_width = span * float(np.sqrt(np.max(np.diag(Sigma_ref))))
    axis = np.linspace(-half_width, half_width, n_side)
    mesh = np.meshgrid(*([axis] * d))
    centers = np.stack([m.ravel() for m in mesh], axis=1) + mu_ref  # (n_side**d, d)
    lengthscale = float(axis[1] - axis[0])
    print(
        f"Config num_basis_functions={num_basis_functions_cfg}; using n_side={n_side} "
        f"-> K={centers.shape[0]} basis functions."
    )

    fixed_centers_cls = BASIS_FUNCTIONS_REGISTRY["FixedCentersRBFBasisFunctionMultidim"]
    basis_function = fixed_centers_cls(centers=centers, lengthscale=lengthscale)

    # Closed-form A_c (theta ~ prior) and A (theta ~ posterior).
    A_c_closed = rbf_gaussian_gram_closed_form(centers, lengthscale, mu=mu_ref, Sigma=Sigma_ref)
    A_closed = rbf_gaussian_gram_closed_form(centers, lengthscale, mu=mu_post, Sigma=Sigma_post)
    gamma_max_closed = _gamma_max_generalized_eig(A_closed, A_c_closed)
    S_closed = radius * gamma_max_closed
    A_closed_norm = float(np.linalg.norm(A_closed, ord="fro"))
    A_c_closed_norm = float(np.linalg.norm(A_c_closed, ord="fro"))
    print(f"Closed-form gamma_max: {gamma_max_closed:.6f}, S_closed: {S_closed:.6f}")

    sample_size_start, sample_size_stop, sample_size_step = 1000, 15000, 2000
    sample_sizes = list(np.linspace(
        sample_size_start, sample_size_stop,
        num=int((sample_size_stop - sample_size_start) / sample_size_step) + 1,
        dtype=int,
    ))

    errors_dir = os.path.join(get_original_cwd(), f"data/{dim_tag}/closed_form_convergence")
    errors_path = os.path.join(errors_dir, f"closed_form_convergence_errors_K{centers.shape[0]}.json")

    def _compute_errors(pairs: list[tuple[int, int]]) -> dict:
        """Per-repeat errors for each (m, l) = (#posterior, #prior samples), keyed by str(m)."""
        errors = {}
        for m, l in pairs:
            errors_full, errors_obj, errors_constr = [], [], []
            errors_A, errors_Ac = [], []
            for _ in range(n_repeats):
                model.prior_samples_init = model.sample_from_base_prior(n_samples=l)
                model.posterior_samples_init = model.sample_posterior(n_samples=m)
                model.m = m
                model.m_prior = l

                estimator_prior = PriorFDNonParametric(model=model)
                estimator_posterior = PosteriorFDNonParametric(model=model)

                A_c_hat, _, _ = estimator_prior.compute_non_parametric_fisher_quadratic_form_prior_only(basis_function)
                A_hat, _, _ = estimator_posterior.compute_non_parametric_fisher_quadratic_form_prior_only(
                    basis_function)

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

            errors[str(m)] = {
                "m": int(m),
                "l": int(l),
                "full": errors_full,
                "objective": errors_obj,
                "constraint": errors_constr,
                "A_matrix": errors_A,
                "Ac_matrix": errors_Ac,
            }
            print(
                f"m={m}, l={l}: full={np.mean(errors_full):.4e}, "
                f"objective={np.mean(errors_obj):.4e}, constraint={np.mean(errors_constr):.4e}, "
                f"||A-Ahat||_F/||A||_F={np.mean(errors_A):.4e}, "
                f"||Ac-Achat||_F/||Ac||_F={np.mean(errors_Ac):.4e}"
            )
        return errors

    def _load_or_compute(path: str, pairs: list[tuple[int, int]]) -> dict:
        if os.path.exists(path):
            print(f"Found existing errors at {path}, skipping computation.")
            return load_results_json(path)
        errors = _compute_errors(pairs)
        save_to_serializable_json(errors, path)
        return errors

    errors_by_l = _load_or_compute(errors_path, [(int(l), int(l)) for l in sample_sizes])

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

    plot_closed_form_sensitivity_error(
        sample_sizes=sample_sizes,
        series=_series_for(errors_by_l, sample_sizes, [
            (r"Full plug-in ($\widehat A$, $\widehat A_c$)", "full"),
        ]),
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"{plot_tag}_closed_form_convergence.pdf",
        xlabel=r"$m = l$",
        y_bottom=0.0,
    )

    plot_closed_form_sensitivity_error(
        sample_sizes=sample_sizes,
        series=_series_for(errors_by_l, sample_sizes, [(r"$\|A - \widehat{A}\|_F$", "A_matrix")]),
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"{plot_tag}_closed_form_A_error.pdf",
        ylabel=r"$\|A - \widehat{A}\|_F / \|A\|_F$",
        xlabel=r"$m$",
        y_bottom=0.0,
    )

    plot_closed_form_sensitivity_error(
        sample_sizes=sample_sizes,
        series=_series_for(errors_by_l, sample_sizes, [(r"$\|A_c - \widehat{A}_c\|_F$", "Ac_matrix")]),
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"{plot_tag}_closed_form_Ac_error.pdf",
        ylabel=r"$\|A_c - \widehat{A}_c\|_F / \|A_c\|_F$",
        xlabel=r"$l$",
        y_bottom=0.0,
    )


if __name__ == "__main__":
    # run_param_nonparam_comparison_skewness()
    # run_param_nonparam_comparison_skewness_matched_radius()
    # run_param_nonparam_comparison_kurtosis()
    # run_param_nonparam_multimodality_comparison()
    # run_gaussian_priors_nonparametric()
    # run_multivariate_gaussian_priors_nonparametric()
    # run_gaussian_priors_nonparametric_diff_radii()
    run_gaussian_priors_nonparametric_diff_center_methods()
    # run_gaussian_priors_nonparametric_diff_kernels()
    # run_multivariate_gaussian_priors_nonparametric_diff_radii()
    # run_gaussian_priors_nonparametric_sensitivity_vs_K()
    # run_gaussian_priors_nonparametric_sensitivity_vs_K_combined()
    # run_gaussian_priors_nonparametric_closed_form_convergence()
    # run_multivariate_gaussian_diff_basis_funcs_num_runtimes()
