from src.nonparametric.optimization import OptimisationNonparametricBase
from src.common.distributions.gaussian import Gaussian, MultivariateGaussian
import numpy as np
from src.parametric.corner_points import *
from src.common.utils.files_operations import *
from paper.toy.plots_nonparametric import *
from src.nonparametric.fisher import PriorFDNonParametric
from src.parametric.fisher import PosteriorFDParametric
from src.nonparametric.fisher import PosteriorFDNonParametric
from src.nonparametric.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.nonparametric.basis_functions import rbf_gaussian_gram_closed_form
from scipy.linalg import eigh as scipy_eigh

import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import OmegaConf
import time
import copy

warnings.filterwarnings("ignore", category=UserWarning)


def _closed_form_conjugate_gaussian_M(mu_ref, Sigma_ref, x_bar, Sigma_over_n) -> float:
    """Closed-form M for the conjugate Gaussian location model, so that S^FD(Q_r) = M * r."""
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
    """Return (K_eff, lengthscale) for a sieve whose bandwidth shrinks with the capped basis count."""
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
    """Precision matrix with the samples' covariance shape and an overall scale set by lengthscale."""
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = False) -> None:
    """Compute FD sensitivity for the univariate Gaussian model across radii with random basis centres."""
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

    # Fresh reference-prior draw used only as the centre pool, never for the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(model.prior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["method"] = "random"

    # Lengthscale shrinks with K so larger K grows the sensitivity instead of plateauing.
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


def _diff_center_methods(
    cfg, basis_funcs_type: str, file_prefix: str, save_samples: bool = False, only_recommended: bool = False,
) -> None:
    """Shared body comparing basis-centre selection methods at a fixed radius for Matern or RBF bases."""
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

    # Fresh reference-prior draw used only as the centre pool, never for the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(original_prior_samples))
    # Likewise a fresh posterior draw, used only as the pool for posterior-based centres.
    centers_pool_posterior_samples = model.sample_posterior(n_samples=len(model.posterior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[basis_funcs_type]
    base_basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    base_basis_kwargs["num_basis_functions"] = 30
    # Matern smoothness nu comes from the config; RBF has no nu.
    if basis_funcs_type != "MaternBasisFunction":
        base_basis_kwargs.pop("nu", None)

    # Lengthscale shrinks with K so larger K grows the sensitivity -- see _k_dependent_basis_settings.
    n_mc_samples = min(len(model.prior_samples_init), len(model.posterior_samples_init))

    def _apply_k_schedule(kwargs, samples_for_scale):
        kwargs["num_basis_functions"], kwargs["lengthscale"] = _k_dependent_basis_settings(
            samples_for_scale, kwargs["num_basis_functions"], n_mc_samples,
        )
        return kwargs

    # Generalised eigenproblem on A_c's well-conditioned subspace (b = b_c = 0), not the SDP.
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

    # Per-method and combined figures are skipped if only_recommended; later figures don't need them.
    if not only_recommended:
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
            filename=f"{file_prefix}_halton_prior_fresh.pdf",
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
            filename=f"{file_prefix}_kmeans_prior_fresh.pdf",
            optimizer=optimizer,
        )

        # (c) Halton, centres selected from the posterior samples
        basis_kwargs = dict(base_basis_kwargs)
        basis_kwargs["prior_samples"] = None
        basis_kwargs["posterior_samples"] = centers_pool_posterior_samples
        basis_kwargs["estimation_samples_source"] = "posterior"
        basis_kwargs["method"] = "halton"
        # Not scheduled: posterior pool is too concentrated, so the K-schedule bandwidth is unstable.
        basis_function_posterior = basis_cls(**basis_kwargs)
        optimizer = OptimisationNonparametricBase(
            estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
            basis_function=basis_function_posterior,
        )
        _run_and_plot(
            method_label="Halton (posterior)",
            filename=f"{file_prefix}_halton_posterior.pdf",
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
            filename=f"{file_prefix}_random_prior_fresh.pdf",
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
            filename=f"{file_prefix}_random_posterior.pdf",
            optimizer=optimizer,
        )

        # (f) K-means, centres from a fresh draw of the normalised likelihood N(x_bar, sigma^2/(n*lr)).
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
        # Not scheduled: the likelihood pool is as concentrated as the posterior samples.
        basis_function_likelihood = basis_cls(**basis_kwargs)
        optimizer = OptimisationNonparametricBase(
            estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
            basis_function=basis_function_likelihood,
        )
        _run_and_plot(
            method_label="K-means (likelihood maximum)",
            filename=f"{file_prefix}_kmeans_likelihood_max.pdf",
            optimizer=optimizer,
        )

        # Combined figure per K: methods' densities overlaid, with colour-matched centre rug strips.
        def _compute(basis_function):
            optimizer = OptimisationNonparametricBase(
                estimator_posterior, estimator_prior, cfg.optimize.nonparametric, radius=radius,
                basis_function=basis_function,
            )
            return optimizer.optimize_through_generalized_eigenvalue(rel_tol=1e-8)

        for K in [100]:
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
                filename=f"{file_prefix}_combined_K{K}.pdf",
                domain=(-6, 12),
                resolution=500,
            )

    # Recommended centres: k-means on a posterior/prior mixture pool; A, A_c from large fresh draws.
    K = 100
    K_max = 10 * K
    mixture_alphas = [0.25, 0.5]
    rec_basis_kwargs = dict(base_basis_kwargs, num_basis_functions=K, method="kmeans")

    def _mixture_pool(alpha):
        n_post = int(round(alpha * K_max))
        return np.concatenate([
            centers_pool_posterior_samples[:n_post],
            centers_pool_samples[:K_max - n_post],
        ], axis=0)

    def _stratified_mixture_centers(alpha, n_centers, min_sep, seed=27):
        # Stratified: round(alpha*K) centres from posterior, rest from prior; near-duplicates replaced.
        from sklearn.cluster import KMeans

        n_post = int(round(alpha * K_max))
        post_part = centers_pool_posterior_samples[:n_post]
        prior_part = centers_pool_samples[:K_max - n_post]
        n_post_centers = int(round(alpha * n_centers))
        n_prior_centers = n_centers - n_post_centers

        post_centers = KMeans(n_clusters=n_post_centers, random_state=0).fit(post_part).cluster_centers_
        prior_centers = KMeans(n_clusters=n_prior_centers, random_state=0).fit(prior_part).cluster_centers_

        dist_to_post = np.min(np.abs(prior_centers[:, None, 0] - post_centers[None, :, 0]), axis=1)
        keep = dist_to_post >= min_sep
        centers = np.concatenate([post_centers, prior_centers[keep]], axis=0)

        rng = np.random.default_rng(seed)
        n_replace = int((~keep).sum())
        for _ in range(n_replace):
            dist_to_centers = np.min(np.abs(prior_part[:, None, 0] - centers[None, :, 0]), axis=1)
            candidates = np.flatnonzero(dist_to_centers >= min_sep)
            if candidates.size == 0:
                print(f"[alpha={alpha:g}] No prior candidate at least {min_sep:.4f} from all centres; "
                      f"using {len(centers)} centres instead of {n_centers}.")
                break
            centers = np.concatenate([centers, prior_part[rng.choice(candidates)][None, :]], axis=0)

        print(f"[alpha={alpha:g}] Stratified centres: {n_post_centers} posterior, {n_prior_centers} prior, "
              f"{n_replace} prior centres within {min_sep:.4f} of a posterior centre replaced.")
        return centers

    pooled_mixture_methods = []
    stratified_mixture_methods = []
    for alpha in mixture_alphas:
        pool = _mixture_pool(alpha)
        kwargs = dict(rec_basis_kwargs, prior_samples=pool, posterior_samples=None, estimation_samples_source="prior")
        kwargs = _apply_k_schedule(kwargs, pool)
        pooled_mixture_methods.append((rf"$\alpha={alpha:g}$", basis_cls(**kwargs)))

        basis_function = basis_cls(**kwargs)
        # Replace pooled centres with stratified ones; keep the lengthscale, min_sep = lengthscale / 2.
        basis_function.centers = _stratified_mixture_centers(
            alpha, kwargs["num_basis_functions"], min_sep=0.5 * kwargs["lengthscale"],
        )
        stratified_mixture_methods.append((rf"$\alpha={alpha:g}$", basis_function))

    baseline_methods = []
    prior_only_pool = centers_pool_samples[:K_max]
    kwargs = dict(rec_basis_kwargs, prior_samples=prior_only_pool, posterior_samples=None,
                  estimation_samples_source="prior")
    kwargs = _apply_k_schedule(kwargs, prior_only_pool)
    baseline_methods.append((plot_cfg.plot.param_latex_names.baseprior, basis_cls(**kwargs)))

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

    pooled_mixture_results = [_compute_rec(basis_function) for _, basis_function in pooled_mixture_methods]
    stratified_mixture_results = [_compute_rec(basis_function) for _, basis_function in stratified_mixture_methods]
    baseline_results = [_compute_rec(basis_function) for _, basis_function in baseline_methods]
    for (label, _), result in zip(
        pooled_mixture_methods + stratified_mixture_methods + baseline_methods,
        pooled_mixture_results + stratified_mixture_results + baseline_results,
    ):
        print(f"[K={K}, recommended] {label}: {result['primal_value']:.4f}")

    # RBF only: exact sensitivity from closed-form A, A_c; the gap to plug-in is pure MC error.
    sensitivity_label_key = "estimatedSensitivityMeasure"
    if basis_funcs_type == "RBFBasisFunction":
        from src.nonparametric.node_sensitivity import _ac_whitening_transform

        mu_post, sigma_post2 = model.compute_posterior_params()

        def _exact_rbf_result(basis_function):
            A_exact = rbf_gaussian_gram_closed_form(
                basis_function.centers, basis_function.lengthscale, mu=mu_post, Sigma=sigma_post2,
            )
            A_c_exact = rbf_gaussian_gram_closed_form(
                basis_function.centers, basis_function.lengthscale, mu=model.prior_init.mu, Sigma=model.prior_init.var,
            )
            W, _ = _ac_whitening_transform(A_c_exact, rel_tol=1e-8, nugget=1e-10)
            omega_vals, Y = np.linalg.eigh(0.5 * (W.T @ A_exact @ W + (W.T @ A_exact @ W).T))  # ascending
            lambda_star = np.sqrt(radius) * (W @ Y[:, -1])  # A_c-normalised, scaled to the radius
            lambda_star = OptimisationNonparametricBase._canonical_sign(lambda_star)
            return {"primal_value": radius * float(omega_vals[-1]), "lambda_star": lambda_star}

        def _exact_results(methods, plug_in_results):
            exact_results = []
            for (label, basis_function), result in zip(methods, plug_in_results):
                exact = _exact_rbf_result(basis_function)
                S_exact, S_hat = exact["primal_value"], result["primal_value"]
                print(
                    f"[K={K}, recommended] {label}: exact {S_exact:.4f}, plug-in {S_hat:.4f}, "
                    f"error {abs(S_hat - S_exact):.4f} ({abs(S_hat - S_exact) / S_exact:.2%})"
                )
                exact_results.append(exact)
            return exact_results

        pooled_mixture_results = _exact_results(pooled_mixture_methods, pooled_mixture_results)
        stratified_mixture_results = _exact_results(stratified_mixture_methods, stratified_mixture_results)
        baseline_results = _exact_results(baseline_methods, baseline_results)
        sensitivity_label_key = "sensitivityMeasure"

    # Population ceiling over all smooth perturbations: r * M, attained at theta_hat.
    M_closed = _closed_form_conjugate_gaussian_M(
        mu_ref=model.prior_init.mu,
        Sigma_ref=model.prior_init.var,
        x_bar=model.x_bar,
        Sigma_over_n=model.loss.var / (model.observations_num * model.loss_lr_init),
    )
    ceiling = radius * M_closed
    ceiling_at = float(np.asarray(model.x_bar).reshape(-1)[0])
    print(f"[K={K}, recommended] Population ceiling r * sup p_post/p_prior: {ceiling:.4f} at theta={ceiling_at:.4f}")

    # One figure for pooled mixtures, one for stratified; both against the prior-only baseline.
    for mixture_methods, mixture_results, filename in [
        (pooled_mixture_methods, pooled_mixture_results,
         f"{file_prefix}_recommended_K{K}.pdf"),
        (stratified_mixture_methods, stratified_mixture_results,
         f"{file_prefix}_recommended_stratified_K{K}.pdf"),
    ]:
        # Pi_ref baseline first, so it heads the legend (and the rug strips).
        rec_methods = baseline_methods + mixture_methods
        rec_results = baseline_results + mixture_results
        plot_sdp_density_with_centers_combined(
            basis_functions=[basis_function for _, basis_function in rec_methods],
            lambda_star_list=[result["lambda_star"] for result in rec_results],
            estimates=[result["primal_value"] for result in rec_results],
            labels=[label for label, _ in rec_methods],
            prior_distribution=model.prior_init,
            plot_cfg=plot_cfg,
            output_dir=plots_output_dir,
            filename=filename,
            domain=(-6, 12),
            resolution=500,
            colors=["#6C936C", "#ADEBDC", "#4d7298"],  # Pi_ref: palette's blue mint
            legend_labels=True,
            upper_bound=ceiling,
            upper_bound_at=ceiling_at,
            sensitivity_label_key=sensitivity_label_key,
        )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_center_methods(cfg, save_samples: bool = False) -> None:
    """Centre-selection comparison with the Matern basis (nu from the config) -- see _diff_center_methods."""
    _diff_center_methods(
        cfg, basis_funcs_type="MaternBasisFunction", file_prefix="gaussian_1d_location_model_centers",
        save_samples=save_samples,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_center_methods_rbf(cfg, save_samples: bool = False) -> None:
    """Compare recommended centre-selection methods with the RBF basis, including exact sensitivities."""
    _diff_center_methods(
        cfg, basis_funcs_type="RBFBasisFunction", file_prefix="gaussian_1d_location_model_centers_rbf",
        save_samples=save_samples, only_recommended=True,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_kernels(cfg, save_samples: bool = False) -> None:
    """Compare Matern and RBF basis kernels at a fixed radius via worst-case candidate density plots."""
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

    # Fresh reference-prior draw used only as the centre pool, never for the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(original_prior_samples))

    num_basis_functions = 100

    # Lengthscale shrinks with K so larger K grows the sensitivity -- see _k_dependent_basis_settings.
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
def run_multivariate_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = False) -> None:
    """Compute FD sensitivity for the multivariate Gaussian model across radii with random basis centres."""
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

    # Fresh reference-prior draw used only as the centre pool, never for the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(model.prior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    basis_kwargs["prior_samples"] = centers_pool_samples
    basis_kwargs["posterior_samples"] = None
    basis_kwargs["estimation_samples_source"] = "prior"
    basis_kwargs["nu"] = 5
    basis_kwargs["method"] = "random"

    # Precision shrinks with K so larger K grows the sensitivity -- see _k_dependent_basis_settings.
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
    """Check that the closed-form RBF sieve sensitivity converges to the exact sensitivity as K grows."""
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

    # Univariate converges fast in K so its grid extends further; multivariate K = n_side^2 limits it.
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
    """Overlay cached univariate and multivariate sensitivity-vs-K approximation errors in one figure."""
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
        radius=radius,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        x_log_scale=x_log_scale,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian_param_nonparam")
def run_param_nonparam_comparison_skewness_matched_radius(cfg) -> None:
    """Compare parametric and nonparametric worst-case priors using the diff_radii r=10 setup."""
    # Use the same saved data as run_gaussian_priors_nonparametric_diff_radii for reproducibility.
    cfg.data.observations_path = "data/univariate_gaussian/observations.npy"
    cfg.data.posterior_samples_path = "data/univariate_gaussian/posterior_samples.npy"
    cfg.data.prior_samples_path = "data/univariate_gaussian/prior_samples.npy"
    cfg.data.posterior_samples_num = 5000
    cfg.data.prior_samples_num = 5000
    model = instantiate(cfg.model, data_config=cfg.data)

    # Parametric optimisation: worst-case Gaussian prior within the (mu, sigma) box.
    posterior_fd = PosteriorFDParametric(model=model)
    param_optimizer = OptimizationCornerPointsUnivariateGaussian(
        posterior_fd,
        cfg.fd.optimize.prior.Gaussian,
        cfg.fd.optimize.loss.GaussianLogLikelihood,
    )
    _, worst_corner = param_optimizer.evaluate_all_prior_corners()
    print(f"Worst parametric prior: mu={worst_corner['mu']:.4f}, sigma={worst_corner['sigma']:.4f}")

    radius = 10.0

    # Nonparametric optimisation -- same setup as run_gaussian_priors_nonparametric_diff_radii, r=10.
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)

    # Fresh reference-prior draw used only as the centre pool, never for the FD estimate.
    np.random.seed(27)
    centers_pool_samples = model.sample_from_base_prior(n_samples=len(model.prior_samples_init))

    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    # Match diff_radii's basis hyperparameters (config defaults: num_basis_functions=10, nu=2.0).
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


def _diff_basis_funcs_num_runtimes(
    cfg,
    dim_tag: str,
    plot_tag: str,
    seed_basis_funcs_num: list[int] | None = None,
    basis_kwargs_overrides: dict | None = None,
    save_samples: bool = False,
) -> None:
    """Time nonparametric optimisation over basis counts K and sample sizes, with per-(m+l, K) caching."""
    total_samples_list = [500, 1000, 5000, 10000]
    basis_funcs_num = [50, 100, 200, 400, 600]
    iters = 500

    def _runtimes_filename(ks, totals):
        return (f"nonparametric_optimisation_times_diff_basis_funcs_K{'_'.join(map(str, ks))}"
                f"_mplusl_{'_'.join(map(str, totals))}.json")

    runtimes_dir = os.path.join(get_original_cwd(), f"data/{dim_tag}/runtimes/nonparam")
    runtimes_path = os.path.join(runtimes_dir, _runtimes_filename(basis_funcs_num, total_samples_list))
    candidate_paths = [runtimes_path]
    if seed_basis_funcs_num is not None:
        candidate_paths.append(
            os.path.join(runtimes_dir, _runtimes_filename(seed_basis_funcs_num, total_samples_list))
        )

    times_list = defaultdict(lambda: defaultdict(dict))
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"Loading existing runtimes from {path}.")
            for total_m, by_k in _json_keys_to_int(load_results_json(path)).items():
                for k, steps in by_k.items():
                    times_list[total_m][k] = steps
            break

    missing = [
        (total_m, k) for total_m in total_samples_list for k in basis_funcs_num
        if len(times_list[total_m].get(k, {})) < iters
    ]

    if not missing:
        print("All (m+l, K) runtimes already computed, skipping computation.")
    else:
        output_dir = os.path.join(get_original_cwd(), f"data/{dim_tag}")

        if save_samples:
            model = instantiate(cfg.model, data_config=cfg.data)
            os.makedirs(output_dir, exist_ok=True)
            np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
            np.save(output_dir + "/observations.npy", model.observations)
            np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

        cfg.data.posterior_samples_path = None
        cfg.data.prior_samples_path = None
        for key, value in (basis_kwargs_overrides or {}).items():
            cfg.optimize.nonparametric.basis_funcs_kwargs[key] = value

        for total_m, k in missing:
            m = total_m // 2
            cfg.data.posterior_samples_num = m
            cfg.data.prior_samples_num = m
            cfg.optimize.nonparametric.basis_funcs_kwargs["num_basis_functions"] = k
            times_list[total_m][k] = {}
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
            # Save after each (m+l, K) so an interrupted run keeps finished combinations.
            save_to_serializable_json(times_list, runtimes_path)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_runtime_nonparametric_diff_basis_funcs_num_diff_samples_with_ci(
        times_list,
        plot_cfg,
        output_dir,
        filename=f"{plot_tag}_runtime_diff_basis_funcs_nums_diff_samples.pdf",
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian_nonparam")
def run_multivariate_gaussian_diff_basis_funcs_num_runtimes(cfg, save_samples: bool = False) -> None:
    """Time nonparametric optimisation vs K for the multivariate Gaussian model."""
    _diff_basis_funcs_num_runtimes(
        cfg, dim_tag="multivariate_gaussian", plot_tag="gaussian_2d_location_model",
        seed_basis_funcs_num=[50, 100, 200, 400], save_samples=save_samples,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_univariate_gaussian_diff_basis_funcs_num_runtimes(cfg, save_samples: bool = False) -> None:
    """Time nonparametric optimisation vs K for the univariate Gaussian model with random centres."""
    _diff_basis_funcs_num_runtimes(
        cfg, dim_tag="univariate_gaussian", plot_tag="gaussian_1d_location_model",
        basis_kwargs_overrides={"method": "random"}, save_samples=save_samples,
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
    """Plot the error of the plug-in RBF FDsens+ sensitivity against its closed form as samples grow."""
    model = instantiate(cfg.model, data_config=cfg.data)

    # Dimension-agnostic: works with univariate (.mu/.var) and multivariate (.mu/.cov) priors.
    mu_ref = np.atleast_1d(np.asarray(model.prior_init.mu, dtype=float))
    cov_ref = getattr(model.prior_init, "cov", model.prior_init.var)
    Sigma_ref = np.atleast_2d(np.asarray(cov_ref, dtype=float))
    mu_post, cov_post = model.compute_posterior_params()
    mu_post = np.atleast_1d(np.asarray(mu_post, dtype=float))
    Sigma_post = np.atleast_2d(np.asarray(cov_post, dtype=float))
    d = mu_ref.shape[0]
    dim_tag = "univariate_gaussian" if d == 1 else "multivariate_gaussian"
    plot_tag = "gaussian_1d_location_model" if d == 1 else "gaussian_2d_location_model"

    # K from the config as a square grid: the closest n_side^d to num_basis_functions.
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
                # Relative Frobenius-norm error of each plug-in matrix: ||A - Ahat||_F / ||A||_F.
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

    # Other m-vs-l allocations (m = l^(3/2), m = 3l), each cached to its own file.
    m_grid = [int(m) for m in sample_sizes if m >= 3000]
    relations = [
        (r"$m = l$", None, lambda m: m),
        (r"$m = l^{3/2}$", "m_eq_l_pow_3_2", lambda m: int(round(m ** (2.0 / 3.0)))),
        (r"$m = 3l$", "m_eq_3l", lambda m: int(round(m / 3.0))),
    ]
    errors_by_relation = {}
    for label, suffix, l_of_m in relations:
        if suffix is None:
            errors_by_relation[label] = errors_by_l
            continue
        path = os.path.join(errors_dir, f"closed_form_convergence_errors_K{centers.shape[0]}_{suffix}.json")
        errors_by_relation[label] = _load_or_compute(path, [(m, l_of_m(m)) for m in m_grid])

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

    # Sensitivity errors divided by r, so the plotted error is independent of the radius.
    sensitivity_series = {}
    for label, errors in errors_by_relation.items():
        means, sems = _series_for(errors, m_grid, [(label, "full")])[label]
        sensitivity_series[label] = ([v / radius for v in means], [v / radius for v in sems])

    plot_closed_form_sensitivity_error(
        sample_sizes=m_grid,
        series=sensitivity_series,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"{plot_tag}_closed_form_convergence.pdf",
        ylabel=r"$|S^{\mathrm{FD}}(\mathcal{Q}_r^K) - \widehat{S}_m^{\mathrm{FD}}(\widehat{\mathcal{Q}}_r^{K,l})| / r$",
        xlabel=r"$m$",
        y_bottom=0.0,
        ylabel_fontsize_scale=0.7,
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
    # run_param_nonparam_comparison_skewness_matched_radius()
    # run_gaussian_priors_nonparametric_diff_radii()
    # run_gaussian_priors_nonparametric_diff_center_methods()
    # run_gaussian_priors_nonparametric_diff_center_methods_rbf()
    # run_gaussian_priors_nonparametric_diff_kernels()
    # run_multivariate_gaussian_priors_nonparametric_diff_radii()
    # run_gaussian_priors_nonparametric_sensitivity_vs_K()
    # run_gaussian_priors_nonparametric_sensitivity_vs_K_combined()
    # run_gaussian_priors_nonparametric_closed_form_convergence()
    # run_multivariate_gaussian_diff_basis_funcs_num_runtimes()
    run_univariate_gaussian_diff_basis_funcs_num_runtimes()
