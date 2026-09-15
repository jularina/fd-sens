import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
from scipy.special import logsumexp

import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.distributions.gaussian import Gaussian
from src.distributions.cauchy import HalfCauchy
from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.utils.files_operations import load_plot_config, save_to_serializable_json
from src.optimization.bnn_node_sensitivity import compute_group_omega_max, compute_node_lambda_star
from src.plots.paper.bnn_paper_funcs import plot_bnn_node_candidate_priors
from src.plots.paper.posterior_db_paper_funcs import (
    plot_component_sensitivity_bar,
    plot_posterior_predictive_with_data,
    _apply_plot_rc,
    _save_fig,
)
from tests.posteriordb.run_ark_kilpisjarvi import (
    x_years,
    y,
    y_centered,
    _ar_posterior_predictive,
    _summarise_bands,
    _mean_acf,
    print_predictive_variance_decomposition,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Kilpisjarvi's composite prior decomposes into 7 independent scalar blocks:
# alpha, beta1..beta5 ~ N(0, 5^2), and sigma ~ HalfCauchy(gamma). This mirrors
# the LaTeX names used by run_ark_kilpisjarvi.py's parametric per-component
# analysis.
LATEX_NAMES = {
    "alpha": r"$\alpha$",
    "beta1": r"$\beta_1$",
    "beta2": r"$\beta_2$",
    "beta3": r"$\beta_3$",
    "beta4": r"$\beta_4$",
    "beta5": r"$\beta_5$",
    "sigma": r"$\sigma$",
}
COMPONENT_ORDER = ["alpha", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma"]

_PIT_EPS = 1e-8


def _to_z_space(prior_dist, x: np.ndarray) -> np.ndarray:
    """
    Probability-integral-transform x -- drawn from `prior_dist` -- to
    z = Phi^{-1}(F_ref(x)), so that z ~ N(0, 1) exactly under the reference
    prior regardless of the original family. This puts every one of
    Kilpisjarvi's 7 components (Gaussian alpha/beta1..5, Half-Cauchy sigma)
    on the same standard-Gaussian reference footing, so a single shared
    neighbourhood radius r_j and a single shared basis-fitting recipe
    (same kernel, same centre-selection/lengthscale rule) represent a
    genuinely comparable degree of prior perturbation for every component --
    unlike comparing raw sensitivities in each parameter's own physical
    units, which are not comparable across differently-scaled/shaped
    reference families (see the omega_max scale-invariance discussion this
    script's earlier iteration was built around).

    For Gaussian components this reduces to the exact closed form
    z = (x - mu) / sigma (no CDF round-trip, avoiding needless precision
    loss from clipping near 0/1 in the far tails). Half-Cauchy has no such
    shortcut, so its CDF F(sigma) = 2*arctan(sigma/gamma)/pi is used
    directly, clipped to [eps, 1-eps] before Phi^{-1} for numerical
    stability.
    """
    x = np.asarray(x, dtype=float)
    if isinstance(prior_dist, Gaussian):
        return (x - prior_dist.mu) / prior_dist.sigma
    if isinstance(prior_dist, HalfCauchy):
        u = (2.0 / np.pi) * np.arctan(np.maximum(x, 0.0) / prior_dist.gamma)
        u = np.clip(u, _PIT_EPS, 1.0 - _PIT_EPS)
        return scipy_norm.ppf(u)
    raise NotImplementedError(f"No reference-prior CDF implemented for {type(prior_dist).__name__}.")


def _log_kef_density_ratio(
    z: np.ndarray, basis, lambda_star: np.ndarray, prior_samples_z: np.ndarray,
) -> np.ndarray:
    """
    log q_K(z) / pi_ref(z) = f(z) - log Z, where f(z) = phi(z) . lambda_star
    and Z = E_{z ~ N(0,1)}[exp(f(z))] is q_K's normalising constant relative
    to the (already-standard-Gaussian, post PIT) reference density -- since
    prior_samples_z are themselves i.i.d. N(0,1) draws under the reference,
    a plain Monte Carlo estimate of Z from them is exact in expectation
    (no separate quadrature grid needed).
    """
    f_at_z = basis.evaluate(np.asarray(z, dtype=float).reshape(-1, 1))[:, 0, :] @ np.asarray(lambda_star)
    f_at_prior = basis.evaluate(prior_samples_z.reshape(-1, 1))[:, 0, :] @ np.asarray(lambda_star)
    log_Z = logsumexp(f_at_prior) - np.log(len(prior_samples_z))
    return f_at_z - log_Z


def _kef_reweight_posterior(
    node_records: list,
    loader,
    posterior_full: np.ndarray,
    prior_samples_z_by_group: dict,
    r_pred: float,
    rng: np.random.Generator,
):
    """
    Self-normalised importance-sample (SNIS) the reference posterior draws
    into a resample representing the posterior under the worst-case KEF
    candidate prior *at radius r_pred* -- not necessarily the same radius
    r_j the per-component sensitivities were computed at. Since
    lambda_star = sqrt(r) * eigenvector and the eigenvector doesn't depend
    on r, rescaling each component's already-fit lambda_star by
    sqrt(r_pred / r_j_used) gives the exact worst-case KEF perturbation at
    r_pred with no re-optimisation.

    Returns (kef_samples, ess, ess_frac).
    """
    node_by_name = {rec["name"]: rec for rec in node_records}
    log_w = np.zeros(posterior_full.shape[0])
    for idx, name in enumerate(COMPONENT_ORDER):
        rec = node_by_name[name]
        prior_dist = loader.groups[rec["group"]]["prior_dist"]
        z_col = _to_z_space(prior_dist, posterior_full[:, idx])
        lam_scaled = np.asarray(rec["lambda_star"]) * np.sqrt(r_pred / rec["r_j"])
        log_w += _log_kef_density_ratio(
            z_col, rec["basis"], lam_scaled, prior_samples_z_by_group[rec["group"]]
        )
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    ess_frac = ess / len(w)

    resample_idx = rng.choice(len(w), size=len(w), replace=True, p=w)
    return posterior_full[resample_idx], ess, ess_frac


# ---------------------------------------------------------------------------
# Parametric-style worst-case corner search performed entirely in z-space,
# using ONE shared neighbourhood for every component -- unlike
# run_ark_kilpisjarvi_param_nonparam.py's parametric analysis, where
# alpha/beta1..5's box (mu in [-2,2], sigma in [0.25,1], original scale) and
# sigma's box (gamma in [0.2,5.0], Half-Cauchy) are declared independently in
# each family's own native units, so they are NOT the same neighbourhood
# budget at all -- they were just chosen separately per family, with no
# principled way to compare their sizes.
#
# Since every component's reference becomes exactly N(0,1) in z-space
# regardless of its original family (Gaussian alpha/beta1..5, Half-Cauchy
# sigma), positing the SAME Gaussian-in-z candidate family and the SAME
# numeric box (mu_z, sigma_z ranges) for every one of the 7 components makes
# them genuinely comparable -- no black-box, family-specific search for
# sigma needed at all, since we no longer require its candidate to stay
# within the Half-Cauchy family.
# ---------------------------------------------------------------------------

def _fd_z_posterior_gaussian_in_z(
    mu_z_cand: float, sigma_z_cand: float, z_post_mean: float, z_post_meansq: float,
) -> float:
    """
    Exact E_{z~posterior_z}[(score_ref_z(z) - score_cand_z(z))^2] for a
    reference N(0,1) and a Gaussian-in-z candidate N(mu_z_cand,
    sigma_z_cand^2), using the posterior's own empirical z-moments (mean,
    second moment) -- diff_z(z) = a*z + b is affine in z (a = 1/sigma_z_cand^2
    - 1, b = -mu_z_cand/sigma_z_cand^2), so E[diff_z(z)^2] = a^2*E[z^2] +
    2ab*E[z] + b^2 exactly (same derivation as run_ark_kilpisjarvi_param_
    nonparam.py's _fd_z_posterior_gaussian_gaussian, reimplemented here,
    parametrised directly by the z-space candidate (mu_z, sigma_z) instead
    of an original-scale Gaussian object, to avoid a circular import between
    the two files).
    """
    a = 1.0 / sigma_z_cand ** 2 - 1.0
    b = -mu_z_cand / sigma_z_cand ** 2
    return float(a ** 2 * z_post_meansq + 2.0 * a * b * z_post_mean + b ** 2)


def compute_uniform_z_neighbourhood_parametric_sensitivity(
    loader,
    mu_z_range: tuple = (-0.4, 0.4),
    sigma_z_range: tuple = (0.8, 1.25),
) -> dict:
    """
    For each of Kilpisjarvi's 7 scalar components, find the worst-case
    Gaussian-in-z candidate within the SAME shared box (mu_z in mu_z_range,
    sigma_z in sigma_z_range) -- the posterior-based z-space FD, maximised
    over the box's 4 corners (the quadratic form is convex in the candidate's
    z-space natural parameters, so its supremum over a box is attained at a
    vertex; see run_ark_kilpisjarvi_param_nonparam.py's
    _posterior_sup_z_gaussian for the same argument).

    Defaults: mu_z in [-0.4,0.4] (reused from alpha/beta1..5's own declared
    mu box, [-2,2], divided by their reference sigma_ref=5), sigma_z in
    [0.8,1.25] -- a genuine neighbourhood of the reference sigma_z=1
    (0.8 = 1/1.25, so it's symmetric in log-scale around 1), unlike
    [0.25,1]/5=[0.05,0.2] (alpha/beta1..5's own declared sigma box), which
    never actually contains sigma_z=1 and so isn't a neighbourhood of the
    reference at all in the scale direction.

    Returns {component_name: fd_z_sup}.
    """
    mu_lo, mu_hi = mu_z_range
    sig_lo, sig_hi = sigma_z_range
    corners = [(mu_lo, sig_lo), (mu_lo, sig_hi), (mu_hi, sig_lo), (mu_hi, sig_hi)]

    fd_z_sup = {}
    for group_name in loader.param_groups:
        g = loader.groups[group_name]
        prior_dist = g["prior_dist"]
        posterior = np.asarray(g["posterior"], dtype=float)
        for local_idx, node_name in enumerate(g["node_names"]):
            z_post = _to_z_space(prior_dist, posterior[:, local_idx])
            z_mean = float(np.mean(z_post))
            z_meansq = float(np.mean(z_post ** 2))
            vals = [
                _fd_z_posterior_gaussian_in_z(mu_z, sig_z, z_mean, z_meansq)
                for mu_z, sig_z in corners
            ]
            fd_z_sup[node_name] = float(max(vals))

    return fd_z_sup


def _fd_z_prior_gaussian_in_z(mu_z_cand: float, sigma_z_cand: float) -> float:
    """
    Exact PRIOR-based FD_z = E_{z~N(0,1)}[(score_ref_z(z)-score_cand_z(z))^2]
    for a Gaussian-in-z candidate against the N(0,1) reference -- unlike
    _fd_z_posterior_gaussian_in_z, this is evaluated under the *reference*
    measure (E[z]=0, E[z^2]=1 exactly), not under any component's posterior,
    so it does not depend on the data at all: diff_z(z)=a*z+b reduces to
    E[diff_z(z)^2] = a^2*1 + 2ab*0 + b^2 = a^2+b^2.
    """
    a = 1.0 / sigma_z_cand ** 2 - 1.0
    b = -mu_z_cand / sigma_z_cand ** 2
    return float(a ** 2 + b ** 2)


def compute_uniform_z_neighbourhood_prior_fd(
    mu_z_range: tuple = (-0.4, 0.4),
    sigma_z_range: tuple = (0.8, 1.25),
) -> dict:
    """
    Prior-based FD_z sup under the SAME shared Gaussian-in-z box as
    compute_uniform_z_neighbourhood_parametric_sensitivity (which, despite
    its name, computes the *posterior*-based sup -- see its docstring).
    Because this quantity is evaluated under the fixed reference measure
    z~N(0,1) rather than any component's own posterior, it does not depend
    on the data at all: with an identical box and an identical (always-
    N(0,1)) z-space reference for every component, this sup is therefore
    exactly identical across all 7 components -- unlike the posterior-based
    sup, which genuinely differs per component (different real posteriors).
    Returned per-component anyway (rather than a single scalar) to make
    that equality explicit/checkable.
    """
    mu_lo, mu_hi = mu_z_range
    sig_lo, sig_hi = sigma_z_range
    corners = [(mu_lo, sig_lo), (mu_lo, sig_hi), (mu_hi, sig_lo), (mu_hi, sig_hi)]
    sup_val = max(_fd_z_prior_gaussian_in_z(mu_z, sig_z) for mu_z, sig_z in corners)
    return {name: sup_val for name in COMPONENT_ORDER}


def compute_nonparametric_sensitivity_at_radii(
    loader,
    basis_cls,
    basis_kwargs: dict,
    r_j_by_component: dict,
    center_samples_num: int = 5000,
) -> dict:
    """
    Transforms a set of per-component radii (e.g. uniform_fd_z_sup from
    compute_uniform_z_neighbourhood_parametric_sensitivity) into the actual
    realised nonparametric KEF sensitivity at those radii: per node, fits
    lambda_star/omega_max at that node's OWN r_j = r_j_by_component[node]
    (instead of one shared r_j applied identically to every node, as
    run_ark_kilpisjarvi_nonparametric_sensitivity's main loop does), then
    sensitivity = r_j * omega_max exactly as elsewhere in this module.

    Returns {"sensitivity": {name: value}, "percentages": {name: pct},
    "node_records": [...]} -- node_records carries lambda_star/basis/r_j per
    node too, reusable for candidate-density plotting or posterior-
    predictive reweighting (see run_ark_kilpisjarvi_param_nonparam.py's
    _kef_reweight_posterior_own_radii).
    """
    node_records = []
    for group_name in loader.param_groups:
        g = loader.groups[group_name]
        prior_dist = g["prior_dist"]

        prior_samples_z = _to_z_space(prior_dist, loader.sample_prior(group_name))
        center_prior_samples_z = _to_z_space(
            prior_dist, loader.sample_prior(group_name, n_samples=center_samples_num)
        )
        posterior_z = _to_z_space(prior_dist, g["posterior"])

        for local_idx, node_name in enumerate(g["node_names"]):
            r_j = float(r_j_by_component[node_name])
            lam_star, omega_max, basis, _diag = compute_node_lambda_star(
                posterior_samples_col=posterior_z[:, local_idx],
                loc=0.0,
                scale=1.0,
                prior_samples=prior_samples_z,
                basis_cls=basis_cls,
                basis_kwargs=basis_kwargs,
                radius_j=r_j,
                center_prior_samples=center_prior_samples_z,
            )
            node_records.append({
                "name": node_name, "group": group_name, "r_j": r_j,
                "lambda_star": lam_star, "basis": basis, "omega_max": omega_max,
                "sensitivity": r_j * omega_max,
            })

    sensitivity = {rec["name"]: rec["sensitivity"] for rec in node_records}
    total = sum(sensitivity.values())
    percentages = {name: v / total * 100.0 for name, v in sensitivity.items()}
    return {"sensitivity": sensitivity, "percentages": percentages, "node_records": node_records}


def plot_posterior_predictive_radius_sweep(
    plot_cfg,
    output_dir: str,
    x_years: np.ndarray,
    y_uncentered: np.ndarray,
    x_pred_years: np.ndarray,
    ref_mean: np.ndarray,
    ref_lo: np.ndarray,
    ref_hi: np.ndarray,
    kef_results: list,
    filename: str,
) -> None:
    """
    Overlays the reference posterior-predictive band with every swept KEF
    worst-case radius's predictive mean on one axes, so the progressive
    shift as r grows is visible at a glance. `kef_results` is a list of
    dicts with keys r, mean, lo, hi (as produced by the radius sweep loop);
    only its mean curve is drawn per radius (all their bands overlaid would
    be unreadable) -- colour darkens from light to dark blue with
    increasing radius. (Each radius's SNIS effective sample size is still
    printed to the console by the caller; it is not shown on this plot.)
    """
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width * 1.7, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    y_mean_offset = float(np.mean(y_uncentered))
    ax.scatter(x_years, y_uncentered, color="black", marker="x", s=14, zorder=5, label=r"$x$")

    ref_color = "#7c397d"
    ax.fill_between(x_pred_years, ref_lo + y_mean_offset, ref_hi + y_mean_offset, color=ref_color, alpha=0.15, zorder=1)
    ax.plot(
        x_pred_years, ref_mean + y_mean_offset, color=ref_color, linewidth=1.1, linestyle="--", zorder=4,
        label=r"$\tilde{x}_{\mathrm{ref}}$",
    )

    cmap = plt.get_cmap("Blues")
    results_sorted = sorted(kef_results, key=lambda d: d["r"])
    n = len(results_sorted)
    for i, res in enumerate(results_sorted):
        color = cmap(0.35 + 0.55 * (i + 1) / n)
        ax.plot(
            x_pred_years, res["mean"] + y_mean_offset, color=color, linewidth=0.9, zorder=3,
            label=rf"$\tilde x_K$ ($r={res['r']:g}$)",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.legend(
        frameon=False, fontsize=plt.rcParams["font.size"] * 0.7,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
    )

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_acf_radius_sweep(
    plot_cfg,
    output_dir: str,
    acf_ref: np.ndarray,
    kef_results: list,
    filename: str,
) -> None:
    """
    Combined ACF comparison across the whole radius sweep on one axes,
    mirroring plot_posterior_predictive_radius_sweep: `kef_results` is the
    same list of dicts (also carrying an "acf" key), coloured light to
    dark blue with increasing radius.
    """
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width * 1.7, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    lags = np.arange(len(acf_ref))
    ax.plot(lags, acf_ref, color="#7c397d", linewidth=2.0,
            linestyle="--", zorder=4, label=r"$\tilde{x}_{\mathrm{ref}}$")

    cmap = plt.get_cmap("Blues")
    results_sorted = sorted(kef_results, key=lambda d: d["r"])
    n = len(results_sorted)
    for i, res in enumerate(results_sorted):
        color = cmap(0.35 + 0.55 * (i + 1) / n)
        ax.plot(
            lags, res["acf"], color=color, linewidth=1.6, zorder=3,
            label=rf"$\tilde x_K$ ($r={res['r']:g}$)",
        )

    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.legend(
        frameon=False, fontsize=plt.rcParams["font.size"] * 0.7,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
    )

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_sensitivity_vs_radius(
    node_records: list,
    r_j_used: float,
    plot_cfg,
    output_dir: str,
    filename: str,
    r_max: float = 20.0,
    n_points: int = 200,
) -> None:
    """
    S_{m,j}^FD(r) = r * omega_max_j is exactly linear in r -- the
    generalised eigenvalue omega_max_j does not depend on r at all (only
    the resulting worst-case coefficient vector lambda_star = sqrt(r) *
    eigenvector does) -- so sweeping the radius costs nothing beyond the
    omega_max_j values already computed: each component's curve here is
    just a straight line through the origin with slope omega_max_j. Also
    note that if the *same* r_j were applied to every component (as done
    throughout this script), each component's *share* of the total,
    S_j(r)/sum_k S_k(r) = omega_max_j / sum_k omega_max_k, would not change
    with r -- only the absolute magnitudes grow, which is what this plot
    shows.
    """
    _apply_plot_rc(plot_cfg)

    radii = np.linspace(0.0, r_max, n_points)
    palette = list(plot_cfg.plot.color_palette.colors)

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    node_by_name = {rec["name"]: rec for rec in node_records}
    for i, name in enumerate(COMPONENT_ORDER):
        rec = node_by_name[name]
        color = palette[i % len(palette)]
        ax.plot(radii, radii * rec["omega_max"], label=LATEX_NAMES[name], color=color, linewidth=1.5)

    ax.axvline(r_j_used, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xlabel(r"neighbourhood radius $r_j$")
    ax.set_ylabel(r"$\widehat{S}_{m,j}^{\mathrm{FD}}(r_j) = r_j \cdot \omega_{\max,j}$")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=plt.rcParams["font.size"] * 0.7, ncol=2)

    _save_fig(fig, output_dir, filename, plot_cfg)


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi_nonparam")
def run_ark_kilpisjarvi_nonparametric_sensitivity(cfg: DictConfig) -> None:
    """
    Per-component nonparametric FD sensitivity analysis for the Kilpisjarvi
    AR(K) regression model (posteriordb), for the parametric counterpart see
    tests/posteriordb/run_ark_kilpisjarvi.py.

    Kilpisjarvi's composite prior decomposes into independent scalar blocks
    (alpha, beta1..beta5 ~ N(0, 5^2); sigma ~ HalfCauchy(gamma)), just like
    the BNN's per-weight/bias decomposable prior in
    tests/bnn/run_bnn_uci_fisher_nonparam.py, so the same per-node
    nonparametric FD sensitivity machinery (src.optimization.
    bnn_node_sensitivity.compute_group_omega_max / compute_node_lambda_star)
    is reused here via KilpisjarviNonparametricLoader.

    Every component's prior and posterior samples are first transformed to
    a common standard-Gaussian reference footing via _to_z_space (the
    reference-prior probability integral transform), so the same shared
    basis-fitting recipe and neighbourhood radius r_j represent a genuinely
    comparable perturbation across all 7 components -- including the
    Half-Cauchy sigma, whose reference is otherwise a different (heavy-
    tailed) shape from the Gaussian alpha/beta1..5 block.
    """
    loader = instantiate(cfg.model, data_config=cfg.data)
    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg.optimize.nonparametric.basis_funcs_kwargs, resolve=True)

    uniform_fd_z_sup = compute_uniform_z_neighbourhood_parametric_sensitivity(loader)
    print("Parametric-style sup sensitivity under a SINGLE shared z-space neighbourhood "
          "(mu_z in [-0.4, 0.4], sigma_z in [0.8, 1.25] for every component, incl. sigma):")
    for name in COMPONENT_ORDER:
        print(f"  {name}: {uniform_fd_z_sup[name]:.4f}")
    print(
        "  (compare sigma's value above against the ~1.30 it got under its own independently-chosen "
        "Half-Cauchy gamma-in-[0.2,5.0] box in run_ark_kilpisjarvi_param_nonparam.py -- with a "
        "genuinely shared neighbourhood, sigma's own sup should no longer be handicapped by that "
        "box having been chosen separately/more conservatively than alpha/beta1..5's.)"
    )
    print(
        "  (NOTE: the values above are POSTERIOR-based -- they use each component's own real "
        "posterior draws, which is why alpha/beta1..5 differ even under the identical box/reference.)"
    )

    prior_fd_z_sup = compute_uniform_z_neighbourhood_prior_fd()
    print("\nPrior-based FD_z sup under the SAME shared z-space neighbourhood (data-independent -- "
          "identical for every component, since the z-space reference is always N(0,1) regardless "
          "of family, and this is evaluated under that reference, not under any posterior):")
    for name in COMPONENT_ORDER:
        print(f"  {name}: {prior_fd_z_sup[name]:.4f}")

    # Transform the uniform-z-neighbourhood parametric sup above into an
    # actual nonparametric KEF radius per component (r_j_by_component =
    # uniform_fd_z_sup), and realise the resulting sensitivity/percentages.
    center_samples_num_uniform = int(cfg.data.get("center_prior_samples_num", 5000))
    uniform_nonparam = compute_nonparametric_sensitivity_at_radii(
        loader, basis_cls, basis_kwargs, uniform_fd_z_sup, center_samples_num=center_samples_num_uniform,
    )
    print("\nNonparametric KEF sensitivity realised at the uniform-z-neighbourhood radii above "
          "(r_j * omega_max, own radius per component):")
    for name in COMPONENT_ORDER:
        print(
            f"  {name}: r_j={uniform_fd_z_sup[name]:.4f}, sensitivity={uniform_nonparam['sensitivity'][name]:.4f} "
            f"({uniform_nonparam['percentages'][name]:.1f}%)"
        )

    J = loader.total_nodes
    r_j = float(cfg.sensitivity.r_j)
    print(f"Total scalar nodes J = {J}. Shared neighbourhood size r_j = {r_j:g} per component.")

    center_samples_num = int(cfg.data.get("center_prior_samples_num", 5000))

    node_records = []
    prior_samples_z_by_group = {}
    start = time.perf_counter()
    for group_name in loader.param_groups:
        g = loader.groups[group_name]
        prior_dist = g["prior_dist"]

        prior_samples_z = _to_z_space(prior_dist, loader.sample_prior(group_name))

        # Fresh, independent prior draw used only to pick basis centres --
        # never the same samples used to estimate the FD constraint matrix
        # A_c (mirrors the toy/BNN models' centers_pool_samples pattern).
        center_prior_samples_z = _to_z_space(
            prior_dist, loader.sample_prior(group_name, n_samples=center_samples_num)
        )
        posterior_z = _to_z_space(prior_dist, g["posterior"])
        prior_samples_z_by_group[group_name] = prior_samples_z

        omega_max = compute_group_omega_max(
            posterior_samples=posterior_z,
            loc=0.0,
            scale=1.0,
            prior_samples=prior_samples_z,
            basis_cls=basis_cls,
            basis_kwargs=basis_kwargs,
            center_prior_samples=center_prior_samples_z,
        )
        sensitivity = r_j * omega_max

        for local_idx, node_name in enumerate(g["node_names"]):
            print(f"{node_name}: omega_max={omega_max[local_idx]:.4f}, sensitivity={sensitivity[local_idx]:.4f}")

            # Exact worst-case KEF coefficient vector, for the candidate-
            # density plots below (compute_group_omega_max only returns the
            # generalised eigenvalue, not the eigenvector).
            lam_star, omega_check, basis, diag = compute_node_lambda_star(
                posterior_samples_col=posterior_z[:, local_idx],
                loc=0.0,
                scale=1.0,
                prior_samples=prior_samples_z,
                basis_cls=basis_cls,
                basis_kwargs=basis_kwargs,
                radius_j=r_j,
                center_prior_samples=center_prior_samples_z,
            )
            node_records.append({
                "name": node_name,
                "group": group_name,
                "r_j": r_j,
                "lambda_star": lam_star,
                "basis": basis,
                "posterior_samples_z": posterior_z[:, local_idx],
                "omega_max": omega_check,
                "sensitivity": r_j * omega_check,
                "ac_cond": diag["ac_cond"],
            })
    elapsed = time.perf_counter() - start
    print(f"Per-node nonparametric FD sensitivity computation time: {elapsed:.3f}s")

    node_by_name = {rec["name"]: rec for rec in node_records}
    total_sensitivity = float(sum(rec["sensitivity"] for rec in node_records))
    print(f"Global FD sensitivity S^FD(Q_r) = {total_sensitivity:.4f} (r={r_j * J:g}, J={J}).")

    contributions = {rec["name"]: rec["sensitivity"] for rec in node_records}
    percentages = {k: v / total_sensitivity * 100.0 for k, v in contributions.items()}
    print("Per-component nonparametric FD sensitivity (PIT-standardized, comparable across components):")
    for name in COMPONENT_ORDER:
        print(f"  {name}: sensitivity={contributions[name]:.4f} ({percentages[name]:.1f}%)")

    results_dir = os.path.join(get_original_cwd(), cfg.flags.results.output_dir)
    os.makedirs(results_dir, exist_ok=True)
    if cfg.playground.get("save_json", True):
        save_to_serializable_json(
            {
                "radius": r_j * J,
                "J": J,
                "r_j": r_j,
                "total_sensitivity": total_sensitivity,
                "components": {
                    rec["name"]: {
                        "group": rec["group"],
                        "r_j": rec["r_j"],
                        "omega_max": rec["omega_max"],
                        "sensitivity": rec["sensitivity"],
                        "percentage": percentages[rec["name"]],
                        "ac_cond": rec["ac_cond"],
                    }
                    for rec in node_records
                },
            },
            os.path.join(results_dir, "kilpisjarvi_node_sensitivity_nonparam.json"),
        )
        print(f"Saved results to {results_dir}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    plot_cfg = load_plot_config(plot_config_path)
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    prefix = cfg.playground.get("output_prefix", "kilpisjarvi_nonparam")

    plot_component_sensitivity_bar(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        names=COMPONENT_ORDER,
        contributions=percentages,
        display_names=LATEX_NAMES,
        order=COMPONENT_ORDER,
        group_tail_betas=False,
        filename=f"{prefix}_component_sensitivity.pdf",
        prefix=prefix,
        ylabel=r"$\widehat{S}_m^{\mathrm{FD}}(\Gamma_j)$ \% (PIT-standardized)",
    )

    # Every component's reference prior is exactly N(0, 1) in z-space, so the
    # same standard-Gaussian candidate-density plot (no sigma-specific
    # special case needed any more) applies uniformly to all 7 nodes.
    all_nodes = [
        {
            "label": LATEX_NAMES.get(rec["name"], rec["name"]),
            "loc": 0.0,
            "scale": 1.0,
            "df": None,
            "lambda_star": rec["lambda_star"],
            "basis": rec["basis"],
            "posterior_samples": rec["posterior_samples_z"],
            "omega_max": rec["omega_max"],
            "sensitivity": rec["sensitivity"],
        }
        for rec in sorted(node_records, key=lambda r: COMPONENT_ORDER.index(r["name"]))
    ]
    plot_bnn_node_candidate_priors(
        nodes=all_nodes,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"{prefix}_candidate_priors.pdf",
        n_cols=4,
        xlabel=r"$z$",
    )

    # ------------------------------------------------------------------
    # Radius sweep: how does each component's sensitivity grow with r_j?
    # ------------------------------------------------------------------
    plot_sensitivity_vs_radius(
        node_records=node_records,
        r_j_used=r_j,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"{prefix}_sensitivity_vs_radius.pdf",
    )

    # ------------------------------------------------------------------
    # Posterior predictive: reference vs. worst-case KEF candidate prior,
    # swept over several radii r_pred (independent of the r_j used for the
    # per-component sensitivity comparison above -- that comparison's
    # percentages are radius-invariant, see plot_sensitivity_vs_radius, so
    # r_pred can be pushed larger here purely to make the KEF-vs-reference
    # predictive gap visible without touching the sensitivity numbers).
    #
    # run_ark_kilpisjarvi.py's parametric version loads *separate* MCMC
    # draws obtained by literally re-running the sampler under the
    # (parametric, closed-form) corner prior. There is no such re-run
    # available for the nonparametric KEF candidate, so instead the
    # reference posterior draws are reweighted via self-normalised
    # importance sampling (SNIS): since posterior(theta) propto
    # prior(theta) * likelihood(theta) and the likelihood is unchanged,
    # importance weights depend only on the KEF/reference prior density
    # ratio, computed in closed form from the already-fit
    # (basis, lambda_star) per component, rescaled to r_pred -- no new
    # sampling or re-optimisation needed. The reweighted draws are then
    # resampled (SIR) into an equally-weighted sample so the existing
    # unweighted posterior-predictive helpers can be reused unmodified.
    #
    # SNIS degrades as r_pred grows: a larger perturbation concentrates the
    # importance weights onto fewer effective reference draws (falling
    # ESS), so the resulting "KEF posterior" resample gets noisier/less
    # reliable the larger r_pred is pushed -- the printed ESS below is the
    # diagnostic to watch.
    # ------------------------------------------------------------------
    K = sum(1 for name in loader.model.prior_init.names if name.startswith("beta"))
    posterior_full = loader.model.posterior_samples_init  # (M, 2+K): alpha, beta[1..K], sigma
    y_full = y_centered
    y_mean_offset = float(np.mean(y))
    x_pred_years = x_years[K:]
    mode, seed = "one_step", 27
    max_lag = 5

    print_predictive_variance_decomposition(y_full, posterior_full, K, name="reference posterior")
    y_rep_ref = _ar_posterior_predictive(y_full=y_full, samples=posterior_full, K=K, mode=mode, seed=seed)
    ref_mean, ref_lo, ref_hi = _summarise_bands(y_rep_ref)
    acf_ref = _mean_acf(y_rep_ref, max_lag)

    plot_posterior_predictive_with_data(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        x_years=x_years,
        y_uncentered=y,
        x_pred_years=x_pred_years,
        pred_mean=ref_mean + y_mean_offset,
        pred_lo=ref_lo + y_mean_offset,
        pred_hi=ref_hi + y_mean_offset,
        pred_label=r"$\tilde{x}_{\mathrm{ref}} \pm 95\%$ CI",
        filename=f"{prefix}_posterior_predictive_ref.pdf",
        pred_color="#7c397d",
    )

    pred_radii = [float(r) for r in cfg.sensitivity.get("posterior_predictive_radii", [r_j])]
    rng = np.random.default_rng(int(cfg.data.get("seed", 0)))
    kef_results = []
    print("Posterior-predictive radius sweep (reference vs. KEF worst-case, SNIS-reweighted):")
    for r_pred in pred_radii:
        kef_samples, ess, ess_frac = _kef_reweight_posterior(
            node_records, loader, posterior_full, prior_samples_z_by_group, r_pred, rng
        )
        print_predictive_variance_decomposition(y_full, kef_samples, K, name=f"KEF worst-case posterior (r={r_pred:g})")

        y_rep_kef = _ar_posterior_predictive(y_full=y_full, samples=kef_samples, K=K, mode=mode, seed=seed)
        kef_mean, kef_lo, kef_hi = _summarise_bands(y_rep_kef)
        mean_abs_shift = float(np.mean(np.abs(kef_mean - ref_mean)))
        print(
            f"  r={r_pred:g}: ESS={ess:.1f}/{len(posterior_full)} ({100.0 * ess_frac:.1f}%), "
            f"mean |kef_mean - ref_mean| = {mean_abs_shift:.4f} degC"
            + ("  [LOW ESS -- reweighting unreliable at this radius]" if ess_frac < 0.05 else "")
        )

        acf_kef = _mean_acf(y_rep_kef, max_lag)
        kef_results.append({
            "r": r_pred, "mean": kef_mean, "lo": kef_lo, "hi": kef_hi, "acf": acf_kef, "ess_frac": ess_frac,
        })

    plot_posterior_predictive_radius_sweep(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        x_years=x_years,
        y_uncentered=y,
        x_pred_years=x_pred_years,
        ref_mean=ref_mean,
        ref_lo=ref_lo,
        ref_hi=ref_hi,
        kef_results=kef_results,
        filename=f"{prefix}_posterior_predictive_radius_sweep.pdf",
    )
    plot_acf_radius_sweep(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        acf_ref=acf_ref,
        kef_results=kef_results,
        filename=f"{prefix}_acf_radius_sweep.pdf",
    )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    run_ark_kilpisjarvi_nonparametric_sensitivity()
