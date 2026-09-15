import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.stats import norm as scipy_norm
from scipy.optimize import dual_annealing
from scipy.special import logsumexp

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.discrepancies.posterior_fisher import PosteriorFDParametric as PosteriorFDBase
from src.optimization.corner_points_fisher import OptimizationCornerPointsCompositePrior
from src.optimization.bnn_node_sensitivity import compute_node_lambda_star
from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.utils.files_operations import load_plot_config
from src.distributions.gaussian import Gaussian
from src.distributions.cauchy import HalfCauchy
from src.plots.paper.posterior_db_paper_funcs import _apply_plot_rc, _save_fig

from tests.posteriordb.run_ark_kilpisjarvi import (
    x_years,
    y,
    y_centered,
    _gaussian_from_eta,
    _param_reweight_posterior,
    _ar_posterior_predictive,
    _summarise_bands,
    print_predictive_variance_decomposition,
)
from tests.posteriordb.run_ark_kilpisjarvi_nonparam import (
    _to_z_space,
    _log_kef_density_ratio,
    COMPONENT_ORDER,
    LATEX_NAMES,
    compute_uniform_z_neighbourhood_parametric_sensitivity,
    compute_uniform_z_neighbourhood_prior_fd,
    compute_nonparametric_sensitivity_at_radii,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Box for sigma's black-box corner search (see _find_sigma_corner_black_box):
# candidate Half-Cauchy scale gamma_cand can shrink to 1/5 or grow to 5x the
# reference gamma_ref=1. Symmetric in log-scale; no existing config declares
# a box for a non-exponential-family candidate, so this is a chosen default.
SIGMA_GAMMA_BOUNDS = (0.2, 5.0)

# ---------------------------------------------------------------------------
# Prior-based Fisher Divergence, in the reference prior's own z-space, between
# a reference component and its parametric worst-case corner candidate.
#
# For Gaussian alpha/beta1..5: exact closed form. Under the affine PIT
# z=(x-ref.mu)/ref.sigma, cand pushes forward to another Gaussian
# N(mu_z, sigma_z^2), so FD_z = E_{z~N(0,1)}[(score_ref_z(z) -
# score_cand_z(z))^2] reduces to the standard Gaussian-Gaussian Fisher
# divergence formula (verified numerically against the general Monte-Carlo
# estimator below in an earlier iteration of this analysis).
#
# For sigma: reference and corner candidate are both Half-Cauchy (see
# _find_sigma_corner_black_box), so no closed form is used -- a general
# PIT-based Monte-Carlo estimate is well-behaved here (no boundary blow-up,
# unlike the Half-Cauchy-vs-Inverse-Gamma pairing this analysis started
# from: score(x) -> 0 as x -> 0 for any Half-Cauchy gamma).
# ---------------------------------------------------------------------------


def _fd_z_gaussian_gaussian(ref: Gaussian, cand: Gaussian) -> float:
    mu_z = (cand.mu - ref.mu) / ref.sigma
    sigma_z = cand.sigma / ref.sigma
    a = 1.0 / sigma_z ** 2 - 1.0
    b = -mu_z / sigma_z ** 2
    return float(a ** 2 + b ** 2)


def _fd_z_at_samples(ref_dist, cand_dist, x_samples: np.ndarray) -> float:
    """
    General change-of-variables estimate of FD_z = mean_i[ (score_ref(x_i) -
    score_cand(x_i))^2 / T'(x_i)^2 ], T'(x) = f_ref(x)/phi(z), z =
    _to_z_space(ref_dist, x) -- exactly E[(score_ref_z(z)-score_cand_z(z))^2]
    under whatever measure x_samples were drawn from, pushed through the
    reference's own PIT. If x_samples ~ ref_dist itself, this estimates the
    *prior*-based FD_z (the E_{z~N(0,1)} quantity _fd_z_gaussian_gaussian
    gives in closed form for the Gaussian case); if x_samples are actual
    posterior draws, this instead estimates the *posterior*-based FD_z (see
    _fd_z_posterior_gaussian_gaussian / _posterior_sup_z_halfcauchy below).
    """
    x = np.asarray(x_samples, dtype=float).reshape(-1)
    z = _to_z_space(ref_dist, x)
    score_ref = np.asarray(ref_dist.grad_log_pdf(x)).reshape(-1)
    score_cand = np.asarray(cand_dist.grad_log_pdf(x)).reshape(-1)
    f_ref = np.asarray(ref_dist.pdf(x)).reshape(-1)
    phi_z = scipy_norm.pdf(z)
    t_prime = f_ref / phi_z
    diff_z = (score_ref - score_cand) / t_prime
    return float(np.mean(diff_z ** 2))


def _fd_z_montecarlo(ref_dist, cand_dist, n_samples: int = 300_000, seed: int = 0) -> float:
    """Prior-based FD_z: _fd_z_at_samples evaluated at fresh draws from
    ref_dist itself (x ~ ref_dist), estimating E_{z~N(0,1)}[...]."""
    x = ref_dist.sample(n_samples)
    return _fd_z_at_samples(ref_dist, cand_dist, x)


def _fd_z_posterior_gaussian_gaussian(
    ref: Gaussian, cand: Gaussian, z_post_mean: float, z_post_meansq: float,
) -> float:
    """
    Exact posterior-based FD_z = E_{z~posterior_z}[(score_ref_z(z) -
    score_cand_z(z))^2] for Gaussian ref/cand, using the posterior's own
    empirical z-moments (mean, second moment) instead of assuming
    z~N(0,1) as the prior-based _fd_z_gaussian_gaussian does. diff_z(z) is
    still exactly affine in z (diff_z(z) = a*z + b, a = 1/sigma_z^2 - 1,
    b = -mu_z/sigma_z^2, mu_z=(cand.mu-ref.mu)/ref.sigma,
    sigma_z=cand.sigma/ref.sigma), so E[diff_z(z)^2] = a^2*E[z^2] +
    2ab*E[z] + b^2 -- exact given the posterior's own first two z-moments,
    no sampling noise beyond estimating those two moments.
    """
    mu_z = (cand.mu - ref.mu) / ref.sigma
    sigma_z = cand.sigma / ref.sigma
    a = 1.0 / sigma_z ** 2 - 1.0
    b = -mu_z / sigma_z ** 2
    return float(a ** 2 * z_post_meansq + 2.0 * a * b * z_post_mean + b ** 2)


def _posterior_sup_z_gaussian(optimizer, comp_idx: int, ref: Gaussian, posterior_col: np.ndarray) -> float:
    """
    Per-component posterior-based sup sensitivity, in z-space: maximises
    _fd_z_posterior_gaussian_gaussian over the component's own eta box
    (reusing optimizer._component_eta_corners(comp_idx), the same 4 eta
    corners evaluate_all_prior_corners_per_component uses). diff_z(z)^2 is a
    convex quadratic in the candidate's *z-space* natural parameters (an
    affine, sigma_ref^2-scaled reparametrisation of the original eta, since
    ref.mu=0 here), so its posterior expectation is a convex quadratic form
    too -- its supremum over a box is always attained at a vertex, exactly
    like the original-scale per-component sup search.
    """
    z_post = _to_z_space(ref, posterior_col)
    z_mean = float(np.mean(z_post))
    z_meansq = float(np.mean(z_post ** 2))
    corners = optimizer._component_eta_corners(comp_idx)
    vals = [
        _fd_z_posterior_gaussian_gaussian(ref, Gaussian(**_gaussian_from_eta(*eta_j)), z_mean, z_meansq)
        for eta_j in corners
    ]
    return float(max(vals))


def _posterior_sup_z_halfcauchy(
    sigma_ref: HalfCauchy,
    sigma_posterior: np.ndarray,
    gamma_bounds=SIGMA_GAMMA_BOUNDS,
    seed: int = 0,
    maxiter: int = 300,
) -> float:
    """
    Per-component posterior-based sup sensitivity for sigma, in z-space:
    black-box (dual_annealing) search over gamma_cand maximising
    _fd_z_at_samples(sigma_ref, HalfCauchy(gamma_cand), sigma_posterior) --
    the posterior-based z-space analogue of _find_sigma_corner_black_box
    (which works in the original sigma scale).
    """
    def neg_fd(x):
        cand = HalfCauchy(gamma=float(x[0]))
        return -_fd_z_at_samples(sigma_ref, cand, sigma_posterior)

    res = dual_annealing(neg_fd, bounds=[gamma_bounds], seed=seed, maxiter=maxiter)
    return -float(res.fun)


def _fd_posterior_halfcauchy(ref: HalfCauchy, cand_gamma: float, posterior_col: np.ndarray) -> float:
    """Original-scale (not z-space) posterior-based FD, used only to find
    the actual worst-case sigma corner used elsewhere in this script
    (candidate_prior_dists / param_dists / fd_z-as-radius) -- see
    _find_sigma_corner_black_box. Kept separate from the z-space
    _posterior_sup_z_halfcauchy above, which is used only for the
    sensitivity-percentage comparison."""
    cand = HalfCauchy(gamma=float(cand_gamma))
    diff = np.asarray(ref.grad_log_pdf(posterior_col)) - np.asarray(cand.grad_log_pdf(posterior_col))
    return float(np.mean(diff ** 2))


def _find_sigma_corner_black_box(
    sigma_ref: HalfCauchy,
    sigma_posterior: np.ndarray,
    gamma_bounds=SIGMA_GAMMA_BOUNDS,
    seed: int = 0,
    maxiter: int = 300,
) -> tuple[HalfCauchy, float]:
    """
    Black-box worst-case search for sigma's Half-Cauchy corner: maximises
    the posterior-based Fisher Divergence FD(sigma_ref, HalfCauchy(gamma))
    over gamma in gamma_bounds via dual_annealing -- mirrors
    OptimizationCornerPointsCompositePrior.black_box_optimize_prior_box_global's
    pattern (src/optimization/corner_points_fisher.py), adapted to operate
    directly on Half-Cauchy's own scale parameter instead of exponential-
    family eta, since Half-Cauchy has no natural-parameter representation
    (src/distributions/cauchy.py raises NotImplementedError there).

    Returns (corner distribution, achieved posterior FD at that corner). The
    achieved infimum over gamma_bounds is exactly 0 (gamma_ref=1 lies inside
    the bounds, giving FD=0 trivially), so the achieved sup alone is already
    the full sup-minus-inf sensitivity range -- no separate minimisation
    needed, mirroring alpha/beta1..5's own per-component sup (see
    run_parametric_optimisation).
    """
    def neg_fd(x):
        return -_fd_posterior_halfcauchy(sigma_ref, x[0], sigma_posterior)

    res = dual_annealing(neg_fd, bounds=[gamma_bounds], seed=seed, maxiter=maxiter)
    gamma_star = float(res.x[0])
    posterior_fd = -float(res.fun)
    print(f"  sigma black-box corner: gamma_star={gamma_star:.4f} (bounds={gamma_bounds}), "
          f"posterior FD={posterior_fd:.4f}")
    return HalfCauchy(gamma=gamma_star), posterior_fd


def _load_configs():
    with initialize(version_base="1.1", config_path="../../configs/paper/real"):
        cfg_param = compose(config_name="ark_kilpisjarvi")
        cfg_nonparam = compose(config_name="ark_kilpisjarvi_nonparam")
    return cfg_param, cfg_nonparam


def run_parametric_optimisation(cfg_param):
    """
    Same corner-point search as run_ark_kilpisjarvi.py's main(): find the
    FD-optimal worst-case composite-prior corner within the declared eta box
    (cfg_param.fd.optimize.prior.Composite). alpha/beta1..5 are taken from
    the optimum eta_star; sigma has no exponential-family representation
    (Half-Cauchy), so its own worst-case corner is found separately via a
    black-box search directly over its scale parameter (see
    _find_sigma_corner_black_box) instead of the eta-box QP machinery.

    Returns two different per-component quantities, deliberately kept
    separate:
      - fd_z: each component's OWN prior-based Fisher Divergence in z-space
        of its slice of the single JOINT worst-case corner eta_star. This is
        a pure prior-geometry quantity (data-independent for the Gaussian
        components, given the corner), used as each component's own
        nonparametric-comparable radius fed into the KEF machinery.
      - posterior_sup: each component's OWN posterior-based sup FD, IN
        Z-SPACE, found by maximising the z-space posterior FD *independently
        per component* within its own box (alpha/beta1..5 via
        _posterior_sup_z_gaussian, reusing the same 4 eta-box corners
        evaluate_all_prior_corners_per_component would use; sigma via
        _posterior_sup_z_halfcauchy), all evaluated at that component's real
        posterior draws pushed through _to_z_space. Unlike fd_z (a *prior*-
        based quantity, evaluated under the reference measure), this
        genuinely differs across alpha/beta1..5 (their posteriors differ,
        even though their boxes and reference priors are identical) and is
        the quantity used for a genuine, standardised "per-parameter global
        sensitivity" comparison against the (already z-space) nonparametric
        side -- the corresponding infimum is 0 in both cases (for sigma,
        gamma_ref=1 is inside the search bounds, so FD=0 is trivially
        achievable there; same logic applies in z-space for alpha/beta1..5),
        so sup-minus-inf reduces to sup alone.

    NOTE: run_ark_kilpisjarvi.py calls evaluate_all_prior_corners_per_component
    with component_names=["beta1",...,"beta5","alpha","sigma"], which does
    NOT match the box config's own declared order (alpha,beta1..5,sigma) --
    since that function zips component_names[j] positionally against
    self.eta_components_cfg[j], this mislabels the first six results by one
    position (e.g. "beta1" there is actually alpha's own-box result). Indexed
    by the config's own positional order below to avoid inheriting that.
    """
    print("Parametric worst-case corner search neighbourhoods (original parametrisation, as declared "
          "in cfg_param.fd.optimize.prior.Composite.components):")
    for comp_cfg in cfg_param.fd.optimize.prior.Composite.components:
        ranges = comp_cfg["parameters_box_range"]["ranges"]
        ranges_str = ", ".join(f"{k}∈[{v[0]}, {v[1]}]" for k, v in ranges.items())
        print(f"  {comp_cfg['name']} ({comp_cfg['family']}): {ranges_str}")
    print(f"  sigma (Half-Cauchy, used instead of the declared Inverse-Gamma box above -- see module "
          f"docstring): gamma∈[{SIGMA_GAMMA_BOUNDS[0]}, {SIGMA_GAMMA_BOUNDS[1]}]")

    model = instantiate(cfg_param.model, data_config=cfg_param.data)
    fisher_estimator = PosteriorFDBase(model=model)
    optimizer = OptimizationCornerPointsCompositePrior(
        fisher_estimator,
        cfg_param.fd.optimize.prior.Composite,
        cfg_param.fd.optimize.loss.GaussianARLogLikelihood,
    )
    _, eta_star = optimizer.evaluate_all_prior_corners()

    gaussian_names = ["alpha"] + [f"beta{k + 1}" for k in range(5)]

    alpha_ref = Gaussian(mu=0.0, sigma=5.0)
    beta_ref = Gaussian(mu=0.0, sigma=5.0)
    sigma_ref = HalfCauchy(gamma=1.0)

    gaussian_posterior_cols = {"alpha": 0, **{f"beta{k + 1}": k + 1 for k in range(5)}}
    posterior_sup = {
        name: _posterior_sup_z_gaussian(
            optimizer, comp_idx, alpha_ref if name == "alpha" else beta_ref,
            model.posterior_samples_init[:, gaussian_posterior_cols[name]],
        )
        for comp_idx, name in enumerate(gaussian_names)
    }

    alpha_cand = Gaussian(**_gaussian_from_eta(*eta_star[0:2]))
    beta_cands = {
        f"beta{k + 1}": Gaussian(**_gaussian_from_eta(*eta_star[2 * (k + 1):2 * (k + 1) + 2]))
        for k in range(5)
    }

    sigma_posterior = model.posterior_samples_init[:, -1]
    sigma_cand, _sigma_posterior_fd_original_scale = _find_sigma_corner_black_box(sigma_ref, sigma_posterior)
    posterior_sup["sigma"] = _posterior_sup_z_halfcauchy(sigma_ref, sigma_posterior)

    fd_z = {"alpha": _fd_z_gaussian_gaussian(alpha_ref, alpha_cand)}
    for name, dist in beta_cands.items():
        fd_z[name] = _fd_z_gaussian_gaussian(beta_ref, dist)
    fd_z["sigma"] = _fd_z_montecarlo(sigma_ref, sigma_cand)

    print("Parametric prior FD, moved to z-scale (= radius r_j fed to the nonparametric KEF "
          "constraint for that same component):")
    for name in ["alpha"] + [f"beta{k + 1}" for k in range(5)] + ["sigma"]:
        print(f"  {name}: fd_z = {fd_z[name]:.4f}  ->  r_j = {fd_z[name]:.4f}")

    total_radius = float(sum(fd_z.values()))
    print(f"Total FD_z over 7 components: {total_radius:.4f}")

    print("Per-component posterior-based sup sensitivity, z-space (own box, inf=0):")
    for name in ["alpha"] + [f"beta{k + 1}" for k in range(5)] + ["sigma"]:
        print(f"  {name}: {posterior_sup[name]:.4f}")

    param_dists = {"alpha": (alpha_ref, alpha_cand)}
    for name, dist in beta_cands.items():
        param_dists[name] = (beta_ref, dist)
    param_dists["sigma"] = (sigma_ref, sigma_cand)

    return model, alpha_cand, beta_cands, sigma_cand, fd_z, total_radius, param_dists, posterior_sup


def _kef_reweight_posterior_own_radii(node_records, loader, posterior_full, prior_samples_z_by_group, rng):
    """
    Like run_ark_kilpisjarvi_nonparam.py's _kef_reweight_posterior, but each
    node's lambda_star was already fit at that node's OWN target radius
    (rec["r_j"] just records it for reference) -- so no further
    sqrt(r_pred/r_j) rescale is applied; every node contributes its own
    already-fit worst-case KEF tilt directly.
    """
    node_by_name = {rec["name"]: rec for rec in node_records}
    log_w = np.zeros(posterior_full.shape[0])
    for idx, name in enumerate(COMPONENT_ORDER):
        rec = node_by_name[name]
        prior_dist = loader.groups[rec["group"]]["prior_dist"]
        z_col = _to_z_space(prior_dist, posterior_full[:, idx])
        log_w += _log_kef_density_ratio(
            z_col, rec["basis"], rec["lambda_star"], prior_samples_z_by_group[rec["group"]]
        )
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    ess_frac = ess / len(w)

    resample_idx = rng.choice(len(w), size=len(w), replace=True, p=w)
    return posterior_full[resample_idx], ess, ess_frac


def run_nonparametric_with_own_radii(cfg_nonparam, r_j_by_component: dict, posterior_full: np.ndarray, seed: int):
    """
    Runs the KEF per-node sensitivity fit with each node's OWN radius
    (r_j_by_component[node_name] -- the parametric worst-case corner's own
    per-component FD_z from run_parametric_optimisation), rather than one
    shared radius applied identically to all nodes, and returns a posterior
    resample under that worst-case KEF candidate.
    """
    loader = instantiate(cfg_nonparam.model, data_config=cfg_nonparam.data)
    basis_cls = BASIS_FUNCTIONS_REGISTRY[cfg_nonparam.optimize.nonparametric.basis_funcs_type]
    basis_kwargs = OmegaConf.to_container(cfg_nonparam.optimize.nonparametric.basis_funcs_kwargs, resolve=True)
    center_samples_num = int(cfg_nonparam.data.get("center_prior_samples_num", 5000))

    node_records = []
    prior_samples_z_by_group = {}
    for group_name in loader.param_groups:
        g = loader.groups[group_name]
        prior_dist = g["prior_dist"]

        prior_samples_z = _to_z_space(prior_dist, loader.sample_prior(group_name))
        center_prior_samples_z = _to_z_space(
            prior_dist, loader.sample_prior(group_name, n_samples=center_samples_num)
        )
        posterior_z = _to_z_space(prior_dist, g["posterior"])
        prior_samples_z_by_group[group_name] = prior_samples_z

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
            })

    rng = np.random.default_rng(seed)
    kef_samples, ess, ess_frac = _kef_reweight_posterior_own_radii(
        node_records, loader, posterior_full, prior_samples_z_by_group, rng,
    )
    print(
        f"Nonparametric KEF worst-case at each component's own radius: "
        f"ESS={ess:.1f}/{len(posterior_full)} ({100.0 * ess_frac:.1f}%)"
        + ("  [LOW ESS -- reweighting unreliable]" if ess_frac < 0.05 else "")
    )
    return kef_samples, node_records


def _draw_component_sensitivity_stack(ax, percentages: dict, title: str, ylabel: str | None) -> None:
    """
    Single stacked-column sensitivity bar drawn onto a given axis, in the
    same visual style as src/plots/paper/posterior_db_paper_funcs.py's
    plot_component_sensitivity_bar (e.g. kilpisjarvi_nonparam_component_
    sensitivity.pdf): one column at x=0, each component a stacked segment,
    coloured by contribution rank (lowest "#4d7298", lightening towards
    white for higher-ranked/larger contributions), with an inline
    "{label} {pct:.1f}%" text for segments >= 4%.
    """
    names = COMPONENT_ORDER
    ranked = sorted(names, key=lambda k: percentages[k])
    n = len(ranked)
    low = np.array(to_rgb("#4d7298"))
    high = low + 0.75 * (np.array([1.0, 1.0, 1.0]) - low)  # same hue, lightened towards white
    color_map = {k: tuple(low + (i / max(n - 1, 1)) * (high - low)) for i, k in enumerate(ranked)}

    bottom = 0.0
    min_label_pct = 4.0
    for k in names:
        pct = percentages[k]
        ax.bar(0, pct, bottom=bottom, color=color_map[k], alpha=0.5, edgecolor="white", linewidth=0.5, width=0.9)
        if pct >= min_label_pct:
            label = LATEX_NAMES.get(k, k)
            ax.text(0, bottom + pct / 2, f"{label} {pct:.1f}%", ha="center", va="center",
                    fontsize="x-small", color="black")
        bottom += pct

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.tick_params(axis="x", length=0, labelcolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=plt.rcParams["font.size"] * 0.85)


def plot_component_sensitivity_bar_param_vs_nonparam(
    plot_cfg,
    output_dir: str,
    percentages_param: dict,
    percentages_nonparam: dict,
    filename: str,
    percentages_omega_max: dict | None = None,
) -> None:
    """
    Side-by-side stacked-column sensitivity panels, each in the same style
    as plot_component_sensitivity_bar (see _draw_component_sensitivity_stack):
      1. Parametric -- own-box posterior-based sup, z-space.
      2. Nonparametric (KEF) -- r_j * omega_max, i.e. the realised worst-case
         sensitivity at each component's own (parametric-implied) radius.
      3. (optional, only if percentages_omega_max is given) Nonparametric
         (KEF), per unit radius -- omega_max alone (panel 2 divided by each
         component's own r_j), the generalised eigenvalue itself: the
         *intrinsic* per-component sensitivity rate, with the (very unequal,
         corner-derived) radii divided out. Omit this panel (leave
         percentages_omega_max=None) when it would be identical to panel 2,
         e.g. under a single shared radius for every component -- there
         r_j cancels out of the normalised percentages exactly, so panel 3
         would just duplicate panel 2.
    """
    _apply_plot_rc(plot_cfg)
    # \FD is not a standard LaTeX command -- _apply_plot_rc's preamble only
    # loads amsmath/type1cm, so define it here (as \mathrm{FD}) rather than
    # substitute a different macro, since the ylabels below use \FD as given.
    plt.rcParams["text.latex.preamble"] += r"\newcommand{\FD}{\mathrm{FD}}"
    os.makedirs(output_dir, exist_ok=True)

    n_panels = 3 if percentages_omega_max is not None else 2
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(plot_cfg.plot.figure.size.width * (n_panels * 0.55), plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    ylabel_param = r"$\widehat{S}_m^{\FD}(\Gamma_j)$ \%"
    ylabel_nonparam = r"$\widehat{S}_m^{\FD}(\widehat{\mathcal{Q}}_{r_j}^{j, K,l})$ \%"
    _draw_component_sensitivity_stack(axes[0], percentages_param, r"\texttt{FDsens}", ylabel=ylabel_param)
    _draw_component_sensitivity_stack(axes[1], percentages_nonparam, r"\texttt{FDsens+}", ylabel=ylabel_nonparam)
    if percentages_omega_max is not None:
        _draw_component_sensitivity_stack(
            axes[2], percentages_omega_max, r"Nonparametric ($\omega_{\max}$, per unit radius)", ylabel=None,
        )

    fig.tight_layout(w_pad=0.6, pad=0.3)
    _save_fig(fig, output_dir, filename, plot_cfg)
    plt.close(fig)
    print(f"Saved plot to {os.path.join(output_dir, filename)}")


def _inverse_to_z_space(ref_dist, z: np.ndarray) -> np.ndarray:
    """Inverse of run_ark_kilpisjarvi_nonparam.py's _to_z_space: x such that
    _to_z_space(ref_dist, x) == z, i.e. x = F_ref^{-1}(Phi(z))."""
    z = np.asarray(z, dtype=float)
    if isinstance(ref_dist, Gaussian):
        return z * ref_dist.sigma + ref_dist.mu
    if isinstance(ref_dist, HalfCauchy):
        u = np.clip(scipy_norm.cdf(z), 1e-12, 1.0 - 1e-12)
        return ref_dist.gamma * np.tan(u * np.pi / 2.0)
    raise NotImplementedError(f"No inverse PIT implemented for {type(ref_dist).__name__}.")


def _param_candidate_density_z(ref_dist, cand_dist, z_grid: np.ndarray) -> np.ndarray:
    """
    Exact density (on z_grid) of the parametric candidate pushed forward
    through the reference's own PIT z=_to_z_space(ref_dist, x). For
    Gaussian ref/cand this is again exactly Gaussian in z (affine map), used
    in closed form; otherwise (Half-Cauchy ref/cand for sigma) evaluated via
    the general change-of-variables identity q_z(z) = f_cand(x) * phi(z) /
    f_ref(x), x = _inverse_to_z_space(ref_dist, z) -- same identity
    _fd_z_montecarlo is built on, just evaluated pointwise on a grid instead
    of averaged over samples.
    """
    if isinstance(ref_dist, Gaussian) and isinstance(cand_dist, Gaussian):
        mu_z = (cand_dist.mu - ref_dist.mu) / ref_dist.sigma
        sigma_z = cand_dist.sigma / ref_dist.sigma
        return scipy_norm.pdf(z_grid, loc=mu_z, scale=sigma_z)
    x = _inverse_to_z_space(ref_dist, z_grid)
    f_ref = np.asarray(ref_dist.pdf(x)).reshape(-1)
    f_cand = np.asarray(cand_dist.pdf(x)).reshape(-1)
    phi_z = scipy_norm.pdf(z_grid)
    return f_cand * phi_z / f_ref


def _kef_candidate_density_z(basis, lambda_star: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """Exact normalised KEF candidate density on z_grid, mirroring
    src/plots/paper/bnn_paper_funcs.py's _plot_bnn_node_candidate_prior:
    log q_K(z) = phi(z).lambda_star + log N(z;0,1), normalised via a
    log-sum-exp Riemann-sum estimate of the log partition function."""
    f = basis.evaluate(z_grid.reshape(-1, 1))[:, 0, :] @ np.asarray(lambda_star)
    log_g = scipy_norm.logpdf(z_grid)
    log_cand = f + log_g
    dz = float(z_grid[1] - z_grid[0])
    log_z_const = logsumexp(log_cand) + np.log(dz)
    return np.exp(log_cand - log_z_const)


def plot_worst_case_priors_param_vs_nonparam(
    plot_cfg,
    output_dir: str,
    param_dists: dict,
    nonparam_records: dict,
    filename: str,
    n_cols: int = 4,
    z_range=(-6.0, 6.0),
    resolution: int = 500,
) -> None:
    """
    One small-multiples figure, one panel per component, each showing the
    standard-Gaussian reference (black dashed, exact by construction of the
    z-space PIT), the parametric worst-case corner (blue) and the
    nonparametric KEF worst-case candidate (orange) -- all on the same
    z-space footing, so the two methods' worst-case densities are directly
    comparable per component.
    """
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)

    n = len(COMPONENT_ORDER)
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))

    fig_w = plot_cfg.plot.figure.size.width
    fig_h = plot_cfg.plot.figure.size.height
    fig_dpi = plot_cfg.plot.figure.dpi
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * fig_w, n_rows * fig_h), dpi=fig_dpi, squeeze=False)

    z_grid = np.linspace(z_range[0], z_range[1], resolution)
    ref_density = scipy_norm.pdf(z_grid)

    for i, name in enumerate(COMPONENT_ORDER):
        ax = axes[i // n_cols][i % n_cols]
        ref_dist, cand_dist = param_dists[name]
        param_density = _param_candidate_density_z(ref_dist, cand_dist, z_grid)

        rec = nonparam_records[name]
        kef_density = _kef_candidate_density_z(rec["basis"], rec["lambda_star"], z_grid)

        ax.plot(z_grid, ref_density, linestyle="--", linewidth=1.3, color="black", label=r"$\Pi_{\mathrm{ref}}$")
        ax.plot(z_grid, param_density, linewidth=1.3, color="#5b9bd5", label="parametric")
        ax.plot(z_grid, kef_density, linewidth=1.3, color="#e08214", label="nonparametric (KEF)")

        ax.set_title(LATEX_NAMES.get(name, name))
        ax.set_xlabel("$z$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(frameon=False, fontsize=plt.rcParams["font.size"] * 0.7)

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    _save_fig(fig, output_dir, filename, plot_cfg)
    plt.close(fig)
    print(f"Saved plot to {os.path.join(output_dir, filename)}")


def plot_posterior_predictive_three_way(
    plot_cfg,
    output_dir: str,
    x_years: np.ndarray,
    y_uncentered: np.ndarray,
    x_pred_years: np.ndarray,
    ref_mean, ref_lo, ref_hi,
    param_mean, param_lo, param_hi,
    kef_mean, kef_lo, kef_hi,
    filename: str,
) -> None:
    """One figure overlaying reference, parametric-corner and nonparametric
    KEF-worst-case posterior predictives (means + 95% bands) against the
    observed data."""
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width * 1.4, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    ax.scatter(x_years, y_uncentered, color="black", marker="x", s=14, zorder=5, label=r"$x$")

    ref_color, param_color, kef_color = "#7c397d", "#5b9bd5", "#e08214"

    ax.fill_between(x_pred_years, ref_lo, ref_hi, color=ref_color, alpha=0.15, zorder=1)
    ax.plot(
        x_pred_years, ref_mean, color=ref_color, linewidth=1.2, linestyle="--", zorder=4,
        label=r"$\tilde{x}_{\mathrm{ref}}$",
    )

    ax.fill_between(x_pred_years, param_lo, param_hi, color=param_color, alpha=0.20, zorder=2)
    ax.plot(
        x_pred_years, param_mean, color=param_color, linewidth=1.3, zorder=4,
        label=r"$\tilde{x}_{\mathrm{param}}$",
    )

    ax.fill_between(x_pred_years, kef_lo, kef_hi, color=kef_color, alpha=0.20, zorder=3)
    ax.plot(
        x_pred_years, kef_mean, color=kef_color, linewidth=1.3, zorder=4,
        label=r"$\tilde{x}_{\mathrm{KEF}}$",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.legend(
        frameon=False, fontsize=plt.rcParams["font.size"] * 0.8,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
    )

    _save_fig(fig, output_dir, filename, plot_cfg)
    print(f"Saved plot to {os.path.join(output_dir, filename)}")


def main() -> None:
    cfg_param, cfg_nonparam = _load_configs()

    print("=== Parametric worst-case corner (per-component radii) ===")
    (model, alpha_cand, beta_cands, sigma_cand, fd_z, total_radius,
     param_dists, posterior_sup) = run_parametric_optimisation(cfg_param)

    K = sum(1 for name in model.prior_init.names if name.startswith("beta"))
    posterior_full = model.posterior_samples_init  # (N, 2+K): alpha, beta[1..K], sigma
    seed = int(cfg_param.data.get("seed", 0))

    print("\n=== Parametric worst-case posterior (SNIS-reweighted) ===")
    base_prior = instantiate(cfg_param.data.base_prior)
    base_prior_dists = dict(zip(base_prior.names, base_prior.components))
    candidate_prior_dists = {"alpha": alpha_cand, **beta_cands, "sigma": sigma_cand}
    rng = np.random.default_rng(seed)
    param_samples, ess, ess_frac = _param_reweight_posterior(
        base_prior_dists, candidate_prior_dists, posterior_full, K, rng,
    )
    print(
        f"SNIS reweighting to parametric corner posterior: ESS={ess:.1f}/{len(posterior_full)} "
        f"({100.0 * ess_frac:.1f}%)"
        + ("  [LOW ESS -- reweighting unreliable]" if ess_frac < 0.05 else "")
    )

    print("\n=== Nonparametric KEF worst-case at each component's own parametric-implied radius ===")
    kef_samples, node_records = run_nonparametric_with_own_radii(cfg_nonparam, fd_z, posterior_full, seed)
    nonparam_records_by_name = {rec["name"]: rec for rec in node_records}

    total_posterior_sup = sum(posterior_sup.values())
    percentages_param = {name: posterior_sup[name] / total_posterior_sup * 100.0 for name in COMPONENT_ORDER}
    nonparam_sensitivity = {
        name: rec["r_j"] * rec["omega_max"] for name, rec in nonparam_records_by_name.items()
    }
    total_nonparam_sensitivity = sum(nonparam_sensitivity.values())
    percentages_nonparam = {
        name: v / total_nonparam_sensitivity * 100.0 for name, v in nonparam_sensitivity.items()
    }

    omega_max_by_name = {name: rec["omega_max"] for name, rec in nonparam_records_by_name.items()}
    total_omega_max = sum(omega_max_by_name.values())
    percentages_omega_max = {
        name: v / total_omega_max * 100.0 for name, v in omega_max_by_name.items()
    }

    print("\nPer-component share of total FD sensitivity (%) -- parametric: own-box posterior-based "
          "sup (inf=0); nonparametric: own-radius KEF sup (r_j * omega_max); "
          "omega_max: per-unit-radius share (r_j divided out):")
    for name in COMPONENT_ORDER:
        print(
            f"  {name}: parametric={percentages_param[name]:.1f}%, "
            f"nonparametric={percentages_nonparam[name]:.1f}%, "
            f"omega_max={percentages_omega_max[name]:.1f}% (omega_max={omega_max_by_name[name]:.6g})"
        )

    y_full = y_centered
    y_mean_offset = float(np.mean(y))
    x_pred_years = x_years[K:]
    mode = "one_step"

    print_predictive_variance_decomposition(y_full, posterior_full, K, name="reference posterior")
    print_predictive_variance_decomposition(y_full, param_samples, K, name="parametric worst-case posterior")
    print_predictive_variance_decomposition(y_full, kef_samples, K, name="nonparametric KEF worst-case posterior")

    y_rep_ref = _ar_posterior_predictive(y_full=y_full, samples=posterior_full, K=K, mode=mode, seed=seed)
    ref_mean, ref_lo, ref_hi = _summarise_bands(y_rep_ref)

    y_rep_param = _ar_posterior_predictive(y_full=y_full, samples=param_samples, K=K, mode=mode, seed=seed)
    param_mean, param_lo, param_hi = _summarise_bands(y_rep_param)

    y_rep_kef = _ar_posterior_predictive(y_full=y_full, samples=kef_samples, K=K, mode=mode, seed=seed)
    kef_mean, kef_lo, kef_hi = _summarise_bands(y_rep_kef)

    plot_config_path = os.path.join(REPO_ROOT, "configs/plots/overleaf_plots_settings.yaml")
    plot_cfg = load_plot_config(plot_config_path)
    output_dir = os.path.join(REPO_ROOT, "outputs/paper/plots/fisher/kilpisjarvi/param_vs_nonparam")

    plot_posterior_predictive_three_way(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        x_years=x_years,
        y_uncentered=y,
        x_pred_years=x_pred_years,
        ref_mean=ref_mean + y_mean_offset, ref_lo=ref_lo + y_mean_offset, ref_hi=ref_hi + y_mean_offset,
        param_mean=param_mean + y_mean_offset, param_lo=param_lo + y_mean_offset, param_hi=param_hi + y_mean_offset,
        kef_mean=kef_mean + y_mean_offset, kef_lo=kef_lo + y_mean_offset, kef_hi=kef_hi + y_mean_offset,
        filename="kilpisjarvi_param_vs_nonparam_posterior_predictive.pdf",
    )

    plot_component_sensitivity_bar_param_vs_nonparam(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        percentages_param=percentages_param,
        percentages_nonparam=percentages_nonparam,
        percentages_omega_max=percentages_omega_max,
        filename="kilpisjarvi_param_vs_nonparam_sensitivity_percentages.pdf",
    )

    # "Both in z-scale" variant: instead of each family's own independently-
    # chosen original-scale box (mu/sigma in [-2,2]/[0.25,1] for alpha/beta,
    # gamma in [0.2,5.0] Half-Cauchy for sigma), the parametric side now
    # searches the SAME shared Gaussian-in-z neighbourhood the nonparametric
    # side already uses (see compute_uniform_z_neighbourhood_parametric_
    # sensitivity / compute_nonparametric_sensitivity_at_radii in
    # run_ark_kilpisjarvi_nonparam.py) -- a genuinely fair comparison.
    loader_uniform = instantiate(cfg_nonparam.model, data_config=cfg_nonparam.data)
    basis_cls_uniform = BASIS_FUNCTIONS_REGISTRY[cfg_nonparam.optimize.nonparametric.basis_funcs_type]
    basis_kwargs_uniform = OmegaConf.to_container(cfg_nonparam.optimize.nonparametric.basis_funcs_kwargs, resolve=True)

    # NOTE: despite the function name, this is the POSTERIOR-based sup (it
    # uses each component's own real posterior draws) -- see
    # compute_uniform_z_neighbourhood_parametric_sensitivity's docstring.
    # That's why it differs across alpha/beta1..5 even under the identical
    # shared box: different components have different actual posteriors.
    uniform_fd_z_sup = compute_uniform_z_neighbourhood_parametric_sensitivity(loader_uniform)

    # The genuinely PRIOR-based FD_z sup under the same shared box (data-
    # independent -- identical for every component, since the z-space
    # reference is always N(0,1) regardless of family). Printed here only
    # for comparison; the radius below still uses the posterior-based sup
    # above, per the max-of-(posterior-based)-sups already in place.
    prior_fd_z_sup = compute_uniform_z_neighbourhood_prior_fd()
    print("\nPrior-based FD_z sup under the shared z-space box (data-independent, identical for "
          "every component -- shown for comparison against the posterior-based sup above):")
    for name in COMPONENT_ORDER:
        print(f"  {name}: {prior_fd_z_sup[name]:.4f}")

    # Instead of feeding each component its OWN posterior-based sup as its
    # own radius (as compute_nonparametric_sensitivity_at_radii would do by
    # default if given uniform_fd_z_sup directly), use a SINGLE shared
    # radius for every component -- the max over the per-component
    # posterior-based sups above -- so the nonparametric side is evaluated
    # at one common worst-case-implied radius rather than 7 different ones.
    r_j_shared_max = max(uniform_fd_z_sup.values())
    print(f"\nShared nonparametric radius (= max over per-component posterior-based FD_z sup above): "
          f"{r_j_shared_max:.4f}")
    r_j_by_component_max = {name: r_j_shared_max for name in COMPONENT_ORDER}

    uniform_nonparam = compute_nonparametric_sensitivity_at_radii(
        loader_uniform, basis_cls_uniform, basis_kwargs_uniform, r_j_by_component_max,
        center_samples_num=int(cfg_nonparam.data.get("center_prior_samples_num", 5000)),
    )

    total_uniform_param = sum(uniform_fd_z_sup.values())
    percentages_param_z = {name: uniform_fd_z_sup[name] / total_uniform_param * 100.0 for name in COMPONENT_ORDER}
    percentages_nonparam_z = uniform_nonparam["percentages"]

    uniform_omega_max = {rec["name"]: rec["omega_max"] for rec in uniform_nonparam["node_records"]}
    total_uniform_omega_max = sum(uniform_omega_max.values())
    percentages_omega_max_z = {
        name: v / total_uniform_omega_max * 100.0 for name, v in uniform_omega_max.items()
    }

    print("\nPer-component share (%) under the SAME shared z-space neighbourhood for both methods:")
    for name in COMPONENT_ORDER:
        print(
            f"  {name}: parametric={percentages_param_z[name]:.1f}%, "
            f"nonparametric={percentages_nonparam_z[name]:.1f}%, "
            f"omega_max={percentages_omega_max_z[name]:.1f}%"
        )

    plot_component_sensitivity_bar_param_vs_nonparam(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        percentages_param=percentages_param_z,
        percentages_nonparam=percentages_nonparam_z,
        filename="kilpisjarvi_param_vs_nonparam_sensitivity_percentages_both_in_z_scale.pdf",
    )

    plot_worst_case_priors_param_vs_nonparam(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        param_dists=param_dists,
        nonparam_records=nonparam_records_by_name,
        filename="kilpisjarvi_param_vs_nonparam_worst_case_priors.pdf",
    )


if __name__ == "__main__":
    main()
