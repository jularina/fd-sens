import os
import time
import numpy as np
from scipy.stats import norm as scipy_norm
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import json

from src.common.utils.files_operations import load_plot_config, save_to_serializable_json
from src.parametric.fisher_divergence import PosteriorFDParametric
from src.parametric.optimization import OptimizationCornerPointsCompositePrior
from paper.posteriordb.plots_parametric import (
    plot_acf_comparison,
    plot_complexity_bar,
    plot_component_sensitivity_bar,
    plot_posterior_predictive_with_data,
    plot_priors_z_scale_one_panel,
)
from src.common.distributions.gaussian import Gaussian
from src.common.distributions.cauchy import HalfCauchy
from src.common.distributions.composite import CompositeProduct

# ---------------------------------------------------------------------------
# Kilpisjarvi dataset
# ---------------------------------------------------------------------------
DATA = {
    "N": 62,
    "x": [
        3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961,
        3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971,
        3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981,
        3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991,
        3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001,
        4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011,
        4012, 4013,
    ],
    "y": [
        8.3, 10.9, 9.4, 8.1, 8.1, 7.7, 8.6, 9.1, 11.0, 10.1,
        7.6, 8.8, 8.3, 7.2, 9.3, 8.8, 7.6, 10.5, 11.0, 8.9,
        11.3, 10.0, 10.1, 6.4, 8.2, 8.4, 9.5, 9.9, 10.6, 7.6,
        7.7, 8.1, 8.4, 9.7, 9.5, 7.3, 10.3, 9.6, 10.3, 9.8,
        9.0, 9.1, 9.5, 8.7, 9.9, 10.5, 9.4, 9.0, 9.0, 9.7,
        11.4, 10.7, 10.1, 10.8, 10.4, 10.3, 8.8, 9.8, 8.8, 10.8,
        8.6, 11.1,
    ],
    "xpred": 2016,
    "pmualpha": 9.31290322580645,
    "psalpha": 100,
    "pmubeta": 0,
    "psbeta": 0.0333333333333333,
}

# x values in DATA are offset: 3952 corresponds to 1952 in the PosteriorDB encoding.
X_OFFSET = 2000
x_years = np.array(DATA["x"]) - X_OFFSET
y = np.array(DATA["y"])
y_centered = y - np.mean(y)


# Fallback z-space box (after the PIT every reference prior is N(0, 1)); cfg.playground overrides it.
Z_MU_RANGE = (-1.0, 1.0)
Z_SIGMA_RANGE = (0.5, 2.0)
PARAM_Z_STAN_DIR = "outputs/paper/results/kilpisjarvi/param/stan"
_PIT_EPS = 1e-8


def _to_z_space(prior_dist, x: np.ndarray) -> np.ndarray:
    """Map x to z = Phi^{-1}(F_ref(x)) so that z ~ N(0, 1) under the reference prior."""
    x = np.asarray(x, dtype=float)
    if isinstance(prior_dist, Gaussian):
        return (x - prior_dist.mu) / prior_dist.sigma
    if isinstance(prior_dist, HalfCauchy):
        u = (2.0 / np.pi) * np.arctan(np.maximum(x, 0.0) / prior_dist.gamma)
        u = np.clip(u, _PIT_EPS, 1.0 - _PIT_EPS)
        return scipy_norm.ppf(u)
    raise NotImplementedError(f"No reference-prior CDF implemented for {type(prior_dist).__name__}.")


def _fd_z_posterior_gaussian_in_z(
    mu_z_cand: float, sigma_z_cand: float, z_post_mean: float, z_post_meansq: float,
) -> float:
    """Exact posterior-based FD_z between N(0, 1) and a Gaussian-in-z candidate from z-moments."""
    a = 1.0 / sigma_z_cand ** 2 - 1.0
    b = -mu_z_cand / sigma_z_cand ** 2
    return float(a ** 2 * z_post_meansq + 2.0 * a * b * z_post_mean + b ** 2)


def _z_box_from_cfg(cfg) -> tuple[tuple[float, float], tuple[float, float]]:
    mu_max = float(cfg.playground.get("z_mu_max", Z_MU_RANGE[1]))
    sig_min = float(cfg.playground.get("z_sigma_min", Z_SIGMA_RANGE[0]))
    sig_max = float(cfg.playground.get("z_sigma_max", Z_SIGMA_RANGE[1]))
    return (-mu_max, mu_max), (sig_min, sig_max)


def _z_box_tag(mu_z_range, sigma_z_range) -> str:
    return f"_mu{mu_z_range[1]:g}_sig{sigma_z_range[0]:g}-{sigma_z_range[1]:g}"


def _run_stage(cfg, stage: str) -> bool:
    return cfg.playground.get("stage", "all") in ("all", stage)


def _export_param_z_stan_data(worst: dict, base_prior, y_full: np.ndarray, K: int, path: str) -> None:
    """Write Stan data for the parametric z-scale Kilpisjarvi model's worst-case priors."""
    ref = dict(zip(base_prior.names, base_prior.components))
    names = list(base_prior.names)
    y_full = np.asarray(y_full, dtype=float).reshape(-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_to_serializable_json(
        {
            "K": int(K),
            "T": int(len(y_full)),
            "y": y_full,
            "mu_ref": [float(ref[n].mu) for n in names[:-1]],
            "sigma_ref": [float(ref[n].sigma) for n in names[:-1]],
            "gamma_sigma": float(ref["sigma"].gamma),
            "pit_eps": _PIT_EPS,
            "mu_z": [float(worst[n][0]) for n in names],
            "sigma_z": [float(worst[n][1]) for n in names],
        },
        path,
    )
    print(f"Saved Stan data for parametric z-space worst-case prior to {path}")


def compute_z_scale_parametric_sensitivity(
    model, base_prior, mu_z_range=Z_MU_RANGE, sigma_z_range=Z_SIGMA_RANGE,
) -> dict:
    """Per component, return (mu_z, sigma_z, fd_z_sup) for the posterior-based FD_z sup over the z-box."""
    corners = [(mu, sig) for mu in mu_z_range for sig in sigma_z_range]
    worst = {}
    for idx, (name, ref) in enumerate(zip(base_prior.names, base_prior.components)):
        z_post = _to_z_space(ref, model.posterior_samples_init[:, idx])
        z_mean, z_meansq = float(np.mean(z_post)), float(np.mean(z_post ** 2))
        vals = [_fd_z_posterior_gaussian_in_z(mu, sig, z_mean, z_meansq) for mu, sig in corners]
        best = int(np.argmax(vals))
        worst[name] = (corners[best][0], corners[best][1], float(vals[best]))
    return worst


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi")
def main(cfg: DictConfig) -> None:
    if not _run_stage(cfg, "optimise"):
        return
    print("=== FD for PosteriorDB model (parametric, z-scale) ===")
    model = instantiate(cfg.model, data_config=cfg.data)
    base_prior = instantiate(cfg.data.base_prior)
    names = list(base_prior.names)
    K = sum(1 for n in names if n.startswith("beta"))

    mu_z_range, sigma_z_range = _z_box_from_cfg(cfg)
    print(f"Shared z-space box: mu_z in {mu_z_range}, sigma_z in {sigma_z_range}")
    worst = compute_z_scale_parametric_sensitivity(model, base_prior, mu_z_range, sigma_z_range)
    total = sum(v[2] for v in worst.values())
    percentages = {k: worst[k][2] / total * 100.0 for k in names}
    print("Per-component worst-case Gaussian-in-z corner (posterior-based FD_z sup):")
    for k in names:
        mu_z, sig_z, fd = worst[k]
        print(f"  {k}: mu_z={mu_z:+.2f}, sigma_z={sig_z:.2f}, FD_z={fd:.4f} ({percentages[k]:.1f}%)")

    _export_param_z_stan_data(
        worst, base_prior, y_centered, K,
        os.path.join(get_original_cwd(), PARAM_Z_STAN_DIR,
                     f"stan_data_param_z{_z_box_tag(mu_z_range, sigma_z_range)}.json"),
    )

    prefix = cfg.playground.get("output_prefix", "kilpisjarvi_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    latex_names = {
        "alpha":  r"$\alpha$",
        "beta1":  r"$\beta_1$",
        "beta2":  r"$\beta_2$",
        "beta3":  r"$\beta_3$",
        "beta4":  r"$\beta_4$",
        "beta5":  r"$\beta_5$",
        "sigma":  r"$\sigma$",
    }
    plot_component_sensitivity_bar(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        names=names,
        contributions=percentages,
        display_names=latex_names,
        order=names,
        group_tail_betas=False,
        tail_beta_keys=["beta3", "beta4", "beta5"],
        filename=f"{prefix}_component_sensitivity.pdf",
        width_scale=0.8,
    )

    plot_priors_z_scale_one_panel(
        worst_corners_z={k: worst[k][:2] for k in names},
        mu_z_range=mu_z_range,
        sigma_z_range=sigma_z_range,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        sample_n=50,
        seed=27,
        filename=f"{prefix}_three_panel_priors.pdf",
    )


class _ZScalePriorFDModel:
    """Prior-only z-space FD problem with N(0, 1) references used for the optimiser runtime comparison."""

    def __init__(self, posterior_z: np.ndarray, names: list[str]):
        self.posterior_samples_init = posterior_z
        self.prior_init = CompositeProduct(distributions={n: Gaussian(mu=0.0, sigma=1.0) for n in names})
        self.prior_candidate = CompositeProduct(distributions={n: Gaussian(mu=0.0, sigma=1.0) for n in names})
        self.loss_lr_init = 1.0
        self.loss_lr = 1.0

    def loss_score(self, x: np.ndarray, multiply_by_lr: bool = True) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=float))


def _z_box_eta_components(names: list[str], mu_z_range, sigma_z_range) -> list[dict]:
    """Return the natural-parameter box spanned by the z-box, identical for every component."""
    mu_max = max(abs(mu_z_range[0]), abs(mu_z_range[1]))
    sig_min, sig_max = sigma_z_range
    eta_1 = [-mu_max / sig_min ** 2, mu_max / sig_min ** 2]
    eta_2 = [-0.5 / sig_min ** 2, -0.5 / sig_max ** 2]
    return [{"name": n, "eta_range": {"eta_1": eta_1, "eta_2": eta_2}} for n in names]


def _timing_dir(cfg) -> str:
    return os.path.join(get_original_cwd(), cfg.playground.get("timing_dir", "data/kilpisjarvi"))


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi")
def time_optimisers(cfg: DictConfig) -> None:
    """Time the corner-enumeration, per-component and dual-annealing optimisers on the z-scale problem."""
    if cfg.playground.get("stage", "all") != "timing":
        return
    n_runs = int(cfg.playground.get("n_timing_runs", 500))
    n_runs_bb = int(cfg.playground.get("n_timing_runs_bb", 100))
    timing_dir = _timing_dir(cfg)
    os.makedirs(timing_dir, exist_ok=True)

    model = instantiate(cfg.model, data_config=cfg.data)
    base_prior = instantiate(cfg.data.base_prior)
    names = list(base_prior.names)
    posterior_z = np.column_stack([
        _to_z_space(ref, model.posterior_samples_init[:, idx])
        for idx, ref in enumerate(base_prior.components)
    ])
    mu_z_range, sigma_z_range = _z_box_from_cfg(cfg)

    fisher_estimator = PosteriorFDParametric(model=_ZScalePriorFDModel(posterior_z, names))
    optimizer = OptimizationCornerPointsCompositePrior(
        fisher_estimator,
        {"eta_components": _z_box_eta_components(names, mu_z_range, sigma_z_range)},
        loss_config={},
    )

    print(f"Starting optimisation of all parameters at once ({n_runs} runs).")
    times_full = np.empty(n_runs)
    for i in range(n_runs):
        start = time.perf_counter()
        optimizer.evaluate_all_prior_corners()
        optimizer.minimize_prior_full_qp()
        times_full[i] = time.perf_counter() - start
    np.savez(os.path.join(timing_dir, "timing_qf_full.npz"), times=times_full)

    print(f"Starting per component optimisation ({n_runs} runs).")
    times_decomp = np.empty(n_runs)
    for i in range(n_runs):
        start = time.perf_counter()
        optimizer.evaluate_all_prior_corners_per_component(component_names=names)
        optimizer.minimize_prior_per_component_qp(names)
        times_decomp[i] = time.perf_counter() - start
    np.savez(os.path.join(timing_dir, "timing_qf_decomp.npz"), times=times_decomp)

    print(f"Starting black-box optimisation ({n_runs_bb} runs).")
    times_bb = np.empty(n_runs_bb)
    for i in range(n_runs_bb):
        start = time.perf_counter()
        optimizer.black_box_optimize_prior_box_global(
            method="dual_annealing",
            seed=i,
            maxiter=150,
            n_restarts=5,
        )
        times_bb[i] = time.perf_counter() - start
        print(f"  Run {i + 1}/{n_runs_bb}: {times_bb[i]:.3f} sec.")
    np.savez(os.path.join(timing_dir, "timing_bb.npz"), times=times_bb)
    print(f"Mean times: full {times_full.mean():.4f}s, per-component {times_decomp.mean():.4f}s, "
          f"black-box {times_bb.mean():.3f}s. Saved to {timing_dir}")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi")
def plot_timing(cfg: DictConfig) -> None:
    if cfg.playground.get("stage", "all") not in ("timing", "plot_timing"):
        return
    timing_dir = _timing_dir(cfg)
    plot_cfg = load_plot_config(os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml"))
    plot_complexity_bar(
        plot_cfg=plot_cfg,
        output_dir=os.path.join(get_original_cwd(), cfg.flags.plots.output_dir),
        filename="kilpisjarvi_computational_cost.pdf",
        use_log10=True,
        qf_full_time_sec=np.load(os.path.join(timing_dir, "timing_qf_full.npz"))["times"],
        qf_decomp_time_sec=np.load(os.path.join(timing_dir, "timing_qf_decomp.npz"))["times"],
        black_box_time_sec=np.load(os.path.join(timing_dir, "timing_bb.npz"))["times"],
    )


# ---------------------------------------------------------------------------
# Posterior predictive helpers
# ---------------------------------------------------------------------------

def _load_corner_draws_json(path: str) -> list[dict]:
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, list) or len(obj) == 0:
        raise ValueError(f"Expected a non-empty list of chains in {path}")
    return obj


def _norm_warmup_chains(warmup, n_chains: int) -> list[int]:
    if isinstance(warmup, (list, tuple)):
        if len(warmup) != n_chains:
            raise ValueError(f"warmup length {len(warmup)} != n_chains {n_chains}")
        return [max(0, int(w)) for w in warmup]
    return [max(0, int(warmup))] * n_chains


def _stack_corner_chains(chains: list[dict], K: int, warmup=0, max_draws: int | None = None) -> np.ndarray:
    """Returns samples as (N, 2+K) = [alpha, beta_1..beta_K, sigma]."""
    n_chains = len(chains)
    warmups = _norm_warmup_chains(warmup, n_chains)
    per_chain = []
    for ci, d in enumerate(chains):
        alpha = np.asarray(d["alpha"], dtype=float).reshape(-1)
        sigma = np.asarray(d["sigma"], dtype=float).reshape(-1)
        beta_cols = []
        if "beta" in d:
            beta = np.asarray(d["beta"], dtype=float)
            if beta.ndim == 1:
                beta = beta.reshape(alpha.shape[0], K)
            beta_cols = [beta[:, j] for j in range(K)]
        else:
            for s in range(1, K + 1):
                key = f"beta[{s}]"
                if key not in d:
                    raise ValueError(f"Chain {ci} missing '{key}'. Keys: {list(d.keys())}")
                beta_cols.append(np.asarray(d[key], dtype=float).reshape(-1))
        w = min(warmups[ci], alpha.shape[0])
        sl = slice(w, None)
        alpha_t = alpha[sl]
        sigma_t = sigma[sl]
        beta_t = [b[sl] for b in beta_cols]
        if max_draws is not None:
            alpha_t = alpha_t[:max_draws]
            sigma_t = sigma_t[:max_draws]
            beta_t = [b[:max_draws] for b in beta_t]
        per_chain.append(np.column_stack([alpha_t] + beta_t + [sigma_t]))
    return np.concatenate(per_chain, axis=0)


def _ar_posterior_predictive(
    y_full: np.ndarray,
    samples: np.ndarray,
    K: int,
    mode: str = "one_step",
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_full = np.asarray(y_full, dtype=float).reshape(-1)
    T = y_full.shape[0]
    m = samples.shape[0]
    alpha = samples[:, 0]
    betas = samples[:, 1:1 + K]
    sigma = samples[:, 1 + K]

    out_len = T - K
    y_rep = np.zeros((m, out_len), dtype=float)

    if mode == "one_step":
        for t in range(K, T):
            lags = np.array([y_full[t - s] for s in range(1, K + 1)], dtype=float)
            mean_t = alpha + (betas @ lags)
            y_rep[:, t - K] = rng.normal(loc=mean_t, scale=sigma)
        return y_rep

    if mode == "rollout":
        y0 = y_full[:K].copy()
        for k in range(m):
            path = np.zeros(T, dtype=float)
            path[:K] = y0
            a, b, s = alpha[k], betas[k], sigma[k]
            for t in range(K, T):
                lags = np.array([path[t - s_] for s_ in range(1, K + 1)], dtype=float)
                path[t] = rng.normal(loc=a + float(np.dot(b, lags)), scale=s)
            y_rep[k, :] = path[K:]
        return y_rep

    raise ValueError(f"Unknown mode='{mode}'.")


def _summarise_bands(y_rep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.mean(y_rep, axis=0), np.quantile(y_rep, 0.025, axis=0), np.quantile(y_rep, 0.975, axis=0)


def _acf_1d(y: np.ndarray, max_lag: int) -> np.ndarray:
    y = y - np.mean(y)
    denom = np.dot(y, y)
    if denom == 0:
        return np.zeros(max_lag + 1)
    n = len(y)
    return np.array([1.0] + [np.dot(y[:n - k], y[k:]) / denom for k in range(1, max_lag + 1)])


def _mean_acf(y_rep: np.ndarray, max_lag: int) -> np.ndarray:
    return np.mean([_acf_1d(y_rep[i], max_lag) for i in range(y_rep.shape[0])], axis=0)


def _summarise_samples(samples, K, name):
    alpha = samples[:, 0]
    betas = samples[:, 1:1 + K]
    sigma = samples[:, 1 + K]
    print(f"\n{name}")
    print("alpha:", np.mean(alpha), np.std(alpha))
    for j in range(K):
        print(f"beta{j + 1}:", np.mean(betas[:, j]), np.std(betas[:, j]))
    print("sigma:", np.mean(sigma), np.std(sigma))


def predictive_variance_law_of_total(
    y_full: np.ndarray,
    samples: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate posterior predictive mean, std, variance components and 95% bounds via total variance."""
    y_full = np.asarray(y_full, dtype=float).reshape(-1)
    T = len(y_full)
    alpha_s = samples[:, 0]        # (S,)
    betas_s = samples[:, 1:1 + K]  # (S, K)
    sigma_s = samples[:, 1 + K]    # (S,)

    # Build lag matrix: row t gives [y_{t-1}, ..., y_{t-K}] for t = K..T-1
    lags = np.stack([y_full[K - s: T - s] for s in range(1, K + 1)], axis=1)  # (T-K, K)

    mu_all = alpha_s[:, None] + betas_s @ lags.T  # (S, T-K)

    aleatoric = float(np.mean(sigma_s ** 2))       # scalar
    epistemic = np.var(mu_all, axis=0, ddof=0)     # (T-K,)
    pred_mean = np.mean(mu_all, axis=0)            # (T-K,)
    pred_std = np.sqrt(aleatoric + epistemic)       # (T-K,)
    lo = pred_mean - 1.96 * pred_std
    hi = pred_mean + 1.96 * pred_std

    return pred_mean, pred_std, aleatoric, epistemic, lo, hi


def print_predictive_variance_decomposition(
    y_full: np.ndarray,
    samples: np.ndarray,
    K: int,
    name: str = "posterior",
) -> None:
    pred_mean, pred_std, aleatoric, epistemic, lo, hi = predictive_variance_law_of_total(
        y_full, samples, K
    )
    total = aleatoric + np.mean(epistemic)
    print(f"\n=== Predictive variance decomposition ({name}) ===")
    print(f"  Aleatoric (E[sigma^2]):          {aleatoric:.6f}")
    print(f"  Epistemic (Var[mu_t]), mean over t: {np.mean(epistemic):.6f}")
    print(f"  Total (mean over t):             {total:.6f}")
    print(f"  Aleatoric fraction:              {aleatoric / total:.3f}")
    print(f"  Epistemic fraction:              {np.mean(epistemic) / total:.3f}")
    print(f"  Predictive std (mean over t):    {np.mean(pred_std):.6f}")
    print(f"  95% CI half-width (mean over t): {np.mean(1.96 * pred_std):.6f}")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi")
def plot_posterior_predictive(cfg: DictConfig) -> None:
    if not _run_stage(cfg, "plot_predictive"):
        return
    prefix = cfg.playground.get("output_prefix", "kilpisjarvi_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    model = instantiate(cfg.model, data_config=cfg.data)
    K = sum(1 for name in model.prior_init.names if name.startswith("beta"))

    y_obs = model.observations.reshape(-1)   # y_centered[K:]
    # y_centered is the full series (length T); y_obs is the target part (length T-K).
    y_full = y_centered

    # Stan posterior under the z-scale worst-case prior; the R sampler saves post-warmup draws only.
    mu_z_range, sigma_z_range = _z_box_from_cfg(cfg)
    corner_draws_path = os.path.join(
        get_original_cwd(), PARAM_Z_STAN_DIR, f"draws_param_z{_z_box_tag(mu_z_range, sigma_z_range)}.json"
    )
    print(f"Corner posterior draws: {corner_draws_path}")
    chains = _load_corner_draws_json(corner_draws_path)
    corner_samples = _stack_corner_chains(chains=chains, K=K, warmup=0, max_draws=None)

    ref_samples = model.posterior_samples_init
    _summarise_samples(ref_samples, K, "reference posterior")
    _summarise_samples(corner_samples, K, "corner posterior")

    print_predictive_variance_decomposition(y_full, ref_samples, K, name="reference posterior")
    print_predictive_variance_decomposition(y_full, corner_samples, K, name="corner posterior")

    mode = "one_step"
    seed = 27

    y_rep_corner = _ar_posterior_predictive(y_full=y_full, samples=corner_samples, K=K, mode=mode, seed=seed)
    corner_mean, corner_lo, corner_hi = _summarise_bands(y_rep_corner)

    y_rep_ref = _ar_posterior_predictive(y_full=y_full, samples=ref_samples, K=K, mode=mode, seed=seed)
    ref_mean, ref_lo, ref_hi = _summarise_bands(y_rep_ref)

    y_mean_offset = float(np.mean(y))
    x_pred_years = x_years[K:]

    all_values = (
        list(y)
        + list(ref_mean + y_mean_offset)
        + list(ref_lo + y_mean_offset)
        + list(ref_hi + y_mean_offset)
        + list(corner_mean + y_mean_offset)
        + list(corner_lo + y_mean_offset)
        + list(corner_hi + y_mean_offset)
    )
    global_ylim = (min(all_values), max(all_values))

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
        filename="kilpisjarvi-posterior-predictive-ref.pdf",
        pred_color="#7c397d",
        ylim=global_ylim,
    )
    plot_posterior_predictive_with_data(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        x_years=x_years,
        y_uncentered=y,
        x_pred_years=x_pred_years,
        pred_mean=corner_mean + y_mean_offset,
        pred_lo=corner_lo + y_mean_offset,
        pred_hi=corner_hi + y_mean_offset,
        pred_label=r"$\tilde{x} \pm 95\%$ CI",
        filename="kilpisjarvi-posterior-predictive-corner.pdf",
        pred_color="#5b9bd5",
        ylim=global_ylim,
        show_ylabel=False,
    )
    # animate_posterior_predictive_with_data(
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     x_years=x_years,
    #     y_uncentered=y,
    #     x_pred_years=x_pred_years,
    #     pred_mean=corner_mean + y_mean_offset,
    #     pred_lo=corner_lo + y_mean_offset,
    #     pred_hi=corner_hi + y_mean_offset,
    #     pred_label=r"$\tilde{x} \pm 95\%$ CI",
    #     filename="kilpisjarvi-posterior-predictive-corner.mp4",
    #     pred_color="#5b9bd5",
    #     ylim=global_ylim,
    #     show_ylabel=False,
    # )

    max_lag = 5
    acf_ref = _mean_acf(y_rep_ref, max_lag)
    acf_corner = _mean_acf(y_rep_corner, max_lag)

    plot_acf_comparison(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        acf_ref=acf_ref,
        acf_corner=acf_corner,
        ref_color="#7c397d",
        corner_color="#5b9bd5",
        ref_label=r"$\tilde{x}_{\mathrm{ref}}$",
        corner_label=r"$\tilde{x}$",
        filename="kilpisjarvi-acf-comparison.pdf",
    )


if __name__ == "__main__":
    main()
    plot_posterior_predictive()
    time_optimisers()
    plot_timing()
