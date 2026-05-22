import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.files_operations import load_plot_config
from src.plots.paper.posterior_db_paper_funcs import _apply_plot_rc, _save_fig
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import json

from src.discrepancies.posterior_fisher import PosteriorFDParametric as PosteriorFDBase
from src.plots.paper.posterior_db_paper_funcs import *
from src.optimization.corner_points_fisher import (
    OptimizationCornerPointsCompositePrior
)
from src.plots.paper.sbi_paper_funcs import *

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

# x values in DATA are offset; actual years start at 1952
# (3952 corresponds to 1952 in the PosteriorDB encoding)
X_OFFSET = 2000
x_years = np.array(DATA["x"]) - X_OFFSET
y = np.array(DATA["y"])
y_centered = y - np.mean(y)


def plot_time_series(output_dir: str, plot_cfg, prefix: str = "kilpisjarvi") -> None:
    _apply_plot_rc(plot_cfg)

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    color = plot_cfg.plot.color_palette.colors[0]
    ax.plot(x_years, y, color=color, linewidth=1.0, zorder=2)
    ax.scatter(x_years, y, color=color, s=14, zorder=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")

    _save_fig(fig, output_dir, f"{prefix}_time_series.pdf", plot_cfg)
    print(f"Saved plot to {os.path.join(output_dir, prefix + '_time_series.pdf')}")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ark_kilpisjarvi")
def main(cfg: DictConfig) -> None:
    print("=== FD for PosteriorDB model ===")
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher for prior: {fisher_estimator.estimate_fisher_prior_only():.4f}")
    print(f"Initial Fisher for lr: {fisher_estimator.estimate_fisher_lr_only():.4f}")

    print("Starting optimisation of all parameters at once.")
    optimizer = OptimizationCornerPointsCompositePrior(
        fisher_estimator,
        cfg.fd.optimize.prior.Composite,
        cfg.fd.optimize.loss.GaussianARLogLikelihood,
    )
    names = ["beta1", "beta2", "beta3", "beta4", "beta5", "alpha", "sigma"]
    qf_corners, eta_star = optimizer.evaluate_all_prior_corners()
    eta_inf, val_inf = optimizer.minimize_prior_full_qp()

    print("Starting per component optimisation.")
    sup_res, eta_sup_blocks = optimizer.evaluate_all_prior_corners_per_component(component_names=names)
    eta_inf_blocks, values_inf = optimizer.minimize_prior_per_component_qp(names)
    print("Per-component argmax corners:")
    for k in names:
        print(k, eta_sup_blocks[k], sup_res[k][0][1])

    print("Per-component infimum:")
    for n in names:
        print(n, eta_inf_blocks[n], values_inf[n])

    contributions = {k: sup_res[k][0][1] - values_inf[k] for k in names}
    total = sum(contributions[k] for k in names)
    percentages = {k: contributions[k] / total * 100.0 for k in names}

    # print(f"Starting black-box optimisation.")
    # bb = optimizer.black_box_optimize_prior_box_global(
    #     method="dual_annealing",
    #     seed=27,
    #     maxiter=150,
    #     n_restarts=5,
    # )
    # print(bb)

    prefix = cfg.playground.get("output_prefix", "kilpisjarvi_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    def _gaussian_from_eta(e1, e2):
        sigma = float(np.sqrt(-1.0 / (2.0 * e2)))
        return {"mu": float(e1 * sigma ** 2), "sigma": sigma}

    def _inv_gamma_from_eta(e1, e2):
        return {"alpha": float(-e1 - 1.0), "beta": float(-e2)}

    alpha_ms = {"family": "Gaussian",     "params": _gaussian_from_eta(*eta_star[0:2])}
    betas_ms = {
        f"beta{k+1}": {"family": "Gaussian", "params": _gaussian_from_eta(*eta_star[2*(k+1):2*(k+1)+2])}
        for k in range(5)
    }
    sigma_ms = {"family": "InverseGamma", "params": _inv_gamma_from_eta(*eta_star[12:14])}
    sigma_inf_prior = {"family": "InverseGamma", "params": _inv_gamma_from_eta(*eta_inf[12:14])}

    alpha_ref = {"family": "Gaussian",   "params": {"mu": 0.0, "sigma": 5.0}}
    betas_ref = {"family": "Gaussian",   "params": {"mu": 0.0, "sigma": 5.0}}
    sigma_ref = {"family": "HalfCauchy", "params": {"gamma": 1.0}}

    alpha_box_ranges = {"mu": (-2.0, 2.0), "sigma": (0.25, 1.0)}
    betas_box_ranges = {f"beta{k+1}": {"mu": (-2.0, 2.0), "sigma": (0.25, 1.0)} for k in range(5)}
    sigma_box_ranges = {"alpha": (2.5, 7.0), "beta": (0.1667, 2.0)}

    latex_names = {
        "alpha":  r"$\alpha$",
        "beta1":  r"$\beta_1$",
        "beta2":  r"$\beta_2$",
        "beta3":  r"$\beta_3$",
        "beta4":  r"$\beta_4$",
        "beta5":  r"$\beta_5$",
        "sigma":  r"$\sigma$",
    }
    percentages = {'alpha': 11.1, 'beta1': 21.9, 'beta2': 18.1,
                   'beta3': 15.6, 'beta4': 14.1, 'beta5': 13.8, 'sigma': 5.4}
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
    )

    plot_three_panel_priors_all_betas_one_plot_explicit(
        alpha_ref=alpha_ref,
        betas_ref=betas_ref,
        sigma_ref=sigma_ref,
        alpha_ms=alpha_ms,
        betas_ms=betas_ms,
        sigma_ms=sigma_ms,
        sigma_inf=sigma_inf_prior,
        alpha_box_ranges=alpha_box_ranges,
        betas_box_ranges=betas_box_ranges,
        sigma_box_ranges=sigma_box_ranges,
        sigma_cand_family="InverseGamma",
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        sample_n_alpha=50,
        sample_n_sigma=30,
        sample_n_beta_total=150,
        seed=27,
        filename="kilpisjarvi_param_three_panel_priors.pdf",
    )


# ---------------------------------------------------------------------------
# Posterior predictive helpers (mirrored from run_ark_fisher)
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
    """
    Estimate posterior predictive mean and variance via the Law of Total Variance.

    Var[y_t | D] = E_theta[Var[y_t | theta]]   (aleatoric)
                 + Var_theta[E[y_t | theta]]    (epistemic)

    For y_t | theta ~ N(alpha + beta' * lags_t, sigma^2):
      - aleatoric = mean(sigma_s^2)                  -- scalar, same for all t
      - epistemic = var_s(mu_t^s)                    -- (T-K,), time-varying

    Returns
    -------
    pred_mean   : (T-K,)  posterior predictive mean
    pred_std    : (T-K,)  total predictive std
    aleatoric   : scalar  expected noise variance
    epistemic   : (T-K,) parameter-uncertainty variance
    lo, hi      : 95% CI bounds (pred_mean +/- 1.96 * pred_std)
    """
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
    prefix = cfg.playground.get("output_prefix", "kilpisjarvi_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    model = instantiate(cfg.model, data_config=cfg.data)
    K = sum(1 for name in model.prior_init.names if name.startswith("beta"))

    y_obs = model.observations.reshape(-1)   # y_centered[K:]
    # y_centered is the full series (length T); y_obs is the target part (length T-K)
    # Reconstruct y_full from the module-level y_centered
    y_full = y_centered

    warmup = getattr(cfg.data, "warmup", 1000)
    corner_draws_path = cfg.data.posterior_draws_corner_prior
    chains = _load_corner_draws_json(corner_draws_path)
    corner_samples = _stack_corner_chains(chains=chains, K=K, warmup=warmup, max_draws=None)

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
    # main()
    plot_posterior_predictive()

    # repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # plot_config_path = os.path.join(repo_root, "configs/plots/overleaf_plots_settings.yaml")
    # output_dir = os.path.join(repo_root, "outputs/paper/plots/fisher/kilpisjarvi")
    #
    # plot_cfg = load_plot_config(plot_config_path)
    # plot_time_series(output_dir=output_dir, plot_cfg=plot_cfg)
