import math
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from utils.distributions import DISTRIBUTION_MAP as _DIST_MAP_EXTPROJ


def _deep_get(cfg, path, default=None):
    """Safe nested getter that works with dicts or OmegaConf/objects."""
    if cfg is None:
        return default
    cur = cfg
    for key in path.split('.'):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{type1cm}",
    })


def _new_fig_ax(plot_cfg):
    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    return fig, ax


def _save_fig(fig, output_dir: str, filename: str, plot_cfg):
    os.makedirs(output_dir, exist_ok=True)
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)


def _palette(plot_cfg, n: int) -> List[str]:
    base = list(plot_cfg.plot.color_palette.colors)
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]


def _latex_name(plot_cfg, key: str, default: str) -> str:
    try:
        return getattr(plot_cfg.plot.param_latex_names, key)
    except Exception:
        return default


def _gaussian_pdf(x, mu, sigma):
    x = np.asarray(x, float)
    s2 = sigma * sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (x - mu) ** 2 / s2)


def _lognormal_pdf(x, mu_log, sigma_log):
    x = np.asarray(x, float)
    pdf = np.zeros_like(x)
    pos = x > 0
    z = (np.log(x[pos]) - mu_log) / sigma_log
    pdf[pos] = (1.0 / (x[pos] * sigma_log * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)
    return pdf


def _get_base_gaussian(cfg, name: str) -> Tuple[float, float]:
    dist = cfg.data.base_prior.distributions[name]
    return float(dist.mu), float(dist.sigma)


def _extract_worst(rows: List[Dict]) -> Dict:
    return max(rows, key=lambda r: float(r["value"]))


def _extract_params(d: Dict, key: str) -> Dict:
    """Return dict of params for given key (beta1/beta3) if present."""
    if not isinstance(d, dict):
        return {}
    return d.get(key, {}).get("params", {})


def plot_ar_time_series(
    y: np.ndarray,
    plot_cfg,
    output_dir: str,
    filename: str = "ar_pred_ribbon.pdf",
):
    """
    y: (T,)
    y_rep: (n_draws, T) or None; if None, only observed series is plotted
    """
    _apply_plot_rc(plot_cfg)
    fig, ax = _new_fig_ax(plot_cfg)

    T = len(y)
    t = np.arange(1, T + 1)

    # lines
    ax.plot(t, y, label=_latex_name(plot_cfg, "observed", "observed"), linewidth=1.5)
    ax.set_xlabel(_latex_name(plot_cfg, "time_t", r"time $t$"))
    ax.set_ylabel(_latex_name(plot_cfg, "y_t", r"$y_t$"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, output_dir, filename, plot_cfg)


def _get_component_cfg(cfg, name: str):
    """Find component config by name from cfg.ksd.optimize.prior.Composite.components."""
    comps = _deep_get(cfg, "ksd.optimize.prior.Composite.components", [])
    for c in comps:
        if (isinstance(c, dict) and c.get("name") == name) or getattr(c, "name", None) == name:
            return c
    raise KeyError(f"Component '{name}' not found in cfg.")


def _parse_family_from_target(target: str) -> str:
    return target.split(".")[-1]


def _get_base_prior_spec(cfg, name: str) -> Tuple[str, Dict[str, float]]:
    """Read family+params from cfg.data.base_prior.distributions[name]."""
    d = _deep_get(cfg, f"data.base_prior.distributions.{name}", None)
    if d is None:
        raise KeyError(f"Base prior for '{name}' not found in cfg.data.base_prior.distributions")
    target = d.get("_target_", "")
    family = _parse_family_from_target(target) if isinstance(target, str) else d.get("family", "")
    # copy params, excluding _target_
    params = {k: float(v) for k, v in d.items() if k != "_target_"}
    return family, params


def _make_pdf(family: str, params: Dict[str, float]):
    """Prefer your DISTRIBUTION_MAP, otherwise fallback for Gaussian/Gamma."""
    fam = family.strip()
    Dist = _DIST_MAP_EXTPROJ[fam]
    dist = Dist(**params)

    def f(x):
        y = dist.pdf(x)
        return np.asarray(y).squeeze()
    return f


def _auto_x_range_from_ref(family: str, ref_params: Dict[str, float]) -> Tuple[float, float]:
    """
    Choose a reasonable x-range for plotting PDFs, based on a reference prior.

    Supported families (case-insensitive):
      - "Gaussian" / "Normal":      (mu - 4*sigma, mu + 4*sigma)
      - "HalfCauchy":               (0, 10*gamma)              # x >= 0
      - "LogNormal":                (exp(mu_log - 5*sigma_log), exp(mu_log + 5*sigma_log)), clipped at > 0
      - "Gamma": (shape α, scale θ) (0, mean + 6*sd) where mean=αθ, sd=√α * θ

    Falls back to (-5, 5) if family is unknown.
    """
    low = family.strip().lower()

    # Gaussian / Normal
    if low in ("gaussian", "normal"):
        mu = float(ref_params["mu"])
        s = float(ref_params["sigma"])
        return (mu - 4.0 * s, mu + 4.0 * s)

    if low in ("halfcauchy"):
        gamma = float(ref_params.get("gamma", ref_params.get("scale", 1.0)))
        return (0.0, 10.0 * gamma)

    # LogNormal with parameters mu_log, sigma_log (accept "sigma-log" typo too)
    if low == "lognormal":
        mu_log = float(ref_params["mu_log"])
        sigma_log = float(ref_params.get("sigma_log", ref_params.get("sigma-log")))
        lo = np.exp(mu_log - 5.0 * sigma_log)
        hi = np.exp(mu_log + 5.0 * sigma_log)
        # Ensure strictly positive lower bound
        return (max(1e-8, lo), hi)

    # Gamma with shape alpha and scale theta; support x > 0
    if low == "gamma":
        alpha = float(ref_params["alpha"])
        theta = float(ref_params["theta"])
        mean = alpha * theta
        sd = np.sqrt(alpha) * theta
        return (0.0, max(1e-8, mean + 6.0 * sd))

    # Fallback
    return (-5.0, 5.0)


def _linspace_pad(xmin, xmax, n=1200, pad=0.05):
    if not np.isfinite([xmin, xmax]).all():
        xmin, xmax = -5.0, 5.0
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    span = xmax - xmin
    return np.linspace(xmin - pad*span, xmax + pad*span, n)


def _sample_param_sets(
    ranges: Dict[str, List[float]],
    n: int,
    rng: np.random.Generator,
):
    """
    Sample `n` parameter dictionaries uniformly from a box defined by `ranges`.
    """
    keys = list(ranges)
    bounds = np.asarray([ranges[k] for k in keys], dtype=float)
    lows, highs = bounds[:, 0], bounds[:, 1]

    samples = rng.uniform(low=lows, high=highs, size=(n, len(keys)))
    return [
        {k: float(v) for k, v in zip(keys, row)}
        for row in samples
    ]


def _canonicalize_params(params: Dict[str, float], tol: float = 1e-10) -> Tuple[Tuple[str, float], ...]:
    # Sort keys for stable ordering; round to tolerance to avoid tiny float diffs
    return tuple(sorted((k, round(float(v), 10)) for k, v in params.items()))


def _group_beta_most_sensitive(rows_all: Dict, beta_names: List[str]) -> List[Tuple[List[str], str, Dict[str, float]]]:
    """
    Returns list of groups: [([beta names], family, params_dict_for_group), ...]
    Group by (family, params) with rounding.
    """
    bykey = {}
    for b in beta_names:
        fam = rows_all[b]["family"]
        p = rows_all[b]["params"]
        key = (fam, _canonicalize_params(p))
        if key not in bykey:
            bykey[key] = ([], fam, p)
        bykey[key][0].append(b)
    # stable order by first beta index
    idx = {"beta1": 1, "beta2": 2, "beta3": 3, "beta4": 4, "beta5": 5}
    groups = list(bykey.values())
    groups.sort(key=lambda g: min(idx.get(n, 99) for n in g[0]))
    return groups


def plot_three_panel_priors(
    rows_all: Dict,               # {'alpha': {...}, 'beta1': {...}, ..., 'sigma': {...}} with 'family' & 'params'
    cfg,                          # config with families & parameter ranges
    plot_cfg,                     # plotting config
    output_dir: str,
    prefix: str = "ark_param",
    sample_n_alpha: int = 30,
    sample_n_sigma: int = 30,
    sample_n_beta: int = 30,      # per-beta cloud count
    seed: int = 123,
    filename: str = None,
    x_alpha=None,                 # optional manual x-ranges
    x_sigma=None,
    x_beta=None,
):
    """
    Three-panel figure:
      row 1: [alpha | sigma]
      row 2: [betas consolidated]

    - Π_ref (dashed black) from cfg midpoint ranges
    - Candidate clouds from cfg ranges (faint)
    - Most sensitive: from rows_all['...']['params'].
      For betas, identical most-sensitive params are GROUPED and drawn in red but with
      distinct line styles; the legend shows grouped beta names.
    """
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_three_panel_priors.pdf"

    rng = np.random.default_rng(seed)
    col_ref = "black"
    col_red = "red"
    alpha_cloud = 0.18
    palette_full = list(plot_cfg.plot.color_palette.colors)

    def _cloud_colors(n: int, skip_first: int = 2) -> List[str]:
        base = palette_full[skip_first:] if skip_first < len(palette_full) else palette_full
        reps = int(np.ceil(n / len(base)))
        return (base * reps)[:n]

    # Figure & grid
    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi
    )
    plt.subplots_adjust(right=0.82, wspace=0.)
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[25, 25, 50],
        wspace=0.25
    )
    ax_alpha = fig.add_subplot(gs[0, 0])  # left (alpha)
    ax_sigma = fig.add_subplot(gs[0, 1])  # middle (sigma)
    ax_betas = fig.add_subplot(gs[0, 2])  # right (betas, double width)

    # ---------------- ALPHA ----------------
    famA_ref, refA_params = _get_base_prior_spec(cfg, "alpha")
    comp_a = _get_component_cfg(cfg, "alpha")
    ranges_a = comp_a["parameters_box_range"]["ranges"]
    famA_ms = rows_all["alpha"]["family"]
    msA_params = rows_all["alpha"]["params"]

    xA_rng = _auto_x_range_from_ref(famA_ref, refA_params) if x_alpha is None else x_alpha
    xA = _linspace_pad(*xA_rng)
    pdf_ref_a = _make_pdf(famA_ref, refA_params)
    pdf_ms_a = _make_pdf(famA_ms,  msA_params)

    ax_alpha.plot(xA, pdf_ref_a(xA), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")
    cand_colors = _cloud_colors(sample_n_alpha, skip_first=2)
    for (p, c) in zip(_sample_param_sets(ranges_a, sample_n_alpha, rng), cand_colors):
        pdf_c = _make_pdf(comp_a["family"], p)
        ax_alpha.plot(xA, pdf_c(xA), linewidth=0.9, alpha=alpha_cloud, color=c)
    ax_alpha.plot(xA, pdf_ms_a(xA), color=col_red, linewidth=1.8, label="Most sensitive")
    ax_alpha.set_ylabel(_latex_name(plot_cfg, "nonparametric_prior", "prior"))
    ax_alpha.set_title(
        _latex_name(plot_cfg, "alpha", r"$\alpha$")
    )
    ax_alpha.spines["top"].set_visible(False)
    ax_alpha.spines["right"].set_visible(False)

    # ---------------- SIGMA ----------------
    famS_ref, refS_params = _get_base_prior_spec(cfg, "sigma")  # HalfCauchy(gamma=2.5) from data.base_prior
    comp_s = _get_component_cfg(cfg, "sigma")                   # candidates from Gamma ranges (per cfg)
    ranges_s = comp_s["parameters_box_range"]["ranges"]
    famS_ms = rows_all["sigma"]["family"]                       # e.g., "Gamma"
    msS_params = rows_all["sigma"]["params"]

    xS_rng = _auto_x_range_from_ref(famS_ref, refS_params) if x_sigma is None else x_sigma
    xS = _linspace_pad(*xS_rng)
    pdf_ref_s = _make_pdf(famS_ref, refS_params)
    pdf_ms_s = _make_pdf(famS_ms,  msS_params)

    ax_sigma.plot(xS, pdf_ref_s(xS), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")
    cand_colors = _cloud_colors(sample_n_sigma, skip_first=2)
    for (p, c) in zip(_sample_param_sets(ranges_s, sample_n_sigma, rng), cand_colors):
        pdf_c = _make_pdf(comp_s["family"], p)
        ax_sigma.plot(xS, pdf_c(xS), linewidth=0.9, alpha=alpha_cloud, color=c)
    ax_sigma.plot(xS, pdf_ms_s(xS), color=col_red, linewidth=1.8, label="Most sensitive")
    ax_sigma.set_title(
        _latex_name(plot_cfg, "sigma", r"$\sigma$")
    )
    ax_sigma.spines["top"].set_visible(False)
    ax_sigma.spines["right"].set_visible(False)

    # ---------------- BETAS (grouped most-sensitive) ----------------
    beta_names = ["beta1", "beta2", "beta3", "beta4", "beta5"]
    beta_group_styles = ["-", ":", "--", "-."]

    # Shared Π_ref from base_prior (assume same family/params across β’s; use beta1 as source)
    famB_ref, refB_params = _get_base_prior_spec(cfg, "beta1")
    xB_rng = _auto_x_range_from_ref(famB_ref, refB_params) if x_beta is None else x_beta
    xB = _linspace_pad(*xB_rng)
    pdf_ref_b = _make_pdf(famB_ref, refB_params)
    ax_betas.plot(xB, pdf_ref_b(xB), linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")

    # candidate clouds from ranges of each beta
    for bname in beta_names:
        comp_b = _get_component_cfg(cfg, bname)
        ranges_b = comp_b["parameters_box_range"]["ranges"]
        fam_b_cand = comp_b["family"]
        cand_colors = _cloud_colors(sample_n_beta, skip_first=2)
        for (p, c) in zip(_sample_param_sets(ranges_b, sample_n_beta, rng), cand_colors):
            pdf_c = _make_pdf(fam_b_cand, p)
            ax_betas.plot(xB, pdf_c(xB), linewidth=0.75, alpha=alpha_cloud, color=c)

    ax_betas.set_title(r"$\beta_{1 \cdots 5}$", pad=-9)
    # ax_betas.set_ylabel(_latex_name(plot_cfg, "nonparametric_prior", "prior"))
    ax_betas.spines["top"].set_visible(False)
    ax_betas.spines["right"].set_visible(False)

    # group equal most-sensitive betas and draw each group with distinct red linestyle
    groups = _group_beta_most_sensitive(rows_all, beta_names)
    legend_handles = []
    legend_labels = []

    # Π_ref (only once)
    legend_handles.append(plt.Line2D([], [], color=col_ref, linestyle="--", linewidth=1.2))
    legend_labels.append(r"$\Pi_{\mathrm{ref}}$")

    legend_handles.append(plt.Line2D([], [], color=col_red, linestyle="-", linewidth=1.2))
    legend_labels.append(
        r"$\Pi$ at " + plot_cfg["plot"]["param_latex_names"]["argoptimisationProblemParam"]
    )

    for gi, (names, fam_g, params_g) in enumerate(groups):
        style = beta_group_styles[min(gi, len(beta_group_styles)-1)]
        pdf_ms = _make_pdf(fam_g, params_g)
        ax_betas.plot(xB, pdf_ms(xB), color=col_red, linestyle=style, linewidth=1.8)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        frameon=False,
        ncol=len(legend_labels),
        bbox_to_anchor=(0.5, -0.23)
    )

    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_three_panel_priors_all_betas_one_plot_explicit(
    # -------- explicit reference priors (no cfg) --------
    alpha_ref: Dict,   # {"family": "Gaussian", "params": {"mu": 0.0, "sigma": 10.0}}
    betas_ref: Dict,   # {"family": "Gaussian", "params": {"mu": 0.0, "sigma": 10.0}}
    sigma_ref: Dict,   # e.g. {"family": "HalfCauchy", "params": {"gamma": 2.5}}

    # -------- explicit candidate priors --------
    alpha_ms: Dict,                  # most-sensitive alpha (sup)
    betas_ms: Dict[str, Dict],       # most-sensitive betas (sup), {"beta1": {...}, ..., "beta5": {...}}
    sigma_ms: Dict,                  # most-sensitive sigma (sup)
    alpha_box_ranges: Dict,
    betas_box_ranges: Dict[str, Dict],
    sigma_box_ranges: Dict,
    sigma_inf: Dict = None,          # OPTIONAL: infimum-optimal sigma (argmin), plotted as dashed red
    alpha_cand_family: str = "Gaussian",
    betas_cand_family: str = "Gaussian",
    sigma_cand_family: str = "Gamma",

    # -------- plotting --------
    plot_cfg=None,
    output_dir: str = ".",
    prefix: str = "ark_param",
    sample_n_alpha: int = 30,
    sample_n_sigma: int = 30,
    sample_n_beta_total: int = 150,   # total cloud draws pooled across betas
    seed: int = 123,
    filename=None,
    x_alpha=None,
    x_sigma=None,
    x_beta=None,

    # -------- hooks to your existing utilities --------
    make_pdf=None,          # if None, uses global _make_pdf
    sample_param_sets=None,  # if None, uses global _sample_param_sets
    save_fig=None,          # if None, uses global _save_fig
    apply_plot_rc=None,     # if None, uses global _apply_plot_rc
):
    """
    3-panel figure in one row (ORDER: alpha | betas | sigma):
      [alpha | betas (β1..β5 overlaid on a single axis) | sigma]

    Curves:
      - reference: black dashed
      - box neighbourhood cloud: faint colored
      - sup / most-sensitive: solid red
      - OPTIONAL sigma infimum: dashed red (sigma panel only)
    """
    # ---------- resolve helper hooks ----------
    if make_pdf is None:
        make_pdf = _make_pdf
    if sample_param_sets is None:
        sample_param_sets = _sample_param_sets
    if save_fig is None:
        save_fig = _save_fig
    if apply_plot_rc is None:
        apply_plot_rc = _apply_plot_rc

    if plot_cfg is not None:
        apply_plot_rc(plot_cfg)

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_three_panel_priors.pdf"

    rng = np.random.default_rng(seed)

    # ---------- cosmetics ----------
    col_ref = "black"
    col_red = "red"
    alpha_cloud = 0.18

    # best-effort palette fallback if plot_cfg is not passed
    if plot_cfg is not None and hasattr(plot_cfg, "plot") and hasattr(plot_cfg.plot, "color_palette"):
        palette_full = list(plot_cfg.plot.color_palette.colors)
    else:
        palette_full = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    def _cloud_colors(n: int, skip_first: int = 0) -> List[str]:
        base = palette_full[skip_first:] if skip_first < len(palette_full) else palette_full
        reps = int(np.ceil(n / len(base)))
        return (base * reps)[:n]

    def _linspace_pad(lo: float, hi: float, n: int = 500, pad: float = 0.03) -> np.ndarray:
        span = hi - lo
        return np.linspace(lo - pad * span, hi + pad * span, n)

    def _auto_x_range_from_prior(family: str, params: Dict) -> Tuple[float, float]:
        fam = family.lower()

        # Normal(mu, sigma)
        if fam in {"normal", "gaussian"}:
            mu = float(params.get("mu", 0.0))
            sig = float(params.get("sigma", params.get("std", 1.0)))
            return mu - 5.0 * sig, mu + 5.0 * sig

        # HalfCauchy(scale) / Cauchy(loc=0, scale)
        if fam in {"halfcauchy", "half-cauchy"}:
            scale = float(params.get("scale", params.get("gamma", 1.0)))
            return 0.0, 20.0 * scale
        if fam in {"cauchy"}:
            loc = float(params.get("loc", 0.0))
            scale = float(params.get("scale", 1.0))
            return loc - 20.0 * scale, loc + 20.0 * scale

        # Gamma: supports either (alpha, theta) (scale) or (a, b) (rate)
        if fam in {"gamma"}:
            if "theta" in params:  # scale
                a = float(params.get("alpha", params.get("shape", 1.0)))
                theta = float(params["theta"])
                mean = a * theta
                std = np.sqrt(a) * theta
            else:  # rate
                a = float(params.get("a", params.get("alpha", params.get("shape", 1.0))))
                b = float(params.get("b", params.get("rate", 1.0)))
                mean = a / b
                std = np.sqrt(a) / b
            lo = max(0.0, mean - 5.0 * std)
            hi = mean + 8.0 * std
            return lo, hi

        # fallback
        return -10.0, 10.0

    def _group_equal_ms_betas(betas_ms_local: Dict[str, Dict]) -> List[Tuple[List[str], str, Dict]]:
        groups = {}
        for name, spec in betas_ms_local.items():
            fam = spec["family"]
            p = spec["params"]
            key = (fam, tuple(sorted(p.items())))
            groups.setdefault(key, {"names": [], "family": fam, "params": p})
            groups[key]["names"].append(name)
        out = sorted(
            [(v["names"], v["family"], v["params"]) for v in groups.values()],
            key=lambda t: (-len(t[0]), t[0][0]),
        )
        return out

    # ---------- figure layout ----------
    if plot_cfg is not None and hasattr(plot_cfg.plot.figure, "size"):
        figsize = (plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        dpi = plot_cfg.plot.figure.dpi
    else:
        figsize = (10.5, 3.2)
        dpi = 200

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(right=0.98, wspace=0.25)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[60, 40],  # sigma now much larger
        wspace=0.2
    )
    ax_alpha_betas = fig.add_subplot(gs[0, 0])
    ax_sigma = fig.add_subplot(gs[0, 1])

    # ===================== ALPHA =====================
    famA_ref, refA_params = alpha_ref["family"], alpha_ref["params"]
    famA_ms, msA_params = alpha_ms["family"], alpha_ms["params"]
    xA_rng = (-7, 7)
    xA = _linspace_pad(*xA_rng)
    pdf_ref_a = make_pdf(famA_ref, refA_params)
    pdf_ms_a = make_pdf(famA_ms, msA_params)

    for p, ccol in zip(sample_param_sets(alpha_box_ranges, sample_n_alpha, rng), _cloud_colors(sample_n_alpha)):
        pdf_c = make_pdf(alpha_cand_family, p)
        ax_alpha_betas.plot(xA, pdf_c(xA), linewidth=0.9, alpha=alpha_cloud, color=ccol)
    ax_alpha_betas.plot(xA, pdf_ref_a(xA), linestyle="--", color=col_ref, linewidth=1.0)
    ax_alpha_betas.plot(xA, pdf_ms_a(xA), color=col_red, linestyle="-.", linewidth=1.0)

    ax_alpha_betas.set_title(r"$\alpha, \beta_{1\cdots 5}$")
    ax_alpha_betas.set_ylabel(r"$\pi$")
    ax_alpha_betas.spines["top"].set_visible(False)
    ax_alpha_betas.spines["right"].set_visible(False)

    # ===================== BETAS (ALL ON ONE AXIS) =====================
    # beta_names = ["beta1", "beta2", "beta3", "beta4", "beta5"]
    beta_group_styles = ["-", ":", "--", "-."]

    famB_ref, refB_params = betas_ref["family"], betas_ref["params"]
    xB = _linspace_pad(*xA_rng)
    # pdf_ref_b = make_pdf(famB_ref, refB_params)
    # ax_betas.plot(xB, pdf_ref_b(xB), linestyle="--", color=col_ref, linewidth=1.2)

    # # pooled cloud budget across betas
    # counts = np.full(len(beta_names), sample_n_beta_total // len(beta_names), dtype=int)
    # counts[: (sample_n_beta_total - counts.sum())] += 1
    #
    # for bname, n_draws in zip(beta_names, counts):
    #     ranges_b = betas_box_ranges[bname]
    #     for p, ccol in zip(sample_param_sets(ranges_b, n_draws, rng), _cloud_colors(n_draws)):
    #         pdf_c = make_pdf(betas_cand_family, p)
    #         ax_betas.plot(xB, pdf_c(xB), linewidth=0.75, alpha=alpha_cloud, color=ccol)

    # most-sensitive betas: group identical curves
    groups = _group_equal_ms_betas(betas_ms)
    for gi, (names, fam_g, params_g) in enumerate(groups):
        style = beta_group_styles[gi % len(beta_group_styles)]
        pdf_ms = make_pdf(fam_g, params_g)
        ax_alpha_betas.plot(xB, pdf_ms(xB), color=col_red, linestyle=style, linewidth=1.0)

    # ax_betas.set_title(r"$\beta_{1\cdots 5}$", pad=-9)
    # ax_betas.spines["top"].set_visible(False)
    # ax_betas.spines["right"].set_visible(False)

    # ===================== SIGMA =====================
    famS_ref, refS_params = sigma_ref["family"], sigma_ref["params"]
    famS_ms, msS_params = sigma_ms["family"], sigma_ms["params"]

    if x_sigma is None:
        lo1, hi1 = _auto_x_range_from_prior(famS_ref, refS_params)
        lo2, hi2 = _auto_x_range_from_prior(famS_ms, msS_params)
        lo, hi = min(lo1, lo2), max(hi1, hi2)
        if sigma_inf is not None:
            lo3, hi3 = _auto_x_range_from_prior(sigma_inf["family"], sigma_inf["params"])
            lo, hi = min(lo, lo3), max(hi, hi3)
        xS_rng = (0.0, 0.8)
    else:
        xS_rng = (0.0, 40.0)

    xS = _linspace_pad(*xS_rng, n=1200)
    pdf_ref_s = make_pdf(famS_ref, refS_params)
    pdf_ms_s = make_pdf(famS_ms, msS_params)

    ax_sigma.plot(xS, pdf_ref_s(xS), linestyle="--", color=col_ref, linewidth=1.0)
    for p, ccol in zip(sample_param_sets(sigma_box_ranges, sample_n_sigma, rng), _cloud_colors(sample_n_sigma)):
        pdf_c = make_pdf(sigma_cand_family, p)
        y = pdf_c(xS)
        if np.max(y) > 10:
            continue
        ax_sigma.plot(xS, y, linewidth=0.9, alpha=alpha_cloud, color=ccol)

    ax_sigma.plot(xS, pdf_ms_s(xS), color=col_red, linestyle="-", linewidth=1.0)

    if sigma_inf is not None:
        famS_inf, infS_params = sigma_inf["family"], sigma_inf["params"]
        pdf_inf_s = make_pdf(famS_inf, infS_params)
        ax_sigma.plot(xS, pdf_inf_s(xS), color="blue", linestyle="--", linewidth=1.8)

    ax_sigma.set_title(r"$\sigma$")
    ax_sigma.spines["top"].set_visible(False)
    ax_sigma.spines["right"].set_visible(False)

    # ---------- legend ----------
    legend_handles = [
        plt.Line2D([], [], color=col_ref, linestyle="--", linewidth=1.2),
        plt.Line2D([], [], color=palette_full[2] if len(palette_full) > 2 else "gray",
                   linestyle="-", linewidth=1.2, alpha=alpha_cloud),
        plt.Line2D([], [], color=col_red, linestyle="-", linewidth=1.8),
    ]
    legend_labels = [r"$\Pi_{\mathrm{ref}}$", r"$\Pi$ in box neighbourhood", r"$\Pi$ at $\eta^\star$"]

    if sigma_inf is not None:
        legend_handles.append(plt.Line2D([], [], color=col_red, linestyle="--", linewidth=1.8))
        legend_labels.append(r"$\Pi$ at $\eta^{\inf}$ (sigma)")

    if len(groups) > 1:
        for gi, (names, _, _) in enumerate(groups):
            style = beta_group_styles[gi % len(beta_group_styles)]
            legend_handles.append(plt.Line2D([], [], color=col_red, linestyle=style, linewidth=1.8))
            idx = ",".join([n.replace("beta", "") for n in names])
            legend_labels.append(r"$\beta_{%s}$" % idx)

    if plot_cfg is not None and getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()

    save_fig(fig, output_dir, filename, plot_cfg)


def _prepare_inputs(y, posterior_samples_init, K):
    y = np.asarray(y).squeeze()
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (got {y.shape}).")
    if posterior_samples_init.ndim != 2:
        raise ValueError("posterior_samples_init must be 2D (S, K+2).")
    S, P = posterior_samples_init.shape
    if P != (K + 2):
        raise ValueError(f"Expected {K+2} columns, got {P}.")
    if len(y) < K + 1:
        raise ValueError(f"y length must be ≥ {K+1}, got {len(y)}.")
    return y, posterior_samples_init


def _compute_in_sample_means(y, alpha, betas):
    K, N = len(betas), len(y)
    mu = np.full(N, np.nan)
    for t in range(K, N):
        mu[t] = alpha + np.dot(betas, y[t - np.arange(1, K + 1)])
    return mu


def _scatter_matrix(params, labels, cfg):
    M = params.shape[1]
    scale = _deep_get(cfg, "ar.fig4.figsize_scale", 2.6)
    bins = _deep_get(cfg, "ar.fig4.hist_bins", 40)
    s = _deep_get(cfg, "ar.fig4.point_size", 4)
    a = _deep_get(cfg, "ar.fig4.scatter_alpha", 0.6)

    fig, axes = plt.subplots(M, M, figsize=(scale*M, scale*M))
    cols = _palette(cfg, 1)
    for i in range(M):
        for j in range(M):
            ax = axes[i, j]
            if i == j:
                ax.hist(params[:, j], bins=bins)
                ax.set_title(f"(med={np.median(params[:, j]):.3f})")
            elif i > j:
                ax.scatter(params[:, j], params[:, i], s=s, alpha=a, c=cols[0])
            else:
                ax.axis("off")
            if i == M-1:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
    if _deep_get(cfg, "plot.figure.tight_layout", True):
        fig.tight_layout()
    return fig


def plot_ar_results(y, posterior_samples_init, K=5, plot_cfg=None, save=False, output_dir=None, prefix="ar"):
    if plot_cfg:
        _apply_plot_rc(plot_cfg)
    show_int = _deep_get(plot_cfg, "ar.show_intervals", True)
    interval = _deep_get(plot_cfg, "ar.interval", 0.9)
    legend_loc = _deep_get(plot_cfg, "plot.legend.loc", "best")
    tight = _deep_get(plot_cfg, "plot.figure.tight_layout", True)

    y, posterior_samples_init = _prepare_inputs(y, posterior_samples_init, K)
    N, S = len(y), posterior_samples_init.shape[0]
    a_s, b_s, s_s = posterior_samples_init[:, 0], posterior_samples_init[:, 1:1+K], posterior_samples_init[:, -1]
    a_med, b_med, s_med = np.median(a_s), np.median(b_s, 0), np.median(s_s)
    mu_med = _compute_in_sample_means(y, a_med, b_med)

    low = high = None
    if show_int and S > 1:
        mu_all = np.array([_compute_in_sample_means(y, a, b) for a, b in zip(a_s, b_s)])
        q = (1-interval)/2
        low, high = np.nanquantile(mu_all, [q, 1-q], 0)

    figs = {}
    cols = _palette(plot_cfg, 3)
    n_alpha = _latex_name(plot_cfg, "alpha", "$\\alpha$")
    n_sigma = _latex_name(plot_cfg, "sigma", "$\\sigma$")

    # fig1
    f1, ax1 = _new_fig_ax(plot_cfg)
    t = np.arange(N)
    ax1.plot(t, y, label=_deep_get(plot_cfg, "ar.fig1.label_y", "Observed y"), c=cols[0])
    ax1.plot(t, mu_med, label=_deep_get(plot_cfg, "ar.fig1.label_mu", "Posterior mean"), c=cols[1])
    if low is not None:
        ax1.fill_between(t, low, high, alpha=_deep_get(plot_cfg, "ar.fig1.band_alpha", 0.25),
                         label=_deep_get(plot_cfg, "ar.fig1.band_label", f"{int(interval*100)}% band"))
    ax1.set(xlabel=_deep_get(plot_cfg, "ar.fig1.x_label", "t"), ylabel=_deep_get(plot_cfg, "ar.fig1.y_label", "y"))
    ax1.legend(loc=legend_loc)
    if tight:
        f1.tight_layout()
        figs["fig1"] = f1

    # fig2
    res = y[~np.isnan(mu_med)]-mu_med[~np.isnan(mu_med)]
    f2, ax2 = _new_fig_ax(plot_cfg)
    ax2.plot(np.arange(len(res)), res, c=cols[2])
    ax2.axhline(0, ls="--")
    ax2.set(xlabel=_deep_get(plot_cfg, "ar.fig2.x_label",
            f"Index t={K}..{N-1}"), ylabel=_deep_get(plot_cfg, "ar.fig2.y_label", "Residual"))
    if tight:
        f2.tight_layout()
        figs["fig2"] = f2

    # fig3
    names = [n_alpha]+[_latex_name(plot_cfg, f"beta_{i}", f"$\\beta_{{{i}}}$") for i in range(1, K+1)]+[n_sigma]
    samps = [a_s]+[b_s[:, i] for i in range(K)]+[s_s]
    bins = _deep_get(plot_cfg, "ar.fig3.bins", 50)
    ncols = _deep_get(plot_cfg, "ar.fig3.ncols", 4)
    nrows = math.ceil(len(samps)/ncols)
    f3, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.5*nrows))
    axes = np.ravel(axes)
    for ax, n, s in zip(axes, names, samps):
        ax.hist(s, bins=bins)
        ax.set_title(f"{n} (med={np.median(s):.3f})")
    for ax in axes[len(samps):]:
        ax.axis("off")
    if tight:
        f3.tight_layout()
        figs["fig3"] = f3

    # fig4
    params = np.column_stack([a_s, b_s, s_s])
    labels = [n_alpha]+[_latex_name(plot_cfg, f"beta_{i}", f"$\\beta_{{{i}}}$") for i in range(1, K+1)]+[n_sigma]
    f4 = _scatter_matrix(params, labels, plot_cfg)
    figs["fig4"] = f4

    for f in figs.values():
        for ax in f.get_axes():
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    if save:
        out = output_dir or _deep_get(plot_cfg, "plot.output.dir", "plots")
        _save_fig(f1, out, _deep_get(plot_cfg, "ar.output.filenames.fig1", f"{prefix}_series.pdf"), plot_cfg)
        _save_fig(f2, out, _deep_get(plot_cfg, "ar.output.filenames.fig2", f"{prefix}_residuals.pdf"), plot_cfg)
        _save_fig(f3, out, _deep_get(plot_cfg, "ar.output.filenames.fig3", f"{prefix}_marginals.pdf"), plot_cfg)
        _save_fig(f4, out, _deep_get(plot_cfg, "ar.output.filenames.fig4", f"{prefix}_pairs.pdf"), plot_cfg)

    return figs


def plot_complexity_bar(
    plot_cfg,
    output_dir: str,
    prefix: str = "ark_param",
    filename: str | None = None,
    use_log10: bool = True,
    qf_full_time_sec=0.0,
    qf_decomp_time_sec=0.0,
    black_box_time_sec=0.0,
):
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f"{prefix}_complexity_bar.pdf"

    labels = [
        "BBO",
        r"$\Gamma_\text{box}$",
        r"$\prod_{j=1}^{d_\Theta}\Gamma_{\text{box}_j}$",
    ]

    def _to_array(v):
        if np.isscalar(v):
            return np.array([float(v)])
        return np.maximum(np.asarray(v, dtype=float), 1e-12)

    data = [_to_array(black_box_time_sec), _to_array(qf_full_time_sec), _to_array(qf_decomp_time_sec)]

    ylab = r"Time (sec.)"

    try:
        palette = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(palette) < len(labels):
        reps = int(np.ceil(len(labels) / len(palette)))
        palette = (palette * reps)[:len(labels)]

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (10, 5),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )

    ax = fig.add_subplot(1, 1, 1)

    if use_log10:
        ax.set_yscale("log")

    print(
        f"Bbox={np.mean(data[0])},  our approach avg. time={np.min(data[1])}, decomposed approach avg. time={np.min(data[2])}")

    for i, (d, color) in enumerate(zip(data, palette)):
        ax.boxplot(
            d,
            positions=[i + 1],
            patch_artist=True,
            widths=0.5,
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.75),
            medianprops=dict(color=color, linewidth=1.0, linestyle="--"),
            whiskerprops=dict(color=color, alpha=0.8),
            capprops=dict(color=color, alpha=0.8),
            flierprops=dict(markerfacecolor=color, markeredgecolor=color, alpha=0.6),
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize="small")
    ax.tick_params(axis="x", length=0, labelcolor="none")

    for i, (d, label) in enumerate(zip(data, labels)):
        if i == 0:
            ax.annotate(label, xy=(i + 1, np.min(d)), xytext=(0, -10),
                        textcoords="offset points", ha="center", va="top", fontsize="small")
        else:
            ax.annotate(label, xy=(i + 1, np.max(d)), xytext=(0, 10),
                        textcoords="offset points", ha="center", va="bottom", fontsize="small")

    ax.set_ylabel(ylab)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")


def plot_component_sensitivity_bar(
    plot_cfg,
    output_dir: str,
    names: list,
    contributions: dict,
    display_names: dict | None = None,
    order: list | None = None,
    group_tail_betas: bool = False,
    tail_beta_keys: list | None = None,
    filename: str | None = None,
    prefix: str = "kilpisjarvi",
):
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f"{prefix}_component_sensitivity.pdf"

    import matplotlib.colors as mcolors

    _display = display_names or {}

    # optionally collapse tail betas into one block
    if group_tail_betas:
        group_keys = tail_beta_keys or ["beta3", "beta4", "beta5"]
        group_val = sum(contributions.get(k, 0.0) for k in group_keys)
        group_label = ", ".join(_display.get(k, k) for k in group_keys)
        _group_key = "__grouped_betas__"
        contributions = {k: v for k, v in contributions.items() if k not in group_keys}
        contributions[_group_key] = group_val
        names = [k for k in names if k not in group_keys] + [_group_key]
        _display = dict(_display)
        _display[_group_key] = group_label

    percentages = contributions

    # display order: low to high from bottom to top; default to names order
    display_order = order if order is not None else list(names)

    # color per component based on contribution rank (highest → #450314, lowest → #5b9bd5)
    c_high = np.array(mcolors.to_rgb("#450314"))
    c_low = np.array(mcolors.to_rgb("#5b9bd5"))
    ranked = sorted(names, key=lambda k: contributions[k])
    n = len(ranked)
    color_map = {
        k: mcolors.to_hex(c_low + t * (c_high - c_low))
        for k, t in zip(ranked, np.linspace(0, 1, n))
    }

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (3, 6),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )
    ax = fig.add_subplot(1, 1, 1)

    bottom = 0.0
    min_label_pct = 4.0
    for k in display_order:
        color = color_map[k]
        pct = percentages[k]
        ax.bar(0, pct, bottom=bottom, color=color, alpha=0.5, edgecolor="white", linewidth=0.5, width=0.6)
        if pct >= min_label_pct:
            label = _display.get(k, k)
            ax.text(
                0, bottom + pct / 2,
                f"{label} {pct:.1f}%",
                ha="center", va="center",
                fontsize="x-small", color="black",
                usetex=False,
            )
        bottom += pct

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"], usetex=False)
    ax.set_ylabel(r"$\widehat{S}_m^{\mathrm{FD}}(\Gamma_j)$ \%")

    ax.tick_params(axis="x", length=0, labelcolor="none")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")


def plot_acf_comparison(
    plot_cfg,
    output_dir: str,
    acf_ref: np.ndarray,
    acf_corner: np.ndarray,
    ref_color: str = "#7c397d",
    corner_color: str = "#5b9bd5",
    ref_label: str = "Reference",
    corner_label: str = "Corner",
    filename: str = "kilpisjarvi_acf_comparison.pdf",
):
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (6, 4),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )

    lags = np.arange(len(acf_ref))
    ax.plot(lags, acf_ref, color=ref_color, linewidth=1.5, label=ref_label)
    ax.plot(lags, acf_corner, color=corner_color, linewidth=1.5, label=corner_label)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")

    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")


def plot_posterior_predictive_bands(
    plot_cfg,
    output_dir: str,
    y_obs: np.ndarray,
    mean: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    filename: str | None = None,
):
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
    t = np.arange(1, y_obs.shape[0] + 1)
    colors = list(plot_cfg.plot.color_palette.colors)
    obs_color = colors[0]
    mean_color = colors[1]
    ci_color = colors[1]

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (10, 4),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t, y_obs, linewidth=1.2, color=obs_color, label=r"$x$")
    ax.fill_between(t, lo, hi, alpha=0.25, color=ci_color)
    ax.plot(t, mean, linewidth=1.5, color=mean_color, label=r"$\tilde{x} \pm 95\%$ CI")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")


def plot_posterior_predictive_bands_compare(
    plot_cfg,
    output_dir: str,
    prefix: str,
    y_obs: np.ndarray,
    ref_mean: np.ndarray,
    ref_lo: np.ndarray,
    ref_hi: np.ndarray,
    corner_mean: np.ndarray,
    corner_lo: np.ndarray,
    corner_hi: np.ndarray,
    filename: str | None = None,
):
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_posterior_predictive_ref_vs_corner.pdf"

    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
    t = np.arange(1, y_obs.shape[0] + 1)

    # palette from config (fallback to matplotlib)
    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(colors) < 3:
        reps = int(np.ceil(3 / len(colors)))
        colors = (colors * reps)[:3]

    ref_color = colors[1]
    corner_color = colors[2]

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height)
        if hasattr(plot_cfg, "plot") else (10, 4),
        dpi=plot_cfg.plot.figure.dpi if hasattr(plot_cfg, "plot") else 120,
    )
    ax = fig.add_subplot(1, 1, 1)

    # ---- observations as black crosses
    ax.plot(
        t[::2],
        y_obs[::2],
        linestyle="None",
        marker="x",
        markersize=3.0,
        markeredgewidth=1.0,
        color="black",
        label=r"$x$",
        zorder=3,
    )

    # # ---- reference posterior predictive
    # ax.fill_between(
    #     t,
    #     ref_lo,
    #     ref_hi,
    #     color=ref_color,
    #     alpha=0.18,
    #     # label=r"Ref. $\tilde{x}$ 95\% CI",
    #     zorder=1,
    # )
    ax.plot(
        t,
        ref_mean,
        linewidth=1.0,
        color=ref_color,
        label=r"$\tilde{x}_\text{ref} \pm 95\% CI$",
        zorder=2,
    )

    # ---- corner posterior predictive
    ax.fill_between(
        t,
        corner_lo,
        corner_hi,
        color=corner_color,
        alpha=0.18,
        # label=r"Corner $\tilde{x}$ 95\% CI",
        zorder=1,
    )
    ax.plot(
        t,
        corner_mean,
        linewidth=1.0,
        color=corner_color,
        label=r"$\tilde{x} \pm 95\% CI$",
        zorder=2,
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, frameon=False, ncol=1,
    #           loc="upper right", bbox_to_anchor=(1, 1.2))

    fig.tight_layout()

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")


def plot_posterior_predictive_with_data(
    plot_cfg,
    output_dir: str,
    x_years: np.ndarray,
    y_uncentered: np.ndarray,
    x_pred_years: np.ndarray,
    pred_mean: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    pred_label: str,
    filename: str,
    pred_color: str | None = None,
    ylim=None,
    show_ylabel: bool = True,
):
    """Plot uncentered observations (line + scatter, real years) with one set of
    posterior predictive bands overlaid."""
    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    x_years = np.asarray(x_years, dtype=float)
    y_uncentered = np.asarray(y_uncentered, dtype=float)
    x_pred_years = np.asarray(x_pred_years, dtype=float)

    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    obs_color = colors[0]
    if pred_color is None:
        pred_color = colors[1]

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    ax.plot(
        x_years, y_uncentered,
        linestyle="None", marker="x", markersize=3.0, markeredgewidth=1.0,
        color="black", zorder=1,
    )

    ax.fill_between(x_pred_years, pred_lo, pred_hi, color=pred_color, alpha=0.25, zorder=2)
    ax.plot(x_pred_years, pred_mean, linewidth=1.5, color=pred_color, label=pred_label, zorder=3)

    ax.set_xlabel("Year")
    if show_ylabel:
        ax.set_ylabel("Temperature (°C)")
    else:
        ax.set_ylabel("Temperature (°C)", color="none")
        ax.tick_params(axis="y", labelcolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    # ax.legend(frameon=False, loc="upper right")

    try:
        _save_fig(fig, output_dir, filename, plot_cfg)
    except Exception:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight")


def animate_posterior_predictive_with_data(
    plot_cfg,
    output_dir: str,
    x_years: np.ndarray,
    y_uncentered: np.ndarray,
    x_pred_years: np.ndarray,
    pred_mean: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    pred_label: str,
    filename: str,
    pred_color: str | None = None,
    ylim=None,
    show_ylabel: bool = True,
    fps: int = 30,
    draw_seconds: float = 4.0,
    hold_seconds: float = 1.5,
):
    """Animated version of plot_posterior_predictive_with_data.

    A single left-to-right sweep through time: at each frame everything
    available up to the current time step is revealed simultaneously —
    observation points, CI band, and mean line all advance together.
    The final frame is held for hold_seconds.

    Saves an MP4 to output_dir/filename.
    """
    import imageio.v3 as iio

    try:
        _apply_plot_rc(plot_cfg)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    x_years = np.asarray(x_years, dtype=float)
    y_uncentered = np.asarray(y_uncentered, dtype=float)
    x_pred_years = np.asarray(x_pred_years, dtype=float)
    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_lo = np.asarray(pred_lo, dtype=float)
    pred_hi = np.asarray(pred_hi, dtype=float)

    try:
        colors = list(plot_cfg.plot.color_palette.colors)
    except Exception:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if pred_color is None:
        pred_color = colors[1]

    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    all_x = np.concatenate([x_years, x_pred_years])
    t_min, t_max = all_x.min(), all_x.max()
    ax.set_xlim(t_min, t_max)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        all_y = np.concatenate([y_uncentered, pred_lo, pred_hi])
        margin = (all_y.max() - all_y.min()) * 0.05
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    ax.set_xlabel("Year")
    if show_ylabel:
        ax.set_ylabel("Temperature (°C)")
    else:
        ax.set_ylabel("Temperature (°C)", color="none")
        ax.tick_params(axis="y", labelcolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    scatter = ax.scatter([], [], marker="x", s=9.0, linewidths=1.0, color="black", zorder=1)
    fill_artist = [ax.fill_between([], [], [], color=pred_color, alpha=0.25, zorder=2)]
    (mean_line,) = ax.plot([], [], linewidth=1.5, color=pred_color, label=pred_label, zorder=3)

    # Apply the same layout the static plot uses via _save_fig
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        fig.tight_layout()

    def _capture():
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    frames = []
    fig.canvas.draw()

    # Sweep frames linearly from t_min to t_max
    n_draw = max(2, int(fps * draw_seconds))
    legend_drawn = False
    for k in range(1, n_draw + 1):
        t_now = t_min + (t_max - t_min) * k / n_draw

        # Observations up to t_now
        obs_mask = x_years <= t_now
        if obs_mask.any():
            scatter.set_offsets(np.column_stack([x_years[obs_mask], y_uncentered[obs_mask]]))

        # CI band and mean line up to t_now
        pred_mask = x_pred_years <= t_now
        fill_artist[0].remove()
        if pred_mask.sum() >= 2:
            fill_artist[0] = ax.fill_between(
                x_pred_years[pred_mask], pred_lo[pred_mask], pred_hi[pred_mask],
                color=pred_color, alpha=0.25, zorder=2,
            )
            mean_line.set_data(x_pred_years[pred_mask], pred_mean[pred_mask])
            if not legend_drawn:
                ax.legend(frameon=False, loc="upper right")
                legend_drawn = True
        else:
            fill_artist[0] = ax.fill_between([], [], [], color=pred_color, alpha=0.25, zorder=2)

        frames.append(_capture())

    # Hold final frame
    n_hold = max(1, int(fps * hold_seconds))
    hold_frame = _capture()
    for _ in range(n_hold):
        frames.append(hold_frame.copy())

    path = os.path.join(output_dir, filename)
    iio.imwrite(path, np.stack(frames), fps=fps, quality=9)
    plt.close(fig)
    print(f"Saved animation to {path}")
