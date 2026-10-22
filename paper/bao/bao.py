"""DESI DR2 BAO likelihood with block scores for data-removal sensitivity.

Model: flat LCDM (theta = omega_b, omega_c, H0) or w0waCDM (+ w0, wa). Sound horizon from the Aubourg et al. (2015)
fitting formula, one massive neutrino (0.06 eV), BBN Gaussian prior on omega_b and box priors elsewhere.
The 13 measurements form 7 independent Gaussian blocks (one with a single point, six with two correlated points).
"""
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "bao")
C_KMS = 299792.458
OMEGA_NU = 0.06 / 93.14                     # omega_nu h^2 for sum m_nu = 0.06 eV
OMEGA_R = 2.469e-5 * (1 + 0.2271 * 2.044)  # photons + ~2 massless neutrino species
BBN_MEAN, BBN_SD = 0.02218, 0.00055         # DESI DR2 BBN prior on omega_b

# ---------------------------------------------------------------- data
_mean = np.loadtxt(os.path.join(DATA_DIR, "desi_gaussian_bao_ALL_GCcomb_mean.txt"), dtype=str, comments="#")
Z = _mean[:, 0].astype(float)
D = _mean[:, 1].astype(float)
KIND = _mean[:, 2]
COV = np.loadtxt(os.path.join(DATA_DIR, "desi_gaussian_bao_ALL_GCcomb_cov.txt"))
BLOCK_NAMES = ["BGS", "LRG1", "LRG2", "LRG3+ELG1", "ELG2", "QSO", "Lya"]
BLOCK_Z = [0.295, 0.51, 0.706, 0.934, 1.321, 1.484, 2.33]
BLOCKS = [np.where(np.isclose(Z, z))[0] for z in BLOCK_Z]
assert sum(len(b) for b in BLOCKS) == len(D)
PREC_BLOCKS = [np.linalg.inv(COV[np.ix_(b, b)]) for b in BLOCKS]
G = len(BLOCKS)

# Gauss-Legendre quadrature on segments between consecutive data redshifts
ZU = np.array(BLOCK_Z)
_edges = np.concatenate([[0.0], ZU])
_x, _w = np.polynomial.legendre.leggauss(24)
ZNODES = np.concatenate([0.5 * (b - a) * _x + 0.5 * (b + a) for a, b in zip(_edges[:-1], _edges[1:])])
WNODES = np.concatenate([0.5 * (b - a) * _w for a, b in zip(_edges[:-1], _edges[1:])])
NSEG = len(ZU)
ZPOS = np.array([int(np.argmin(np.abs(ZU - z))) for z in Z])  # data point -> segment end index

MODELS = {
    "LCDM": dict(names=["omega_b", "omega_c", "H0"],
                 lo=np.array([0.005, 0.01, 20.0]), hi=np.array([0.1, 0.99, 100.0])),
    "w0waCDM": dict(names=["omega_b", "omega_c", "H0", "w0", "wa"],
                    lo=np.array([0.005, 0.01, 20.0, -3.0, -3.0]),
                    hi=np.array([0.1, 0.99, 100.0, 1.0, 2.0])),
}


def theory(theta):
    """BAO predictions (..., 13) and r_d for theta (..., d), real or complex (for complex-step derivatives)."""
    theta = np.asarray(theta)
    wb, wc, H0 = theta[..., 0], theta[..., 1], theta[..., 2]
    d = theta.shape[-1]
    h = H0 / 100.0
    Om = (wb + wc + OMEGA_NU) / h ** 2
    Or = OMEGA_R / h ** 2
    Ode = 1.0 - Om - Or

    def Efun(z):
        zp = 1.0 + z
        if d == 5:
            w0, wa = theta[..., 3], theta[..., 4]
            fde = zp[None, :] ** (3 * (1 + w0[..., None] + wa[..., None])) * \
                np.exp(-3 * wa[..., None] * z[None, :] / zp[None, :])
        else:
            fde = np.ones_like(zp)[None, :]
        return np.sqrt(Or[..., None] * zp ** 4 + Om[..., None] * zp ** 3 + Ode[..., None] * fde)

    seg = (WNODES / Efun(ZNODES)).reshape(Om.shape + (NSEG, -1)).sum(-1)
    chi = np.cumsum(seg, axis=-1)[..., ZPOS]
    DH = C_KMS / H0[..., None] * (1.0 / Efun(Z))
    DM = C_KMS / H0[..., None] * chi
    DV = (Z * DM ** 2 * DH) ** (1.0 / 3.0)
    wcb = wb + wc
    rd = 55.154 * np.exp(-72.3 * (OMEGA_NU + 0.0006) ** 2) / (wcb ** 0.25351 * wb ** 0.12807)
    out = np.where(KIND == "DV_over_rs", DV, np.where(KIND == "DM_over_rs", DM, DH)) / rd[..., None]
    return out, rd


def loglike_blocks(theta):
    """Per-block Gaussian log-likelihoods, shape (..., G)."""
    M, _ = theory(theta)
    r = D - M
    return np.stack([-0.5 * np.einsum("...i,ij,...j->...", r[..., b], P, r[..., b])
                     for b, P in zip(BLOCKS, PREC_BLOCKS)], axis=-1)


def logprior(theta, model):
    lo, hi = MODELS[model]["lo"], MODELS[model]["hi"]
    inside = np.all((theta > lo) & (theta < hi), axis=-1)
    if theta.shape[-1] == 5:
        inside &= (theta[..., 3] + theta[..., 4]) < 0
    lp = -0.5 * ((theta[..., 0] - BBN_MEAN) / BBN_SD) ** 2
    return np.where(inside, lp, -np.inf)


def logpost(theta, model, keep):
    """Log posterior (up to a constant) keeping the blocks where the boolean mask `keep` (G,) is True."""
    lp = logprior(theta, model)
    ok = np.isfinite(lp)
    out = np.full(theta.shape[:-1], -np.inf)
    if ok.any():
        ll = loglike_blocks(theta[ok])
        out[ok] = lp[ok] + ll[..., keep].sum(-1)
    return out


def jacobian(theta, h=1e-20):
    """Complex-step Jacobian dM/dtheta, shape (N, 13, d)."""
    theta = np.atleast_2d(theta)
    N, d = theta.shape
    J = np.empty((N, len(D), d))
    for k in range(d):
        step = h * np.maximum(np.abs(theta[:, k]), 1e-3)
        tc = theta.astype(complex)
        tc[:, k] += 1j * step
        M, _ = theory(tc)
        J[:, :, k] = M.imag / step[:, None]
    return J


def block_scores(theta, batch_size=2000):
    """gamma_g(theta) = J_g^T Sigma_g^{-1} (D_g - M_g(theta)) for every block g, shape (N, G, d)."""
    theta = np.atleast_2d(theta)
    out = []
    for i in range(0, len(theta), batch_size):
        th = theta[i:i + batch_size]
        M, _ = theory(th)
        J = jacobian(th)
        r = D - M
        out.append(np.stack([np.einsum("nik,ij,nj->nk", J[:, b, :], P, r[:, b])
                             for b, P in zip(BLOCKS, PREC_BLOCKS)], axis=1))
    return np.concatenate(out)


# ---------------------------------------------------------------- sampler
def stretch_mcmc(logp, p0, nsteps, a=2.0, rng=None):
    """Vectorised affine-invariant ensemble sampler (Goodman & Weare 2010)."""
    rng = rng or np.random.default_rng(0)
    W, d = p0.shape
    x = p0.copy()
    lp = logp(x)
    chain = np.empty((nsteps, W, d))
    acc = 0
    half = W // 2
    for t in range(nsteps):
        for s, other in ((slice(0, half), slice(half, W)), (slice(half, W), slice(0, half))):
            xs, xo = x[s], x[other]
            n = xs.shape[0]
            zz = ((a - 1) * rng.random(n) + 1) ** 2 / a
            partners = xo[rng.integers(0, xo.shape[0], n)]
            prop = partners + zz[:, None] * (xs - partners)
            lpp = logp(prop)
            accept = np.log(rng.random(n)) < (d - 1) * np.log(zz) + lpp - lp[s]
            x[s] = np.where(accept[:, None], prop, xs)
            lp[s] = np.where(accept, lpp, lp[s])
            acc += accept.sum()
        chain[t] = x
    return chain, acc / (nsteps * W)


def run_posterior(model="LCDM", keep=None, nwalkers=64, nsteps=5000, burn=1500, thin=10, seed=0, p_init=None):
    """Sample the posterior keeping the blocks in `keep` (all by default); returns draws (N, d) and acceptance."""
    rng = np.random.default_rng(seed)
    keep = np.ones(G, bool) if keep is None else np.asarray(keep, bool)
    d = len(MODELS[model]["names"])
    if p_init is None:
        centre = np.array([0.0222, 0.12, 68.0, -1.0, 0.0][:d])
        scale = np.array([5e-4, 5e-3, 1.0, 0.1, 0.3][:d])
    else:
        centre, scale = p_init

    def lp(th):
        return logpost(th, model, keep)

    p0 = centre + scale * rng.standard_normal((nwalkers, d))
    for _ in range(1000):  # resample starting points until all are inside the prior
        bad = ~np.isfinite(lp(p0))
        if not bad.any():
            break
        p0[bad] = centre + scale * rng.standard_normal((bad.sum(), d))
    chain, acc = stretch_mcmc(lp, p0, nsteps, rng=rng)
    return chain[burn::thin].reshape(-1, d), acc
