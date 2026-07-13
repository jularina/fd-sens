from scipy.spatial.distance import pdist
from abc import ABC, abstractmethod
from scipy.integrate import nquad
import numpy as np
import sympy as sp
from typing import Optional, Literal, Sequence
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from numpy.linalg import eigh
from scipy.special import gamma, kv
from scipy.stats.qmc import Halton


class BaseBasisFunction(ABC):
    @abstractmethod
    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """Compute basis features φ(θ)."""
        pass

    @abstractmethod
    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """Compute ∇θ φ(θ)."""
        pass

    def check_C1(self, dim: int) -> bool:
        expr = self.symbolic_expression(dim)
        if expr is None:
            raise NotImplementedError("Symbolic expression not implemented.")
        variables = sp.symbols(f"x0:{dim}")
        try:
            for var in variables:
                sp.diff(expr, var)
            return True
        except Exception:
            return False

    def check_L2(self, dim: int) -> bool:
        def func(*args):
            x = np.array(args)[None, :]  # shape (1, d)
            try:
                val = self.evaluate(x)[0, 0, 0]  # just the first basis function at the first sample
                return float(val ** 2)
            except Exception:
                return np.inf

        try:
            region = [(-10, 10)] * dim
            result, _ = nquad(func, region)
            return np.isfinite(result)
        except Exception:
            return False


class PolynomialBasisFunction(BaseBasisFunction):
    def __init__(self, degree: int):
        self.degree = degree

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        m, d = samples.shape
        values = np.zeros((m, d, self.degree))

        for k in range(1, self.degree + 1):
            values[:, :, k - 1] = samples ** k

        return values  # (m, d, degree)

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        m, d = samples.shape
        grads = []

        for k in range(1, self.degree + 1):
            grad_k = k * samples ** (k - 1)  # (m, d)
            grad_block = np.zeros((m, d, d * self.degree))

            for dim in range(d):
                grad_block[:, dim, dim + d * (k - 1)] = grad_k[:, dim]

            grads.append(grad_block)

        return np.sum(grads, axis=0)  # (m, d, d * degree)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        x = sp.symbols(f"x0:{dim}")
        return sum([x[i] ** self.degree for i in range(dim)])


class MaternBasisFunction(BaseBasisFunction):
    r"""
    Matérn basis functions: φ_k(θ) = φ( ||θ - c_k|| ), with smoothness ν > 1.

    Shapes (to match your RBFBasisFunction interface):
      - evaluate(samples) -> (m, d, K)   (scalar φ replicated across d)
      - gradient(samples) -> (m, d, K)   (true ∇θ φ_k)

    Kernel (variance σ^2, lengthscale ℓ):
        φ(r) = σ^2 * 2^{1-ν} / Γ(ν) * (a r)^ν K_ν(a r),
        where a = sqrt(2ν) / ℓ, and K_ν is modified Bessel K.

    Radial derivative identity used:
        d/dr [ x^ν K_ν(x) ] = - x^ν K_{ν-1}(x).
    """

    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        lengthscale: Optional[float] = None,
        nu: float = 1.5,
        variance: float = 1.0,
        method: Literal["kmeans", "farthest", "halton"] = "kmeans",
        estimation_samples_source: Optional[str] = "prior",
        scale_multiplier: float = 1.0,
        B: Optional[float] = None,
        eps: float = 1e-12,
    ):
        if nu <= 1.0:
            raise ValueError(f"Need nu > 1 for C^1; got nu={nu}.")
        if variance <= 0:
            raise ValueError(f"variance must be > 0; got {variance}.")
        self.nu = float(nu)
        self.variance = float(variance)
        self.eps = float(eps)
        self.rng = np.random.default_rng(27)

        if estimation_samples_source == "prior":
            estimation_samples = prior_samples
        else:
            estimation_samples = posterior_samples

        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
        )

        # Optional check: centres in Θ_B
        self.B = None if B is None else float(B)
        if self.B is not None:
            norms = np.linalg.norm(self.centers, axis=1)
            if np.any(norms > self.B + 1e-10):
                raise ValueError(
                    "Some centres lie outside Θ_B. "
                    f"max ||c_k||={norms.max():.4g} > B={self.B:.4g}."
                )

        if lengthscale is None:
            if estimation_samples is None:
                raise ValueError(
                    "lengthscale=None but estimation_samples_source requires samples; "
                    "provide prior_samples or posterior_samples accordingly."
                )
            self.lengthscale = self._estimate_lengthscale(
                centers=self.centers,
                samples=estimation_samples,
                multiplier=scale_multiplier,
            )
        else:
            self.lengthscale = float(lengthscale)

        self._a = np.sqrt(2.0 * self.nu) / self.lengthscale
        self._prefactor = (2.0 ** (1.0 - self.nu)) / gamma(self.nu)  # scalar

    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
    ) -> np.ndarray:
        if prior_samples is not None:
            X = np.asarray(prior_samples, dtype=float)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, d = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        if method == "halton":
            return self._halton_centers(X, num_centers)

        raise ValueError(f"Unknown center selection method: {method}")

    def _halton_centers(self, X: np.ndarray, num_centers: int) -> np.ndarray:
        """
        Quasi-uniform centers via a scrambled Halton sequence mapped through
        the empirical quantiles of X (dimension-wise).

        Halton points u_k ∈ [0,1]^d are mapped as c_{k,j} = Q_j(u_{k,j}),
        where Q_j is the empirical quantile function of X[:,j].
        This preserves the O(K^{-1/d}) fill-distance guarantee of the
        Halton sequence while adapting centers to the data support.
        """
        d = X.shape[1]
        sampler = Halton(d=d, scramble=True, seed=27)
        u = sampler.random(n=num_centers)          # (K, d) in [0, 1]^d
        centers = np.empty((num_centers, d), dtype=float)
        for j in range(d):
            centers[:, j] = np.quantile(X[:, j], u[:, j])
        return centers

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    def _estimate_lengthscale(
        self,
        centers: np.ndarray,
        samples: np.ndarray,
        source: str = "samples",
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
    ) -> float:
        if centers.shape[0] < 2:
            return 1.0
        if source == "samples":
            m = np.median(pdist(samples.reshape(-1, samples.shape[-1])))
        else:
            m = np.median(pdist(centers))
        ell = multiplier * m
        floor = floor_frac * np.std(samples, axis=0).mean()
        ls = float(max(ell, floor))
        print(f"Selected lengthscale for Matérn basis function: {ls}.")
        return ls

    def _matern(self, r: np.ndarray) -> np.ndarray:
        """
        r: (...,) nonnegative distances
        returns φ(r) with φ(0)=variance
        """
        r = np.asarray(r, dtype=float)
        x = self._a * r

        out = np.empty_like(x, dtype=float)

        # For r=0, define φ(0)=σ^2.
        mask0 = x <= self.eps
        out[mask0] = self.variance

        # For r>0, use Matérn formula.
        xm = x[~mask0]
        # σ^2 * c * x^ν K_ν(x)
        out[~mask0] = (
            self.variance
            * self._prefactor
            * (xm ** self.nu)
            * kv(self.nu, xm)
        )
        return out

    def _matern_dr(self, r: np.ndarray) -> np.ndarray:
        """
        Radial derivative dφ/dr.

        For r>0:
          dφ/dr = -σ^2 * c * a * x^ν K_{ν-1}(x),   x=a r
        and define dφ/dr(0)=0.
        """
        r = np.asarray(r, dtype=float)
        x = self._a * r

        out = np.zeros_like(x, dtype=float)
        mask0 = x <= self.eps
        if np.any(~mask0):
            xm = x[~mask0]
            out[~mask0] = (
                -self.variance
                * self._prefactor
                * self._a
                * (xm ** self.nu)
                * kv(self.nu - 1.0, xm)
            )
        return out

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """
        Returns shape (m, d, K): φ(||θ-c_k||) replicated across d.
        """
        X = np.asarray(samples, dtype=float)  # (m,d)
        diffs = X[:, None, :] - self.centers[None, :, :]  # (m,K,d)
        r = np.linalg.norm(diffs, axis=-1)  # (m,K)
        vals = self._matern(r)  # (m,K)

        m, d = X.shape
        return np.broadcast_to(vals[:, None, :], (m, d, vals.shape[1])).copy()

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """
        ∇θ φ_k(θ) = (dφ/dr)(r) * (θ - c_k) / r, with 0 at r=0.
        Returns shape (m, d, K).
        """
        X = np.asarray(samples, dtype=float)  # (m,d)
        diffs = X[:, None, :] - self.centers[None, :, :]  # (m,K,d)
        r = np.linalg.norm(diffs, axis=-1)  # (m,K)

        dphi = self._matern_dr(r)  # (m,K)

        # safe division by r
        inv_r = np.zeros_like(r)
        mask = r > self.eps
        inv_r[mask] = 1.0 / r[mask]

        # (m,K,d) = (m,K,1) * (m,K,1) * (m,K,d)
        grad = (dphi[:, :, None] * inv_r[:, :, None]) * diffs  # (m,K,d)
        return np.transpose(grad, (0, 2, 1))  # (m,d,K)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        """
        Symbolic expression for φ(||x-c||) using the first centre c_1.
        Note: Sympy uses besselk for K_ν.
        """
        x = sp.symbols(f"x0:{dim}")
        c = np.asarray(self.centers[0], dtype=float)
        nu = sp.Float(self.nu)
        sig2 = sp.Float(self.variance)
        ell = sp.Float(self.lengthscale)

        r = sp.sqrt(sum((x[i] - sp.Float(c[i])) ** 2 for i in range(dim)))
        a = sp.sqrt(2 * nu) / ell
        z = a * r
        pref = 2 ** (1 - nu) / sp.gamma(nu)

        # Define φ(0)=σ^2; symbolic expression won’t special-case r=0, but differentiability checks
        # in sympy typically still work for ν>1.
        return sig2 * pref * (z ** nu) * sp.besselk(nu, z)


class MaternBasisFunctionMultidim(BaseBasisFunction):
    r"""
    Multidim Matérn basis with either per-dimension (diag) or full (non-diagonal) metric.

    Two modes (mirrors your RBFBasisFunctionMultidim API):

    metric="diag":
      Per-dimension (1D) Matérn features:
        phi_{i,b}(x) = Matérn_nu( |x_i - c_{b,i}| / l_i )
      evaluate(samples) -> (m, d, B)
      gradient(samples) -> (m, d, B) with 1D derivative wrt x_i

    metric="full":
      Scalar Matérn feature using Mahalanobis distance:
        r_b(x) = sqrt( (x - c_b)^T P (x - c_b) )
        phi_b(x) = Matérn_nu( r_b(x) )
      evaluate(samples) -> (m, d, B) by repeating scalar across d
      gradient(samples) -> (m, d, B): ∇_x phi_b = (dphi/dr) * P(x-c_b)/r_b

    Matérn (scaled-distance form):
      k(r) = σ^2 * 2^{1-ν}/Γ(ν) * (x^ν K_ν(x)),  x = sqrt(2ν) r

    Radial derivative identity:
      d/dr [ x^ν K_ν(x) ] = - sqrt(2ν) * x^ν K_{ν-1}(x)   (chain rule via x = sqrt(2ν) r)

    Notes:
      - You need ν > 1 if you want C^1 basis functions (your Assumption A1).
      - In metric="full", P plays the role of a precision / inverse lengthscale matrix.
    """

    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        lengthscale: Optional[np.ndarray] = None,      # (d,) for diag
        precision: Optional[np.ndarray] = None,        # (d,d) for full
        metric: Literal["diag", "full"] = "diag",
        nu: float = 1.5,
        variance: float = 1.0,
        method: Literal["kmeans", "farthest", "halton", "quantile_grid"] = "kmeans",
        estimation_samples_source: Optional[str] = "prior",  # "prior" or "posterior"
        estimation_centers_source: Optional[str] = "prior",
        scale_multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
        eps: float = 1e-12,
    ):
        if nu <= 1.0:
            raise ValueError(f"Need nu > 1 for C^1; got nu={nu}.")
        if variance <= 0.0:
            raise ValueError(f"variance must be > 0; got {variance}.")

        self.rng = np.random.default_rng(27)
        self.metric = metric
        self.nu = float(nu)
        self.variance = float(variance)
        self.eps = float(eps)

        # Matérn constants for scaled-distance form (x = sqrt(2ν) r)
        self._sqrt_2nu = float(np.sqrt(2.0 * self.nu))
        self._prefactor = float((2.0 ** (1.0 - self.nu)) / gamma(self.nu))

        if estimation_samples_source == "prior":
            estimation_samples = prior_samples
        elif estimation_samples_source == "min_scale":
            estimation_samples = self._pick_min_scale_samples(prior_samples, posterior_samples)
        elif estimation_samples_source == "both":
            estimation_samples = self._concat_samples(prior_samples, posterior_samples)
        else:
            estimation_samples = posterior_samples

        if estimation_centers_source == "min_scale":
            estimation_centers_source_resolved = "prior" if estimation_samples is prior_samples else "posterior"
        elif estimation_centers_source == "both":
            estimation_centers_source_resolved = "both"
        else:
            estimation_centers_source_resolved = estimation_centers_source

        # Choose centers (B, d)
        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
            estimation_centers_source=estimation_centers_source_resolved,
        )
        _, self.dim = self.centers.shape
        self.num_basis = int(self.centers.shape[0])

        if self.metric == "diag":
            if lengthscale is None:
                if estimation_samples is None:
                    ls = self._estimate_lengthscale_vector_from_centers(
                        centers=self.centers,
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
                else:
                    ls = self._estimate_lengthscale_vector_from_samples(
                        samples=np.asarray(estimation_samples, dtype=float),
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
            else:
                ls = np.asarray(lengthscale, dtype=float)
                if ls.ndim != 1 or ls.shape[0] != self.dim:
                    raise ValueError(f"lengthscale must be shape (d,), got {ls.shape}.")
                ls = np.maximum(ls, jitter)

            self.lengthscale = ls.astype(float)
            self.precision = None

        elif self.metric == "full":
            if precision is None:
                if estimation_samples is None:
                    P = self._estimate_precision_from_centers(
                        centers=self.centers,
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
                else:
                    P = self._estimate_precision_from_samples(
                        samples=np.asarray(estimation_samples, dtype=float),
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
            else:
                P = np.asarray(precision, dtype=float)
                if P.shape != (self.dim, self.dim):
                    raise ValueError(f"precision must be (d,d), got {P.shape}.")
                P = 0.5 * (P + P.T)
                w, V = eigh(P)
                w = np.maximum(w, jitter)
                P = (V * w) @ V.T

            self.precision = 0.5 * (P + P.T)
            self.lengthscale = None

        else:
            raise ValueError("metric must be 'diag' or 'full'.")

        # Fill distance estimate over all available samples
        all_samples = self._concat_samples(prior_samples, posterior_samples)
        dists = cdist(all_samples, self.centers)          # (n, B)
        fill_distance = float(dists.min(axis=1).max())
        print(f"Fill distance estimate (h): {fill_distance:.4f}")

        # Print estimated lengthscale
        if self.metric == "diag":
            print(f"Estimated lengthscales (per dim): {np.array2string(self.lengthscale, precision=4)}")
        else:
            print(f"Estimated precision matrix P:\n{np.array2string(self.precision, precision=4)}")

    # ---------------- scale selection ----------------
    @staticmethod
    def _concat_samples(
        prior_samples: Optional[np.ndarray],
        posterior_samples: Optional[np.ndarray],
    ) -> np.ndarray:
        if prior_samples is None:
            return np.asarray(posterior_samples, dtype=float)
        if posterior_samples is None:
            return np.asarray(prior_samples, dtype=float)
        return np.concatenate([
            np.asarray(prior_samples, dtype=float),
            np.asarray(posterior_samples, dtype=float),
        ], axis=0)

    @staticmethod
    def _pick_min_scale_samples(
        prior_samples: Optional[np.ndarray],
        posterior_samples: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return whichever sample set has the smaller mean variance (trace of cov / d)."""
        if prior_samples is None:
            return np.asarray(posterior_samples, dtype=float)
        if posterior_samples is None:
            return np.asarray(prior_samples, dtype=float)
        prior_arr = np.asarray(prior_samples, dtype=float)
        post_arr = np.asarray(posterior_samples, dtype=float)
        prior_scale = float(np.mean(np.var(prior_arr, axis=0)))
        post_scale = float(np.mean(np.var(post_arr, axis=0)))
        return post_arr if post_scale <= prior_scale else prior_arr

    # ---------------- center selection ----------------
    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
        estimation_centers_source: str,
    ) -> np.ndarray:
        if estimation_centers_source == "prior":
            X = np.asarray(prior_samples, dtype=float)
        elif estimation_centers_source == "both":
            X = self._concat_samples(prior_samples, posterior_samples)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, _ = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        if method == "halton":
            return self._halton_centers(X, num_centers)

        if method == "quantile_grid":
            return self._select_centers_quantile_grid(X, num_centers=num_centers)

        raise ValueError(f"Unknown center selection method: {method}")

    def _select_centers_quantile_grid(
            self,
            X: np.ndarray,
            num_centers: int,
            q_low: float = 0.05,
            q_high: float = 0.95,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        if d != 2:
            raise ValueError("quantile_grid currently implemented for d=2 (ECMO).")

        # pick grid sizes close to sqrt(num_centers)
        g1 = int(np.floor(np.sqrt(num_centers)))
        g2 = int(np.ceil(num_centers / g1))

        qx = np.linspace(q_low, q_high, g1)
        qy = np.linspace(q_low, q_high, g2)

        xs = np.quantile(X[:, 0], qx)
        ys = np.quantile(X[:, 1], qy)

        Gx, Gy = np.meshgrid(xs, ys, indexing="xy")
        C = np.column_stack([Gx.ravel(), Gy.ravel()])

        # trim if too many
        return C[:num_centers]

    def _halton_centers(self, X: np.ndarray, num_centers: int) -> np.ndarray:
        """
        Quasi-uniform centers via a scrambled Halton sequence mapped through
        the empirical quantiles of X (dimension-wise).
        """
        d = X.shape[1]
        sampler = Halton(d=d, scramble=True, seed=27)
        u = sampler.random(n=num_centers)
        centers = np.empty((num_centers, d), dtype=float)
        for j in range(d):
            centers[:, j] = np.quantile(X[:, j], u[:, j])
        return centers

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    # ---------------- estimation helpers ----------------
    def _median_heuristic_per_dim(self, x: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
        """
        Per-dimension median heuristic:
          l_i = sqrt(median_{p<q} (x_{p,i} - x_{q,i})^2 + jitter)
        x: (n, d)
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be (n,d).")
        n, d = x.shape
        if n < 2:
            return np.sqrt(np.var(x, axis=0) + jitter)

        diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
        iu = np.triu_indices(n, k=1)
        diffs = diffs[iu]                      # (n*(n-1)/2, d)
        med_sq = np.median(diffs**2, axis=0)    # (d,)
        return np.sqrt(med_sq + jitter)

    def _estimate_lengthscale_vector_from_samples(
        self,
        samples: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-12,
    ) -> np.ndarray:
        ls = self._median_heuristic_per_dim(samples, jitter=jitter)
        floor = floor_frac * np.std(samples, axis=0)
        ls = np.maximum(multiplier * ls, floor)
        return np.maximum(ls, jitter).astype(float)

    def _estimate_lengthscale_vector_from_centers(
        self,
        centers: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-12,
    ) -> np.ndarray:
        ls = self._median_heuristic_per_dim(centers, jitter=jitter)
        floor = floor_frac * np.std(centers, axis=0)
        ls = np.maximum(multiplier * ls, floor)
        return np.maximum(ls, jitter).astype(float)

    def _estimate_precision_from_samples(
        self,
        samples: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ) -> np.ndarray:
        """
        Estimate shared precision P ≻ 0:
          - compute covariance Σ
          - floor eigenvalues
          - invert, then scale by 1/multiplier^2
        """
        X = np.asarray(samples, dtype=float)
        Σ = np.cov(X, rowvar=False)
        Σ = 0.5 * (Σ + Σ.T)

        w, V = eigh(Σ)
        mean_eig = float(np.mean(np.maximum(w, 0.0)))
        w_reg = np.maximum(w, floor_frac * mean_eig + jitter)
        w_inv = 1.0 / w_reg
        Σ_inv = (V * w_inv) @ V.T

        P = Σ_inv / (multiplier**2)
        return 0.5 * (P + P.T)

    def _estimate_precision_from_centers(
        self,
        centers: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ) -> np.ndarray:
        return self._estimate_precision_from_samples(
            samples=centers,
            multiplier=multiplier,
            floor_frac=floor_frac,
            jitter=jitter,
        )

    # ---------------- Matérn core in scaled-distance form ----------------
    def _matern_scaled(self, r: np.ndarray) -> np.ndarray:
        """
        r: (...,) scaled distance (dimensionless). Uses x = sqrt(2ν) r.
        Returns k(r) with k(0)=variance.
        """
        r = np.asarray(r, dtype=float)
        x = self._sqrt_2nu * r

        out = np.empty_like(x, dtype=float)
        mask0 = x <= self.eps
        out[mask0] = self.variance

        xm = x[~mask0]
        out[~mask0] = (
            self.variance
            * self._prefactor
            * (xm ** self.nu)
            * kv(self.nu, xm)
        )
        return out

    def _matern_scaled_dr(self, r: np.ndarray) -> np.ndarray:
        """
        d/dr k(r) in scaled-distance form.
        For r>0:
          dk/dr = -σ^2 * c * sqrt(2ν) * x^ν K_{ν-1}(x),  x = sqrt(2ν) r
        with dk/dr(0)=0.
        """
        r = np.asarray(r, dtype=float)
        x = self._sqrt_2nu * r

        out = np.zeros_like(x, dtype=float)
        mask0 = x <= self.eps
        if np.any(~mask0):
            xm = x[~mask0]
            out[~mask0] = (
                -self.variance
                * self._prefactor
                * self._sqrt_2nu
                * (xm ** self.nu)
                * kv(self.nu - 1.0, xm)
            )
        return out

    # ---------------- API: evaluate / gradient ----------------
    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """
        metric="diag": returns (m, d, B) with per-dim Matérn.
        metric="full": returns (m, d, B) by repeating scalar across d.
        """
        X = np.asarray(samples, dtype=float)  # (m,d)
        m, d = X.shape
        if d != self.dim:
            raise ValueError(f"Sample dim {d} != center dim {self.dim}.")

        if self.metric == "diag":
            # diffs: (m,B,d)
            diffs = X[:, None, :] - self.centers[None, :, :]
            r = np.abs(diffs) / self.lengthscale[None, None, :]  # (m,B,d)
            vals = self._matern_scaled(r)                         # (m,B,d)
            return np.transpose(vals, (0, 2, 1))                  # (m,d,B)

        # full
        diffs = X[:, None, :] - self.centers[None, :, :]          # (m,B,d)
        r2 = np.einsum("mbi,ij,mbj->mb", diffs, self.precision, diffs, optimize=True)
        r = np.sqrt(np.maximum(r2, 0.0))                          # (m,B)
        phi = self._matern_scaled(r)                              # (m,B)
        return np.repeat(phi[:, None, :], d, axis=1)              # (m,d,B)

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """
        metric="diag":
          phi_{i,b}(x) = k( |x_i - c_{b,i}| / l_i )
          d/dx_i phi_{i,b}(x) = k'(r) * sign(x_i - c_{b,i}) / l_i   (with 0 at diff=0)

        metric="full":
          r_b(x)=sqrt((x-c)^T P (x-c))
          ∇ phi_b = k'(r) * P(x-c)/r, with 0 at r=0

        Returns: (m, d, B)
        """
        X = np.asarray(samples, dtype=float)
        m, d = X.shape
        if d != self.dim:
            raise ValueError(f"Sample dim {d} != center dim {self.dim}.")

        if self.metric == "diag":
            diffs = X[:, None, :] - self.centers[None, :, :]           # (m,B,d)
            absdiff = np.abs(diffs)
            r = absdiff / self.lengthscale[None, None, :]              # (m,B,d)
            dkdr = self._matern_scaled_dr(r)                            # (m,B,d)

            # sign(diff) safely (0 at diff=0)
            sign = diffs / np.maximum(absdiff, self.eps)               # (m,B,d)
            drdx = sign / self.lengthscale[None, None, :]              # (m,B,d)

            grad = dkdr * drdx                                         # (m,B,d)
            return np.transpose(grad, (0, 2, 1))                       # (m,d,B)

        # full
        diffs = X[:, None, :] - self.centers[None, :, :]               # (m,B,d)
        Px = np.einsum("ij,mbj->mbi", self.precision, diffs, optimize=True)  # (m,B,d)
        r2 = np.einsum("mbi,ij,mbj->mb", diffs, self.precision, diffs, optimize=True)  # (m,B)
        r = np.sqrt(np.maximum(r2, 0.0))                               # (m,B)

        dkdr = self._matern_scaled_dr(r)                               # (m,B)

        inv_r = np.zeros_like(r)
        mask = r > self.eps
        inv_r[mask] = 1.0 / r[mask]

        grad = (dkdr[:, :, None] * inv_r[:, :, None]) * Px             # (m,B,d)
        return np.transpose(grad, (0, 2, 1))                           # (m,d,B)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        """
        Symbolic form for the first centre (b=0).
        - diag: Tuple of per-dim Matérn expressions
        - full: scalar Matérn(r) repeated in a Tuple length d (shape-compat)
        """
        x = sp.symbols(f"x0:{dim}")
        c = np.asarray(self.centers[0], dtype=float)

        nu = sp.Float(self.nu)
        sig2 = sp.Float(self.variance)
        pref = sp.Float(self._prefactor)
        s2nu = sp.sqrt(2 * nu)

        if self.metric == "diag":
            ls = np.asarray(self.lengthscale, dtype=float)
            exprs = []
            for i in range(dim):
                r_i = sp.Abs(x[i] - sp.Float(c[i])) / sp.Float(ls[i])
                z = s2nu * r_i
                exprs.append(sig2 * pref * (z**nu) * sp.besselk(nu, z))
            return sp.Tuple(*exprs)

        # full
        P = sp.Matrix(np.asarray(self.precision, dtype=float))
        xm = sp.Matrix(x)
        cm = sp.Matrix([sp.Float(ci) for ci in c.tolist()])
        r = sp.sqrt(((xm - cm).T * P * (xm - cm))[0])
        z = s2nu * r
        expr = sig2 * pref * (z**nu) * sp.besselk(nu, z)
        return sp.Tuple(*[expr for _ in range(dim)])


class RBFBasisFunction(BaseBasisFunction):
    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        lengthscale: Optional[float] = None,
        method: Literal["kmeans", "farthest"] = "kmeans",
        estimation_samples_source: Optional[str] = "prior",
        scale_multiplier: float = 1.0,
    ):
        self.rng = np.random.default_rng(27)

        if estimation_samples_source == "prior":
            estimation_samples = prior_samples
        else:
            estimation_samples = posterior_samples

        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
        )

        if lengthscale is None:
            self.lengthscale = self._estimate_lengthscale(
                centers=self.centers,
                samples=estimation_samples,
                multiplier=scale_multiplier,
            )
        else:
            self.lengthscale = float(lengthscale)

    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
    ) -> np.ndarray:
        if prior_samples is not None:
            X = np.asarray(prior_samples, dtype=float)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, d = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        raise ValueError(f"Unknown center selection method: {method}")

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    def _estimate_lengthscale(self, centers: np.ndarray, samples: np.ndarray,
                              source: str = "samples", multiplier: float = 1.0,
                              floor_frac: float = 0.1) -> float:
        if centers.shape[0] < 2:
            return 1.0
        if source == "samples":
            m = np.median(pdist(samples.reshape(-1, samples.shape[-1])))
        else:
            m = np.median(pdist(centers))

        ell = multiplier * m
        floor = floor_frac * np.std(samples, axis=0).mean()
        ls = float(max(ell, floor))
        print(f"Selected lengthscale for RBF basis function: {ls}.")

        return ls

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        diffs = samples[:, None, :] - self.centers[None, :, :]
        squared = diffs ** 2
        squared = np.transpose(squared, (0, 2, 1))

        return np.exp(-squared / (2 * self.lengthscale ** 2))  # (m, d, B)

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        diffs = samples[:, None, :] - self.centers[None, :, :]  # (m, B, d)
        rbf_vals = np.exp(-np.sum(diffs ** 2, axis=-1) / (2 * self.lengthscale ** 2))  # (m, B)
        grad = (-1 / self.lengthscale ** 2) * (diffs * rbf_vals[:, :, None])  # (m, B, d)

        return np.transpose(grad, (0, 2, 1))  # (m, d, B)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        x = sp.symbols(f"x0:{dim}")
        c = self.centers[0]  # just validate the first center
        return sp.exp(-sum([(x[i] - c[i])**2 for i in range(dim)]) / (2 * self.lengthscale**2))


class SigmoidBasisFunction(BaseBasisFunction):
    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
        method: Literal["kmeans", "farthest"] = "kmeans",
        estimation_samples_source: Optional[str] = "prior",
        scale_multiplier: float = 1.0,
    ):
        self.rng = np.random.default_rng(None)

        if estimation_samples_source == "prior":
            estimation_samples = prior_samples
        else:
            estimation_samples = posterior_samples

        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
        )

        if scale is None:
            self.scale = self._estimate_scale(
                centers=self.centers,
                samples=estimation_samples,
                multiplier=scale_multiplier,
            )
            print(f"Selected scale for Sigmoid basis function: {self.scale}.")
        else:
            self.scale = float(scale)

    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
    ) -> np.ndarray:
        if prior_samples is not None:
            X = np.asarray(prior_samples, dtype=float)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, d = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        raise ValueError(f"Unknown center selection method: {method}")

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    def _estimate_scale(
        self,
        centers: np.ndarray,
        samples: np.ndarray,
        *,
        source: str = "samples",
        multiplier: float = 0.01,
        floor_frac: float = 0.1,
    ) -> float:
        if centers.shape[0] < 2:
            return 1.0
        if source == "samples":
            m = np.median(pdist(samples.reshape(-1, samples.shape[-1])))
        else:
            m = np.median(pdist(centers))
        s = multiplier * m
        floor = floor_frac * np.std(samples, axis=0).mean()
        return float(max(s, floor))

    @staticmethod
    def _sigmoid(u: np.ndarray) -> np.ndarray:
        # stable logistic
        return 1.0 / (1.0 + np.exp(-u))

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """
        Returns φ(x) with shape (m, d, B),
        where the last axis indexes basis functions.
        """
        # diffs: (m, B, d)
        diffs = samples[:, None, :] - self.centers[None, :, :]
        u = diffs / self.scale                       # (m, B, d)
        vals = self._sigmoid(u)                      # (m, B, d)
        return np.transpose(vals, (0, 2, 1))         # (m, d, B)

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """
        ∂φ/∂x: shape (m, d, B).
        d/dx σ((x-c)/s) = (1/s) * σ(u)*(1-σ(u)) component-wise.
        """
        diffs = samples[:, None, :] - self.centers[None, :, :]  # (m, B, d)
        u = diffs / self.scale                                  # (m, B, d)
        sig = self._sigmoid(u)
        d_sig_du = sig * (1.0 - sig)                            # (m, B, d)
        grad_mBd = (1.0 / self.scale) * d_sig_du                # (m, B, d)
        return np.transpose(grad_mBd, (0, 2, 1))                # (m, d, B)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        x = sp.symbols(f"x0:{dim}")
        c = self.centers[0]
        s = sp.Float(self.scale)

        return 1 / (1 + sp.exp(-(sum((x[i] - c[i]) for i in range(dim)) / s)))


class RBFBasisFunctionMultidim(BaseBasisFunction):
    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        lengthscale: Optional[np.ndarray] = None,
        precision: Optional[np.ndarray] = None,
        metric: Literal["diag", "full"] = "diag",
        method: Literal["kmeans", "farthest"] = "kmeans",
        estimation_samples_source: Optional[str] = "prior",  # "prior" or "posterior"
        scale_multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ):
        """
        Multidim RBF basis with either per-dimension (diag) or full (non-diagonal) bandwidth.

        metric="diag":
          phi_{i,b}(x) = exp( - (x_i - c_{b,i})^2 / (2 l_i^2) ), per-dimension separable basis.
          evaluate(samples) -> (m, d, B)
          gradient(samples) -> (m, d, B) with per-dim formula.

        metric="full":
          phi_b(x)       = exp( -1/2 (x - c_b)^T P (x - c_b) ), with shared precision matrix P ≻ 0.
          evaluate       -> (m, d, B) by repeating the scalar across d to keep shape compatibility.
          gradient       -> (m, d, B) using full-matrix derivative: -P(x-c_b) * phi_b(x)
        """
        self.rng = np.random.default_rng(None)
        self.metric = metric

        # Pick samples that control the bandwidth/precision estimation
        if estimation_samples_source == "prior":
            estimation_samples = prior_samples
        else:
            estimation_samples = posterior_samples

        # Choose centers (B, d)
        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
        )
        _, self.dim = self.centers.shape
        self.num_basis = num_basis_functions

        if self.metric == "diag":
            if lengthscale is None:
                if estimation_samples is None:
                    ls = self._estimate_lengthscale_vector_from_centers(
                        centers=self.centers,
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                    )
                else:
                    ls = self._estimate_lengthscale_vector_from_samples(
                        samples=np.asarray(estimation_samples, dtype=float),
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                    )
            else:
                ls = np.asarray(lengthscale, dtype=float)
                if ls.ndim != 1 or ls.shape[0] != self.dim:
                    raise ValueError(f"lengthscale must be shape (d,), got {ls.shape}.")
            self.lengthscale = ls
            self.precision = None
            ls_str = np.array2string(self.lengthscale, precision=4, separator=", ")
            print(f"[RBF diag] lengthscales l: {ls_str}")

        elif self.metric == "full":
            if precision is None:
                if estimation_samples is None:
                    P = self._estimate_precision_from_centers(
                        centers=self.centers,
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
                else:
                    P = self._estimate_precision_from_samples(
                        samples=np.asarray(estimation_samples, dtype=float),
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
            else:
                P = np.asarray(precision, dtype=float)
                if P.shape != (self.dim, self.dim):
                    raise ValueError(f"precision must be (d,d), got {P.shape}.")
                P = 0.5 * (P + P.T)
                w, V = eigh(P)
                w = np.maximum(w, jitter)
                P = (V * w) @ V.T

            self.precision = P
            self.lengthscale = None
            w, _ = eigh(self.precision)
            w_str = np.array2string(w, precision=4, separator=", ")
            print(f"[RBF full] precision eigvals: {w_str}")
        else:
            raise ValueError("metric must be 'diag' or 'full'.")

    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
    ) -> np.ndarray:
        if prior_samples is not None:
            X = np.asarray(prior_samples, dtype=float)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, d = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        raise ValueError(f"Unknown center selection method: {method}")

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    # ---------- diagonal (per-dim) lengthscale estimation ----------
    def _median_heuristic_per_dim(self, x: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
        """
        Per-dimension median heuristic:
            l_i = sqrt(median_{i<j} (x_i - x_j)^2 + jitter)
        x: (n, d)
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("reference_data must be a 2D array of shape (n, d).")
        n, d = x.shape
        if n < 2:
            return np.sqrt(np.var(x, axis=0) + jitter)

        diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
        iu = np.triu_indices(n, k=1)
        diffs = diffs[iu]                       # (n*(n-1)/2, d)
        med_sq = np.median(diffs**2, axis=0)    # (d,)
        return np.sqrt(med_sq + jitter)         # (d,)

    def _estimate_lengthscale_vector_from_samples(
        self,
        samples: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
    ) -> np.ndarray:
        ls = self._median_heuristic_per_dim(samples)
        floor = floor_frac * np.std(samples, axis=0)
        ls = np.maximum(multiplier * ls, floor)
        return ls.astype(float)

    def _estimate_lengthscale_vector_from_centers(
        self,
        centers: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
    ) -> np.ndarray:
        ls = self._median_heuristic_per_dim(centers)
        floor = floor_frac * np.std(centers, axis=0)
        ls = np.maximum(multiplier * ls, floor)
        return ls.astype(float)

    def _estimate_precision_from_samples(
        self,
        samples: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ) -> np.ndarray:
        """
        Estimate a shared precision matrix P ≻ 0:
          - compute covariance Σ from samples
          - regularize eigenvalues (floor_frac * mean_eig + jitter)
          - invert and scale: P = (1 / multiplier^2) * Σ^{-1}
        """
        X = np.asarray(samples, dtype=float)
        Σ = np.cov(X, rowvar=False)  # (d,d)
        Σ = 0.5 * (Σ + Σ.T)

        w, V = eigh(Σ)
        mean_eig = float(np.mean(np.maximum(w, 0.0)))
        w_reg = np.maximum(w, floor_frac * mean_eig + jitter)
        w_inv = 1.0 / w_reg
        Σ_inv = (V * w_inv) @ V.T
        P = Σ_inv / (multiplier**2)

        return 0.5 * (P + P.T)

    def _estimate_precision_from_centers(
        self,
        centers: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ) -> np.ndarray:
        return self._estimate_precision_from_samples(
            samples=centers,
            multiplier=multiplier,
            floor_frac=floor_frac,
            jitter=jitter,
        )

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """
        metric="diag":
            phi_{i,b}(x) = exp( - (x_i - c_{b,i})^2 / (2 l_i^2) )
            return shape: (m, d, B)
        metric="full":
            phi_b(x)     = exp( -1/2 (x - c_b)^T P (x - c_b) ), shared P
            return shape: (m, d, B) with the scalar repeated along d to match downstream API
        """
        X = np.asarray(samples, dtype=float)                # (m, d)
        d = self.dim

        if self.metric == "diag":
            # diffs: (m, B, d)
            diffs = X[:, None, :] - self.centers[None, :, :]
            denom = 2.0 * (self.lengthscale**2)[None, None, :]  # (1,1,d)
            vals = np.exp(-(diffs**2) / denom)                  # (m, B, d)
            return np.transpose(vals, (0, 2, 1))                # (m, d, B)

        diffs = X[:, None, :] - self.centers[None, :, :]
        r2 = np.einsum("mbi,ij,mbj->mb", diffs, self.precision, diffs, optimize=True)
        phi = np.exp(-0.5 * r2)                              # (m, B)
        phi_rep = np.repeat(phi[:, None, :], d, axis=1)      # (m, d, B)
        return phi_rep

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """
        metric="diag":
            d/dx_i phi_{i,b}(x) = -(x_i - c_{b,i}) / l_i^2 * phi_{i,b}(x)
        metric="full":
            ∇_x phi_b(x) = -P (x - c_b) * phi_b(x)
        Returns shape (m, d, B)
        """
        X = np.asarray(samples, dtype=float)                # (m, d)

        if self.metric == "diag":
            diffs = X[:, None, :] - self.centers[None, :, :]                 # (m, B, d)
            denom = 2.0 * (self.lengthscale**2)[None, None, :]               # (1,1,d)
            phi = np.exp(-(diffs**2) / denom)                                # (m, B, d)
            factor = -diffs / (self.lengthscale**2)[None, None, :]           # (m, B, d)
            grad = factor * phi                                              # (m, B, d)
            return np.transpose(grad, (0, 2, 1))                             # (m, d, B)

        # metric == "full"
        diffs = X[:, None, :] - self.centers[None, :, :]                     # (m, B, d)
        # compute P (x - c_b) for all b: (d,d) @ (m,B,d)^T -> (m,B,d)
        Px = np.einsum("ij,mbj->mbi", self.precision, diffs, optimize=True)  # (m, B, d)
        r2 = np.einsum("mbi,ij,mbj->mb", diffs, self.precision, diffs, optimize=True)  # (m,B)
        phi = np.exp(-0.5 * r2)                                              # (m, B)
        grad = -Px * phi[:, :, None]                                         # (m, B, d)
        return np.transpose(grad, (0, 2, 1))                                 # (m, d, B)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        """
        Returns a symbolic form for the FIRST center b=0.
        metric="diag": vector of per-dim terms exp( - (x_i - c_i)^2 / (2 l_i^2) )
        metric="full": scalar exp( -1/2 (x - c)^T P (x - c) ) replicated conceptually across dims
        """
        x = sp.symbols(f"x0:{dim}")
        c = self.centers[0]
        if self.metric == "diag":
            exprs = [
                sp.exp(-((x[i] - c[i])**2) / (2 * (self.lengthscale[i]**2)))
                for i in range(dim)
            ]
            return sp.Tuple(*exprs)
        # full
        P = sp.Matrix(self.precision)
        xm = sp.Matrix(x)
        cm = sp.Matrix(c)
        quad = (xm - cm).T * P * (xm - cm)
        expr = sp.exp(-sp.Rational(1, 2) * quad[0])

        return sp.Tuple(*[expr for _ in range(dim)])


class SigmoidBasisFunctionMultidim(BaseBasisFunction):
    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,              # (d,) for diag
        precision: Optional[np.ndarray] = None,          # (d,d) for full
        metric: Literal["diag", "full"] = "diag",
        method: Literal["kmeans", "farthest"] = "kmeans",
        estimation_samples_source: Optional[str] = "prior",  # "prior" or "posterior"
        scale_multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ):
        """
        Multidimensional sigmoid basis (logistic).

        metric="diag":
          φ_{i,b}(x) = σ((x_i - c_{b,i}) / s_i), separable per-dimension basis.
          evaluate(samples) -> (m, d, B)
          gradient(samples) -> (m, d, B) with (1/s_i) σ(u)(1-σ(u)).

        metric="full":
          φ_b(x) = σ( -1/2 (x - c_b)^T P (x - c_b) ), shared P ≻ 0.
          evaluate -> (m, d, B) by repeating the scalar across d to match downstream API.
          gradient -> (m, d, B) via chain rule: σ(u)(1-σ(u)) * ( -P (x - c_b) ).
        """
        self.rng = np.random.default_rng(None)
        self.metric = metric

        # Which samples to use for scale/precision estimation
        estimation_samples = prior_samples if estimation_samples_source == "prior" else posterior_samples

        # Choose centers (B, d)
        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
        )
        _, self.dim = self.centers.shape
        self.num_basis = num_basis_functions

        if self.metric == "diag":
            # per-dimension scales s_i
            if scale is None:
                if estimation_samples is None:
                    s = self._estimate_scale_vector_from_centers(
                        centers=self.centers,
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                    )
                else:
                    s = self._estimate_scale_vector_from_samples(
                        samples=np.asarray(estimation_samples, dtype=float),
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                    )
            else:
                s = np.asarray(scale, dtype=float)
                if s.ndim != 1 or s.shape[0] != self.dim:
                    raise ValueError(f"scale must be shape (d,), got {s.shape}.")
            self.scale = s
            self.precision = None
            s_str = np.array2string(self.scale, precision=4, separator=", ")
            print(f"[Sigmoid diag] scales s: {s_str}")

        elif self.metric == "full":
            if precision is None:
                if estimation_samples is None:
                    P = self._estimate_precision_from_centers(
                        centers=self.centers,
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
                else:
                    P = self._estimate_precision_from_samples(
                        samples=np.asarray(estimation_samples, dtype=float),
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
            else:
                P = np.asarray(precision, dtype=float)
                if P.shape != (self.dim, self.dim):
                    raise ValueError(f"precision must be (d,d), got {P.shape}.")
                P = 0.5 * (P + P.T)
                w, V = eigh(P)
                w = np.maximum(w, jitter)
                P = (V * w) @ V.T

            self.precision = P
            self.scale = None
            w, _ = eigh(self.precision)
            w_str = np.array2string(w, precision=4, separator=", ")
            print(f"[Sigmoid full] precision: {self.precision}")
            print(f"[Sigmoid full] precision eigvals: {w_str}")
        else:
            raise ValueError("metric must be 'diag' or 'full'.")

    # ---------- center selection ----------
    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
    ) -> np.ndarray:
        if prior_samples is not None:
            X = np.asarray(prior_samples, dtype=float)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, d = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        raise ValueError(f"Unknown center selection method: {method}")

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    # ---------- diag scale estimation ----------
    def _median_heuristic_per_dim(self, x: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
        """
        Per-dimension median heuristic:
            s_i ≈ sqrt(median_{i<j} (x_i - x_j)^2 + jitter)
        x: (n, d)
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("reference_data must be a 2D array of shape (n, d).")
        n, d = x.shape
        if n < 2:
            return np.sqrt(np.var(x, axis=0) + jitter)

        diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
        iu = np.triu_indices(n, k=1)
        diffs = diffs[iu]                       # (n*(n-1)/2, d)
        med_sq = np.median(diffs**2, axis=0)    # (d,)
        return np.sqrt(med_sq + jitter)         # (d,)

    def _estimate_scale_vector_from_samples(
        self,
        samples: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
    ) -> np.ndarray:
        s = self._median_heuristic_per_dim(samples)
        floor = floor_frac * np.std(samples, axis=0)
        s = np.maximum(multiplier * s, floor)
        return s.astype(float)

    def _estimate_scale_vector_from_centers(
        self,
        centers: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
    ) -> np.ndarray:
        s = self._median_heuristic_per_dim(centers)
        floor = floor_frac * np.std(centers, axis=0)
        s = np.maximum(multiplier * s, floor)
        return s.astype(float)

    # ---------- full (non-diagonal) precision estimation ----------
    def _estimate_precision_from_samples(
        self,
        samples: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ) -> np.ndarray:
        """
        Estimate a shared precision matrix P ≻ 0:
          - compute covariance Σ from samples
          - regularize eigenvalues (floor_frac * mean_eig + jitter)
          - invert and scale: P = (1 / multiplier^2) * Σ^{-1}
        """
        X = np.asarray(samples, dtype=float)
        Σ = np.cov(X, rowvar=False)  # (d,d)
        Σ = 0.5 * (Σ + Σ.T)

        w, V = eigh(Σ)
        mean_eig = float(np.mean(np.maximum(w, 0.0)))
        w_reg = np.maximum(w, floor_frac * mean_eig + jitter)
        w_inv = 1.0 / w_reg
        Σ_inv = (V * w_inv) @ V.T
        P = Σ_inv / (multiplier**2)

        return 0.5 * (P + P.T)

    def _estimate_precision_from_centers(
        self,
        centers: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ) -> np.ndarray:
        return self._estimate_precision_from_samples(
            samples=centers,
            multiplier=multiplier,
            floor_frac=floor_frac,
            jitter=jitter,
        )

    # ---------- basis & gradient ----------
    @staticmethod
    def _sigmoid(u: np.ndarray) -> np.ndarray:
        # stable logistic
        return 1.0 / (1.0 + np.exp(-u))

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """
        metric="diag":
            φ_{i,b}(x) = σ( (x_i - c_{b,i}) / s_i )
            return shape: (m, d, B)

        metric="full":
            φ_b(x) = σ( -1/2 (x - c_b)^T P (x - c_b) )
            return shape: (m, d, B) with the scalar repeated along d.
        """
        X = np.asarray(samples, dtype=float)  # (m, d)
        m, d = X.shape
        B = self.num_basis

        if self.metric == "diag":
            diffs = X[:, None, :] - self.centers[None, :, :]           # (m, B, d)
            u = diffs / (self.scale[None, None, :])                    # (m, B, d)
            vals = self._sigmoid(u)                                    # (m, B, d)
            return np.transpose(vals, (0, 2, 1))                       # (m, d, B)

        # metric == "full"
        diffs = X[:, None, :] - self.centers[None, :, :]               # (m, B, d)
        r2 = np.einsum("mbi,ij,mbj->mb", diffs, self.precision, diffs, optimize=True)  # (m,B)
        u = -0.5 * r2                                                  # (m,B)
        phi = self._sigmoid(u)                                         # (m,B)
        phi_rep = np.repeat(phi[:, None, :], d, axis=1)                # (m, d, B)
        return phi_rep

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """
        metric="diag":
            ∂/∂x_i σ((x_i-c_{b,i})/s_i) = (1/s_i) σ(u_{i,b})(1-σ(u_{i,b}))
        metric="full":
            u_b(x) = -(1/2) (x-c_b)^T P (x-c_b)
            ∇u_b(x) = -P (x-c_b)
            ∇φ_b(x) = σ(u_b)(1-σ(u_b)) * ∇u_b(x)
        Returns shape (m, d, B).
        """
        X = np.asarray(samples, dtype=float)  # (m, d)

        if self.metric == "diag":
            diffs = X[:, None, :] - self.centers[None, :, :]            # (m, B, d)
            u = diffs / (self.scale[None, None, :])                     # (m, B, d)
            sig = self._sigmoid(u)                                      # (m, B, d)
            d_sig_du = sig * (1.0 - sig)                                # (m, B, d)
            grad_mBd = (1.0 / self.scale[None, None, :]) * d_sig_du     # (m, B, d)
            return np.transpose(grad_mBd, (0, 2, 1))                    # (m, d, B)

        diffs = X[:, None, :] - self.centers[None, :, :]                # (m, B, d)
        r2 = np.einsum("mbi,ij,mbj->mb", diffs, self.precision, diffs, optimize=True)  # (m,B)
        u = -0.5 * r2                                                   # (m,B)
        sig = self._sigmoid(u)                                          # (m,B)
        d_sig_du = sig * (1.0 - sig)                                    # (m,B)
        Px = np.einsum("ij,mbj->mbi", self.precision, diffs, optimize=True)            # (m,B,d)
        grad = -d_sig_du[:, :, None] * Px                               # (m,B,d)
        return np.transpose(grad, (0, 2, 1))                            # (m, d, B)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        """
        Returns a symbolic form for the FIRST center b=0.
        metric="diag": tuple of σ( (x_i - c_i)/s_i ) for i=1..d.
        metric="full": tuple of the same scalar σ( -1/2 (x-c)^T P (x-c) ) repeated d times.
        """
        x = sp.symbols(f"x0:{dim}")
        c = self.centers[0]
        if self.metric == "diag":
            s = self.scale
            exprs = [1 / (1 + sp.exp(-((x[i] - c[i]) / s[i]))) for i in range(dim)]
            return sp.Tuple(*exprs)

        # full
        P = sp.Matrix(self.precision)
        xm = sp.Matrix(x)
        cm = sp.Matrix(c)
        quad = (xm - cm).T * P * (xm - cm)
        u = -sp.Rational(1, 2) * quad[0]
        expr = 1 / (1 + sp.exp(-u))
        return sp.Tuple(*[expr for _ in range(dim)])


class PolynomialBasisFunctionMultidim(BaseBasisFunction):
    def __init__(self, degree: int):
        if degree < 1:
            raise ValueError("degree must be >= 1.")
        self.degree = int(degree)

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """
        Diagonal polynomial basis (no cross-terms).
        φ_{i,k}(x) = x_i^k,  k=1..degree
        Returns: (m, d, degree)
        """
        X = np.asarray(samples, dtype=float)     # (m, d)
        m, d = X.shape
        vals = np.empty((m, d, self.degree), dtype=float)
        # powers 1..degree (exclude constant term)
        acc = np.ones_like(X)
        for k in range(1, self.degree + 1):
            acc = acc * X                        # acc = X**k without re-pow
            vals[:, :, k - 1] = acc
        return vals

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """
        ∂/∂x_i x_i^k = k * x_i^{k-1}
        Returns: (m, d, degree)
        """
        X = np.asarray(samples, dtype=float)     # (m, d)
        m, d = X.shape
        grads = np.empty((m, d, self.degree), dtype=float)

        # k = 1 term: derivative is 1
        grads[:, :, 0] = 1.0

        # For k >= 2: k * x^{k-1}. Build powers incrementally for stability/speed.
        if self.degree >= 2:
            acc = np.ones_like(X)                # will hold X**(k-1)
            for k in range(2, self.degree + 1):
                acc = acc * X                    # now acc = X**(k-1)
                grads[:, :, k - 1] = k * acc
        return grads

    def symbolic_expression(self, dim: int) -> sp.Expr:
        """
        Returns a tuple of per-dimension monomials:
        (x0^1, ..., x0^K, x1^1, ..., x1^K, ..., x_{d-1}^1, ..., x_{d-1}^K)
        """
        x = sp.symbols(f"x0:{dim}")
        exprs = []
        for i in range(dim):
            for k in range(1, self.degree + 1):
                exprs.append(x[i] ** k)
        return sp.Tuple(*exprs)


class FixedCentersRBFBasisFunction(BaseBasisFunction):
    r"""
    Separable Gaussian-RBF basis with explicit, shared centres:

        phi_k(theta) = exp( -(theta - c_k)^2 / (2 l^2) ),   c_k = loc + mult_k * scale,

    used for the per-node KEF fit against a Gaussian base measure
    N(loc, scale^2), with fixed multiples (default {-2,-1,0,1,2}, so K=5)
    rather than centres estimated from data via kmeans/farthest/halton as in
    RBFBasisFunction.

    Every column of `samples` (m, d) is treated as an *independent* scalar
    node sharing the same K centres/lengthscale -- unlike
    RBFBasisFunctionMultidim(metric="full"), there is no cross-column
    interaction. This is the additive-kernel structure needed to decompose a
    d-dimensional FD sensitivity computation into d independent 1D
    sub-problems: d=1 recovers the single-node case, d>1 lets one call
    evaluate/score a whole group of nodes that share the same reference
    prior (e.g. all weights of one BNN layer) at once.

    evaluate/gradient shapes match BaseBasisFunction: (m, d, K).
    """

    # Marks this basis as safe to call with a (m, n_independent_nodes) sample
    # matrix and get back n_independent_nodes worth of *independent* per-node
    # gradients in one call (see src.optimization.bnn_node_sensitivity, which
    # checks this flag to decide whether it can batch many nodes through one
    # gradient() call or must loop node-by-node). Data-driven bases whose
    # kernel couples all columns jointly (e.g. MaternBasisFunction,
    # RBFBasisFunction) do NOT set this.
    SEPARABLE = True

    def __init__(
        self,
        posterior_samples: Optional[np.ndarray] = None,
        prior_samples: Optional[np.ndarray] = None,
        num_basis_functions: Optional[int] = None,
        loc: float = 0.0,
        scale: float = 1.0,
        center_multiples: Optional[Sequence[float]] = None,
        lengthscale: Optional[float] = None,
    ):
        if scale <= 0:
            raise ValueError(f"scale must be > 0; got {scale}.")
        if center_multiples is None:
            center_multiples = [-2.0, -1.0, 0.0, 1.0, 2.0]
        center_multiples = np.asarray(center_multiples, dtype=float)
        if num_basis_functions is not None and len(center_multiples) != num_basis_functions:
            raise ValueError(
                f"len(center_multiples)={len(center_multiples)} != "
                f"num_basis_functions={num_basis_functions}."
            )

        self.loc = float(loc)
        self.scale = float(scale)
        self.center_multiples = center_multiples
        self.centers = self.loc + self.scale * center_multiples  # (K,)
        self.lengthscale = float(lengthscale) if lengthscale is not None else self.scale
        self.num_basis = int(self.centers.shape[0])

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """Returns phi(theta) with shape (m, d, K)."""
        X = np.asarray(samples, dtype=float)  # (m, d)
        diff = X[:, :, None] - self.centers[None, None, :]  # (m, d, K)
        return np.exp(-(diff ** 2) / (2.0 * self.lengthscale ** 2))

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """d/dtheta phi_k(theta), shape (m, d, K), independent per column of samples."""
        X = np.asarray(samples, dtype=float)  # (m, d)
        diff = X[:, :, None] - self.centers[None, None, :]  # (m, d, K)
        phi = np.exp(-(diff ** 2) / (2.0 * self.lengthscale ** 2))
        return (-diff / (self.lengthscale ** 2)) * phi

    def symbolic_expression(self, dim: int) -> sp.Expr:
        x = sp.symbols(f"x0:{dim}")
        c0 = sp.Float(float(self.centers[0]))
        ell = sp.Float(self.lengthscale)
        exprs = [sp.exp(-((xi - c0) ** 2) / (2 * ell ** 2)) for xi in x]
        return sp.Tuple(*exprs)
