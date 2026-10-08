from scipy.spatial.distance import pdist
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Literal
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
        method: Literal["kmeans", "halton", "random"] = "kmeans",
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

        if method == "halton":
            return self._halton_centers(X, num_centers)

        if method == "random":
            return self._random_centers(X, num_centers)

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

    def _random_centers(self, X: np.ndarray, num_centers: int) -> np.ndarray:
        """
        Centers via random subsampling of the distinct rows of X, rejecting
        a candidate that lands too close to an already-chosen center and
        drawing another one instead.

        Plain uniform-without-replacement subsampling can (and empirically
        does) draw two centers within a small fraction of the eventual
        lengthscale of each other; for a smooth radial basis their gradients
        are then nearly identical functions of theta, which collapses
        several directions of the K x K gradient Gram matrix A_c to
        near-zero eigenvalues (verified: two random centers within ~5% of
        the fitted lengthscale were enough to wipe out 4-5 of 50 gradient
        directions). That numerically inflates the FD-sensitivity
        generalized-eigenvalue ratio omega_max for any node with even a
        small projection onto that spurious near-null direction -- exact
        deduplication alone does not catch this, since the offending centers
        are all distinct floats, just clustered.
        """
        X_unique = np.unique(X, axis=0)
        n = X_unique.shape[0]
        num_centers = min(num_centers, n)
        spread = float(np.linalg.norm(X_unique.max(axis=0) - X_unique.min(axis=0)))
        min_sep = spread / (2 * max(num_centers, 1))

        order = self.rng.permutation(n)
        chosen: list[int] = []
        for i in order:
            if len(chosen) == num_centers:
                break
            cand = X_unique[i]
            if chosen and np.linalg.norm(X_unique[chosen] - cand, axis=1).min() < min_sep:
                continue
            chosen.append(int(i))

        if len(chosen) < num_centers:
            leftover = [int(i) for i in order if i not in chosen]
            chosen.extend(leftover[: num_centers - len(chosen)])

        return X_unique[chosen]


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
        method: Literal["kmeans", "halton", "random"] = "kmeans",
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
        elif estimation_samples_source == "both":
            estimation_samples = self._concat_samples(prior_samples, posterior_samples)
        else:
            estimation_samples = posterior_samples

        # Choose centers (B, d)
        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
            estimation_centers_source=estimation_centers_source,
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

        if method == "halton":
            return self._halton_centers(X, num_centers)

        if method == "random":
            return self._random_centers(X, num_centers)

        raise ValueError(f"Unknown center selection method: {method}")


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


    def _random_centers(self, X: np.ndarray, num_centers: int) -> np.ndarray:
        """
        Centers via random subsampling of the distinct rows of X, rejecting
        a candidate that lands too close to an already-chosen center and
        drawing another one instead.

        Plain uniform-without-replacement subsampling can (and empirically
        does) draw two centers within a small fraction of the eventual
        lengthscale of each other; for a smooth radial basis their gradients
        are then nearly identical functions of theta, which collapses
        several directions of the K x K gradient Gram matrix A_c to
        near-zero eigenvalues (verified: two random centers within ~5% of
        the fitted lengthscale were enough to wipe out 4-5 of 50 gradient
        directions). That numerically inflates the FD-sensitivity
        generalized-eigenvalue ratio omega_max for any node with even a
        small projection onto that spurious near-null direction -- exact
        deduplication alone does not catch this, since the offending centers
        are all distinct floats, just clustered.
        """
        X_unique = np.unique(X, axis=0)
        n = X_unique.shape[0]
        num_centers = min(num_centers, n)
        spread = float(np.linalg.norm(X_unique.max(axis=0) - X_unique.min(axis=0)))
        min_sep = spread / (2 * max(num_centers, 1))

        order = self.rng.permutation(n)
        chosen: list[int] = []
        for i in order:
            if len(chosen) == num_centers:
                break
            cand = X_unique[i]
            if chosen and np.linalg.norm(X_unique[chosen] - cand, axis=1).min() < min_sep:
                continue
            chosen.append(int(i))

        if len(chosen) < num_centers:
            leftover = [int(i) for i in order if i not in chosen]
            chosen.extend(leftover[: num_centers - len(chosen)])

        return X_unique[chosen]

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


class RBFBasisFunction(BaseBasisFunction):
    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        lengthscale: Optional[float] = None,
        method: Literal["kmeans", "halton", "random"] = "kmeans",
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

        if method == "halton":
            return self._halton_centers(X, num_centers)

        if method == "random":
            return self._random_centers(X, num_centers)

        raise ValueError(f"Unknown center selection method: {method}")

    def _halton_centers(self, X: np.ndarray, num_centers: int) -> np.ndarray:
        """
        Quasi-uniform centers via a scrambled Halton sequence mapped through
        the empirical quantiles of X (dimension-wise). See
        MaternBasisFunction._halton_centers for details.
        """
        d = X.shape[1]
        sampler = Halton(d=d, scramble=True, seed=27)
        u = sampler.random(n=num_centers)          # (K, d) in [0, 1]^d
        centers = np.empty((num_centers, d), dtype=float)
        for j in range(d):
            centers[:, j] = np.quantile(X[:, j], u[:, j])
        return centers

    def _random_centers(self, X: np.ndarray, num_centers: int) -> np.ndarray:
        """
        Centers via random subsampling of the distinct rows of X, rejecting
        a candidate that lands too close to an already-chosen center and
        drawing another one instead.

        Plain uniform-without-replacement subsampling can (and empirically
        does) draw two centers within a small fraction of the eventual
        lengthscale of each other; for a smooth radial basis their gradients
        are then nearly identical functions of theta, which collapses
        several directions of the K x K gradient Gram matrix A_c to
        near-zero eigenvalues (verified: two random centers within ~5% of
        the fitted lengthscale were enough to wipe out 4-5 of 50 gradient
        directions). That numerically inflates the FD-sensitivity
        generalized-eigenvalue ratio omega_max for any node with even a
        small projection onto that spurious near-null direction -- exact
        deduplication alone does not catch this, since the offending centers
        are all distinct floats, just clustered.
        """
        X_unique = np.unique(X, axis=0)
        n = X_unique.shape[0]
        num_centers = min(num_centers, n)
        spread = float(np.linalg.norm(X_unique.max(axis=0) - X_unique.min(axis=0)))
        min_sep = spread / (2 * max(num_centers, 1))

        order = self.rng.permutation(n)
        chosen: list[int] = []
        for i in order:
            if len(chosen) == num_centers:
                break
            cand = X_unique[i]
            if chosen and np.linalg.norm(X_unique[chosen] - cand, axis=1).min() < min_sep:
                continue
            chosen.append(int(i))

        if len(chosen) < num_centers:
            leftover = [int(i) for i in order if i not in chosen]
            chosen.extend(leftover[: num_centers - len(chosen)])

        return X_unique[chosen]


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


def rbf_gaussian_gram_closed_form(
    centers: np.ndarray,
    lengthscale: float,
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> np.ndarray:
    r"""
    Closed-form K x K Gram matrix

        M_{ij} = E_{theta ~ N(mu, Sigma)}[ grad phi_i(theta) . grad phi_j(theta) ],

    for the RBF kernel phi_i(theta) = exp(-||theta - c_i||^2 / (2 ell^2)), when
    the reference measure is Gaussian. This is the closed form used for both
    A (theta ~ posterior) and A_c (theta ~ prior) in the FDsens+ RBF setting:
    call this once with (mu_ref, Sigma_ref) for A_c, and once with
    (mu_post, Sigma_post) for A.

    Args:
        centers: (K, d) kernel centres bar_theta_1, ..., bar_theta_K.
        lengthscale: RBF lengthscale ell.
        mu: (d,) mean of the Gaussian reference measure.
        Sigma: (d, d) covariance of the Gaussian reference measure.

    Returns:
        (K, K) symmetric PSD matrix M.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    K, d = centers.shape
    mu = np.asarray(mu, dtype=float).reshape(d)
    Sigma = np.asarray(Sigma, dtype=float).reshape(d, d)
    ell = float(lengthscale)

    theta_bar = centers.T  # (d, K), bar_Theta in the derivation
    U = mu.reshape(d, 1) - theta_bar  # (d, K), mu 1_K^T - bar_Theta

    def G(P: np.ndarray) -> np.ndarray:
        return U.T @ P @ U  # (K, K)

    def Q(P: np.ndarray) -> np.ndarray:
        g = G(P)
        diag_g = np.diag(g)
        return 0.25 * (diag_g[:, None] + diag_g[None, :] + 2.0 * g)

    sq_norms = np.sum(theta_bar ** 2, axis=0)  # (K,)
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (theta_bar.T @ theta_bar)
    D = np.maximum(D, 0.0)  # (K, K), squared pairwise distances between centres

    B = np.eye(d) + (2.0 / ell ** 2) * Sigma
    B_inv = np.linalg.inv(B)
    B_inv2 = B_inv @ B_inv
    det_B = np.linalg.det(B)
    trace_term = float(np.trace(Sigma @ B_inv))

    exponent = -D / (4.0 * ell ** 2) - Q(B_inv) / ell ** 2
    bracket = trace_term * np.ones((K, K)) + Q(B_inv2) - D / 4.0

    M = (det_B ** (-0.5) / ell ** 4) * np.exp(exponent) * bracket
    return 0.5 * (M + M.T)


class FixedCentersRBFBasisFunctionMultidim(BaseBasisFunction):
    r"""
    Joint (non-separable) isotropic Gaussian-RBF basis with explicit, shared
    d-dimensional centres:

        phi_b(theta) = exp( -||theta - c_b||^2 / (2 ell^2) ),   c_b in R^d,

    matching exactly the kernel assumed by `rbf_gaussian_gram_closed_form`
    (isotropic precision I / ell^2), unlike FixedCentersRBFBasisFunction
    (which is separable per-column/1D) or RBFBasisFunctionMultidim (whose
    centres are estimated from data via kmeans/halton/random, not passed in
    directly). Used to Monte-Carlo estimate A, A_c for a basis whose exact
    closed-form Gram matrices under a Gaussian reference measure can be
    computed independently via `rbf_gaussian_gram_closed_form` for the same
    centres/lengthscale, so the two can be compared directly.

    evaluate/gradient shapes match BaseBasisFunction: (m, d, K).
    """

    def __init__(self, centers: np.ndarray, lengthscale: float):
        centers = np.atleast_2d(np.asarray(centers, dtype=float))
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0; got {lengthscale}.")
        self.centers = centers  # (K, d)
        self.lengthscale = float(lengthscale)
        self.num_basis, self.dim = self.centers.shape

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """Returns phi(theta) with shape (m, d, K), scalar per centre repeated across d."""
        X = np.asarray(samples, dtype=float)  # (m, d)
        diff = X[:, None, :] - self.centers[None, :, :]  # (m, K, d)
        r2 = np.sum(diff ** 2, axis=-1)  # (m, K)
        phi = np.exp(-r2 / (2.0 * self.lengthscale ** 2))  # (m, K)
        return np.repeat(phi[:, None, :], self.dim, axis=1)  # (m, d, K)

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """d/dtheta phi_b(theta) = -(theta - c_b) / ell^2 * phi_b(theta), shape (m, d, K)."""
        X = np.asarray(samples, dtype=float)  # (m, d)
        diff = X[:, None, :] - self.centers[None, :, :]  # (m, K, d)
        r2 = np.sum(diff ** 2, axis=-1)  # (m, K)
        phi = np.exp(-r2 / (2.0 * self.lengthscale ** 2))  # (m, K)
        grad = (-diff / (self.lengthscale ** 2)) * phi[:, :, None]  # (m, K, d)
        return np.transpose(grad, (0, 2, 1))  # (m, d, K)

