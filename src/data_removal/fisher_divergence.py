from typing import Optional, Sequence

import numpy as np

from src.common.bayesian_model.samples import PosteriorSamplesModel


def aggregate_group_scores(scores: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Sum per-observation scores (m, n, d) over disjoint groups of observation indices, giving (m, G, d).

    Valid when the likelihood factorises over the observations; for correlated observations within a group give the
    group scores directly instead.
    """
    groups = [np.asarray(g, dtype=int).reshape(-1) for g in groups]
    flat = np.concatenate(groups) if groups else np.array([], dtype=int)
    if len(np.unique(flat)) != len(flat):
        raise ValueError("groups must be disjoint.")
    if flat.size and (flat.min() < 0 or flat.max() >= scores.shape[1]):
        raise ValueError(f"group indices must lie in [0, {scores.shape[1]}).")
    return np.stack([scores[:, g, :].sum(axis=1) for g in groups], axis=1)


class PosteriorFDDataRemoval:
    def __init__(
        self,
        model: "PosteriorSamplesModel",
        groups: Optional[Sequence[Sequence[int]]] = None,
        whiten: bool = False,
    ):
        """Fisher divergence between the reference posterior and the posterior with a subset of observations removed.

        The prior score cancels in the score difference, so only the per-observation log-likelihood gradients at the
        reference-posterior draws are needed: J[j] = [grad log l(theta_j; x_1), ..., grad log l(theta_j; x_n)].
        """
        self.model = model
        self.samples: np.ndarray = self.model.posterior_samples_init
        self.m = self.samples.shape[0]

        # Per-observation (or per-group) scores at the samples — shape: (m, n, paramdim)
        scores = self.model.data_score(self.samples)
        if groups is not None:
            scores = aggregate_group_scores(scores, groups)

        # Whitening measures the FD in z = L^{-1}(theta - mean), L L^T = posterior covariance, so the score becomes
        # L^T grad and parameters on different scales contribute comparably.
        self.whiten = whiten
        if whiten:
            cov = np.atleast_2d(np.cov(self.samples, rowvar=False))
            scores = np.einsum("mnk,kl->mnl", scores, np.linalg.cholesky(cov))
        self.scores = scores
        self.n = scores.shape[1]

    def compute_fisher_quadratic_form(self) -> np.ndarray:
        """Return A = (1/m) sum_j J_j^T J_j, shape (n, n), so that the FD of removing v in {0,1}^n is v^T A v."""
        A = np.einsum("mik,mlk->il", self.scores, self.scores) / self.m
        return 0.5 * (A + A.T)

    def _removal_vector(self, removed: Sequence[int]) -> np.ndarray:
        v = np.zeros(self.n)
        v[np.asarray(removed, dtype=int)] = 1.0
        return v

    def fd_per_draw(self, removed: Sequence[int]) -> np.ndarray:
        """Return ||J_j v||^2 at each draw for the removal indicator v of `removed`, shape (m,)."""
        diff = np.einsum("mik,i->mk", self.scores, self._removal_vector(removed))
        return np.sum(diff * diff, axis=1)

    def fd_given_removal(self, removed: Sequence[int]) -> float:
        """Empirical FD of removing the observations with indices `removed`."""
        return float(np.mean(self.fd_per_draw(removed)))

    def fd_standard_error(self, removed: Sequence[int]) -> float:
        """Monte Carlo standard error of the empirical FD, treating the draws as independent."""
        per_draw = self.fd_per_draw(removed)
        return float(per_draw.std(ddof=1) / np.sqrt(self.m)) if self.m > 1 else float("nan")
