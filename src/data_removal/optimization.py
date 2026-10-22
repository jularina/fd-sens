import itertools
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np


def number_of_corner_points(n: int, max_removed: int) -> int:
    """N(n, K) = sum_{k=0}^{K} binom(n, k): the vertices of {v in [0,1]^n : 1^T v <= K}."""
    return sum(comb(n, k) for k in range(max_removed + 1))


class OptimizationCornerPointsDataRemoval:
    def __init__(self, A: np.ndarray, chunk_size: int = 100_000):
        """Maximise v^T A v over binary v with at most K ones by evaluating every corner point of the relaxation.

        v -> v^T A v is convex (A is PSD), so its maximum over the polytope {v in [0,1]^n : 1^T v <= K} is attained
        at a vertex, i.e. a binary v with at most K ones; enumeration is exact.
        """
        self.A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
        self.n = self.A.shape[0]
        self.chunk_size = int(chunk_size)

    def _evaluate_size(self, k: int, keep_all: bool) -> Tuple[float, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Evaluate all subsets of size k in chunks; return the best value, its subset and (optionally) all of them."""
        combos = itertools.combinations(range(self.n), k)
        best_val, best_set = -np.inf, None
        all_sets, all_vals = [], []
        while True:
            chunk = np.fromiter(itertools.chain.from_iterable(itertools.islice(combos, self.chunk_size)),
                                dtype=int).reshape(-1, k)
            if chunk.shape[0] == 0:
                break
            vals = self.A[chunk[:, :, None], chunk[:, None, :]].sum(axis=(1, 2))
            i = int(np.argmax(vals))
            if vals[i] > best_val:
                best_val, best_set = float(vals[i]), chunk[i].copy()
            if keep_all:
                all_sets.append(chunk)
                all_vals.append(vals)
        if keep_all:
            return best_val, best_set, np.concatenate(all_sets), np.concatenate(all_vals)
        return best_val, best_set, None, None

    def evaluate_all_corners(
        self, max_removed: int, keep_all: bool = False,
    ) -> Tuple[Dict[int, Dict[str, object]], Optional[List[Tuple[np.ndarray, np.ndarray]]]]:
        """Evaluate every binary v with 1 <= 1^T v <= max_removed.

        Returns per removal size k the maximal FD and the removed indices attaining it, and, when keep_all=True,
        a list over k of (subsets (N_k, k), FD values (N_k,)).
        """
        per_size, everything = {}, []
        for k in range(1, max_removed + 1):
            val, removed, sets, vals = self._evaluate_size(k, keep_all)
            per_size[k] = {"fd": val, "removed": removed}
            if keep_all:
                everything.append((sets, vals))
        return per_size, (everything if keep_all else None)
