import json
import warnings
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.common.utils.typing import ArrayLike

# ---------------------------------------------------------------------------
# Hardcoded Kilpisjarvi dataset
# ---------------------------------------------------------------------------
_DATA = {
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
}


class KilpisjarviBayesianModel:
    """
    Bayesian model for Kilpisjarvi simple linear regression loaded from posteriordb.
    Model: y_i ~ Normal(alpha + beta * x_i, sigma)
    Parameter vector order: [alpha, beta, sigma].
    """

    def __init__(self, data_config: Any):
        self.true_dgp = data_config.true_dgp
        self.loss_lr: float = data_config.loss_lr
        self.loss: Any = data_config.loss
        self.prior: Any = data_config.candidate_prior
        self.prior_init: Any = data_config.base_prior
        self.prior_candidate: Any = data_config.candidate_prior
        self.loss_lr_init: float = data_config.loss_lr
        self.posterior_draws_path: str = data_config.posterior_draws_path
        self.warmup = getattr(data_config, "warmup", 0)

        (self.observations,
         self.posterior_samples_init,
         self.posterior_sample_colnames) = self._prepare_data()
        self.observations_num = self.observations.shape[0]
        self.x_bar: np.ndarray = np.mean(self.observations, axis=0)
        self.m = self.posterior_samples_init.shape[0]

    def _prepare_observations(self) -> np.ndarray:
        """
        Build observation matrix from hardcoded data with centered y and pass it to the
        GaussianARLogLikelihood via set_data(y, K), where K is inferred from prior names.
        """
        y = np.asarray(_DATA["y"], dtype=float)
        y_centered = y - y.mean()
        K = sum(1 for name in self.prior_init.names if name.startswith("beta"))
        self.loss.set_data(y_centered, K)
        return y_centered[K:].reshape(-1, 1)

    def _norm_warmup(self, warmup, n_chains: int):
        if isinstance(warmup, (list, tuple)):
            if len(warmup) != n_chains:
                raise ValueError(f"warmup length {len(warmup)} != n_chains {n_chains}")
            return [max(0, int(w)) for w in warmup]
        return [max(0, int(warmup))] * n_chains

    def _concat_chains(self, series_with_chains: pd.Series,
                       warmup=0, max_draws=None) -> np.ndarray:
        raw = [np.asarray(a) for a in series_with_chains.to_numpy()]
        n_chains = len(raw)
        warmups = self._norm_warmup(warmup, n_chains)
        trimmed = []
        for i, arr in enumerate(raw):
            w = min(warmups[i], arr.shape[0])
            if warmups[i] > arr.shape[0]:
                warnings.warn(f"Warmup ({warmups[i]}) exceeds chain {i} draws ({arr.shape[0]}). Using {w}.")
            sl = arr[w:]
            if max_draws is not None:
                sl = sl[:max_draws]
            trimmed.append(sl)
        return np.concatenate(trimmed, axis=0)

    def _flatten_param_draws(self, name: str, arr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if arr.ndim == 1:
            flat = arr.reshape(-1, 1)
            cols = [name]
        else:
            total_draws = arr.shape[0]
            param_shape = arr.shape[1:]
            k = int(np.prod(param_shape))
            flat = arr.reshape(total_draws, k)
            if len(param_shape) == 1:
                cols = [f"{name}[{i+1}]" for i in range(param_shape[0])]
            else:
                from itertools import product
                indices = product(*[range(1, s+1) for s in param_shape])
                cols = [f"{name}[{','.join(map(str, idx))}]" for idx in indices]
        return flat, cols

    def _prepare_posterior_draws(self, df: pd.DataFrame,
                                 warmup=0, max_draws=None) -> Tuple[np.ndarray, List[str]]:
        per_param_mats, per_param_cols = [], []
        for param in df.columns:
            concatenated = self._concat_chains(df[param], warmup=warmup, max_draws=max_draws)
            flat, cols = self._flatten_param_draws(param, concatenated)
            per_param_mats.append(flat)
            per_param_cols.extend(cols)
        samples = np.hstack(per_param_mats).astype(float)
        return samples, per_param_cols

    def _reorder_to_prior_order(self, samples: np.ndarray,
                                colnames: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Reorder sample columns to [alpha, beta[1]..beta[K], sigma]."""
        name_to_idx = {n: i for i, n in enumerate(colnames)}
        # Detect number of beta components: try beta[1]..beta[K] first, then single beta
        K = 0
        for k in range(1, 100):
            if f"beta[{k}]" in name_to_idx:
                K = k
            else:
                break
        if K > 0:
            want = ["alpha"] + [f"beta[{k}]" for k in range(1, K + 1)] + ["sigma"]
        else:
            want = ["alpha", "beta", "sigma"]
        missing = [n for n in want if n not in name_to_idx]
        if missing:
            raise ValueError(f"Missing parameters in posterior draws: {missing}. Got {colnames}")
        idx = [name_to_idx[n] for n in want]
        return samples[:, idx], want

    def _load_draws_json(self) -> List[Dict]:
        with open(self.posterior_draws_path, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, list) or len(obj) == 0:
            raise ValueError(f"Expected a non-empty list of chains in {self.posterior_draws_path}")
        return obj

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        observations = self._prepare_observations()

        chains = self._load_draws_json()
        reference_df = pd.DataFrame(chains)
        posterior_samples, colnames = self._prepare_posterior_draws(
            reference_df, warmup=self.warmup
        )
        posterior_samples, ordered_names = self._reorder_to_prior_order(posterior_samples, colnames)

        return observations, posterior_samples, ordered_names


    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        grad = self.loss.grad_log_pdf(x)
        return self.loss_lr * grad if multiply_by_lr else grad
