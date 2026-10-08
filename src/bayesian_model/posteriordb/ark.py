import warnings
from abc import ABC
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from posteriordb import PosteriorDatabase


class ArkBayesianModel(ABC):
    """
    Bayesian model for AR(K) time series loaded from posteriordb.
    Parameters vector order used here: [alpha, beta[1]..beta[K], sigma or log_sigma].
    """

    def __init__(self, data_config: Any):
        self.true_dgp = data_config.true_dgp
        self.loss_lr: float = data_config.loss_lr
        self.loss: Any = data_config.loss
        self.prior: Any = data_config.candidate_prior
        self.prior_init: Any = data_config.base_prior
        self.prior_candidate: Any = data_config.candidate_prior
        self.loss_lr_init: float = data_config.loss_lr
        self.pdb_path = data_config.pdb_path
        self.pdb_model_name = data_config.pdb_model_name
        self.warmup = getattr(data_config, "warmup", 0)

        (self.observations,
         self.x,
         self.posterior_samples_init,
         self.posterior_sample_colnames,
         self.K) = self._prepare_data()
        self.observations_num = self.observations.shape[0]
        self.x_bar: np.ndarray = np.mean(self.observations, axis=0)
        self.m = self.posterior_samples_init.shape[0]

    def _prepare_observations(self, datavals: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        For AR(K): datavals expected keys: 'K', 'T', 'y'.
        We set loss data with y and K; observations are y[K:] as (n,1) for bookkeeping.
        """
        K = int(datavals["K"])
        y = np.asarray(datavals["y"], dtype=float).reshape(-1)
        T = int(datavals.get("T", len(y)))

        if len(y) != T:
            raise ValueError(f"Mismatch: T={T} but len(y)={len(y)}.")

        if hasattr(self.loss, "set_data"):
            self.loss.set_data(y, K)

        obs = y[K:].reshape(-1, 1)
        return obs, y, K

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
            n = arr.shape[0]
            shape = arr.shape[1:]
            k = int(np.prod(shape))
            flat = arr.reshape(n, k)
            if len(shape) == 1:
                cols = [f"{name}[{i+1}]" for i in range(shape[0])]
            else:
                from itertools import product
                idx = product(*[range(1, s+1) for s in shape])
                cols = [f"{name}[{','.join(map(str, t))}]" for t in idx]
        return flat, cols

    def _prepare_posterior_draws(self, df: pd.DataFrame,
                                 warmup=0, max_draws=None) -> Tuple[np.ndarray, List[str]]:
        mats, names = [], []
        for param in df.columns:
            concatenated = self._concat_chains(df[param], warmup=warmup, max_draws=max_draws)
            flat, cols = self._flatten_param_draws(param, concatenated)
            mats.append(flat)
            names.extend(cols)
        samples = np.hstack(mats).astype(float)
        return samples, names

    def _reorder_to_prior_order(self, samples: np.ndarray, colnames: List[str], K: int) -> Tuple[np.ndarray, List[str]]:
        """
        Reorder sample columns to [alpha, beta[1],...,beta[K], sigma] if present.
        """
        want = ["alpha"] + [f"beta[{i}]" for i in range(1, K+1)] + ["sigma"]
        name_to_idx = {n: i for i, n in enumerate(colnames)}
        missing = [n for n in want if n not in name_to_idx]
        if missing:
            # Some databases name sigma as "sigma" already; if log-sigma, adapt here if needed.
            raise ValueError(f"Missing parameters in posterior draws: {missing}. Got {colnames}")
        idx = [name_to_idx[n] for n in want]
        return samples[:, idx], want

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]:
        my_pdb = PosteriorDatabase(self.pdb_path)
        posterior = my_pdb.posterior(self.pdb_model_name)

        datavals = posterior.data.values()
        observations, x, K = self._prepare_observations(datavals)

        reference_df = pd.DataFrame(posterior.reference_draws())
        posterior_samples, colnames = self._prepare_posterior_draws(
            reference_df, warmup=self.warmup
        )
        posterior_samples, ordered_names = self._reorder_to_prior_order(posterior_samples, colnames, K)

        return observations, x, posterior_samples, ordered_names, K

    def sample_from_base_prior(self, n_samples: int = 1000) -> np.ndarray:
        return self.prior_init.sample(n_samples)

    def set_lr_parameter(self, lr: float) -> None:
        self.loss_lr = lr

    def prior_score(self, x: np.ndarray) -> np.ndarray:
        return self.prior.grad_log_pdf(x)

    def reference_prior_score(self, x) -> np.ndarray:
        return self.prior_init.grad_log_pdf(x)

    def loss_score(self, x: np.ndarray, multiply_by_lr: bool = True) -> np.ndarray:
        grad = self.loss.grad_log_pdf(x)
        return self.loss_lr * grad if multiply_by_lr else grad

    def reference_loss_score(self, x, multiply_by_lr: bool = True) -> np.ndarray:
        grad = self.loss.grad_log_pdf(x)
        return self.loss_lr_init * grad if multiply_by_lr else grad

    def posterior_score(self, x: np.ndarray) -> np.ndarray:
        return self.prior_score(x) + self.loss_score(x)

    def jacobian_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        return self.prior.grad_sufficient_statistics(x)

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        return self.prior.grad_log_base_measure(x)
