from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.common.bayesian_model.posteriordb.kilpisjarvi import KilpisjarviBayesianModel


class KilpisjarviNonparametricLoader:
    """
    Adapts KilpisjarviBayesianModel's composite (alpha, beta1..5, sigma)
    prior/posterior to the per-node "groups" interface used by
    src.common.bayesian_model.bnn.uci_bnn.BNNPosteriorSamples, so the same per-node
    nonparametric FD sensitivity machinery used for the BNN's decomposable
    weight/bias parameters (src.nonparametric.node_sensitivity.
    compute_group_omega_max / compute_node_lambda_star) can be reused here:
    Kilpisjarvi's prior also decomposes into independent scalar blocks
    (alpha, beta1..beta5 ~ N(0, 5^2); sigma ~ HalfCauchy(gamma)).

    Two groups, split by reference-prior family:
      "gaussian": alpha, beta1..beta5 -- share one reference prior.
      "sigma":    the lone Half-Cauchy(gamma) scalar.
    Column order within each group matches KilpisjarviBayesianModel.
    posterior_samples_init's column order (alpha, beta[1..5], sigma).
    """

    def __init__(self, data_config: Any):
        self.model = KilpisjarviBayesianModel(data_config)
        self.prior_samples_num = int(getattr(data_config, "prior_samples_num", 5000))

        # Component `sample()` methods draw from the shared global numpy RNG
        # (they take no local generator), so `sample_prior` calls are only
        # reproducible across runs if that global state is seeded once here.
        np.random.seed(int(getattr(data_config, "seed", 0)))

        names = self.model.prior_init.names
        components = self.model.prior_init.components
        samples = self.model.posterior_samples_init

        sigma_pos = names.index("sigma")
        gaussian_idx = [i for i in range(len(names)) if i != sigma_pos]

        self.groups: Dict[str, Dict[str, Any]] = {
            "gaussian": {
                "loc": float(components[gaussian_idx[0]].mu),
                "scale": float(components[gaussian_idx[0]].sigma),
                "df": None,
                "posterior": samples[:, gaussian_idx],
                "n_nodes": len(gaussian_idx),
                "shape": (len(gaussian_idx),),
                "node_names": [names[i] for i in gaussian_idx],
                "prior_dist": components[gaussian_idx[0]],
            },
            "sigma": {
                "loc": 0.0,
                "scale": float(components[sigma_pos].gamma),
                "df": None,
                "posterior": samples[:, [sigma_pos]],
                "n_nodes": 1,
                "shape": (1,),
                "node_names": [names[sigma_pos]],
                "prior_dist": components[sigma_pos],
            },
        }
        self.param_groups: Tuple[str, ...] = tuple(self.groups.keys())

    @property
    def total_nodes(self) -> int:
        return sum(g["n_nodes"] for g in self.groups.values())

    def sample_prior(self, group_name: str, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Draw i.i.d. samples from the reference prior shared by every scalar
        node in `group_name`. Every node in a group has the same family and
        parameters (independent product prior), so any one component's own
        `sample` method can be used directly.
        """
        n = self.prior_samples_num if n_samples is None else int(n_samples)
        return np.asarray(self.groups[group_name]["prior_dist"].sample(n), dtype=float).reshape(-1)
