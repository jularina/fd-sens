from typing import Any, Dict, Optional, Tuple

import numpy as np

from paper.posteriordb.kilpisjarvi import KilpisjarviBayesianModel


class KilpisjarviNonparametricLoader:
    """Adapt the Kilpisjarvi model to the per-node groups interface used for nonparametric FD sensitivity."""

    def __init__(self, data_config: Any):
        self.model = KilpisjarviBayesianModel(data_config)
        self.prior_samples_num = int(getattr(data_config, "prior_samples_num", 5000))

        # Component samplers use the global numpy RNG, so seed it once here for reproducibility.
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
        """Draw i.i.d. samples from the reference prior shared by every scalar node in the group."""
        n = self.prior_samples_num if n_samples is None else int(n_samples)
        return np.asarray(self.groups[group_name]["prior_dist"].sample(n), dtype=float).reshape(-1)
