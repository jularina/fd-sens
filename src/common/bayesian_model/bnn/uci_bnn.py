import os
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

try:
    from hydra.utils import get_original_cwd
except ImportError:  # pragma: no cover
    get_original_cwd = None


# The *_prior.p datasets in samples.pt that correspond to actual network
# weights/biases of the 3-layer densenet of Fortuin et al. (2022). `noise_std`
# is excluded: it is an observation-noise nuisance parameter, not a network
# weight/bias covered by the per-node reference prior.
DEFAULT_PARAM_GROUPS: Tuple[str, ...] = (
    "net.module.0.weight_prior",
    "net.module.0.bias_prior",
    "net.module.2.weight_prior",
    "net.module.2.bias_prior",
    "net.module.4.weight_prior",
    "net.module.4.bias_prior",
)


class BNNPosteriorSamples:
    """
    Loads posterior draws for a bnn_priors run and exposes, per weight/bias
    tensor, the scalar reference prior (loc, scale[, df]) shared by every
    entry, plus the (n_samples, n_nodes) matrix of flattened posterior draws.

    Reference prior family is set by `data_config.reference_prior`
    ("gaussian" or "studentt"); a "studentt" run's weight tensors carry a
    Student-t prior (loc + scale * T_df) while biases stay Gaussian
    N(loc, scale^2).
    """

    def __init__(self, data_config: Any):
        samples_path = data_config.samples_path
        if not os.path.isabs(samples_path) and get_original_cwd is not None:
            try:
                samples_path = os.path.join(get_original_cwd(), samples_path)
            except ValueError:
                pass
        self.samples_path = samples_path
        self.param_groups: Tuple[str, ...] = tuple(
            getattr(data_config, "param_groups", DEFAULT_PARAM_GROUPS)
        )
        self.prior_samples_num = int(getattr(data_config, "prior_samples_num", 1000))
        self.seed = int(getattr(data_config, "seed", 0))
        self.reference_prior = str(getattr(data_config, "reference_prior", "gaussian")).lower()
        self.rng = np.random.default_rng(self.seed)

        self.groups: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.samples_path):
            raise FileNotFoundError(f"BNN samples file not found: {self.samples_path}")

        with h5py.File(self.samples_path, "r", swmr=True) as f:
            for name in self.param_groups:
                p = np.asarray(f[f"{name}.p"], dtype=np.float64)  # (m, *node_shape)
                loc = float(np.asarray(f[f"{name}.loc"])[0])
                scale = float(np.asarray(f[f"{name}.scale"])[0])
                df = None
                if self.reference_prior == "studentt":
                    df_dataset = f.get(f"{name}.df")
                    df = float(np.asarray(df_dataset)[0]) if df_dataset is not None else None
                m = p.shape[0]
                node_shape = p.shape[1:]
                n_nodes = int(np.prod(node_shape)) if node_shape else 1
                self.groups[name] = {
                    "loc": loc,
                    "scale": scale,
                    "df": df,
                    "posterior": p.reshape(m, n_nodes),
                    "n_nodes": n_nodes,
                    "shape": node_shape,
                }

    @property
    def total_nodes(self) -> int:
        return sum(g["n_nodes"] for g in self.groups.values())

    def sample_prior(self, group_name: str, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Draw i.i.d. samples from the reference prior shared by every scalar
        node in `group_name` (Gaussian, or Student-t if `g["df"]` is set).
        """
        g = self.groups[group_name]
        n = self.prior_samples_num if n_samples is None else int(n_samples)
        if g.get("df") is not None:
            return g["loc"] + g["scale"] * self.rng.standard_t(g["df"], size=n)
        return self.rng.normal(g["loc"], g["scale"], size=n)
