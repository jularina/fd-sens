import importlib
from typing import Dict, Any, List, Union
import numpy as np

from .base import BaseDistribution


def _instantiate_from_target_str(target: str, kwargs: Dict[str, Any]):
    """Instantiate a class from a full import path string."""
    module_name, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(**kwargs)


def _is_basedistribution(obj) -> bool:
    """Best-effort check to recognize a BaseDistribution instance without hard import deps."""
    try:
        from src.common.distributions.base import BaseDistribution as _BD
        return isinstance(obj, _BD)
    except Exception:
        required = (
            "sample", "pdf", "log_pdf", "grad_log_pdf",
            "grad_log_base_measure", "natural_parameters",
            "grad_sufficient_statistics",
        )
        return all(hasattr(obj, m) for m in required)


def _maybe_build_component(spec: Union["BaseDistribution", Dict[str, Any]]):
    """Accept an instantiated distribution or a Hydra-style spec with _target_."""
    if isinstance(spec, dict) and "_target_" in spec:
        target = spec["_target_"]
        kwargs = {k: v for k, v in spec.items() if k != "_target_"}
        return _instantiate_from_target_str(target, kwargs)
    if _is_basedistribution(spec):
        return spec
    raise ValueError("Each component must be a BaseDistribution instance or a dict with a '_target_' key.")


class CompositeProduct(BaseDistribution):
    """
    Product (independent) composite distribution over concatenated parameters.
    The joint density factorizes:  p(x) = ∏_k p_k(x_k),  with x = concat(x_1, ..., x_K).

    Parameters
    ----------
    distributions : Dict[str, Union[BaseDistribution, Dict]]
        Mapping name -> component distribution or Hydra-style spec.
        (No other kwargs are accepted.)
    """

    def __init__(self, *, distributions: Dict[str, Union["BaseDistribution", Dict[str, Any]]]):
        if distributions is None:
            raise ValueError("CompositeProduct requires a non-empty keyword-only argument 'distributions'.")

        self.names: List[str] = []
        self.components: List[Any] = []

        for name, spec in distributions.items():
            self.names.append(name)
            self.components.append(_maybe_build_component(spec))

        self.comp_dims: List[int] = []
        self._slices: List[slice] = []
        offset = 0

        for dist in self.components:
            dim = self._infer_component_dim(dist)
            self.comp_dims.append(dim)
            self._slices.append(slice(offset, offset + dim))
            offset += dim
        self.dim: int = offset

    @staticmethod
    def _infer_component_dim(dist) -> int:
        try:
            s = dist.sample(1)
            return int(np.atleast_2d(s).shape[-1])
        except Exception:
            return int(getattr(dist, "dim", 1))

    def _as_batch(self, x: Union[np.ndarray, list, float]) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input last dimension {x.shape[-1]} != composite dim {self.dim}")
        return x

    def _split(self, x_batch: np.ndarray) -> List[np.ndarray]:
        return [x_batch[:, s] for s in self._slices]

    def sample(self, n_samples: int = 1) -> np.ndarray:
        parts = []
        for dist in self.components:
            s = np.atleast_2d(dist.sample(n_samples))
            if s.shape[0] != n_samples:
                s = s.reshape(n_samples, -1)
            parts.append(s)
        return np.concatenate(parts, axis=1)

    def pdf(self, x: Union[np.ndarray, list, float]) -> np.ndarray:
        xb = self._as_batch(x)
        xs = self._split(xb)
        out = None
        for name, dist, xk in zip(self.names, self.components, xs):
            try:
                pk = dist.pdf(xk)
            except NotImplementedError as e:
                raise NotImplementedError(f"Component '{name}' pdf unavailable: {e}")
            pk = np.asarray(pk).reshape(xb.shape[0])
            out = pk if out is None else (out * pk)  # product, not sum
        return out

    def log_pdf(self, x: Union[np.ndarray, list, float]) -> np.ndarray:
        xb = self._as_batch(x)
        xs = self._split(xb)
        out = None
        for name, dist, xk in zip(self.names, self.components, xs):
            try:
                lk = dist.log_pdf(xk)
            except NotImplementedError as e:
                raise NotImplementedError(f"Component '{name}' log_pdf unavailable: {e}")
            lk = np.asarray(lk).reshape(xb.shape[0])
            out = lk if out is None else (out + lk)
        return out

    def grad_log_pdf(self, x: Union[np.ndarray, list, float]) -> np.ndarray:
        xb = self._as_batch(x)  # (m, d)
        xs = self._split(xb)  # list of (m, d)
        grads = []
        for name, dist, xk in zip(self.names, self.components, xs):
            try:
                gk = dist.grad_log_pdf(xk)
            except NotImplementedError as e:
                raise NotImplementedError(f"Component '{name}' grad_log_pdf unavailable: {e}")
            grads.append(gk)
        return np.concatenate(grads, axis=1)  # (m, d_k)

    def grad_log_base_measure(self, x):
        xb = self._as_batch(x)  # (m, d_total)
        xs = self._split(xb)  # list of (m, d_k)
        parts = []
        for name, dist, xk in zip(self.names, self.components, xs):
            gk = np.asarray(dist.grad_log_base_measure(xk))
            m = xb.shape[0]
            if gk.ndim == 1:
                gk = gk.reshape(m, -1)
            elif gk.ndim != 2:
                gk = gk.reshape(m, -1)
            parts.append(gk)  # each (m, d_k)
        return np.concatenate(parts, axis=1)  # (m, d_total)

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        xb = self._as_batch(x)  # (m, d_total)
        xs = self._split(xb)  # list of (m, d_k)
        m, d_total = xb.shape
        gks = []
        pk_list = []

        # Collect per-component Jacobians and sizes
        for name, dist, xk in zip(self.names, self.components, xs):
            gk = np.asarray(dist.grad_sufficient_statistics(xk))
            # Accept (m, p_k), (m, 1, p_k), or (m, d_k, p_k)
            if gk.ndim == 1:
                gk = gk.reshape(m, 1, -1)
            elif gk.ndim == 2:
                gk = gk.reshape(m, 1, -1)
            elif gk.ndim != 3:
                gk = gk.reshape(m, 1, -1)
            d_k = xk.shape[1]
            if gk.shape[1] not in (1, d_k):
                raise ValueError(
                    f"Component '{name}' grad_sufficient_statistics has shape {gk.shape}; "
                    f"expected (m,{d_k},p_k) or (m,1,p_k)."
                )
            gks.append(gk)
            pk_list.append(gk.shape[2])

        p_total = int(np.sum(pk_list))
        J = np.zeros((m, d_total, p_total), dtype=np.float64)

        # Fill block-diagonal: rows for this component's dims, cols for its stats
        col = 0
        for (name, dist, xk), gk, p_k, s in zip(
                zip(self.names, self.components, xs), gks, pk_list, self._slices
        ):
            d_k = xk.shape[1]
            if gk.shape[1] == 1:
                if d_k != 1:
                    raise ValueError(
                        f"Component '{name}' is {d_k}-dimensional but returned (m,1,p_k); "
                        "please return (m,d_k,p_k) from grad_sufficient_statistics."
                    )
                J[:, s, col:col + p_k] = gk  # (m,1,p_k) -> rows s
            else:
                J[:, s, col:col + p_k] = gk  # (m,d_k,p_k)
            col += p_k

        return J  # shape (m, d_total, p_total)

    # def augmented_natural_parameters(self) -> np.ndarray:
    #     parts = []
    #     for name, dist in zip(self.names, self.components):
    #         try:
    #             v = dist.augmented_natural_parameters()
    #         except NotImplementedError as e:
    #             raise NotImplementedError(f"Component '{name}' augmented_natural_parameters unavailable: {e}")
    #         v = np.asarray(v).reshape(-1)
    #         parts.append(v)
    #     return np.concatenate(parts, axis=0)
    #
    # def natural_parameters(self) -> np.ndarray:
    #     parts = []
    #     for name, dist in zip(self.names, self.components):
    #         try:
    #             v = dist.natural_parameters()
    #         except NotImplementedError as e:
    #             raise NotImplementedError(f"Component '{name}' natural_parameters unavailable: {e}")
    #         v = np.asarray(v).reshape(-1)
    #         parts.append(v)
    #     return np.concatenate(parts, axis=0)

    def natural_parameters(self) -> np.ndarray:
        parts = []
        for name, dist in zip(self.names, self.components):
            v = np.asarray(dist.natural_parameters()).reshape(-1)
            parts.append(v)
        return np.concatenate(parts, axis=0)

    def augmented_natural_parameters(self) -> np.ndarray:
        eta = self.natural_parameters()  # length = sum_k p_k
        return np.concatenate([eta, np.array([1.0])], axis=0)  # length = sum_k p_k + 1
