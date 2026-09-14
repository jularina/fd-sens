from src.distributions.gaussian import Gaussian, MultivariateGaussian
from src.distributions.log_normal import LogNormal
from src.distributions.cauchy import Cauchy, HalfCauchy
from src.distributions.uniform import Uniform
from src.distributions.gamma import Gamma
from src.distributions.laplace import Laplace
from src.distributions.chi_squared import ChiSquared
from src.distributions.beta import Beta
from src.distributions.inverse_gamma import InverseGamma

DISTRIBUTION_MAP = {
    "Gaussian": Gaussian,
    "LogNormal": LogNormal,
    "MultivariateGaussian": MultivariateGaussian,
    "Cauchy": Cauchy,
    "HalfCauchy": HalfCauchy,
    "Uniform": Uniform,
    "Gamma": Gamma,
    "Laplace": Laplace,
    "ChiSquared": ChiSquared,
    "Beta": Beta,
    "InverseGamma": InverseGamma,
}


def is_basedistribution_like(obj) -> bool:
    try:
        from src.distributions.base import BaseDistribution as _BD
        return isinstance(obj, _BD)
    except Exception:
        required = (
            "sample", "pdf", "log_pdf", "grad_log_pdf",
            "grad_log_base_measure", "natural_parameters",
            "grad_sufficient_statistics",
        )
        return all(hasattr(obj, m) for m in required)
