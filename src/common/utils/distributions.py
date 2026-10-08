from src.common.distributions.gaussian import Gaussian, MultivariateGaussian
from src.common.distributions.log_normal import LogNormal
from src.common.distributions.cauchy import HalfCauchy
from src.common.distributions.uniform import Uniform
from src.common.distributions.gamma import Gamma
from src.common.distributions.chi_squared import ChiSquared
from src.common.distributions.beta import Beta
from src.common.distributions.inverse_gamma import InverseGamma

DISTRIBUTION_MAP = {
    "Gaussian": Gaussian,
    "LogNormal": LogNormal,
    "MultivariateGaussian": MultivariateGaussian,
    "HalfCauchy": HalfCauchy,
    "Uniform": Uniform,
    "Gamma": Gamma,
    "ChiSquared": ChiSquared,
    "Beta": Beta,
    "InverseGamma": InverseGamma,
}


