from src.distributions.gaussian import Gaussian, MultivariateGaussian
from src.distributions.log_normal import LogNormal
from src.distributions.cauchy import HalfCauchy
from src.distributions.uniform import Uniform
from src.distributions.gamma import Gamma
from src.distributions.chi_squared import ChiSquared
from src.distributions.beta import Beta
from src.distributions.inverse_gamma import InverseGamma

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


