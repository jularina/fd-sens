from omegaconf import DictConfig

from src.optimization.corner_points import (
    OptimizationCornerPointsUnivariateGaussian,
    OptimizationCornerPointsMultivariateGaussian,
    OptimizationCornerPointsCompositePrior
)
from src.discrepancies.posterior_ksd import PosteriorKSDParametric


def pick_optimizer(cfg: DictConfig, ksd_estimator: PosteriorKSDParametric):
    prior_cfg = cfg.ksd.optimize.prior
    if hasattr(prior_cfg, "Gaussian"):
        return OptimizationCornerPointsUnivariateGaussian(ksd_estimator, prior_cfg.Gaussian)
    if hasattr(prior_cfg, "MultivariateGaussian"):
        return OptimizationCornerPointsMultivariateGaussian(ksd_estimator, prior_cfg.MultivariateGaussian)
    if hasattr(prior_cfg, "Composite"):
        return OptimizationCornerPointsCompositePrior(ksd_estimator, prior_cfg.Composite, precomputed_qfs=True)
    raise ValueError(
        "No supported prior found under cfg.ksd.optimize.prior. "
        "Supported: Gaussian, MultivariateGaussian, Composite"
    )

