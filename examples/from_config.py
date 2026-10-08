"""Run FDsens or FDsens+ on a model defined by a Hydra config (see configs/README.md)."""
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.nonparametric.sensitivity import nonparametric_prior_sensitivity
from src.parametric.sensitivity import lr_sensitivity, prior_sensitivity


@hydra.main(version_base="1.1", config_path="../configs/examples", config_name="gaussian_location")
def main(cfg: DictConfig) -> None:
    np.random.seed(0)  # the example simulates its data and posterior draws
    model = instantiate(cfg.model, data_config=cfg.data)

    if "sensitivity" in cfg:
        natural_box = OmegaConf.to_container(cfg.sensitivity.natural_box, resolve=True)
        print(prior_sensitivity(model, natural_box))
        lower, upper = cfg.sensitivity.lr_interval
        print(lr_sensitivity(model, lower=lower, upper=upper))

    if "nonparametric" in cfg:
        print(nonparametric_prior_sensitivity(
            model,
            radius=cfg.nonparametric.radius,
            basis=cfg.nonparametric.basis,
            basis_kwargs=OmegaConf.to_container(cfg.nonparametric.basis_kwargs, resolve=True),
        ))


if __name__ == "__main__":
    main()
