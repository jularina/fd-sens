from typing import Optional

import numpy as np

from src.common.result import FDSensitivityResult


def lr_sensitivity(model, lower: float, upper: float, lr_ref: Optional[float] = None) -> FDSensitivityResult:
    """Learning-rate sensitivity over [lower, upper], using FD(lr) = (lr - lr_ref)^2 E||grad loss||^2.

    Only the prior is shared by the reference and candidate posteriors, so its score cancels and no prior is needed.
    """
    if lower > upper:
        raise ValueError("lower must not exceed upper.")
    if model._loss_grad is None:
        raise ValueError("Learning-rate sensitivity needs model.loss_grad (the gradient of the unscaled loss).")
    lr_ref = model.loss_lr_init if lr_ref is None else float(lr_ref)
    grad = model.loss_score(model.posterior_samples_init, multiply_by_lr=False)
    mean_sq_grad = float(np.mean(np.sum(grad * grad, axis=1)))

    def fd(lr):
        return (lr - lr_ref) ** 2 * mean_sq_grad

    lr_max = lower if fd(lower) >= fd(upper) else upper
    lr_min = min(max(lr_ref, lower), upper)
    return FDSensitivityResult(
        sensitivity=fd(lr_max) - fd(lr_min), fd_min=fd(lr_min), fd_max=fd(lr_max),
        lambda_min=np.array([lr_min]), lambda_max=np.array([lr_max]),
        analysis="learning_rate", method="closed_form",
    )
