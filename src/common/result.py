from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FDSensitivityResult:
    """Global FD sensitivity together with the hyperparameters attaining its maximum and minimum."""

    sensitivity: float
    fd_min: float
    fd_max: float
    lambda_min: np.ndarray
    lambda_max: np.ndarray
    analysis: str
    method: str
    components: Optional[Dict[str, Dict[str, Any]]] = None

    def __str__(self) -> str:
        lines = [
            f"FD {self.analysis} sensitivity",
            f"  optimisation: {self.method}",
            f"  sensitivity: {self.sensitivity:.6g}",
            f"  minimum FD:  {self.fd_min:.6g} at lambda_min = {np.round(self.lambda_min, 6)}",
            f"  maximum FD:  {self.fd_max:.6g} at lambda_max = {np.round(self.lambda_max, 6)}",
        ]
        if self.components:
            lines.append("  components:")
            for name, comp in self.components.items():
                lines.append(f"    {name}: sensitivity={comp['sensitivity']:.6g} ({comp['sensitivity_share']:.1%})")
        return "\n".join(lines)
