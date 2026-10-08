import torch


class IsingGradients:
    """Gradient functions for 2D Ising model (no sampler dependency)."""

    def __init__(self, size: int):
        self.size = size
        self.P_mat = self._generate_edge_mat(size)

    def _generate_edge_mat(self, dim: int) -> torch.Tensor:
        padmat = torch.zeros(dim, dim, dim, dim)
        for i in range(dim):
            for j in range(dim):
                if i - 1 != -1:
                    padmat[i][j][i - 1, j] = 1
                if j - 1 != -1:
                    padmat[i][j][i, j - 1] = 1
                if i + 1 != dim:
                    padmat[i][j][i + 1, j] = 1
                if j + 1 != dim:
                    padmat[i][j][i, j + 1] = 1
        return padmat.reshape(dim * dim, dim * dim)

    def stat_m(self, X: torch.Tensor) -> torch.Tensor:
        return (-1 - X) * (X @ self.P_mat)

    def stat_p(self, X: torch.Tensor) -> torch.Tensor:
        return (1 - X) * (X @ self.P_mat)

    def grad_pseudologlikelihood(self, param, X, eps: float = 1e-12) -> torch.Tensor:
        """Gradient of summed log pseudo-likelihood wrt param. Returns shape (M, 1)."""
        X = torch.as_tensor(X, dtype=torch.get_default_dtype())
        H = X @ self.P_mat
        I_minus = (X - 1).abs() / 2
        I_plus = (X + 1).abs() / 2
        p3 = torch.as_tensor(param, dtype=X.dtype).view(-1, 1, 1)
        Hb = H.unsqueeze(0)
        Imb = I_minus.unsqueeze(0)
        Ipb = I_plus.unsqueeze(0)
        a = torch.exp(-Hb / p3)
        b = torch.exp(Hb / p3)
        Nnum = a * Imb + b * Ipb
        Dden = a + b
        term_num = (Imb * a - Ipb * b) / (Nnum + eps)
        term_den = (a - b) / (Dden + eps)
        grad = ((Hb / p3 ** 2) * (term_num - term_den)).sum(dim=(1, 2)).view(-1, 1)
        return grad

    def grad_dfd_loss(self, param, X) -> torch.Tensor:
        """Analytic gradient of DFD loss wrt param. Returns shape (M, 1)."""
        X = torch.as_tensor(X, dtype=torch.get_default_dtype())
        p = torch.as_tensor(param, dtype=X.dtype).view(-1, 1, 1)
        SX_m = self.stat_m(X).unsqueeze(0)
        SX_p = self.stat_p(X).unsqueeze(0)
        ratio_m = torch.exp(SX_m / p)
        inv_ratio_p = torch.exp(-SX_p / p)
        term1 = -(2.0 * SX_m / p ** 2) * ratio_m ** 2
        term2 = -2.0 * (SX_p / p ** 2) * inv_ratio_p
        return ((term1 + term2).sum(dim=(1, 2)) / X.shape[0]).view(-1, 1)
