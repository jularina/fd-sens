import numpy as np


def is_symmetric(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    return np.allclose(matrix, matrix.T, atol=tol)


def is_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    eigvals = np.linalg.eigvalsh(matrix)
    return np.all(eigvals >= -tol)


def is_symmetric_and_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    return is_symmetric(matrix, tol) and is_psd(matrix, tol)
