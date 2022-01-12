import math
import numpy as np

from ..linalg import normalize

from ..log import get_logger
logger = get_logger()


def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        logger.warning((
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians"))


def is_rot_mat(mat, tol=1e-6):
    r"""Checks if a matrix is a valid rotation matrix.

    Args:
        mat (numpy.ndarray): A :math:`3\times 3` matrix.
        tol (float, optional): Tolerance for checking if all close.

    Returns:
        bool: Whether this is a valid rotation matrix.
    """
    mat_t = np.transpose(mat)
    should_be_identity = np.dot(mat_t, mat)
    identity = np.identity(3, dtype=mat.dtype)
    return np.allclose(identity, should_be_identity, atol=tol)
