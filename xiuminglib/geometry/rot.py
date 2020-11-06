import numpy as np

from ..linalg import normalize

from ..log import get_logger
logger = get_logger()


def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        logger.warning((
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians"))


def get_rot_mat(axis, theta):
    """Gets rotation matrix that rotates points around an arbitrary axis by any
    angle.

    Rotating around the X/Y/Z-axis are special cases of this, where you simply
    specify the axis to be one of those axes.

    Args:
        axis (array_like): 3-vector that specifies the end point of the
            rotation axis (start point is the origin). This will be normalized
            to be unit-length.
        theta (float): Angle in radians, prescribed by the right-hand rule, so
            a negative value means flipping the rotation axis.

    Returns:
        numpy.ndarray: 3-by-3 rotation matrix, to be pre-multiplied with the
            vector to rotate.
    """
    # TODO: not tested thoroughly. Use with caution!
    axis = np.array(axis)

    ux, uy, uz = normalize(axis)
    cos = np.cos(theta)
    sin = np.sin(theta)

    r11 = cos + (ux ** 2) * (1 - cos)
    r12 = ux * uy * (1 - cos) - uz * sin
    r13 = ux * uz * (1 - cos) + uy * sin
    r21 = uy * ux * (1 - cos) + uz * sin
    r22 = cos + (uy ** 2) * (1 - cos)
    r23 = uy * uz * (1 - cos) - ux * sin
    r31 = uz * ux * (1 - cos) - uy * sin
    r32 = uz * uy * (1 - cos) + ux * sin
    r33 = cos + (uz ** 2) * (1 - cos)

    rmat = np.array(((r11, r12, r13), (r21, r22, r23), (r31, r32, r33)))
    return rmat
