import numpy as np


def get(arr, top=True, n=1, n_std=None):
    """Gets top (or bottom) N value(s) from an M-D array.

    Args:
        arr (array_like): Array, which will be flattened if high-D.
        top (bool, optional): Whether to find the top or bottom N.
        n (int, optional): Number of values to return.
        n_std (float, optional): Definition of outliers to exclude, assuming Gaussian.
            ``None`` means assuming no outlier.

    Returns:
        tuple:
            - **ind** (*tuple*) -- Indices that give the extrema, M-tuple of arrays of N integers.
            - **val** (*numpy.ndarray*) -- Extremum values, i.e., ``arr[ind]``.
    """
    arr = np.array(arr, dtype=float)

    if top:
        arr_to_sort = -arr.flatten()
    else:
        arr_to_sort = arr.flatten()

    if n_std is not None:
        meanv = np.mean(arr_to_sort)
        stdv = np.std(arr_to_sort)
        arr_to_sort[np.logical_or(
            arr_to_sort < meanv - n_std * stdv,
            arr_to_sort > meanv + n_std * stdv,
        )] = np.nan # considered greater than numbers

    ind = [x for x in np.argsort(arr_to_sort)
           if not np.isnan(arr_to_sort[x])][:n] # 1D indices
    ind = np.unravel_index(ind, arr.shape) # Back to high-D
    val = arr[ind]

    return ind, val


def is_symmetric(mat, eps=None):
    """Checks if a matrix is symmetric.

    If the input is not even square, ``False`` is returned.

    Args:
        mat (numpy.ndarray): Input matrix.
        eps (float, optional): Numerical tolerance for equality. ``None`` means
            ``np.finfo(mat.dtype).eps``.

    Returns:
        bool: Whether the input is symmetric.
    """
    if eps is None:
        eps = np.finfo(mat.dtype).eps

    assert mat.ndim == 2
    if mat.shape[0] != mat.shape[1]:
        return False

    return np.allclose(mat, mat.T, atol=eps)


def is_identity(mat, eps=None):
    """Checks if a matrix is an identity matrix.

    If the input is not even square, ``False`` is returned.

    Args:
        mat (numpy.ndarray): Input matrix.
        eps (float, optional): Numerical tolerance for equality. ``None`` means
            ``np.finfo(mat.dtype).eps``.

    Returns:
        bool: Whether the input is an identity matrix.
    """
    if eps is None:
        eps = np.finfo(mat.dtype).eps

    assert mat.ndim == 2
    if mat.shape[0] != mat.shape[1]:
        return False

    return np.allclose(mat, np.eye(mat.shape[0]), atol=eps)


def angle_between(vec1, vec2, radian=True):
    r"""Computes the angle between two vectors.

    Args:
        vec1 (numpy.ndarray): Vector 1.
        vec2
        radian (bool, optional): Whether to use radians.

    Returns:
        float: The angle :math:`\in [0,\pi]`.
    """
    cos = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    angle = np.arccos(np.clip(cos, -1, 1))
    if not radian:
        angle = angle / np.pi * 180
    return angle


def to_homo(pts):
    """Pads 2D or 3D points to homogeneous, by guessing which dimension to pad.

    Args:
        pts (numpy.ndarray): Input array of 2D or 3D points.

    Raises:
        ValueError: If ``pts`` is ambiguous to guess.

    Returns:
        numpy.ndarray: Homogeneous coordinates of the input points.
    """
    if pts.ndim == 1:
        pts_homo = np.hstack((pts, 1))
    elif pts.ndim == 2:
        err_str = " (assumed to be # points) must be >3 to be not ambiguous"
        h, w = pts.shape
        if h > w: # tall
            assert h > 3, "Input has height (%d) > width (%d); the height" % (h, w) + err_str
            pts_homo = np.hstack((pts, np.ones((h, 1))))
        elif h < w: # fat
            assert w > 3, "If input has width (%d) > height (%d); the width" % (w, h) + err_str
            pts_homo = np.vstack((pts, np.ones((1, w))))
        else: # square
            raise ValueError("Ambiguous square matrix that I can't guess how to pad")
    else:
        raise ValueError(pts.ndim)
    return pts_homo


def from_homo(pts, axis):
    """Converts from homogeneous to non-homogeneous coordinates.

    Args:
        pts (numpy.ndarray): Input array of 2D or 3D points.
        axis (int): The last slice of which dimension holds the w values.

    Returns:
        numpy.ndarray: Non-homogeneous coordinates of the input points.
    """
    arr = np.take(pts, range(pts.shape[axis] - 1), axis=axis)
    w = np.take(pts, -1, axis=axis)
    pts_nonhomo = np.divide(arr, w) # by broadcasting
    return pts_nonhomo


def main(func_name):
    """Unit tests that can also serve as example usage."""
    if func_name == 'is_symmetric':
        mat = np.random.random((10, 9))
        print(is_symmetric(mat))
        mat = np.random.random((10, 10))
        print(is_symmetric(mat))
        mat = np.random.random((10, 10))
        mat = mat + mat.T
        print(is_symmetric(mat))
    elif func_name == 'to_homo':
        arr = np.array([2, 3, 4])
        print(arr)
        print("to")
        print(to_homo(arr))
        print("~~~~~~")
        # arr = np.array([[2, 3, 4]])
        # print(arr)
        # print("to")
        # print(to_homo(arr))
        # print("~~~~~~")
        arr = np.array([[2, 3, 4, 5]])
        print(arr)
        print("to")
        print(to_homo(arr))
        print("~~~~~~")
        # arr = np.array([[2, 3, 4], [2, 8, 3], [2, 9, 3]])
        # print(arr)
        # print("to")
        # print(to_homo(arr))
        # print("~~~~~~")
        arr = np.array([[2, 3, 4], [2, 8, 3], [2, 9, 3], [2, 9, 3]])
        print(arr)
        print("to")
        print(to_homo(arr))
    else:
        raise NotImplementedError("Unit tests for %s" % func_name)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('func', type=str, help="function to test")
    args = parser.parse_args()

    main(args.func)
