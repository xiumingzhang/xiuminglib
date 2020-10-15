import numpy as np

from ..linalg import normalize as normalize_vec


def normalize(normal_map, norm_thres=0.5):
    """Normalizes the normal vector at each pixel of the normal map.

    The normal maps rendered by Blender are *almost* normalized, so this
    function is called by :func:`xiuminglib.io.exr.EXR.extract_normal`.

    Args:
        normal_map (numpy.ndarray): H-by-W-by-3 array of normal vectors.
        norm_thres (float, optional): Normalize only vectors with a norm
            greater than this; helpful to avoid errors at the boundary or
            in the background.

    Returns:
        numpy.ndarray: Normalized normal map.
    """
    norm = np.linalg.norm(normal_map, axis=-1)
    valid = norm > norm_thres
    normal_map[valid] = normalize_vec(normal_map[valid], axis=1)
    return normal_map


def transform_space(normal_map, rotmat):
    """Transforms the normal vectors from one space to another.

    Args:
        normal_map (numpy.ndarray): H-by-W-by-3 array of normal vectors.
        rotmat (numpy.ndarray or mathutils.Matrix): 3-by-3 rotation
            matrix, which is left-multiplied to the vectors.

    Returns:
        numpy.ndarray: Transformed normal map.
    """
    rotmat = np.array(rotmat)
    orig_shape = normal_map.shape
    normal = normal_map.reshape(-1, 3).T # 3-by-N

    normal_trans = rotmat.dot(normal)

    normal_map_trans = normal_trans.T.reshape(orig_shape)
    return normal_map_trans
