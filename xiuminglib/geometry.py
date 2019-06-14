from os.path import abspath
import numpy as np

from .config import create_logger
logger, thisfile = create_logger(abspath(__file__))


def cartesian2spherical(pts_cartesian, convention='lat-lng'):
    r"""Converts 3D Cartesian coordinates to spherical coordinates.

    Args:
        pts_cartesian (array_like): Cartesian x, y and z. Of shape N-by-3
            or length 3 if just one point.
        convention (str, optional): Convention for spherical coordinates:
            ``'lat-lng'`` or ``'theta-phi'``:

            .. code-block:: none

                   lat-lng
                                            ^ z (lat = 90)
                                            |
                                            |
                       (lng = -90) ---------+---------> y (lng = 90)
                                          ,'|
                                        ,'  |
                   (lat = 0, lng = 0) x     | (lat = -90)

            .. code-block:: none

                theta-phi
                                            ^ z (theta = 0)
                                            |
                                            |
                       (phi = 270) ---------+---------> y (phi = 90)
                                          ,'|
                                        ,'  |
                (theta = 90, phi = 0) x     | (theta = 180)

    Raises:
        ValueError: If input is of a wrong shape.
        NotImplementedError: If convention is neither ``'lat-lng'`` nor ``'theta-phi'``.

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)` in radians.
    """
    pts_cartesian = np.array(pts_cartesian)

    # Validate inputs
    is_one_point = False
    if pts_cartesian.shape == (3,):
        is_one_point = True
        pts_cartesian = pts_cartesian.reshape(1, 3)
    elif pts_cartesian.ndim != 2 or pts_cartesian.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Compute r
    r = np.sqrt(np.sum(np.square(pts_cartesian), axis=1))

    # Compute latitude
    z = pts_cartesian[:, 2]
    lat = np.arcsin(z / r)

    # Compute longitude
    x = pts_cartesian[:, 0]
    y = pts_cartesian[:, 1]
    lng = np.arctan2(y, x) # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = np.stack((r, lat, lng), axis=-1)

    # Select output convention
    if convention == 'lat-lng':
        pts_spherical = pts_r_lat_lng
    elif convention == 'theta-phi':
        pts_spherical = _convert_spherical_conventions(
            pts_r_lat_lng, 'lat-lng_to_theta-phi')
    else:
        raise NotImplementedError(convention)

    if is_one_point:
        pts_spherical = pts_spherical.reshape(3)

    return pts_spherical


def _convert_spherical_conventions(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions
    for spherical coordinates. See :func:`cartesian2spherical` for conventions.
    """
    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_theta_phi

    elif what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > np.pi
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * np.pi
        pts_r_lat_lng[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_lat_lng

    else:
        raise NotImplementedError(what2what)


def spherical2cartesian(pts_spherical, convention='lat-lng'):
    """Inverse of :func:`cartesian2spherical`.

    See :func:`cartesian2spherical`.
    """
    pts_spherical = np.array(pts_spherical)

    # Validate inputs
    is_one_point = False
    if pts_spherical.shape == (3,):
        is_one_point = True
        pts_spherical = pts_spherical.reshape(1, 3)
    elif pts_spherical.ndim != 2 or pts_spherical.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    if (np.abs(pts_spherical[:, 1:]) > 2 * np.pi).any():
        logger.warning(("Some input value falls outside [-2pi, 2pi]. "
                        "Sure inputs are in radians?"))

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_spherical
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_spherical_conventions(
            pts_spherical, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cartesian = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cartesian = pts_cartesian.reshape(3)

    return pts_cartesian


def moeller_trumbore(ray_orig, ray_dir, tri_v0, tri_v1, tri_v2):
    r"""Decides if a ray intersects with a triangle using Moeller-Trumbore algorithm.

    :math:`O + D = (1-u-v)V_0 + uV_1 + vV_2`.

    Args:
        ray_orig (array_like): 3D coordinates of the ray origin :math:`O`.
        ray_dir (array_like): Ray direction :math:`D` (not necessarily normalized).
        tri_v0 (array_like): Triangle vertex :math:`V_0`.
        tri_v1 (array_like): Triangle vertex :math:`V_1`.
        tri_v2 (array_like): Triangle vertex :math:`V_2`.

    Returns:
        tuple:
            - **u** (*float*) -- The :math:`u` component of the Barycentric coordinates
              of the intersection. Intersection is in-triangle (including on an edge
              or at a vertex), if :math:`u\geq 0`, :math:`v\geq 0`, and :math:`u+v\leq 1`.
            - **v** (*float*) -- The :math:`v` component.
            - **t** (*float*) -- Distance coefficient from :math:`O` to the intersection along
              :math:`D`. Intersection is between :math:`O` and :math:`O+D`, if :math:`0 < t < 1`.
    """
    # Validate inputs
    ray_orig = np.array(ray_orig)
    ray_dir = np.array(ray_dir)
    tri_v0 = np.array(tri_v0)
    tri_v1 = np.array(tri_v1)
    tri_v2 = np.array(tri_v2)
    assert (ray_orig.shape == (3,)), "'ray_orig' must be of length 3"
    assert (ray_dir.shape == (3,)), "'ray_dir' must be of length 3"
    assert (tri_v0.shape == (3,)), "'tri_v0' must be of length 3"
    assert (tri_v1.shape == (3,)), "'tri_v1' must be of length 3"
    assert (tri_v2.shape == (3,)), "'tri_v2' must be of length 3"

    M = np.array([-ray_dir, tri_v1 - tri_v0, tri_v2 - tri_v0]).T # noqa: N806
    y = (ray_orig - tri_v0).T
    t, u, v = np.linalg.solve(M, y)

    return u, v, t


def ptcld2tdf(pts, res=128, center=False):
    """Converts point cloud to truncated distance function (TDF).

    Maximum distance is capped at 1 / ``res``.

    Args:
        pts (array_like): Cartesian coordinates in object space. Of shape N-by-3.
        res (int, optional): Resolution of the TDF.
        center (bool, optional): Whether to center these points around the object
            space origin.

    Returns:
        numpy.ndarray: Output TDF.
    """
    pts = np.array(pts)

    n_pts = pts.shape[0]

    if center:
        pts_center = np.mean(pts, axis=0)
        pts -= np.tile(pts_center, (n_pts, 1))

    tdf = np.ones((res, res, res)) / res
    cnt = np.zeros((res, res, res))

    # -0.5 to 0.5 in every dimension
    extent = 2 * np.abs(pts).max()
    pts_scaled = pts / extent

    # Compute distance from center of each involved voxel to its surface points
    for i in range(n_pts):
        pt = pts_scaled[i, :]
        ind = np.floor((pt + 0.5) * (res - 1)).astype(int)
        v_ctr = (ind + 0.5) / (res - 1) - 0.5
        dist = np.linalg.norm(pt - v_ctr)
        n = cnt[ind[0], ind[1], ind[2]]
        tdf[ind[0], ind[1], ind[2]] = (tdf[ind[0], ind[1], ind[2]] * n + dist) / (n + 1)
        cnt[ind[0], ind[1], ind[2]] += 1

    return tdf


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


def normalize(vecs, axis=0):
    """Normalizes one or multiple vectors.

    Args:
        vecs (array_like): 1D array for one vector; 2D array for multiple vectors.
        axis (int, optional): Along which axis normalization is done. Use ``0`` when vectors
            are columns of the 2D array, or ``1`` when vectors are rows.

    Raises:
        ValueError: If ``vecs`` is neither 1D nor 2D, or ``axis`` is illegal.

    Returns:
        numpy.ndarray: Normalized vector(s) of the same shape.
    """
    vecs = np.array(vecs)

    n_dims = vecs.ndim
    if axis < 0:
        raise ValueError("Negative index not allowed for safety")
    elif axis >= n_dims:
        raise ValueError("Can't normalize along axis %d when you only have %d dimension(s)"
                         % (axis, n_dims))

    if n_dims == 1:
        vecs_2d = vecs.reshape((-1, 1))
    elif n_dims == 2:
        vecs_2d = vecs
    else:
        raise ValueError("Input is neither 1D nor 2D, but %dD" % n_dims)
    # Guaranteed to be 2D now

    norms = np.linalg.norm(vecs_2d, axis=axis)
    shape_for_broadcast = [-1, -1]
    shape_for_broadcast[axis] = 1
    vecs_normalized = np.divide(vecs_2d, norms.reshape(shape_for_broadcast)) # normalize

    return vecs_normalized.reshape(vecs.shape)


def barycentric(pts, tvs):
    """Computes barycentric coordinates of 3D point(s) w.r.t. a triangle.

    Args:
        pts (array_like): 3-array for one point; N-by-3 array for multiple points.
        tvs (array_like): 3-by-3 array with rows being the triangle's vertices.

    Returns:
        numpy.ndarray: Barycentric coordinates of the same shape as the input.
    """
    pts = np.array(pts)
    tvs = np.array(tvs)
    input_shape = pts.shape
    if pts.ndim == 1:
        pts = pts.reshape((1, -1))

    vec0 = tvs[1] - tvs[0]
    vec1 = tvs[2] - tvs[0]
    vec2 = pts - tvs[0]
    d00 = vec0.dot(vec0)
    d01 = vec0.dot(vec1)
    d11 = vec1.dot(vec1)
    d20 = vec2.dot(vec0)
    d21 = vec2.dot(vec1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    uvw = np.hstack((u.reshape((-1, 1)), v.reshape((-1, 1)), w.reshape((-1, 1))))
    return uvw.reshape(input_shape)


def main(func_name):
    """Unit tests that can also serve as example usage."""
    if func_name in ('spherical2cartesian', 'cartesian2spherical'):
        # cartesian2spherical() and spherical2cartesian()
        pts_car = np.array([[-1, 2, 3],
                            [4, -5, 6],
                            [3, 5, -8],
                            [-2, -5, 2],
                            [4, -2, -23]])
        print(pts_car)
        pts_sph = cartesian2spherical(pts_car)
        print(pts_sph)
        pts_car_recover = spherical2cartesian(pts_sph)
        print(pts_car_recover)
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
