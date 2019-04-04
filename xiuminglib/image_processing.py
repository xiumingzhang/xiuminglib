from os.path import abspath
from copy import deepcopy
import numpy as np
import cv2

from xiuminglib import config
logger, thisfile = config.create_logger(abspath(__file__))


def binarize(im, threshold=None):
    """Binarizes images.

    Args:
        im (numpy.ndarray): Image to binarize. Of any integer type (``uint8``, ``uint16``, etc.).
            If H-by-W-by-3, will be converted to grayscale and treated as H-by-W.
        threshold (float, optional): Threshold for binarization. ``None`` means midpoint of the ``dtype``.

    Raises:
        ValueError: If ``im`` has wrong dimensions.

    Returns:
        numpy.ndarray: Binarized image. Of only 0's and 1's.
    """
    im_copy = deepcopy(im)

    # RGB to grayscale
    if im_copy.ndim == 3 and im_copy.shape[2] == 3: # h-by-w-by-3
        im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2GRAY)

    if im_copy.ndim == 2: # h-by-w
        # Compute threshold from data type
        if threshold is None:
            maxval = np.iinfo(im_copy.dtype).max
            threshold = maxval / 2.
        im_bin = im_copy
        logicalmap = im_copy > threshold
        im_bin[logicalmap] = 1
        im_bin[np.logical_not(logicalmap)] = 0
    else:
        raise ValueError("'im' is neither h-by-w nor h-by-w-by-3")

    return im_bin


def remove_islands(im, min_n_pixels, connectivity=4):
    """Removes small islands of pixels from a binary image.

    Args:
        im (numpy.ndarray): Input binary image. Of only 0's and 1's.
        min_n_pixels (int): Minimum island size to keep.
        connectivity (int, optional): Definition of "connected": either 4 or 8.

    Returns:
        numpy.ndarray: Output image with small islands removed.
    """
    # Validate inputs
    assert (len(im.shape) == 2), "'im' needs to have exactly two dimensions"
    assert np.array_equal(np.unique(im), np.array([0, 1])), "'im' needs to contain only 0's and 1's"
    assert (connectivity == 4 or connectivity == 8), "'connectivity' must be either 4 or 8"

    # Find islands, big or small
    nlabels, labelmap, leftx_topy_bbw_bbh_npix, _ = \
        cv2.connectedComponentsWithStats(im, connectivity)

    # Figure out background is 0 or 1
    bgval = im[labelmap == 0][0]

    # Set small islands to background value
    im_clean = im
    for i in range(1, nlabels): # skip the 0th island -- background
        island_size = leftx_topy_bbw_bbh_npix[i, -1]
        if island_size < min_n_pixels:
            im_clean[labelmap == i] = bgval

    return im_clean


def query_float_locations(im, query_pts, method='bilinear'):
    """Queries interpolated values of float lactions on image.

    Uses
        -- Bilinear interpolation: Can break big matrices into patches and work locally, or
        -- Bivariate spline interpolation: Fitting a global spline, so memory-intensive
           and shows global effects.
    Pixel values are considered as values at pixel centers. E.g., if ``im[0, 1]`` is :math:``0.68``,
    then :math:`f(0.5, 1.5)` is deemed to evaluate to :math:`0.68` exactly.

    Args:
        im (numpy.ndarray): H-by-W or H-by-W-by-C Rectangular grid of data.
            Each of C channels is interpolated independently.
        query_pts (array_like): Query locations of shape N-by-2 or length 2::

            +---------> dim1
            |
            |
            |
            v dim0

        method (str, optional): Interpolation method: ``'spline'`` or ``'bilinear'``.

    Raises:
        ValueError: If input is of a wrong shape or dimensions.
        NotImplementedError: If interpolation method is not implemented.

    Returns:
        numpy.ndarray: Interpolated values at query locations.
    """
    from scipy.interpolate import RectBivariateSpline, interp2d

    logger_name = thisfile + '->query_float_locations()'

    # Figure out image size and number of channels
    if im.ndim == 3:
        h, w, c = im.shape
        if c == 1: # single dimension
            im = im[:, :, 0]
    elif im.ndim == 2:
        h, w = im.shape
        c = 1
    else:
        raise ValueError("'im' must have either two or three dimensions")

    # Validate inputs
    query_pts = np.array(query_pts)
    is_one_point = False
    if query_pts.shape == (2,):
        is_one_point = True
        query_pts = query_pts.reshape(1, 2)
    elif query_pts.ndim != 2 or query_pts.shape[1] != 2:
        raise ValueError("Shape of input must be either (2,) or (n, 2)")

    # Querying one point, very likely in a loop -- no printing
    if is_one_point:
        logger.name = logger_name
        logger.setLevel(config.logging_warn)

    x = np.arange(h) + 0.5 # pixel center
    y = np.arange(w) + 0.5
    query_x = query_pts[:, 0]
    query_y = query_pts[:, 1]

    if np.min(query_x) < 0 or np.max(query_x) > h or \
            np.min(query_y) < 0 or np.max(query_y) > w:
        logger.name = logger_name
        logger.warning("Sure you want to query points outside 'im'?")

    if c == 1:
        # Single channel
        z = im

        logger.name = logger_name
        logger.info("Interpolation (method: %s) started", method)

        if method == 'spline':
            spline_obj = RectBivariateSpline(x, y, z)
            interp_val = spline_obj(query_x, query_y, grid=False)

        elif method == 'bilinear':
            f = interp2d(y, x, z, kind='linear')
            interp_val = f(query_y, query_x)

        else:
            raise NotImplementedError("Other interplation methods")

        logger.name = logger_name
        logger.info("    ... done")

    else:
        # Multiple channels
        logger.name = logger_name
        logger.warning("Support for 'im' having multiple channels has not been thoroughly tested!")

        interp_val = np.zeros((len(query_x), c))
        for i in range(c):

            z = im[:, :, i]

            logger.name = logger_name
            logger.info("Interpolation (method: %s) started for channel %d/%d", method, i + 1, c)

            if method == 'spline':
                spline_obj = RectBivariateSpline(x, y, z)
                interp_val[:, i] = spline_obj(query_x, query_y, grid=False)

            elif method == 'bilinear':
                f = interp2d(y, x, z, kind='linear')
                interp_val[:, i] = f(query_y, query_x)

            else:
                raise NotImplementedError("Other interplation methods")

            logger.name = logger_name
            logger.info("    ... done")

    if is_one_point:
        interp_val = interp_val.reshape(c)

    return interp_val


def find_local_extrema(im, want_maxima, kernel_size=3):
    """Finds local maxima or minima in an image.

    Args:
        im (numpy.ndarray): H-by-W if single-channel (e.g., grayscale) or H-by-W-by-C for multi-channel
            (e.g., RGB) images. Extrema are found independently for each of the C channels.
        want_maxima (bool): Whether maxima or minima are wanted.
        kernel_size (int, optional): Side length of the square window under consideration. Must be
            larger than 1.

    Raises:
        ValueError: If the input image has wrong dimensions.

    Returns:
        numpy.ndarray: Binary map indicating if each pixel is a local extremum.
    """
    from scipy.ndimage.filters import minimum_filter, maximum_filter

    logger_name = thisfile + '->find_local_extrema()'

    logger.name = logger_name
    logger.error("find_local_extrema() not tested yet!")

    # Figure out image size and number of channels
    if im.ndim == 3:
        h, w, c = im.shape
        expanded = False
    elif im.ndim == 2:
        h, w = im.shape
        c = 1
        im = np.expand_dims(im, axis=2) # adds singleton dimension
        expanded = True
    else:
        raise ValueError("'im' must have either two or three dimensions")

    kernel = np.ones((kernel_size, kernel_size)).astype(bool)

    is_extremum = np.zeros((h, w, c), dtype=bool)

    for i in range(c):
        z = im[:, :, i]

        if want_maxima:
            equals_extremum = maximum_filter(z, footprint=kernel) == z
        else:
            equals_extremum = minimum_filter(z, footprint=kernel) == z

        is_extremum[:, :, i] = equals_extremum

    if expanded:
        is_extremum = is_extremum[:, :, 0]

    return is_extremum


def compute_gradients(im):
    """Computes magnitudes and orientations of image gradients.

    With Scharr operators::

        [ 3 0 -3 ]           [ 3  10  3]
        [10 0 -10]    and    [ 0   0  0]
        [ 3 0 -3 ]           [-3 -10 -3]

    Args:
        im (numpy.ndarray): H-by-W if single-channel (e.g., grayscale) or H-by-W-by-C if multi-channel
            (e.g., RGB) images. Gradients are computed independently for each of the C channels.

    Raises:
        ValueError: If ``im`` has wrong dimensions.

    Returns:
        tuple:
            - **grad_mag** (*numpy.ndarray*) -- Magnitude image of the gradients.
            - **grad_orient** (*numpy.ndarray*) -- Orientation image of the gradients (in radians)::

                       y ^ pi/2
                         |
                pi       |
                 --------+--------> 0
                -pi      |       x
                         | -pi/2
    """
    # Figure out image size and number of channels
    if im.ndim == 3:
        h, w, c = im.shape
        expanded = False
    elif im.ndim == 2:
        h, w = im.shape
        c = 1
        im = np.expand_dims(im, axis=2) # adds singleton dimension
        expanded = True
    else:
        raise ValueError("'im' must have either two or three dimensions")

    grad_mag = np.zeros((h, w, c))
    grad_orient = np.zeros((h, w, c))

    for i in range(c):
        z = im[:, :, i]
        ddepth = -1 # same depth as the source

        # Along horizontal direction
        xorder, yorder = 1, 0
        grad_h = cv2.Sobel(z, ddepth, xorder, yorder, ksize=-1) # 3x3 Scharr
        grad_h = grad_h.astype(float)

        # Along vertical direction
        xorder, yorder = 0, 1
        grad_v = cv2.Sobel(z, ddepth, xorder, yorder, ksize=-1) # 3x3 Scharr
        grad_v = grad_v.astype(float)

        # Magnitude
        grad_mag[:, :, i] = np.sqrt(np.square(grad_h) + np.square(grad_v))

        # Orientation
        grad_orient[:, :, i] = np.arctan2(grad_v, grad_h)

    if expanded:
        grad_mag = grad_mag[:, :, 0]
        grad_orient = grad_orient[:, :, 0]

    return grad_mag, grad_orient


def gamma_correct(im, gamma):
    r"""Applies gamma correction to image.

    Args:
        im (numpy.ndarray): H-by-W if single-channel (e.g., grayscale) or H-by-W-by-C multi-channel
            (e.g., RGB) images.
        gamma (float): Gamma value :math:`< 1` shifts image towards the darker end of the spectrum,
            while value :math:`> 1` towards the brighter.

    Returns:
        numpy.ndarray: Gamma-corrected image.
    """
    assert im.dtype in ('uint8', 'uint16')

    # Don't correct alpha channel, if exists
    alpha = None
    if im.ndim == 2 and im.shape[2] == 4:
        alpha = im[:, :, 3]
        im = im[:, :, :3]

    # Correct with lookup table
    type_max = np.iinfo(im.dtype).max
    table = np.array([
        ((x / type_max) ** (1 / gamma)) * type_max for x in np.arange(0, type_max + 1)
    ]).astype(im.dtype)
    im_corrected = cv2.LUT(im, table)

    # Concat alpha channel back
    if alpha is not None:
        im_corrected = np.dstack((im_corrected, alpha))

    return im_corrected
