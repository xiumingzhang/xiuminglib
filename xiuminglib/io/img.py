from os.path import dirname
import numpy as np
from PIL import Image

from ..log import get_logger
logger = get_logger()

from ..imprt import preset_import
gfile = preset_import('gfile')

from ..os import makedirs


def load(path):
    """Loads an image.

    Args:
        path (str): Path to the image file. Supported formats: whatever Pillow
            supports and HDR.

    Returns:
        numpy.ndarray: Loaded image.
    """
    cv2 = preset_import('cv2')

    open_func = open if gfile is None else gfile.Open

    # EXR
    if path.endswith('.exr'):
        raise ValueError("Use the dedicated `io.exr.EXR()` class for .exr")

    # HDR
    elif path.endswith('.hdr'):
        with open_func(path, 'rb') as h:
            buffer_ = np.fromstring(h.read(), np.uint8)
        img = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Whatever supported by Pillow
    else:
        with open_func(path, 'rb') as h:
            img = Image.open(h)
            img.load()
        img = np.array(img)

    logger.debug("Image loaded from:\n\t%s", path)

    return img


def write_img(arr_uint, outpath):
    r"""Writes an ``uint`` array/image to disk.

    Args:
        arr_uint (numpy.ndarray): A ``uint`` array.
        outpath (str): Output path.

    Writes
        - The resultant image.
    """
    if arr_uint.ndim == 3 and arr_uint.shape[2] == 1:
        arr_uint = np.dstack([arr_uint] * 3)

    img = Image.fromarray(arr_uint)

    # Write to disk
    open_func = open if gfile is None else gfile.Open
    makedirs(dirname(outpath))
    with open_func(outpath, 'wb') as h:
        img.save(h)

    logger.debug("Image written to:\n\t%s", outpath)


def write_arr(arr_0to1, outpath, img_dtype='uint8', clip=False):
    r"""Writes an array to disk as an image.

    Args:
        arr_0to1 (numpy.ndarray): Array with values roughly :math:`\in [0,1]`.
        outpath (str): Output path.
        img_dtype (str, optional): Image data type. Defaults to ``'uint8'``.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Writes
        - The resultant image.

    Returns:
        numpy.ndarray: The resultant image array.
    """
    arr_min, arr_max = arr_0to1.min(), arr_0to1.max()
    if clip:
        if arr_max > 1:
            logger.info("Maximum before clipping: %f", arr_max)
        if arr_min < 0:
            logger.info("Minimum before clipping: %f", arr_min)
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    else:
        assert arr_min >= 0 and arr_max <= 1, \
            "Input should be in [0, 1], or allow it to be clipped"

    # Float array to image
    img_arr = (arr_0to1 * np.iinfo(img_dtype).max).astype(img_dtype)

    write_img(img_arr, outpath)

    return img_arr
