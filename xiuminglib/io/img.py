from os.path import abspath, dirname
import numpy as np
from PIL import Image

from .. import config
logger, thisfile = config.create_logger(abspath(__file__))

from ..os import makedirs


def load_as_array(path):
    """Loads the image file as an array.

    Args:
        path (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded image.
    """
    logger_name = thisfile + '->load_as_array()'

    with open(path, 'rb') as h:
        img = Image.open(h)
        img.load()

    logger.name = logger_name
    logger.debug("Image loaded from:\n\t%s", path)
    return np.array(img)


def write_array(arr_0to1, outpath, img_dtype='uint8', clip=False):
    r"""Writes an array to disk as an image.

    Args:
        arr_0to1 (numpy.ndarray): Array with values roughly :math:`\in [0,1]`.
        outpath (str): Output path.
        img_dtype (str, optional): Image data type. Defaults to ``'uint8'``.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Returns:
        numpy.ndarray: The resultant image array.
    """
    logger_name = thisfile + '->write_array()'

    if clip:
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    assert arr_0to1.min() >= 0 and arr_0to1.max() <= 1, \
        "Input should be in [0, 1], or allow it to be clipped"

    # Float array to image
    img_arr = (arr_0to1 * np.iinfo(img_dtype).max).astype(img_dtype)
    img = Image.fromarray(img_arr)

    # Write to disk
    makedirs(dirname(outpath))
    with open(outpath, 'wb') as h:
        img.save(h)

    logger.name = logger_name
    logger.debug("Image written to:\n\t%s", outpath)
    return img_arr
