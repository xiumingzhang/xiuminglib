from os.path import abspath
import numpy as np
from PIL import Image

from .. import config
logger, thisfile = config.create_logger(abspath(__file__))


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
