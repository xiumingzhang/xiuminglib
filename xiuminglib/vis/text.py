from os.path import dirname, abspath, join
import numpy as np

from ..config import create_logger
logger, thisfile = create_logger(abspath(__file__))

from .. import const, os as xm_os
from ..imprt import preset_import


def text_as_image(text, imsize=256, thickness=2, outpath=None):
    """Rasterizes a text string into an image.

    The text will be drawn in black to the center of a white canvas.
    Text size gets automatically figured out based on the provided
    thickness and image size.

    Args:
        text (str): Text to be drawn.
        imsize (float or tuple(float), optional): Output image size.
        thickness (float, optional): Text thickness.
        outpath (str, optional): Where to dump the result to. ``None``
            means ``os.path.join(const.Dir.tmp, 'text_as_image.png')``.

    Writes
        - An image of the text.
    """
    cv2 = preset_import('cv2')

    logger_name = thisfile + '->text_as_image()'

    if isinstance(imsize, int):
        imsize = (imsize, imsize)
    assert isinstance(imsize, tuple), \
        "`imsize` must be an int or a 2-tuple of ints"

    if outpath is None:
        outpath = join(const.Dir.tmp, 'text_as_image.png')

    # Unimportant constants not exposed to the user
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)

    # Base canvas
    dtype = 'uint8'
    dtype_max = np.iinfo(dtype).max
    im = np.ones(imsize, dtype=dtype) * dtype_max

    # Figure out the correct font scale
    font_scale = 1 / 128 # real small
    while True:
        text_size, bl_y = cv2.getTextSize(
            text, font_face, font_scale, thickness)
        if bl_y + text_size[1] >= imsize[1] or text_size[0] >= imsize[0]:
            # Undo the destroying step before breaking
            font_scale /= 2
            text_size, bl_y = cv2.getTextSize(
                text, font_face, font_scale, thickness)
            break
        font_scale *= 2

    # Such that the text is at the center
    bottom_left_corner = (
        (imsize[0] - text_size[0]) // 2,
        (imsize[1] - text_size[1]) // 2 + text_size[1])
    cv2.putText(
        im, text, bottom_left_corner, font_face, font_scale, color, thickness)

    # Write
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)
    cv2.imwrite(outpath, im)

    logger.name = logger_name
    logger.info("Text rasterized into image to:\n%s", outpath)
