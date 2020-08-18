from os.path import join, dirname
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..log import get_logger
logger = get_logger()

from .. import const
from ..io import img as imgio
from ..os import makedirs
from ..imprt import preset_import


def make_anim(
        imgs, labels=None, label_top_left_xy=None, font_size=24,
        font_color=(1, 0, 0), font_ttf=None, duration=1, outpath=None):
    r"""Writes a list of (optionally labeled) images into an animation.

    Args:
        imgs (list(numpy.ndarray or str)): An image is either a path or an
            array (mixing ok, but arrays will need to be written to a temporary
            directory). If array, should be of type ``uint`` and of shape H-by-W
            (grayscale) or H-by-W-by-3 (RGB).
        labels (list(str), optional): Labels used to annotate the images.
        label_top_left_xy (tuple(int), optional): The XY coordinate of the
            label's top left corner.
        font_size (int, optional): Font size.
        font_color (tuple(float), optional): Font RGB, normalized to
            :math:`[0,1]`.
        font_ttf (str, optional): Path to the .ttf font file. Defaults to Arial.
        duration (float, optional): Duration of each frame in seconds.
        outpath (str, optional): Where to write the output to (a .apng or .gif
            file). ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_anim.gif')``.

    Raises:
        TypeError: If any input image is neither a string nor an array.

    Writes
        - An animation of the images.
    """
    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_anim.gif')
    if not outpath.endswith(('.apng', '.gif')):
        outpath += '.gif'
    makedirs(dirname(outpath))

    # Font
    if font_ttf is None:
        font = ImageFont.truetype(const.Path.open_sans_regular, font_size)
    else:
        gfile = preset_import('gfile')
        open_func = open if gfile is None else gfile.Open
        with open_func(font_ttf, 'rb') as h:
            font_bytes = BytesIO(h.read())
        font = ImageFont.truetype(font_bytes, font_size)

    def put_text(img, text):
        if label_top_left_xy is None:
            top_left_xy = (int(0.1 * img.width), int(0.05 * img.height))
        dtype_max = np.iinfo(np.array(img).dtype).max
        color = tuple(int(x * dtype_max) for x in font_color)
        ImageDraw.Draw(img).text(top_left_xy, text, fill=color, font=font)
        return img

    imgs_loaded = []
    for img_i, img in enumerate(imgs):
        if isinstance(img, str):
            # Path
            img = imgio.load(img)
            if labels is not None:
                img = put_text(img, labels[img_i])
            imgs_loaded.append(img)
        elif isinstance(img, np.ndarray):
            # Array
            assert np.issubdtype(img.dtype, np.unsignedinteger), \
                "If image is provided as an array, it has to be `uint`"
            if (img.ndim == 3 and img.shape[2] == 1) or img.ndim == 2:
                img = np.dstack([img] * 3)
            img = Image.fromarray(img)
            if labels is not None:
                img = put_text(img, labels[img_i])
            imgs_loaded.append(img)
        else:
            raise TypeError(type(img))

    duration = duration * 1000 # because in ms

    gfile = preset_import('gfile')
    open_func = open if gfile is None else gfile.Open
    with open_func(outpath, 'wb') as h:
        imgs_loaded[0].save(
            h, save_all=True, append_images=imgs_loaded[1:],
            duration=duration, loop=0)

    logger.info("Images written as an animation to:\n\t%s", outpath)
