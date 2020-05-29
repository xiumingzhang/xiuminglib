from os.path import abspath
import numpy as np

from .log import create_logger
logger, thisfile = create_logger(abspath(__file__))

from .imprt import preset_import
tf = preset_import('tf')


class PSNR():
    """Peak Signal-to-Noise Ratio (PSNR).
    """
    def __init__(self):
        pass

    def compute(self, im1, im2):
        r"""Computes the metric between a pair of images.

        Args:
            im1 (numpy.ndarray): A grayscale or RGB image.
            im2

        Returns:
            float: SSIM score :math:`\in [0, 1]` (higher is better).
        """
        from IPython import embed; embed()


class SSIM():
    """The Structural Similarity Index (SSIM).

    Args:
        drange (float): Dynamic range, i.e., difference between the maximum and
            minimum allowed.

    Attributes:
        drange (float)
    """
    def __init__(self, drange):
        self.drange = drange
        assert tf is not None, "TensorFlow import failed"

    def compute(self, im1, im2):
        r"""Computes the metric between a pair of images.

        Args:
            im1 (numpy.ndarray): An H-by-W-by-1/3 image.
            im2

        Returns:
            float: SSIM score :math:`\in [0, 1]` (higher is better).
        """
        assert im1.dtype == im2.dtype, \
            "The two images are not even of the same data type"
        for im in (im1, im2):
            assert self.drange >= im.max() - im.min(), \
                "The actual dynamic range is larger than what was specified"
        im1 = tf.convert_to_tensor(im1)
        im2 = tf.convert_to_tensor(im2)
        similarity = tf.image.ssim(im1, im2, max_val=self.drange).numpy()
        return similarity
