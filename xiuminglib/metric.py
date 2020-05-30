from os.path import abspath
import numpy as np

from .log import create_logger
logger, thisfile = create_logger(abspath(__file__))

from .imprt import preset_import
tf = preset_import('tf')


class PSNR():
    """Peak Signal-to-Noise Ratio (PSNR).
    """
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        self.dtype_max = np.iinfo(self.dtype).max

    def compute(self, im1, im2, mask=None):
        r"""Computes the metric between a pair of images.

        Args:
            im1 (numpy.ndarray): A grayscale or RGB image.
            im2

        Returns:
            float: PSNR in dB (higher is better).
        """
        # Checks
        for im in (im1, im2):
            assert im.dtype == self.dtype, (
                "Input data type ({in_dtype}) different from what was "
                "specified ({dtype})"
            ).format(in_dtype=im.dtype, dtype=self.dtype)
        assert im1.shape == im2.shape, \
            "The two images are not even of the same shape"
        if mask is None:
            mask = np.ones(im1.shape, dtype=bool)
        else:
            assert mask.shape == im1.shape[:2], \
                "Mask must be 2D and of the same spatial resolution as inputs"
            if im1.ndim == 3:
                mask = np.dstack([mask] * 3)
            mask = mask.astype(bool)
        # Actual computation
        se = np.square(im1[mask] - im2[mask])
        if mask.ndim == 2:
            n = np.sum(mask)
        else:
            n = np.sum(mask[:, :, 0])
        mse = np.sum(se) / n
        psnr = 10 * np.log10((self.dtype_max ** 2) / mse) # dB
        return psnr


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
        # Checks
        assert im1.dtype == im2.dtype, \
            "The two images are not even of the same data type"
        assert im1.shape == im2.shape, \
            "The two images are not even of the same shape"
        for im in (im1, im2):
            assert self.drange >= im.max() - im.min(), \
                "The actual dynamic range is larger than what was specified"
        # Actual computation
        im1 = tf.convert_to_tensor(im1)
        im2 = tf.convert_to_tensor(im2)
        similarity = tf.image.ssim(im1, im2, max_val=self.drange).numpy()
        return similarity
