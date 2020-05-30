from os.path import abspath
import numpy as np

from .log import create_logger
logger, thisfile = create_logger(abspath(__file__))

from .img import rgb2lum
from .imprt import preset_import
tf = preset_import('tf')


class Base():
    """The base metric.

    Args:
        dtype (str): Data type, from which dynamic range will be derived.

    Attributes:
        dtype (str): Data type, with which data dynamic range is derived.
        drange (float): Dynamic range, i.e., difference between the maximum and
            minimum allowed.
    """
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        if self.dtype.kind == 'f':
            self.drange = 1
            logger.name = thisfile + '->Base'
            logger.warning(
                "Input type is float, so assuming dynamic range to be 1")
        elif self.dtype.kind == 'u':
            iinfo = np.iinfo(self.dtype)
            self.drange = iinfo.max - iinfo.min
        else:
            raise NotImplementedError(self.dtype.kind)

    def _assert_type(self, im):
        assert im.dtype == self.dtype, (
            "Input data type ({in_dtype}) different from what was "
            "specified ({dtype})"
        ).format(in_dtype=im.dtype, dtype=self.dtype)

    def _assert_drange(self, im):
        actual = im.max() - im.min()
        assert self.drange >= actual, (
            "The actual dynamic range ({actual}) is larger than what was "
            "derived from the data type ({derived})"
        ).format(actual=actual, derived=self.drange)

    @staticmethod
    def _assert_same_shape(im1, im2):
        assert im1.shape == im2.shape, \
            "The two images are not even of the same shape"

    @staticmethod
    def _assert_ch(im):
        assert im.ndim == 3, \
            "Input must be 3D (H-by-W-by-C), but is %dD" % im.ndim
        assert im.shape[2] in (1, 3), \
            "Input must have either 1 or 3 channels, but has %d" % im.shape[2]

    def compute(self, im1, im2, **kwargs):
        r"""Computes the metric between a pair of images.

        Args:
            im1 (numpy.ndarray): An H-by-W-by-3 (or 1) image.
            im2

        Returns:
            float: The metric.
        """
        raise NotImplementedError


class PSNR(Base):
    """Peak Signal-to-Noise Ratio (PSNR) in dB (higher is better).

    If the inputs are RGB, they are first converted to luma (or relative
    luminance, if the inputs are not gamma-corrected). PSNR is computed
    on the luma.
    """
    def compute(self, im1, im2, mask=None):
        r"""Computes the metric between a pair of images.

        Args:
            im1
            im2
            mask (numpy.ndarray): A grayscale or RGB image.

        Returns:
            float: PSNR in dB (higher is better).
        """
        # Checks
        self._assert_type(im1)
        self._assert_type(im2)
        self._assert_same_shape(im1, im2)
        self._assert_ch(im1)
        self._assert_ch(im2)
        self._assert_drange(im1)
        self._assert_drange(im2)
        # Mask
        if mask is None:
            mask = np.ones(im1.shape, dtype=bool)
        else:
            assert mask.shape == im1.shape[:2], \
                "Mask must be 2D and of the same spatial resolutions as inputs"
            if im1.ndim == 3:
                mask = np.dstack([mask] * 3)
            mask = mask.astype(bool)
        # Number of spatial locations
        if mask.ndim == 2:
            n = np.sum(mask)
        else:
            n = np.sum(mask[:, :, 0])
        # Actual computation
        se = np.square(im1[mask] - im2[mask])
        mse = np.sum(se) / n
        psnr = 10 * np.log10((self.drange ** 2) / mse) # dB
        return psnr


class SSIM(Base):
    r"""The Structural Similarity Index (SSIM) :math:`\in [0,1]`
    (higher is better).

    If the inputs are RGB, they are first converted to luma (or relative
    luminance, if the inputs are not gamma-corrected). SSIM is computed
    on the luma.
    """
    def __init__(self, dtype):
        super().__init__(dtype)
        assert tf is not None, "TensorFlow import failed"

    def compute(self, im1, im2):
        # Checks
        self._assert_type(im1)
        self._assert_type(im2)
        self._assert_same_shape(im1, im2)
        self._assert_ch(im1)
        self._assert_ch(im2)
        self._assert_drange(im1)
        self._assert_drange(im2)
        # To luma
        if im1.shape[2] == 3:
            im1 = rgb2lum(im1)
            im2 = rgb2lum(im2)
        # Guaranteed to be H-by-W-by-1 now
        im1 = tf.convert_to_tensor(im1)
        im2 = tf.convert_to_tensor(im2)
        similarity = tf.image.ssim(im1, im2, max_val=self.drange).numpy()
        return similarity
