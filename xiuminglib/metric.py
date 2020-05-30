from os.path import abspath
import numpy as np

from .log import create_logger
logger, thisfile = create_logger(abspath(__file__))

from .img import rgb2lum
from .const import Path

from .imprt import preset_import
tf = preset_import('tf')
gfile = preset_import('gfile')


class Base():
    """The base metric.

    Attributes:
        dtype (str): Data type, with which data dynamic range is derived.
        drange (float): Dynamic range, i.e., difference between the maximum and
            minimum allowed.
    """
    def __init__(self, dtype):
        """
        Args:
            dtype (str): Data type, from which dynamic range will be derived.
        """
        self.dtype = np.dtype(dtype)
        if self.dtype.kind == 'f':
            self.drange = 1.
            logger.name = thisfile + '->Base'
            logger.warning(
                "Input type is float, so assuming dynamic range to be 1")
        elif self.dtype.kind == 'u':
            iinfo = np.iinfo(self.dtype)
            self.drange = float(iinfo.max - iinfo.min)
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

    def __call__(self, im1, im2, **kwargs):
        """
        Args:
            im1 (numpy.ndarray): An H-by-W-by-3 (or 1) image.
            im2

        Returns:
            float: The metric computed.
        """
        raise NotImplementedError


class PSNR(Base):
    """Peak Signal-to-Noise Ratio (PSNR) in dB (higher is better).

    If the inputs are RGB, they are first converted to luma (or relative
    luminance, if the inputs are not gamma-corrected). PSNR is computed
    on the luma.
    """
    def __call__(self, im1, im2, mask=None):
        """
        Args:
            im1
            im2
            mask (numpy.ndarray, optional): An H-by-W-by-1 mask.

        Returns:
            float: PSNR in dB.
        """
        self._assert_type(im1)
        self._assert_type(im2)
        im1 = im1.astype(float) # must be cast to an unbounded type
        im2 = im2.astype(float)
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
        if mask is None:
            mask = np.ones(im1.shape)
        mask = mask.astype(bool)
        assert mask.shape == im1.shape, (
            "Mask must be of shape {input_shape}, but is of shape "
            "{actual}"
        ).format(input_shape=im1.shape, actual=mask.shape)
        se = np.square(im1[mask] - im2[mask])
        mse = np.sum(se) / np.sum(mask)
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

    def __call__(self, im1, im2):
        self._assert_type(im1)
        self._assert_type(im2)
        im1 = im1.astype(float) # must be cast to an unbounded type
        im2 = im2.astype(float)
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


class LPIPS(Base):
    r"""The Learned Perceptual Image Patch Similarity (LPIPS) metric (lower is
    better).

    Project page: https://richzhang.github.io/PerceptualSimilarity/

    Note:
        This implementation assumes the minimum value allowed is :math:`0`, so
        data dynamic range becomes the maximum value allowed.

    Attributes:
        dtype (str): Data type, with which data dynamic range is derived.
        drange (float): Dynamic range, i.e., difference between the maximum and
            minimum allowed.
        lpips_func (tf.function): The LPIPS network packed into a function.
    """
    def __init__(self, dtype, weight_pb=None):
        """
        Args:
            dtype (str): Data type, from which maximum allowed will be derived.
            weight_pb (str, optional): Path to the network weight protobuf.
                Defaults to the bundled ``net-lin_alex_v0.1.pb``.
        """
        super().__init__(dtype)
        assert tf is not None, "TensorFlow import failed"
        open_func = open if gfile is None else gfile.Open
        if weight_pb is None:
            weight_pb = Path.lpips_weights
        # Pack LPIPS network into a tf function
        graph_def = tf.compat.v1.GraphDef()
        with open_func(weight_pb, 'rb') as h:
            graph_def.ParseFromString(h.read())
        self.lpips_func = tf.function(self._wrap_frozen_graph(
            graph_def, inputs=['0:0', '1:0'], outputs='Reshape_10:0'))

    @staticmethod
    def _wrap_frozen_graph(graph_def, inputs, outputs):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))

    def __call__(self, im1, im2):
        """
        Args:
            im1 (numpy.ndarray): An H-by-W-by-3 image.
            im2

        Returns:
            float: The LPIPS metric computed (lower is better).
        """
        self._assert_type(im1)
        self._assert_type(im2)
        im1 = im1.astype(float) # must be cast to an unbounded type
        im2 = im2.astype(float)
        self._assert_same_shape(im1, im2)
        self._assert_ch(im1)
        self._assert_ch(im2)
        self._assert_drange(im1)
        self._assert_drange(im2)
        maxv = self.drange + 0 # NOTE: assumes the minimum value allowed is 0
        im1t = tf.convert_to_tensor(
            np.expand_dims(im1, axis=0), dtype=float) / maxv * 2 - 1
        im2t = tf.convert_to_tensor(
            np.expand_dims(im2, axis=0), dtype=float) / maxv * 2 - 1
        # Now 1xHxWxC and all values in [-1, 1]
        lpips = self.lpips_func(
            tf.transpose(im1t, [0, 3, 1, 2]), # to NxCxHxW
            tf.transpose(im2t, [0, 3, 1, 2])
        ).numpy().squeeze()[()]
        return lpips
