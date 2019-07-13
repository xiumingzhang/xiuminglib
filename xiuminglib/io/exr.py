from os.path import abspath, join
import numpy as np

try:
    import OpenEXR
    import Imath
except ModuleNotFoundError:
    pass

from .. import config
logger, thisfile = config.create_logger(abspath(__file__))

from ..imprt import import_from_google3
from ..vis.geometry import depth_as_image, normal_as_image
from ..vis.matrix import matrix_as_image
from ..geometry.normal import normalize


class EXR():
    """Reads EXR files.

    EXR files can be generic or physically meaningful, such as depth, normal,
    etc. When data loaded are physically meaningful, these methods assume the
    EXR files are produced by :mod:`xiuminglib.blender.render` and hence
    follow certain formats.

    Args:
        exr_path (str, optional): Path to the EXR file.

    Attributes:
        exr_f (str): Path to the EXR file.
        data (dict): Data loaded.
    """
    def __init__(self, exr_path=None):
        self.exr_f = exr_path
        if self.exr_f is None:
            self.data = None
        else:
            self.data = self.load()

    def load(self):
        r"""Loads an EXR as a dictionary of NumPy arrays.

        Uses OpenEXR for :math:`> 3` channels. If :math:`\leq 3` channels,
        can just use OpenCV.

        Returns:
            dict: Loaded EXR data.
        """
        logger_name = thisfile + '->EXR:load()'
        assert self.exr_f is not None, "You need to set exr_f first"
        f = OpenEXR.InputFile(self.exr_f)
        pix_type = Imath.PixelType(Imath.PixelType.FLOAT)
        data_win = f.header()['dataWindow']
        win_size = (data_win.max.y - data_win.min.y + 1,
                    data_win.max.x - data_win.min.x + 1)
        data = {}
        for c in f.header()['channels']:
            arr = np.fromstring(f.channel(c, pix_type), dtype=np.float32)
            data[c] = arr.reshape(win_size)
        logger.name = logger_name
        logger.info("Loaded %s", self.exr_f)
        return data

    def extract_rgb(self, outpath, vis=False):
        """Extracts a RGB(A) array to .npy from .exr.

        All values remain raw in the conversion from .exr to .npy.
        Tonemapping and then visualizing it as an image is a separate issue.

        Args:
            outpath (str): Path to the result .npy file.
            vis (bool, optional): Whether to tonemap and visualize it as an
                image.

        Writes
            - A .npy file containing the raw RGB(A) values.
            - If ``vis``, a .png image, naively tonemapped from the raw values.
        """
        logger_name = thisfile + '->EXR:extract_rgb()'
        data = [self.data['R'], self.data['G'], self.data['B']]
        if 'A' in self.data.keys():
            data.append(self.data['A'])
        data = np.dstack(data)
        if not outpath.endswith('.npy'):
            outpath += '.npy'
        np.save(outpath, data)
        if vis:
            matrix_as_image(data, outpath[:-4] + '.png')
        logger.name = logger_name
        logger.info("RGB image extractd to %s", outpath)

    def extract_depth(self, alpha_exr, outpath, vis=False):
        """Combines an aliased .exr depth map and an anti-aliased .exr alpha
        map into a single RGBA .npy depth map.

        All values remain raw in the conversion from .exr to .npy.

        Args:
            alpha_exr (str): Path to the EXR file of the anti-aliased alpha
                map.
            outpath (str): Path to the result .npy file.
            vis (bool, optional): Whether to visualize it as an image.

        Writes
            - A .npy file containing an aliased depth map and its alpha map.
            - If ``vis``, a .png image of anti-aliased depth.
        """
        cv2 = import_from_google3('cv2')
        logger_name = thisfile + '->EXR:extract_depth()'

        def assert_all_channels_same(arr):
            for i in range(1, arr.shape[-1]):
                assert (arr[..., i - 1] == arr[..., i]).all(), \
                    "A valid map must have all channels the same"
            return arr[..., 0]

        # Load alpha
        arr = cv2.imread(alpha_exr, cv2.IMREAD_UNCHANGED)
        alpha = assert_all_channels_same(arr)
        # Load depth
        arr = cv2.imread(self.exr_f, cv2.IMREAD_UNCHANGED)
        depth = assert_all_channels_same(arr) # these raw values are aliased,
        # so only one crazy big value for the background
        if not outpath.endswith('.npy'):
            outpath += '.npy'
        np.save(outpath, np.dstack((arr, alpha)))
        if vis:
            depth_as_image(depth, alpha, outpath[:-4] + '.png')
        logger.name = logger_name
        logger.info("Depth image extractd to %s", outpath)

    def extract_normal(self, outpath, negate=False, vis=False):
        """Converts an RGBA EXR normal map to an RGBA .npy normal map.

        Args:
            outpath (str): Path to the result .npy file.
            negate (bool, optional): Whether to negate the normals. If the
                loaded data are from a non-world-space (i.e., "camera-space")
                .exr, then you need to set this to ``True`` to get the normals
                really in the camera space. See the warning in
                :func:`xiuminglib.blender.render.render_normal`.
            vis (bool, optional): Whether to visualize it as an image.

        Writes
            - A .npy file containing an aliased normal map and its alpha map.
            - If ``vis``, a .png visualization of anti-aliased normals.
        """
        logger_name = thisfile + '->extract_normal()'
        # Load RGBA .exr
        data = self.data
        arr = np.dstack((data['R'], data['G'], data['B']))
        if negate:
            arr = -arr
        arr = normalize(arr)
        alpha = data['A']
        if not outpath.endswith('.npy'):
            outpath += '.npy'
        np.save(outpath, np.dstack((arr, alpha)))
        if vis:
            normal_as_image(arr, alpha, outpath[:-4] + '.png')
        logger.name = logger_name
        logger.info("Normal image extractd to %s", outpath)

    def extract_intrinsic_images_from_lighting_passes(self, outdir,
                                                      vis=False):
        """Extract intrinsic images from an EXR of lighting passes into
        multiple .npy files.

        Args:
            outdir (str): Directory to save the result .npy files to.
            vis (bool, optional): Whether to visualize the values as images.

        Writes
            - albedo.npy (and its visualization if ``vis``).
            - shading.npy (ditto).
            - specularity.npy (ditto).
            - recon.npy (ditto): reconstruction by combining albedo, shading,
              and specularity.
            - composite.npy (ditto): composite by Blender.
        """
        from ..vis.matrix import matrix_as_image, os as xm_os
        logger_name = thisfile + \
            '->extract_intrinsic_images_from_lighting_passes()'
        xm_os.makedirs(outdir)
        data = self.data

        def collapse_passes(components):
            ch_arrays = []
            for ch in ['R', 'G', 'B']:
                comp_arrs = []
                for comp in components:
                    comp_arrs.append(data[comp + '.' + ch])
                ch_array = np.sum(comp_arrs, axis=0) # sum components
                ch_arrays.append(ch_array)
            # Handle alpha channel
            first_alpha = data[components[0] + '.A']
            for ci in range(1, len(components)):
                assert (first_alpha == data[components[ci] + '.A']).all(), \
                    "Alpha channels of all passes must be the same"
            ch_arrays.append(first_alpha)
            return np.dstack(ch_arrays)

        # Albedo
        albedo = collapse_passes(['diffuse_color', 'glossy_color'])
        np.save(join(outdir, 'albedo.npy'), albedo)
        if vis:
            matrix_as_image(albedo, outpath=join(outdir, 'albedo.png'))
        # Shading
        shading = collapse_passes(['diffuse_indirect', 'diffuse_direct'])
        np.save(join(outdir, 'shading.npy'), shading)
        if vis:
            matrix_as_image(shading, outpath=join(outdir, 'shading.png'))
        # Specularity
        specularity = collapse_passes(['glossy_indirect', 'glossy_direct'])
        np.save(join(outdir, 'specularity.npy'), specularity)
        if vis:
            matrix_as_image(
                specularity, outpath=join(outdir, 'specularity.png'))
        # Reconstruction vs. ...
        recon = np.multiply(albedo, shading) + specularity
        recon[:, :, 3] = albedo[:, :, 3] # can't add up alpha channels
        np.save(join(outdir, 'recon.npy'), recon)
        if vis:
            matrix_as_image(recon, outpath=join(outdir, 'recon.png'))
        # ... composite from Blender, just for sanity check
        composite = collapse_passes(['composite'])
        np.save(join(outdir, 'composite.npy'), composite)
        if vis:
            matrix_as_image(composite, outpath=join(outdir, 'composite.png'))
        logger.name = logger_name
        logger.info("Intrinsic images extracted to %s", outdir)


def main():
    """Unit tests that can also serve as example usage."""
    from .. import constants
    exr_f = join(constants.Dir.tmp, 'test.exr')
    exr = EXR(exr_f)
    exr.extract_normal(exr_f.replace('.exr', '.npy'), vis=True)


if __name__ == '__main__':
    main()
