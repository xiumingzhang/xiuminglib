from os.path import abspath, join
import numpy as np

try:
    import OpenEXR
    import Imath
except ModuleNotFoundError:
    pass

from .. import config
logger, thisfile = config.create_logger(abspath(__file__))


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

    @staticmethod
    def vis_depth(depth_map, alpha_map, outpath):
        """Visualizes a(n) (aliased) depth map and an (anti-aliased) alpha map
        as a single depth image.

        Output has black background, with bright values for closeness to the
        camera. If the alpha map is anti-aliased, the result depth map will
        be nicely anti-aliased.

        Args:
            depth_map (numpy.ndarray): 2D array of (aliased) raw depth values.
            alpha_map (numpy.ndarray): 2D array of (anti-aliased) alpha
                values.
            outpath (str): Path to the result .png file.

        Writes
            - The (anti-aliased) depth image.
        """
        cv2 = config.import_from_google3('cv2')
        dtype = 'uint8'
        dtype_max = np.iinfo(dtype).max
        is_fg = depth_map < depth_map.max()
        max_val = depth_map[is_fg].max()
        depth_map[depth_map > max_val] = max_val # cap background depth at the
        # object maximum depth
        min_val = depth_map.min()
        im = dtype_max * (max_val - depth_map) / (max_val - min_val)
        # Now [0, dtype_max]
        # Anti-aliasing
        bg = np.zeros(im.shape)
        im = np.multiply(alpha_map, im) + np.multiply(1 - alpha_map, bg)
        cv2.imwrite(outpath, im.astype(dtype))

    def extract_depth(self, alpha_exr, outpath, vis=False):
        """Combines an aliased .exr depth map and an anti-aliased .exr alpha
        map into a single RGBA .npy depth map.

        All values remain raw in the conversion from .exr to .npy.

        Args:
            alpha_exr (str): Path to the EXR file of the anti-aliased alpha
                map.
            outpath (str): Path to the result .npy file.
            vis (bool, optional): Whether to visualize the raw values as an
                image.

        Writes
            - A .npy file containing an aliased depth map and its alpha map.
            - If ``vis``, a .png image of anti-aliased depth.
        """
        cv2 = config.import_from_google3('cv2')
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
            self.vis_depth(depth, alpha, outpath[:-4] + '.png')
        logger.name = logger_name
        logger.info("Depth image extractd to %s", outpath)

    @staticmethod
    def normalize_normal(normal_map):
        """Normalizes the normal vector at each pixel of the normal map.

        The normal maps rendered by Blender are *almost* normalized, so this
        function is called by :func:`xiuminglib.io.exr.EXR.extract_normal`.

        Args:
            normal_map (numpy.ndarray): H-by-W-by-3 array of normal vectors.

        Returns:
            numpy.ndarray: Normalized normal map.
        """
        norm = np.linalg.norm(normal_map, axis=-1)
        valid = norm > 0.5
        normal_map[valid] = normal_map[valid] / norm[valid][..., None]
        return normal_map

    @staticmethod
    def vis_normal(normal_map, alpha_map, outpath):
        """Visualizes the normal map by converting vectors to pixel values.

        The background is black, complying with industry standards (e.g.,
        Adobe AE).

        Args:
            normal_map (numpy.ndarray): H-by-W-by-3 array of normal vectors.
            alpha_map (numpy.ndarray): H-by-W array of alpha values.
            outpath (str): Where to save the visualization to.

        Writes
            - The normal image.
        """
        cv2 = config.import_from_google3('cv2')
        dtype = 'uint8'
        dtype_max = np.iinfo(dtype).max
        # [-1, 1]
        im = (normal_map / 2 + 0.5) * dtype_max
        # [0, dtype_max]
        bg = np.zeros(im.shape)
        alpha = np.dstack([alpha_map] * 3)
        im = np.multiply(alpha, im) + np.multiply(1 - alpha, bg)
        cv2.imwrite(outpath, im.astype(dtype)[..., ::-1])

    @staticmethod
    def transform_normal(normal_map, rotmat):
        """Transforms the normal vectors from one space to another.

        Args:
            normal_map (numpy.ndarray): H-by-W-by-3 array of normal vectors.
            rotmat (numpy.ndarray or mathutils.Matrix): 3-by-3 rotation
                matrix.

        Returns:
            numpy.ndarray: Transformed normal map.
        """
        rotmat = np.array(rotmat)
        orig_shape = normal_map.shape
        normal = normal_map.reshape(-1, 3).T # 3-by-N
        normal_trans = rotmat.dot(normal)
        normal_map_trans = normal_trans.T.reshape(orig_shape)
        return normal_map_trans

    def extract_normal(self, outpath, negate=False, vis=False):
        """Converts an RGBA EXR normal map to an RGBA .npy normal map.

        Args:
            outpath (str): Path to the result .npy file.
            negate (bool, optional): Whether to negate the normals. If the
                loaded data are from a non-world-space (i.e., "camera-space")
                .exr, then you need to set this to ``True`` to get the normals
                really in the camera space. See the warning in
                :func:`xiuminglib.blender.render.render_normal`.
            vis (bool, optional): Whether to visualize the normal vectors as
                an image.

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
        arr = self.normalize_normal(arr)
        alpha = data['A']
        if not outpath.endswith('.npy'):
            outpath += '.npy'
        np.save(outpath, np.dstack((arr, alpha)))
        if vis:
            self.vis_normal(arr, alpha, outpath[:-4] + '.png')
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
        from .. import vis as xm_vis, os as xm_os
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
            xm_vis.matrix_as_image(albedo,
                                   outpath=join(outdir, 'albedo.png'))
        # Shading
        shading = collapse_passes(['diffuse_indirect', 'diffuse_direct'])
        np.save(join(outdir, 'shading.npy'), shading)
        if vis:
            xm_vis.matrix_as_image(shading,
                                   outpath=join(outdir, 'shading.png'))
        # Specularity
        specularity = collapse_passes(['glossy_indirect', 'glossy_direct'])
        np.save(join(outdir, 'specularity.npy'), specularity)
        if vis:
            xm_vis.matrix_as_image(specularity,
                                   outpath=join(outdir, 'specularity.png'))
        # Reconstruction vs. ...
        recon = np.multiply(albedo, shading) + specularity
        recon[:, :, 3] = albedo[:, :, 3] # can't add up alpha channels
        np.save(join(outdir, 'recon.npy'), recon)
        if vis:
            xm_vis.matrix_as_image(recon, outpath=join(outdir, 'recon.png'))
        # ... composite from Blender, just for sanity check
        composite = collapse_passes(['composite'])
        np.save(join(outdir, 'composite.npy'), composite)
        if vis:
            xm_vis.matrix_as_image(composite,
                                   outpath=join(outdir, 'composite.png'))
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
