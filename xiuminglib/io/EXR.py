from os import makedirs
from os.path import abspath, dirname, exists, join, basename
import numpy as np
import cv2

import config
logger, thisfile = config.create_logger(abspath(__file__))


def load(exr_path):
    """
    Load .exr as a dict, by converting it to a .npz and loading that .npz

    Args:
        exr_path: Path to the .exr file
            String

    Returns:
        data: Loaded OpenEXR data
            dict
    """
    from time import time
    import subprocess

    npz_f = '/tmp/%s_t%s.npz' % \
        (basename(exr_path).replace('.exr', ''), time())

    # Convert to .npz
    cwd = join(
        dirname(abspath(__file__)),
        '..', '..', 'commandline'
    )
    bash_cmd = 'python2 exr2npz.py %s %s' % (exr_path, npz_f)
    process = subprocess.Popen(
        bash_cmd.split(),
        stdout=subprocess.PIPE,
        cwd=cwd,
    )
    _, _ = process.communicate()

    # Load this .npz
    data = np.load(npz_f)
    return data


def extract_depth(exr_prefix, outpath, vis=False):
    """
    Combine an aliased, raw depth map and an anti-aliased alpha map into a .png image,
        with background being black, close to camera being bright, and far away being dark

    Args:
        exr_prefix: Common prefix of .exr paths
            String
        outpath: Path to the result .npy file
            String
        vis: Whether to visualize the raw values as an image
            Boolean
            Optional; defaults to False
    """
    logger_name = thisfile + '->extract_depth()'

    dtype = 'uint8'
    dtype_max = np.iinfo(dtype).max

    # Load alpha
    arr = cv2.imread(exr_prefix + '_a.exr', cv2.IMREAD_UNCHANGED)
    assert (arr[:, :, 0] == arr[:, :, 1]).all() and (arr[:, :, 1] == arr[:, :, 2]).all(), \
        "A valid alpha map must have all three channels the same"
    alpha = arr[:, :, 0]

    # Load depth
    arr = cv2.imread(exr_prefix + '_z.exr', cv2.IMREAD_UNCHANGED)
    assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all(), \
        "A valid depth map must have all three channels the same"
    depth = arr[..., 0] # these raw values are aliased, so only one crazy big value

    if not outpath.endswith('.npy'):
        outpath += '.npy'
    np.save(outpath, np.dstack((arr, alpha)))

    if vis:
        is_fg = depth < depth.max()
        max_val = depth[is_fg].max()
        depth[depth > max_val] = max_val # cap background depth at the object maximum depth
        min_val = depth.min()

        im = dtype_max * (max_val - depth) / (max_val - min_val) # [0, dtype_max]

        # Anti-aliasing
        bg = np.zeros(im.shape)
        im = np.multiply(alpha, im) + np.multiply(1 - alpha, bg)
        cv2.imwrite(outpath[:-4] + '.png', im.astype(dtype))

    logger.name = logger_name
    logger.info("Depth image extractd at %s", outpath)


def extract_normal(exr_path, outpath, vis=False):
    """
    Convert an RGBA .exr normal map to a .png normal image, with background being black
        and complying with industry standards (e.g., Adobe AE)

    Args:
        exr_path: Path to the .exr file to convert
            String
        outpath: Path to the result .npy file
            String
        vis: Whether to visualize the raw values as an image
            Boolean
            Optional; defaults to False
    """
    logger_name = thisfile + '->extract_normal()'

    dtype = 'uint8'
    dtype_max = np.iinfo(dtype).max

    # Load RGBA .exr
    # cv2.imread() can't load more than three channels from .exr even with IMREAD_UNCHANGED
    # Has to go through IO. Maybe there's a better way?
    data = load(exr_path)
    arr = np.dstack((data['R'], data['G'], data['B']))
    alpha = data['A']

    if not outpath.endswith('.npy'):
        outpath += '.npy'
    np.save(outpath, np.dstack((arr, alpha)))

    if vis:
        # [-1, 1]
        im = (1 - (arr / 2 + 0.5)) * dtype_max
        # [0, dtype_max]
        bg = np.zeros(im.shape)
        alpha = np.dstack((alpha, alpha, alpha))
        im = np.multiply(alpha, im) + np.multiply(1 - alpha, bg)
        cv2.imwrite(outpath[:-4] + '.png', im.astype(dtype)[..., ::-1])

    logger.name = logger_name
    logger.info("Normal image extractd at %s", outpath)


def extract_intrinsic_images_from_lighting_passes(exr_path, outdir, vis=False):
    """
    Extract intrinsic images from a multi-layer .exr of lighting passes
        into multiple .npy files

    Args:
        exr_path: Path to the multi-layer .exr file
            String
        outdir: Directory to the result .png files to
            String
        vis: Whether to visualize the raw values as images
            Boolean
            Optional; defaults to False
    """
    from xiuminglib import visualization as xv

    logger_name = thisfile + '->extract_intrinsic_images_from_lighting_passes()'

    if not exists(outdir):
        makedirs(outdir)

    data = load(exr_path)

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
        xv.matrix_as_image(albedo, outpath=join(outdir, 'albedo.png'))

    # Shading
    shading = collapse_passes(['diffuse_indirect', 'diffuse_direct'])
    np.save(join(outdir, 'shading.npy'), shading)
    if vis:
        xv.matrix_as_image(shading, join(outdir, 'shading.png'))

    # Specularity
    specularity = collapse_passes(['glossy_indirect', 'glossy_direct'])
    np.save(join(outdir, 'specularity.npy'), specularity)
    if vis:
        xv.matrix_as_image(specularity, join(outdir, 'specularity.png'))

    # Reconstruction vs.
    recon = np.multiply(albedo, shading) + specularity
    recon[:, :, 3] = albedo[:, :, 3] # can't add up alpha channels
    np.save(join(outdir, 'recon.npy'), recon)
    if vis:
        xv.matrix_as_image(recon, join(outdir, 'recon.png'))

    # ... composite from Blender, just for sanity check
    composite = collapse_passes(['composite'])
    np.save(join(outdir, 'composite.npy'), composite)
    if vis:
        xv.matrix_as_image(composite, join(outdir, 'composite.png'))

    logger.name = logger_name
    logger.info("Intrinsic images extracted to %s", outdir)
