from os.path import join, dirname
import numpy as np

from .. import constants, config, os as xm_os
from .general import _savefig

try:
    cv2 = config.import_from_google3('cv2')
except ModuleNotFoundError:
    pass


def rgb_on_uv_canvas(uvs, rgbs,
                     canvas_rgb=(0, 0, 0), canvas_res=(256, 256),
                     max_l1_interp=None, outpath=None):
    r"""Paints colors on a canvas according to their UV locations.

    Args:
        uvs (numpy.ndarray): N-by-2 array of UV coordinates where we have
            values (i.e., colors). See
            :func:`xiuminglib.blender.object.smart_uv_unwrap` for the UV
            coordinate convention.
        rgbs (numpy.ndarray): N-by-3 array of RGB values :math:`\in [0, 1]`.
        canvas_rgb (array_like, optional): Color of the base canvas. Will be
            used to fill in pixels outside the convex hulls formed by the UV
            locations, and if ``max_l1_interp`` is provided, also the
            pixels whose interpolation is too much of a stretch to be
            trusted.
        canvas_size (array_like, optional): Resolution (height first; then
            width) of the visualization. Essentially controls how dense the
            query grid is.
        max_l1_interp (int, optional): Maximum :math:`\ell_1` distance, which
            we can trust in interpolation, to pixels who have values.
            Interpolation over a larger range will not be trusted and hence
            not be painted. ``None`` means trusting (and hence showing) all
            interpolations.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means
            ``os.path.join(constants.Dir.tmp, 'rgb_on_uv_canvas.png')``.

    Writes
        - An interpolated image of the UV-indexed colors.
    """
    from scipy.interpolate import griddata

    dtype = np.uint8
    dtype_max = np.iinfo(dtype).max

    if outpath is None:
        outpath = join(constants.Dir.tmp, 'rgb_on_uv_canvas.png')

    if max_l1_interp is None:
        max_l1_interp = np.inf # trust everything

    h, w = canvas_res
    # Generate query coordinates
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # +---> x
    # |
    # v y
    grid_u, grid_v = grid_x, 1 - grid_y
    # ^ v
    # |
    # +---> u

    # Figure out which pixels can be trusted
    has_value = np.zeros((h, w), dtype=np.uint8)
    ri = ((1 - uvs[:, 1]) * h).astype(int).ravel()
    ci = (uvs[:, 0] * w).astype(int).ravel()
    has_value[ri, ci] = 1
    dist2val = cv2.distanceTransform(1 - has_value, cv2.DIST_L1, 3)
    trusted = dist2val <= max_l1_interp

    # Process each color channel separately
    interps = []
    for ch_i in range(3):
        v_fill = canvas_rgb[ch_i]
        v = rgbs[:, ch_i]
        interp = griddata(uvs, v, (grid_u, grid_v), fill_value=v_fill)
        interp[~trusted] = v_fill
        interps.append(interp)
    rgb = np.dstack(interps) # [0, 1]

    img = (rgb * dtype_max).astype(dtype)
    cv2.imwrite(outpath, img[:, :, ::-1])


def uv_on_texmap(uvs, texmap, ft=None, outpath=None,
                 dotsize=4, dotcolor='r', linewidth=1, linecolor='b'):
    """Visualizes which points on texture map the vertices map to.

    Args:
        uvs (numpy.ndarray): N-by-2 array of UV coordinates. See
            :func:`xiuminglib.blender.object.smart_uv_unwrap` for the UV
            coordinate convention.
        texmap (numpy.ndarray or str): Loaded texture map or its path. If
            *numpy.ndarray*, can be H-by-W (grayscale) or H-by-W-by-3 (color).
        ft (list(list(int)), optional): Texture faces used to connect the
            UV points. Values start from 1, e.g., ``'[[1, 2, 3], [],
            [2, 3, 4, 5], ...]'``.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means
            ``os.path.join(constants.Dir.tmp, 'uv_on_texmap.png')``.
        dotsize (int or list(int), optional): Size(s) of the UV dots.
        dotcolor (str or list(str), optional): Their color(s).
        linewidth (float, optional): Width of the lines connecting the dots.
        linecolor (str, optional): Their color.

    Raises:
        ValueError: If ``texmap`` is of the wrong value.

    Writes
        - An image of where the vertices map to on the texture map.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if outpath is None:
        outpath = join(constants.Dir.tmp, 'uv_on_texmap.png')

    # Preprocess input
    if isinstance(texmap, str):
        texmap = cv2.imread(
            texmap, cv2.IMREAD_UNCHANGED)[:, :, ::-1] # made RGB
    if len(texmap.shape) == 2:
        add_colorbar = True # for grayscale
    elif len(texmap.shape) == 3:
        add_colorbar = False # for color texture maps
    else:
        raise ValueError(
            ("texmap must be either H-by-W (grayscale) or H-by-W-by-3 "
             "(color), or a path to such images"))

    dpi = 96 # assumed
    h, w = texmap.shape[:2]
    w_in, h_in = w / dpi, h / dpi
    fig = plt.figure(figsize=(w_in, h_in))

    u, v = uvs[:, 0], uvs[:, 1]
    # ^ v
    # |
    # +---> u
    x, y = u * w, (1 - v) * h
    #   +---> x
    #   |
    #   v y

    # UV dots
    ax = fig.gca()
    ax.set_xlim([min(0, min(x)), max(w, max(x))])
    ax.set_ylim([max(h, max(y)), min(0, min(y))])
    im = ax.imshow(texmap, cmap='gray')
    ax.scatter(x, y, c=dotcolor, s=dotsize, zorder=2)
    ax.set_aspect('equal')

    # Connect these dots
    if ft is not None:
        lines = []
        for vert_id in [x for x in ft if x]: # non-empty ones
            assert min(vert_id) >= 1, "Indices in ft are 1-indexed"
            # For each face
            ind = [i - 1 for i in vert_id]
            n_verts = len(ind)
            for i in range(n_verts):
                lines.append([
                    (x[ind[i]], y[ind[i]]),
                    (x[ind[(i + 1) % n_verts]], y[ind[(i + 1) % n_verts]])
                ]) # line start and end
        line_collection = LineCollection(
            lines, linewidths=linewidth, colors=linecolor, zorder=1)
        ax.add_collection(line_collection)

    # Make directory, if necessary
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    # Colorbar
    if add_colorbar:
        # Create an axes on the right side of ax. The width of cax will be 2%
        # of ax and the padding between cax and ax will be fixed at 0.1 inch.
        cax = make_axes_locatable(ax).append_axes('right', size='2%', pad=0.2)
        plt.colorbar(im, cax=cax)

    # Save
    contents_only = not add_colorbar
    _savefig(outpath, contents_only=contents_only)

    plt.close('all')
