from os.path import join, dirname

from .. import const, os as xm_os
from .general import _savefig

from ..imprt import preset_import
cv2 = preset_import('cv2')


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
            ``os.path.join(const.Dir.tmp, 'uv_on_texmap.png')``.
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
        outpath = join(const.Dir.tmp, 'uv_on_texmap.png')

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
