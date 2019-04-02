"""This module should be imported before ``skimage`` to
avoid the ``matplotlib`` backend problem."""

from os import makedirs, environ
from os.path import dirname, exists, abspath, join
from pickle import dump
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D # noqa; pylint: disable=unused-import

from xiuminglib import config
logger, thisfile = config.create_logger(abspath(__file__))


def pyplot_wrapper(*args,
                   ci=None,
                   func='plot',
                   labels=None,
                   legend_fontsize=20,
                   legend_loc=0,
                   figsize=(14, 14),
                   figtitle=None,
                   figtitle_fontsize=20,
                   xlabel=None,
                   xlabel_fontsize=20,
                   ylabel=None,
                   ylabel_fontsize=20,
                   xticks=None,
                   xticks_locations=None,
                   xticks_fontsize=10,
                   xticks_rotation=0,
                   yticks=None,
                   yticks_locations=None,
                   yticks_fontsize=10,
                   yticks_rotation=0,
                   xlim=None,
                   ylim=None,
                   grid=True,
                   outpath=None,
                   **kwargs):
    """
    Convinience wrapper for matplotlib.pyplot functions that saves plots directly to the disk
        without displaying

    Args:
        *args, **kwargs: Positional and/or keyword parameters that the wrapped function takes
            See documentation for matplotlib.pyplot
        ci: Confidence interval for x_i[j] is y_i[j] +/- ci[i][j]; effective only when `func` is 'plot'
            List of floats for one line; list of lists of floats for multiple lines
            Optional; defaults to None
        func: Which pyplot function to invoke
            'plot', 'hist', or 'bar'
            Optional; defaults to 'plot'
        labels: Labels for plot objects, to appear in the legend
            List of strings or None (no label for this object)
            Optional; defaults to None (no legend)
        legend_loc: Legend location; effective only when `labels` is not None
            'best' | 'upper right' | 'lower left' | 'right' | 'center left' | 'lower center' |
                'upper center' | 'center' | etc.
            Optional; defaults to 'best'
        figsize: Width and height of the figure in inches
            Tuple of two positive floats
            Optional; defaults to (14, 14)
        figtitle: Figure title
            String
            Optional; defaults to None (no title)
        xlabel, ylabel: Label of x- or y-axis
            String
            Optional; defaults to None (no label)
        xticks, yticks: Tick values of x- or y-axis
            Array_like
            Optional; defaults to None (auto)
        xticks_locations, yticks_locations: Locations of the ticks
            Array_like of floats
            Optional; defaults to None (starting from 0, one next to another)
        *_fontsize: Font size
            Positive integer
            Optional
        *_rotation: Tick rotation in degrees
            Float
            Optional; defaults to 0
        xlim, ylim: Start and end values for x- and y-axes
            List of two float(s) or None(s) (auto)
            Optional; defaults to None (auto)
        grid: Whether to draw grid
            Boolean
            Optional; defaults to True
        outpath: Path to which the visualization is saved
            String
            Optional; defaults to '$TMP_DIR/plot.png'
    """
    if ci is not None:
        assert func == 'plot', "CI makes sense only for `plot`"

    if outpath is None:
        outpath = join(environ.get('TMP_DIR', '~'), 'plot.png')

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Set title
    if figtitle is not None:
        ax.set_title(figtitle, fontsize=figtitle_fontsize)

    if func == 'plot':
        func = plt.plot
    elif func == 'hist':
        func = plt.hist
    elif func == 'bar':
        func = plt.bar
    elif func == 'boxplot':
        func = plt.boxplot
    else:
        raise NotImplementedError(func)

    plot_objs = func(*args, **kwargs)

    # Confidence intervals
    if ci is not None:
        # `func` is 'plot'
        if isinstance(ci[0], (int, float)):
            # List of numbers -> only one line
            assert len(plot_objs) == 1, "Only one CI is provided, but there are more than one lines"
            ci = np.array(ci)
            assert (ci > 0).all(), "CI should be positive"
            ci = [ci]
        elif isinstance(ci[0], (list, np.ndarray)):
            # List of lists -> multiple lines
            assert len(ci) == len(plot_objs), "Numbers of CI's and lines are different"
            ci = [np.array(x) for x in ci]
            for x in ci:
                assert (x > 0).all(), "CI should be positive"
        else:
            raise TypeError(ci)
        # `ci` is now a list of numpy array(s)
        for i, plot_obj in enumerate(plot_objs):
            x, y = plot_obj.get_data()
            ub = y + ci[i]
            lb = y - ci[i]
            plt.fill_between(x, ub, lb, color=plot_obj.get_c(), alpha=.5)

    # Legend
    if labels is not None:
        n_plot_objs = len(plot_objs)
        assert (len(labels) == n_plot_objs), \
            "Number of labels must equal number of plot objects; use None for object without a label"
        for i in range(n_plot_objs):
            plot_objs[i].set_label(labels[i])
        plt.legend(fontsize=legend_fontsize, loc=legend_loc)

    # Grid
    plt.grid(grid)

    # Axis limits
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

    # Axis labels
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=ylabel_fontsize)

    # Axis ticks
    if xticks is None:
        plt.xticks(fontsize=xticks_fontsize, rotation=xticks_rotation)
    else:
        if xticks_locations is None:
            xticks_locations = range(len(xticks))
        plt.xticks(xticks_locations, xticks, fontsize=xticks_fontsize, rotation=xticks_rotation)
    if yticks is None:
        plt.yticks(fontsize=yticks_fontsize, rotation=yticks_rotation)
    else:
        if yticks_locations is None:
            yticks_locations = range(len(yticks))
        plt.yticks(yticks_locations, yticks, fontsize=yticks_fontsize, rotation=yticks_rotation)

    # Make directory, if necessary
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir, exist_ok=True)

    # Save plot
    plt.savefig(outpath, bbox_inches='tight')

    plt.close('all')


def scatter_on_image(im, pts, size=2, bgr=(0, 0, 255), outpath=None):
    """
    Scatter plot on top of an image

    Args:
        im: Image to scatter on
            h-by-w (grayscale) or h-by-w-by-3 (RGB) numpy array of type np.uint8 or np.uint16
        pts: Coordinates of the scatter point(s)
            +-----------> dim1
            |
            |
            |
            v dim0
            Array_like of length 2 or shape (n, 2)
        size: Size(s) of scatter points
            Positive float or array_like thereof of length n
            Optional; defaults to 2
        bgr: BGR color(s) of scatter points
            3-tuple of integers ranging from 0 to 255 or array_like thereof of shape (n, 3)
            Optional; defaults to (0, 0, 255), i.e., all red
        outpath: Path to which the visualization is saved
            String
            Optional; defaults to '$TMP_DIR/scatter_on_image.png'
    """
    import cv2

    logger_name = thisfile + '->scatter_on_image()'

    if outpath is None:
        outpath = join(environ.get('TMP_DIR', '~'), 'scatter_on_image.png')

    thickness = -1 # for filled circles

    # Standardize inputs
    if im.ndim == 2: # grayscale
        im = np.dstack((im, im, im)) # to BGR
    pts = np.array(pts)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    n_pts = pts.shape[0]

    if im.dtype != 'uint8' and im.dtype != 'uint16':
        logger.name = logger_name
        logger.warning("Input image type may cause obscure cv2 errors")

    if isinstance(size, int):
        size = np.array([size] * n_pts)
    else:
        size = np.array(size)

    bgr = np.array(bgr)
    if bgr.ndim == 1:
        bgr = np.tile(bgr, (n_pts, 1))

    # FIXME: necessary, probably due to OpenCV bugs?
    im = im.copy()

    # Put on scatter points
    for i in range(pts.shape[0]):
        uv = tuple(pts[i, ::-1].astype(int))
        color = (int(bgr[i, 0]), int(bgr[i, 1]), int(bgr[i, 2]))
        cv2.circle(im, uv, size[i], color, thickness)

    # Make directory, if necessary
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir, exist_ok=True)

    # Write to disk
    cv2.imwrite(outpath, im)


def matrix_as_image(arr, outpath=None, gamma=None):
    """
    Visualize an array into an image by putting minimum (across all channels) at 0
        and maximum at dtype_max

    Args:
        arr: Array to be transformed into an image
            2D or 3D numpy array with one or three channels in the third dimension (RGB)
        outpath: Where to visualize the result
            String
            Optional; defaults to '$TMP_DIR/matrix_as_image.png'
        gamma: For gamma correction
            Positive float
            Optional; defaults to None (no correction)
    """
    import cv2
    from xiuminglib import image_processing as xi

    logger_name = thisfile + '->matrix_as_image()'

    if outpath is None:
        outpath = join(environ.get('TMP_DIR', '~'), 'matrix_as_image.png')

    dtype = 'uint8'
    dtype_max = np.iinfo(dtype).max

    arr = arr.astype(float)
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape + (1,))
    elif arr.ndim == 3:
        assert (arr.shape[-1] in (1, 3, 4)), \
            ("Only single- (grayscale), three- (RGB), "
             "or four-channel (RGBA) matrices are supported")
    else:
        raise ValueError("'arr' needs to be either 2D or 3D")

    n_chs = arr.shape[-1]
    if n_chs == 4:
        arr, a = arr[:, :, :3], arr[:, :, 3]
        assert a.min() >= 0 and a.max() == 1, "Alpha must be [0, 1]"
        im_a = (a * dtype_max).astype(dtype)
    else:
        a = None

    minv, maxv = arr.min(), arr.max()
    if minv == maxv:
        im = (arr * dtype_max).astype(dtype)
        logger.name = logger_name
        logger.warning(
            ("RGB channels contain only a single value: %f, so only operations "
             "performed: multiplied by dtype_max and cast to integer"), maxv)
    else:
        im = (dtype_max * (arr - minv) / (maxv - minv)).astype(dtype)
        # astype() safe only because we know it's [0, dtype_max]

    if a is not None:
        im = np.dstack((im, im_a))

    if gamma is not None:
        im = xi.gamma_correct(im, gamma)

    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir)

    if im.shape[-1] == 4:
        # RGBA
        cv2.imwrite(outpath, im[:, :, [2, 1, 0, 3]]) # OpenCV uses BGR
    else:
        # Grayscale or RGB
        cv2.imwrite(outpath, im[:, :, ::-1])


def make_colormap(low, high):
    """
    Generates your own colormap for heatmap

    Args:
        low, high: Colors for the lowest/highest value
            String, such as 'red', or 3-tuple, such as (1, 0, 0)

    Returns:
        cmap: Generated colormap
            matplotlib.colors.LinearSegmentedColormap
    """
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, str):
        low = c(low)
    if isinstance(high, str):
        high = c(high)
    seq = [(None,) * 3, 0.0] + [low, high] + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, x in enumerate(seq):
        if isinstance(x, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([x, r1, r2])
            cdict['green'].append([x, g1, g2])
            cdict['blue'].append([x, b1, b2])
    cmap = mcolors.LinearSegmentedColormap('CustomMap', cdict)
    return cmap


def matrix_as_heatmap(mat, cmap='viridis', center_around_zero=False,
                      outpath=None, contents_only=False, figtitle=None):
    """
    Visualizes a matrix as heatmap
        Functional with matplotlib 2.0.2, but buggy with 3.0.0

    Args:
        mat: Matrix to visualize as heatmp
            2D numpy array that may contain NaN's, which will be plotted white
        cmap: Colormap to use
            String or any colormap type
            Optional; defaults to 'viridis'
        center_around_zero: Whether to center colorbar around 0 (so that zero is no color, i.e., white)
            Useful when matrix consists of both positive and negative values, and 0 means "nothing"
            Boolean
            Optional; defaults to False (default colormap and auto range)
        outpath: Path to which the visualization is saved
            String
            Optional; defaults to '$TMP_DIR/matrix_as_heatmap.png'
        contents_only: Whether to plot only the contents (i.e., no borders, axes, etc.)
            Boolean
            Optional; defaults to False
        figtitle: Figure title
            String
            Optional; defaults to None (no title)
    """
    if outpath is None:
        outpath = join(environ.get('TMP_DIR', '~'), 'matrix_as_heatmap.png')

    if mat.ndim != 2:
        raise ValueError("'mat' must have exactly 2 dimensions, but has %d" % mat.ndim)

    # Figure
    if contents_only:
        # Output heatmap will have the exact same shape as input matrix
        dpi = 96 # assumed
        fig = plt.figure(frameon=False)
        w_in = mat.shape[1] / dpi
        h_in = mat.shape[0] / dpi
        fig.set_size_inches(w_in, h_in)
    else:
        figsize = 14
        plt.figure(figsize=(figsize, figsize))

    # Axis
    if contents_only:
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        ax = plt.gca()

    # Set title
    if (not contents_only) and (figtitle is not None):
        ax.set_title(figtitle)

    if center_around_zero:
        v_abs_max = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
        v_max, v_min = v_abs_max, -v_abs_max
        im = ax.imshow(mat, cmap=cmap, interpolation='none', vmin=v_min, vmax=v_max)
    else:
        im = ax.imshow(mat, interpolation='none')

    if not contents_only:
        # Colorbar
        # Create an axes on the right side of ax; width will be 4% of ax,
        # and the padding between cax and ax will be fixed at 0.1 inch
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        plt.colorbar(im, cax=cax)

    # Make directory, if necessary
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir, exist_ok=True)

    # Save plot
    if contents_only:
        fig.savefig(outpath, dpi=dpi)
        fig.clf()
    else:
        plt.savefig(outpath, bbox_inches='tight')
        plt.close('all')


def uv_on_texmap(u, v, texmap, ft=None, outpath=None, figtitle=None):
    """
    Visualizes which points on texture map the vertices map to

    Args:
        u, v: UV coordinates of the vertices
            1D numpy array
                (0, 1)
                    ^ v
                    |
                    |
                    |
                    |
                    +-----------> (1, 0)
                (0, 0)        u
        texmap: Loaded texture map or its path
            h-by-w (grayscale) or h-by-w-by-3 (color) numpy array or string
        ft: Texture faces
            List of lists of integers starting from 1, e.g., '[[1, 2, 3], [], [2, 3, 4, 5], ...]'
            Optional; defaults to None. If provided, use it to connect UV points
        outpath: Path to which the visualization is saved
            String
            Optional; defaults to '$TMP_DIR/uv_on_texmap.png'
        figtitle: Figure title
            String
            Optional; defaults to None (no title)
    """
    import cv2

    if outpath is None:
        outpath = join(environ.get('TMP_DIR', '~'), 'uv_on_texmap.png')

    figsize = 50
    dc = 'r' # color
    ds = 4 # size of UV dots
    lc = 'b' # color
    lw = 1 # width of edges connecting UV dots

    fig = plt.figure(figsize=(figsize, figsize))
    if figtitle is not None:
        fig.title(figtitle)

    # Preprocess input
    if isinstance(texmap, str):
        texmap = cv2.imread(texmap, cv2.IMREAD_UNCHANGED)
    elif isinstance(texmap, np.ndarray):
        assert (len(texmap.shape) == 2 or len(texmap.shape) == 3), \
            "'texmap' must be either h-by-w (grayscale) or h-by-w-by-3 (color)"
    else:
        raise TypeError("Wrong input format for 'texmap'")

    h, w = texmap.shape[:2]
    x = u * w
    y = (1 - v) * h
    # (0, 0)
    #   +----------->
    #   |          x
    #   |
    #   |
    #   v y

    # UV dots
    ax = fig.gca()
    ax.set_xlim([min(0, min(x)), max(w, max(x))])
    ax.set_ylim([max(h, max(y)), min(0, min(y))])
    im = ax.imshow(texmap, cmap='gray')
    ax.scatter(x, y, c=dc, s=ds)
    ax.set_aspect('equal')

    # Also connect these dots
    if ft is not None:
        lines = []
        for vert_id in ft:
            if vert_id: # not empty
                # For each face
                ind = [i - 1 for i in vert_id]
                n_verts = len(ind)
                for i in range(n_verts):
                    lines.append([
                        (x[ind[i]], y[ind[i]]), # starting point
                        (x[ind[(i + 1) % n_verts]], y[ind[(i + 1) % n_verts]]) # ending point
                    ])
        line_collection = LineCollection(lines, linewidths=lw, colors=lc)
        ax.add_collection(line_collection)

    # Colorbar
    # Create an axes on the right side of ax. The width of cax will be 2%
    # of ax and the padding between cax and ax will be fixed at 0.1 inch.
    cax = make_axes_locatable(ax).append_axes('right', size='2%', pad=0.2)
    plt.colorbar(im, cax=cax)

    # Make directory, if necessary
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir, exist_ok=True)

    # Save plot
    plt.savefig(outpath, bbox_inches='tight')

    plt.close('all')


def axes3d_wrapper(
        *args,
        func='scatter',
        labels=None,
        legend_fontsize=20,
        legend_loc=0,
        figsize=(14, 14),
        figtitle=None,
        figtitle_fontsize=20,
        xlabel=None,
        xlabel_fontsize=20,
        ylabel=None,
        ylabel_fontsize=20,
        zlabel=None,
        zlabel_fontsize=20,
        xticks=None,
        xticks_fontsize=10,
        xticks_rotation=0,
        yticks=None,
        yticks_fontsize=10,
        yticks_rotation=0,
        zticks=None,
        zticks_fontsize=10,
        zticks_rotation=0,
        grid=True,
        views=None,
        equal_axes=False,
        outpath=None,
        **kwargs):
    """
    Convinience wrapper for mpl_toolkits.mplot3d.Axes3D functions that saves plots directly to the disk
        without displaying

    Args:
        *args, **kwargs: Positional and/or keyword parameters that the wrapped function takes
            See documentation for mpl_toolkits.mplot3d.Axes3D
        func: Which pyplot function to invoke
            'scatter'
            Optional; defaults to 'scatter'
        labels: Labels for plot objects, to appear in the legend
            List of strings or None (no label for this object)
            Optional; defaults to None (no legend)
        legend_loc: Legend location; effective only when labels is not None
            'best', 'upper right', 'lower left', 'right', 'center left',
                'lower center', 'upper center', 'center, etc.
            Optional; defaults to 'best'
        figsize: Width and height of the figure in inches
            Tuple of two positive floats
            Optional; defaults to (14, 14)
        figtitle: Figure title
            String
            Optional; defaults to None (no title)
        xlabel, ylabel, zlabel: Labels of x-, y- or z-axis
            String
            Optional; defaults to None (no label)
        xticks, yticks, zticks: Tick values of x-, y- or z-axis
            Array_like
            Optional; defaults to None (auto)
        *_fontsize: Font size
            Positive integer
            Optional
        *_rotation: Tick rotation in degrees
            Float
            Optional; defaults to 0
        grid: Whether to draw grid
            Boolean
            Optional; defaults to True
        views: List of elevation-azimuth angle pairs (in degree)
            List of 2-tuples of floats
            Optional; defaults to None
        equal_axes: Whether to have the same scale for all axes
            Boolean
            Optional; defaults to False
        outpath: Path to which the visualization is saved
            String ending with '.png' or '.pkl' (for offline interactive viewing)
            Optional; defaults to '$TMP_DIR/plot.png'
    """
    logger_name = thisfile + '->axes3d_wrapper()'

    if outpath is not None:
        outpath = join(environ.get('TMP_DIR', '~'), 'plot.png')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Set title
    if figtitle is not None:
        ax.set_title(figtitle, fontsize=figtitle_fontsize)

    if func == 'scatter':
        func = ax.scatter
    elif func == 'plot':
        func = ax.plot
    else:
        raise NotImplementedError(func)

    plot_objs = func(*args, **kwargs)

    # Legend
    if labels is not None:
        n_plot_objs = len(plot_objs)
        assert (len(labels) == n_plot_objs), \
            "Number of labels must equal number of plot objects; use None for object without a label"
        for i in range(n_plot_objs):
            plot_objs[i].set_label(labels[i])
        plt.legend(fontsize=legend_fontsize, loc=legend_loc)

    # Grid
    plt.grid(grid)

    # Axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if zlabel is not None:
        ax.set_zlabel(zlabel, fontsize=zlabel_fontsize)

    # Axis ticks
    if xticks is None:
        ax.set_xticklabels(ax.get_xticks(), fontsize=xticks_fontsize, rotation=xticks_rotation)
    else:
        ax.set_xticklabels(xticks, fontsize=xticks_fontsize, rotation=xticks_rotation)
    if yticks is None:
        ax.set_yticklabels(ax.get_yticks(), fontsize=yticks_fontsize, rotation=yticks_rotation)
    else:
        ax.set_yticklabels(yticks, fontsize=yticks_fontsize, rotation=yticks_rotation)
    if zticks is None:
        ax.set_zticklabels(ax.get_zticks(), fontsize=zticks_fontsize, rotation=zticks_rotation)
    else:
        ax.set_zticklabels(zticks, fontsize=zticks_fontsize, rotation=zticks_rotation)

    # Make directory, if necessary
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir, exist_ok=True)

    if equal_axes:
        # plt.axis('equal') # not working, hence the hack of creating a cubic bounding box
        x_data, y_data, z_data = np.array([]), np.array([]), np.array([])

        logger.name = logger_name
        logger.warning("Assuming args are x1, y1, z1, x2, y2, z2, ...")

        for i in range(0, len(args), 3):
            x_data = np.hstack((x_data, args[i]))
            y_data = np.hstack((y_data, args[i + 1]))
            z_data = np.hstack((z_data, args[i + 2]))
        max_range = np.array([
            x_data.max() - x_data.min(),
            y_data.max() - y_data.min(),
            z_data.max() - z_data.min()]).max()
        xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() \
            + 0.5 * (x_data.max() + x_data.min())
        yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() \
            + 0.5 * (y_data.max() + y_data.min())
        zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() \
            + 0.5 * (z_data.max() + z_data.min())
        for xb_, yb_, zb_ in zip(xb, yb, zb):
            ax.plot([xb_], [yb_], [zb_], 'w')

    # Save plot
    if outpath.endswith('.png'):
        if views is None:
            plt.savefig(outpath, bbox_inches='tight')
        else:
            for elev, azim in views:
                ax.view_init(elev, azim)
                plt.draw()
                plt.savefig(outpath.replace('.png', '_elev%d_azim%d.png' % (elev, azim)),
                            bbox_inches='tight')
    elif outpath.endswith('.pkl'):
        # FIXME: can't laod
        with open(outpath, 'wb') as h:
            dump(ax, h)
    else:
        raise ValueError("`outpath` must end with either '.png' or '.pkl'")

    plt.close('all')


def ptcld_as_isosurf(pts, out_obj, res=128, center=False):
    """
    Visualize point cloud as isosurface of its TDF

    Args:
        pts: Cartesian coordinates in object space
            n-by-3 array_like of floats
        out_obj: The output path of the surface .obj
            String
        res: Resolution of the TDF
            Integer
            Optional; defaults to 128
        center: Whether to center these points around object space origin
            Boolean
            Optional; defaults to False
    """
    from skimage.measure import marching_cubes_lewiner
    from trimesh import Trimesh
    from trimesh.io.export import export_mesh
    from xiuminglib import geometry as xg

    # Point cloud to TDF
    tdf = xg.ptcld2tdf(pts, res=res, center=center)

    # Isosurface of TDF
    vs, fs, ns, _ = marching_cubes_lewiner(
        tdf, 0.999 / res, spacing=(1 / res, 1 / res, 1 / res))

    mesh = Trimesh(vertices=vs, faces=fs, normals=ns)
    export_mesh(mesh, out_obj)
