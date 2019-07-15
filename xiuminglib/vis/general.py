"""This module should be imported before ``skimage`` to avoid the ``matplotlib`` backend problem."""

from os.path import dirname, abspath, join
from pickle import dump
import numpy as np

from ..config import create_logger
logger, thisfile = create_logger(abspath(__file__))

from .. import const, os as xm_os
from ..imprt import preset_import


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
    """Convinience wrapper for :mod:`matplotlib.pyplot` functions.

    It saves plots directly to the disk without displaying.

    Args:
        *args: Positional parameters that the wrapped function takes. See
            :mod:`matplotlib.pyplot`.
        **kwargs: Keyword parameters.
        ci (list(float) or list(list(float)), optional): Confidence interval
            for ``x_i[j]`` is ``y_i[j] +/- ci[i][j]``. Effective only when
            ``func`` is ``'plot'``. List of floats for one line, and list of
            lists of floats for multiple lines.
        func (str, optional): Which ``pyplot`` function to invoke, e.g.,
            ``'plot'`` or ``'bar'``.
        labels (list, optional): Labels for plot objects, to appear in the
            legend. ``None`` means no label for this object.
        legend_loc (str, optional): Legend location: ``'best'``,
            ``'upper right'``, ``'lower left'``, ``'right'``,
            ``'center left'``, ``'lower center'``, ``'upper center'``,
            ``'center'``, etc. Effective only when ``labels`` is not ``None``.
        figsize (tuple, optional): Width and height of the figure in inches.
        figtitle (str, optional): Figure title.
        xlabel (str, optional): Label of x-axis.
        ylabel
        xticks (array_like, optional): Tick values of x-axis. ``None`` means
            auto.
        yticks
        xticks_locations (array_like, optional): Locations of the ticks.
            ``None`` means starting from 0 and one next to another.
        yticks_locations
        *_fontsize (int, optional): Font size.
        *_rotation (float, optional): Tick rotation in degrees.
        xlim (list, optional): Start and end values for x-axis. ``None``
            means auto.
        ylim
        grid (bool, optional): Whether to draw grid.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'pyplot_wrapper.png')``.

    Raises:
        NotImplementedError: If ``func`` is not implemented.
        TypeError: If ``ci`` is of a wrong type.

    Writes
        - The plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if ci is not None:
        assert func == 'plot', "CI makes sense only for `plot`"

    if outpath is None:
        outpath = join(const.Dir.tmp, 'pyplot_wrapper.png')

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
    elif func == 'scatter':
        func = plt.scatter
    else:
        raise NotImplementedError(func)

    plot_objs = func(*args, **kwargs)

    # Confidence intervals
    if ci is not None:
        # `func` is 'plot'
        if isinstance(ci[0], (int, float)):
            # List of numbers -> only one line
            assert len(plot_objs) == 1, \
                "Only one CI is provided, but there are more than one lines"
            ci = np.array(ci)
            assert (ci > 0).all(), "CI should be positive"
            ci = [ci]
        elif isinstance(ci[0], (list, np.ndarray)):
            # List of lists -> multiple lines
            assert len(ci) == len(plot_objs), \
                "Numbers of CI's and lines are different"
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
            ("Number of labels must equal number of plot objects; "
             "use None for object without a label")
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
        plt.xticks(xticks_locations, xticks,
                   fontsize=xticks_fontsize, rotation=xticks_rotation)
    if yticks is None:
        plt.yticks(fontsize=yticks_fontsize, rotation=yticks_rotation)
    else:
        if yticks_locations is None:
            yticks_locations = range(len(yticks))
        plt.yticks(yticks_locations, yticks,
                   fontsize=yticks_fontsize, rotation=yticks_rotation)

    # Make directory, if necessary
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    # Save plot
    _savefig(outpath)

    plt.close('all')


def scatter_on_image(im, pts, size=2, bgr=(0, 0, 255), outpath=None):
    r"""Plots scatter on top of an image.

    Args:
        im (numpy.ndarray): Image to scatter on. H-by-W (grayscale) or
            H-by-W-by-3 (RGB) arrays of type ``numpy.uint8`` or
            ``numpy.uint16``.
        pts (array_like): Coordinates of the scatter point(s), of length 2
            for just one point or shape N-by-2 for multiple points.
            Convention:

            .. code-block:: none

                +-----------> dim1
                |
                |
                |
                v dim0

        size (float or array_like(float), optional): Size(s) of scatter
            points. If *array_like*, must be of length N.
        bgr (tuple or array_like(tuple), optional): BGR color(s) of scatter
            points. Each element :math:`\in [0, 255]`. If *array_like*, must
            be of shape N-by-3.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means ``os.path.join(const.Dir.tmp,
            'scatter_on_image.png')``.

    Writes
        - The scatter plot overlaid over the image.
    """
    cv2 = preset_import('cv2')

    logger_name = thisfile + '->scatter_on_image()'

    if outpath is None:
        outpath = join(const.Dir.tmp, 'scatter_on_image.png')

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
        xy = tuple(pts[i, ::-1].astype(int))
        color = (int(bgr[i, 0]), int(bgr[i, 1]), int(bgr[i, 2]))
        cv2.circle(im, xy, size[i], color, thickness)

    # Make directory, if necessary
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    # Write to disk
    cv2.imwrite(outpath, im)


def make_colormap(low, high):
    """Generates your own colormap for heatmap.

    Args:
        low (str or tuple): Color for the lowest value, such as ``'red'`` or
            ``(1, 0, 0)``.
        high

    Returns:
        matplotlib.colors.LinearSegmentedColormap: Generated colormap.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.colors as mcolors

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


def _savefig(outpath, contents_only=False, dpi=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if contents_only:
        ax = plt.gca()
        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()
        plt.savefig(outpath, dpi=dpi)
    else:
        plt.savefig(outpath, bbox_inches='tight', dpi=dpi)


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
    """Convinience wrapper for :class:`mpl_toolkits.mplot3d.Axes3D` functions.

    It saves plots directly to the disk without displaying.

    Args:
        *args: Positional parameters that the wrapped function takes. See
            :class:`mpl_toolkits.mplot3d.Axes3D`.
        **kwargs: Keyword parameters.
        func (str, optional): Which pyplot function to invoke, e.g.,
            ``'scatter'``.
        labels (list(str), optional): Labels for plot objects, to appear in
            the legend. Use ``None`` for no label for a certain object.
            ``None`` means no legend at all.
        legend_loc (str, optional): Legend location: ``'best'``,
            ``'upper right'``, ``'lower left'``, ``'right'``,
            ``'center left'``, ``'lower center'``, ``'upper center'``,
            ``'center'``, etc. Effective only when ``labels`` is not ``None``.
        figsize (tuple, optional): Width and height of the figure in inches.
        figtitle (str, optional): Figure title.
        xlabel (str, optional): Label of x-axis.
        ylabel
        zlabel
        xticks (array_like, optional): Tick values of x-axis. ``None`` means
            auto.
        yticks
        zticks
        *_fontsize (int, optional): Font size.
        *_rotation (float, optional): Tick rotation in degrees.
        grid (bool, optional): Whether to draw grid.
        views (list(tuple), optional): List of elevation-azimuth angle pairs
            (in degrees). A good set of views is ``[(30, 0), (30, 45),
            (30, 90), (30, 135)]``.
        equal_axes (bool, optional): Whether to have the same scale for all
            axes.
        outpath (str, optional): Path to which the visualization is saved to.
            Should end with ``'.png'`` or ``'.pkl'`` (for offline interactive
            viewing). ``None`` means ``os.path.join(const.Dir.tmp,
            'axes3d_wrapper.png')``.

    Raises:
        NotImplementedError: If ``func`` is not yet implemented.
        ValueError: If ``outpath`` has a wrong extension.

    Writes
        - One or multiple (if ``views`` is provided) views of the 3D plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # noqa; pylint: disable=unused-import

    logger_name = thisfile + '->axes3d_wrapper()'

    if outpath is None:
        outpath = join(const.Dir.tmp, 'axes3d_wrapper.png')

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
            ("Number of labels must equal number of plot objects; "
             "use None for object without a label")
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
        ax.set_xticklabels(ax.get_xticks(), fontsize=xticks_fontsize,
                           rotation=xticks_rotation)
    else:
        ax.set_xticklabels(xticks, fontsize=xticks_fontsize,
                           rotation=xticks_rotation)
    if yticks is None:
        ax.set_yticklabels(ax.get_yticks(), fontsize=yticks_fontsize,
                           rotation=yticks_rotation)
    else:
        ax.set_yticklabels(yticks, fontsize=yticks_fontsize,
                           rotation=yticks_rotation)
    if zticks is None:
        ax.set_zticklabels(ax.get_zticks(), fontsize=zticks_fontsize,
                           rotation=zticks_rotation)
    else:
        ax.set_zticklabels(zticks, fontsize=zticks_fontsize,
                           rotation=zticks_rotation)

    # Make directory, if necessary
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    if equal_axes:
        # plt.axis('equal') # not working, hence the hack of creating a cubic
        # bounding box
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
            _savefig(outpath)
        else:
            for elev, azim in views:
                ax.view_init(elev, azim)
                plt.draw()
                _savefig(outpath.replace(
                    '.png', '_elev%d_azim%d.png' % (elev, azim)))
    elif outpath.endswith('.pkl'):
        # FIXME: can't laod
        with open(outpath, 'wb') as h:
            dump(ax, h)
    else:
        raise ValueError("`outpath` must end with either '.png' or '.pkl'")

    plt.close('all')
