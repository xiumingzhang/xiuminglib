from os.path import join, dirname
import pickle as pk
import numpy as np

from ..log import get_logger
logger = get_logger()

from .. import const, os as xm_os
from ..imprt import preset_import


def scatter3d(
        *args,
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
    """Convinience wrapper for 3D scatterplot.

    It saves plots directly to the disk without displaying.

    Args:
        *args: Positional parameters that ``ax.scatter()`` takes.
        **kwargs: Keyword parameters that ``ax.scatter()`` takes.
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
        ValueError: If ``outpath`` has a wrong extension.

    Writes
        - One or multiple (if ``views`` is provided) views of the 3D plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # noqa; pylint: disable=unused-import

    if outpath is None:
        outpath = join(const.Dir.tmp, 'axes3d_wrapper.png')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Set title
    if figtitle is not None:
        ax.set_title(figtitle, fontsize=figtitle_fontsize)

    plot_objs = ax.scatter(*args, **kwargs)

    # Legend
    if labels is not None:
        n_plot_objs = len(plot_objs)
        assert (len(labels) == n_plot_objs), (
            "Number of labels must equal number of plot objects; "
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
        ax.set_xticklabels(
            ax.get_xticks(), fontsize=xticks_fontsize, rotation=xticks_rotation)
    else:
        ax.set_xticklabels(
            xticks, fontsize=xticks_fontsize, rotation=xticks_rotation)
    if yticks is None:
        ax.set_yticklabels(
            ax.get_yticks(), fontsize=yticks_fontsize, rotation=yticks_rotation)
    else:
        ax.set_yticklabels(
            yticks, fontsize=yticks_fontsize, rotation=yticks_rotation)
    if zticks is None:
        ax.set_zticklabels(
            ax.get_zticks(), fontsize=zticks_fontsize, rotation=zticks_rotation)
    else:
        ax.set_zticklabels(
            zticks, fontsize=zticks_fontsize, rotation=zticks_rotation)

    # Make directory, if necessary
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    if equal_axes:
        # plt.axis('equal') # not working, hence the hack of creating a cubic
        # bounding box
        x_data, y_data, z_data = np.array([]), np.array([]), np.array([])

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
                _savefig(
                    outpath.replace(
                        '.png', '_elev%d_azim%d.png' % (elev, azim)))
    elif outpath.endswith('.pkl'):
        # FIXME: can't load
        with open(outpath, 'wb') as h:
            pk.dump(ax, h)
    else:
        raise ValueError("`outpath` must end with either '.png' or '.pkl'")

    plt.close('all')


def _savefig(outpath, contents_only=False, dpi=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gfile = preset_import('gfile')
    open_func = open if gfile is None else gfile.Open

    if contents_only:
        ax = plt.gca()
        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()
        with open_func(outpath, 'wb') as h:
            plt.savefig(h, dpi=dpi)
    else:
        with open_func(outpath, 'wb') as h:
            plt.savefig(h, bbox_inches='tight', dpi=dpi)
