from os.path import join, dirname
import numpy as np

from ..log import get_logger
logger = get_logger()

from .. import const, os as xm_os
from ..imprt import preset_import


class Plot:
    def __init__(
            self,
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
            xlim=None,
            ylim=None,
            zlim=None,
            xticks=None,
            xticks_fontsize=10,
            xticks_rotation=0,
            yticks=None,
            yticks_fontsize=10,
            yticks_rotation=0,
            zticks=None,
            zticks_fontsize=10,
            zticks_rotation=0,
            grid=True):
        # FIXME: docstring
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
        #
        self.plt = plt
        self.figsize = figsize
        self.figtitle = figtitle
        self.figtitle_fontsize = figtitle_fontsize
        self.legend_fontsize = legend_fontsize
        self.legend_loc = legend_loc
        self.grid = grid
        self.xlabel = xlabel
        self.xlabel_fontsize = xlabel_fontsize
        self.ylabel = ylabel
        self.ylabel_fontsize = ylabel_fontsize
        self.zlabel = zlabel
        self.zlabel_fontsize = zlabel_fontsize
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.xticks = xticks
        self.xticks_rotation = xticks_rotation
        self.xticks_fontsize = xticks_fontsize
        self.yticks = yticks
        self.yticks_rotation = yticks_rotation
        self.yticks_fontsize = yticks_fontsize
        self.zticks = zticks
        self.zticks_rotation = zticks_rotation
        self.zticks_fontsize = zticks_fontsize

    def _set_title(self, ax):
        if self.figtitle is None:
            return
        ax.set_title(self.figtitle, fontsize=self.figtitle_fontsize)

    def _savefig(self, outpath, contents_only=False, dpi=None):
        # Make directory, if necessary
        outdir = dirname(outpath)
        xm_os.makedirs(outdir)
        # Figure out open function, depending on filesystem
        gfile = preset_import('gfile')
        open_func = open if gfile is None else gfile.Open
        #
        if contents_only:
            ax = self.plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            with open_func(outpath, 'wb') as h:
                self.plt.savefig(h, dpi=dpi)
        else:
            with open_func(outpath, 'wb') as h:
                self.plt.savefig(h, bbox_inches='tight', dpi=dpi)

    def _add_legend(self, plot_objs, labels):
        if labels is None:
            return
        n_plot_objs = len(plot_objs)
        assert (len(labels) == n_plot_objs), (
            "Number of labels must equal number of plot objects; "
            "use None for object without a label")
        for i in range(n_plot_objs):
            plot_objs[i].set_label(labels[i])
        self.plt.legend(fontsize=self.legend_fontsize, loc=self.legend_loc)

    def _add_axis_labels(self, ax):
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel, fontsize=self.xlabel_fontsize)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel, fontsize=self.ylabel_fontsize)
        if self.zlabel is not None:
            ax.set_zlabel(self.zlabel, fontsize=self.zlabel_fontsize)

    def _set_axis_ticks(self, ax):
        # FIXME: if xticks is not provided, xticks_fontsize and xticks_rotation have
        # no effect, which shouldn't be the case
        if self.xticks is not None:
            ax.set_xticklabels(
                self.xticks, fontsize=self.xticks_fontsize,
                rotation=self.xticks_rotation)
        if self.yticks is not None:
            ax.set_yticklabels(
                self.yticks, fontsize=self.yticks_fontsize,
                rotation=self.yticks_rotation)
        if self.zticks is not None:
            ax.set_zticklabels(
                self.zticks, fontsize=self.zticks_fontsize,
                rotation=self.zticks_rotation)

    def _set_axis_lim(self, ax):
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        if self.zlim is not None:
            ax.set_zlim(*self.zlim)

    @staticmethod
    def _set_axes_equal(ax, xyz):
        # plt.axis('equal') # not working, hence the hack of creating a cubic
        # bounding box
        x_data, y_data, z_data = xyz[:, 0], xyz[:, 1], xyz[:, 2]
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

    def bar(self, y, group_width=0.8, labels=None, outpath=None):
        if outpath is None:
            outpath = join(const.Dir.tmp, 'bar.png')
        fig = self.plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        self._set_title(ax)
        # Ensure y is 2D, with columns representing values within groups
        # and rows across groups
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        n, n_grp = y.shape
        # Group width is shared by all groups
        bar_width = group_width / n_grp
        # Assume x is evenly spaced
        x = np.arange(n)
        # Plot
        plot_objs = []
        for i in range(n_grp):
            x_ = x - 0.5 * group_width + 0.5 * bar_width + i * bar_width
            plot_obj = ax.bar(x_, y[:, i], bar_width)
            plot_objs.append(plot_obj)
        # Legend to plot objects
        self._add_legend(plot_objs, labels)
        # Grid
        self.plt.grid(self.grid)
        # Axis labels
        self._add_axis_labels(ax)
        # Axis ticks
        self._set_axis_ticks(ax)
        # Axis limits
        self._set_axis_lim(ax)
        # Save plot
        self._savefig(outpath)
        # Clear all when done
        self.plt.close('all')
        return outpath

    def scatter3d(
            self, xyz, color=None, size=None, labels=None, equal_axes=False,
            views=None, outpath=None):
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D # noqa; pylint: disable=unused-import
        #
        if outpath is None:
            outpath = join(const.Dir.tmp, 'scatter3d.png')
        fig = self.plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        self._set_title(ax)
        # Prepare kwargs to scatter()
        kwargs = {}
        if isinstance(color, np.ndarray):
            color = (color - color.min()) / (color.max() - color.min())
            color = cm.Reds(color)
            kwargs['c'] = color
        elif color is not None:
            kwargs['c'] = color
        if size is not None:
            kwargs['s'] = size
        # Plot
        plot_objs = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kwargs)
        # Legend to plot objects
        self._add_legend(plot_objs, labels)
        # Grid
        self.plt.grid(self.grid)
        # Axis labels
        self._add_axis_labels(ax)
        # Axis ticks
        self._set_axis_ticks(ax)
        # Axis limits
        self._set_axis_lim(ax)
        # Equal axes
        if equal_axes:
            self._set_axes_equal(ax, xyz)
        # Save plot
        outpaths = []
        if outpath.endswith('.png'):
            if views is None:
                self._savefig(outpath)
                outpaths.append(outpath)
            else:
                for elev, azim in views:
                    ax.view_init(elev, azim)
                    self.plt.draw()
                    outpath_ = outpath[:-len('.png')] + \
                        '_elev%03d_azim%03d.png' % (elev, azim)
                    self._savefig(outpath_)
                    outpaths.append(outpath_)
        else:
            raise ValueError("`outpath` must end with '.png'")
        # Clear all when done
        self.plt.close('all')
        return outpaths
