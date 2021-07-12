# pylint: disable=blacklisted-name

from os.path import join, dirname
import numpy as np

from ..log import get_logger
logger = get_logger()

from .. import const
from ..os import makedirs, open_file


class Plot:
    def __init__(
            self,
            figsize=None,
            legend_labels=None, legend_fontsize=None, legend_loc='best',
            figtitle=None, figtitle_fontsize=None,
            axis_labels=None, axis_label_fontsizes=None,
            axis_lim=None,
            axis_ticks=None, axis_tick_labels=None,
            axis_tick_label_fontsizes=None, axis_tick_label_rotations=None,
            grid=True,
            outpath=None):
        """Plotter.

        Args:
            figsize (tuple, optional): Width and height of the figure in inches.
            legend_labels (list, optional): Legend labels.
            legend_fontsize (int, optional): Legend font size.
            legend_loc (str, optional): Legend location: ``'best'``,
                ``'upper right'``, ``'lower left'``, ``'right'``,
                ``'center left'``, ``'lower center'``, ``'upper center'``,
                ``'center'``, etc. Effective only when ``legend_labels`` is not
                ``None``.
            figtitle (str, optional): Figure title.
            figtitle_fontsize (int, optional): Font size.
            axis_labels (dict, optional): Axis labels with ``'x'``, ``'y'``,
                and/or ``'z'`` as keys.
            axis_label_fontsizes (dict, optional): Axis label font sizes with
                ``'x'``, ``'y'``, and/or ``'z'`` as keys.
            axis_lim (dict, optional): Mapping ``'x'``, ``'y'``, or ``'z'`` to
                an ``array_like`` of axis min. and max.
            axis_ticks (dict, optional): Axis tick locations, mapping ``'x'``,
                ``'y'``, or ``'z'`` to an ``array_like`` of floats.
            axis_tick_labels (dict, optional): Axis tick labels, mapping
                ``'x'``, ``'y'``, or ``'z'`` to a list of strings.
            axis_tick_label_fontsizes (dict, optional): Axis tick label font
                sizes, mapping ``'x'``, ``'y'``, or ``'z'`` to a float.
            axis_tick_label_rotations (dict, optional): Axis tick label
                rotations in degrees, mapping ``'x'``, ``'y'``, or ``'z'``
                to a float.
            grid (bool, optional): Whether to draw grid.
            outpath (str, optional): Path to which the plot is saved to. Should
                end with ``'.png'``, and ``None`` means to ``const.Dir.tmp``.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        #
        self.plt = plt
        self.legend_labels = legend_labels
        self.legend_fontsize = legend_fontsize
        self.legend_loc = legend_loc
        self.figsize = figsize
        self.figtitle = figtitle
        self.figtitle_fontsize = figtitle_fontsize
        self.axis_labels = self._init_axis_dict(axis_labels)
        self.axis_label_fontsizes = self._init_axis_dict(axis_label_fontsizes)
        self.axis_lim = self._init_axis_dict(axis_lim)
        self.axis_ticks = self._init_axis_dict(axis_ticks)
        self.axis_tick_labels = self._init_axis_dict(axis_tick_labels)
        self.axis_tick_label_fontsizes = self._init_axis_dict(
            axis_tick_label_fontsizes)
        self.axis_tick_label_rotations = self._init_axis_dict(
            axis_tick_label_rotations)
        self.grid = grid
        self.outpath = outpath

    @staticmethod
    def _init_axis_dict(overrides):
        default = {'x': None, 'y': None, 'z': None}
        if overrides is None:
            return default
        for k, v in overrides.items():
            default[k] = v
        return default

    def _create_fig(self):
        if self.figsize is None:
            fig = self.plt.figure()
        else:
            fig = self.plt.figure(figsize=self.figsize)
        return fig

    def _savefig(self, outpath, contents_only=False, dpi=None):
        # Make directory, if necessary
        outdir = dirname(outpath)
        makedirs(outdir)
        #
        if contents_only:
            ax = self.plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            with open_file(outpath, 'wb') as h:
                self.plt.savefig(h, dpi=dpi)
        else:
            with open_file(outpath, 'wb') as h:
                self.plt.savefig(h, bbox_inches='tight', dpi=dpi)

    def _add_legend(self, plot_objs):
        if self.legend_labels is None:
            return
        n_plot_objs = len(plot_objs)
        assert (len(self.legend_labels) == n_plot_objs), (
            "Number of legend labels must equal number of plot objects; "
            "use None for object without a legend label")
        for i in range(n_plot_objs):
            plot_objs[i].set_label(self.legend_labels[i])
        if self.legend_fontsize is None:
            self.plt.legend(loc=self.legend_loc)
        else:
            self.plt.legend(fontsize=self.legend_fontsize, loc=self.legend_loc)

    def _add_axis_labels(self, ax):
        for axis, label in self.axis_labels.items():
            if label is None:
                continue
            fontsize = self.axis_label_fontsizes[axis]
            set_func = getattr(ax, f'set_{axis}label')
            set_func(label, fontsize=fontsize)

    def _set_axis_ticks(self, ax):
        # Tick locations
        for axis, ticks in self.axis_ticks.items():
            if ticks is None:
                continue
            set_func = getattr(ax, f'set_{axis}ticks')
            set_func(ticks)
        # Tick labels
        for axis, tick_labels in self.axis_tick_labels.items():
            if tick_labels is None:
                continue
            set_func = getattr(ax, f'set_{axis}ticklabels')
            set_func(tick_labels)
        # Tick label font size
        for axis, tick_label_fontsize in self.axis_tick_label_fontsizes.items():
            if tick_label_fontsize is None:
                continue
            ax.tick_params(axis=axis, labelsize=tick_label_fontsize)
        # Tick rotation
        for axis, tick_label_rotation in self.axis_tick_label_rotations.items():
            if tick_label_rotation is None:
                continue
            ax.tick_params(axis=axis, labelrotation=tick_label_rotation)

    def _set_axis_lim(self, ax):
        for axis, lim in self.axis_lim.items():
            if lim is None:
                continue
            set_func = getattr(ax, f'set_{axis}lim')
            set_func(*lim)

    @staticmethod
    def _set_axes_equal(ax, xyz):
        # plt.axis('equal') not working, hence the hack of creating a cubic
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

    def _set_title(self, ax):
        if self.figtitle is not None:
            if self.figtitle_fontsize is None:
                ax.set_title(self.figtitle)
            else:
                ax.set_title(self.figtitle, fontsize=self.figtitle_fontsize)

    def bar(self, y, group_width=0.8):
        """Bar plot.

        Args:
            y (array_like): N-by-M array of N groups, each with M bars,
                or N-array of N groups, each with one bar.
            group_width (float, optional): Width allocated to each group,
                shared by all bars within the group.

        Writes
            - The bar plot.
        """
        outpath = join(const.Dir.tmp, 'bar.png') if self.outpath is None \
            else self.outpath
        fig = self._create_fig()
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
        #
        self._add_legend(plot_objs)
        self.plt.grid(self.grid)
        self._add_axis_labels(ax)
        self._set_axis_ticks(ax)
        self._set_axis_lim(ax)
        self._savefig(outpath)
        self.plt.close('all')
        return outpath

    def scatter3d(
            self, xyz, colors=None, size=None, equal_axes=False, views=None):
        """3D scatter plot.

        Args:
            xyz (array_like): N-by-3 array of N points.
            colors (array_like or list(str) or str, optional): If N-array, these
                values are colormapped. If N-list, its elements should be color
                strings. If a single color string, all points use that color.
            size (int, optional): Scatter size.
            equal_axes (bool, optional): Whether to have the same scale for all
                axes.
            views (list(tuple), optional): List of elevation-azimuth angle pairs
                (in degrees). A good set of views is ``[(30, 0), (30, 45),
                (30, 90), (30, 135)]``.

        Writes
            - One or multiple (if ``views`` is provided) views of the 3D plot.
        """
        from mpl_toolkits.mplot3d import Axes3D # noqa; pylint: disable=unused-variable
        #
        outpath = join(const.Dir.tmp, 'scatter3d.png') if self.outpath is None \
            else self.outpath
        fig = self._create_fig()
        ax = fig.add_subplot(111, projection='3d')
        self._set_title(ax)
        # Prepare kwargs to scatter()
        kwargs = {}
        need_colorbar = False
        if isinstance(colors, np.ndarray):
            kwargs['c'] = colors # will be colormapped with color map
            kwargs['cmap'] = 'viridis'
            need_colorbar = True
        elif colors is not None:
            kwargs['c'] = colors
        if size is not None:
            kwargs['s'] = size
        # Plot
        plot_objs = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kwargs)
        #
        self._add_legend(plot_objs)
        self.plt.grid(self.grid)
        self._add_axis_labels(ax)
        self._set_axis_ticks(ax)
        self._set_axis_lim(ax)
        if equal_axes:
            self._set_axes_equal(ax, xyz)
        if need_colorbar:
            self.plt.colorbar(plot_objs)
            # FIXME: This seems to mess up equal axes
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
        self.plt.close('all')
        return outpaths

    def line(self, xy, width=None, marker=None, marker_size=None):
        """Line/curve plot.

        Args:
            xy (array_like): N-by-M array of N x-values (first column) and
                their corresponding y-values (the remaining M-1 columns).
            width (float, optional): Line width.
            marker (str, optional): Marker.
            marker_size (float, optional): Marker size.

        Writes
            - The line plot.
        """
        outpath = join(const.Dir.tmp, 'line.png') if self.outpath is None \
            else self.outpath
        fig = self._create_fig()
        ax = fig.add_subplot(111)
        self._set_title(ax)
        # Prepare kwargs to scatter()
        kwargs_list = []
        n_lines = xy.shape[1] - 1
        for i in range(n_lines):
            kwargs = {}
            if width is not None:
                kwargs['linewidth'] = width
            if marker is not None:
                kwargs['marker'] = marker
            if marker_size is not None:
                kwargs['markersize'] = marker_size
            kwargs_list.append(kwargs)
        # Plot
        plot_objs = []
        for i in range(n_lines):
            plot_obj = self.plt.plot(xy[:, 0], xy[:, 1 + i], **kwargs_list[i])
            assert len(plot_obj) == 1
            plot_obj = plot_obj[0]
            plot_objs.append(plot_obj)
        #
        self._add_legend(plot_objs)
        self.plt.grid(self.grid)
        self._add_axis_labels(ax)
        self._set_axis_ticks(ax)
        self._set_axis_lim(ax)
        self._savefig(outpath)
        self.plt.close('all')
        return outpath
