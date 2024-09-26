from __future__ import annotations

from abc import ABC
from typing import Tuple
import re
from numpy import nan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._config import LegendConfig

from .._misc import ColorGenerator, MarkerShapeGenerator, is_latex_formatted
from .._core import (
    BasePlot,
    LinePlot,
    VLinePlot,
    ScatterPlot,
    BaseMSPlot,
    ChromatogramPlot,
    MobilogramPlot,
    SpectrumPlot,
    PeakMapPlot,
    APPEND_PLOT_DOC,
)


class MATPLOTLIBPlot(BasePlot, ABC):
    """
    Base class for assembling a Matplotlib plot.

    Attributes:
        data (DataFrame): The input data frame.
    """

    @property
    def _interactive(self):
        return False

    def _load_extension(self):
        """
        Load the matplotlib extension.
        """
        try:
            from matplotlib import pyplot
        except ImportError:
            raise ImportError(
                f"matplotlib is not installed. Please install using `pip install matplotlib` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self):
        """
        Create a figure and axes objects,
        for consistency with other backends, the fig object stores the matplotlib axes object
        """
        # TODO why is self.heigh and self.width checked if no alternatives
        if self.width is not None and self.height is not None and not self.plot_3d:
            superFig, fig = plt.subplots(
                figsize=(self.width / 100, self.height / 100), dpi=100
            )
            fig.set_title(self.title)
            fig.set_xlabel(self.xlabel)
            fig.set_ylabel(self.ylabel)
        elif self.width is not None and self.height is not None and self.plot_3d:
            superFig = plt.figure(
                figsize=(self.width / 100, self.height / 100), layout="constrained"
            )
            fig = superFig.add_subplot(111, projection="3d")
            fig.set_title(self.title)
            fig.set_xlabel(
                self.xlabel,
                fontsize=9,
                labelpad=-2,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
                style="italic",
            )
            fig.set_ylabel(
                self.ylabel,
                fontsize=9,
                labelpad=-2,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
            )
            fig.set_zlabel(
                self.zlabel,
                fontsize=10,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
                labelpad=-2,
            )

            for axis in ("x", "y", "z"):
                fig.tick_params(
                    axis=axis,
                    labelsize=8,
                    pad=-2,
                    colors=ColorGenerator.color_blind_friendly_map[
                        ColorGenerator.Colors.DARKGRAY
                    ],
                )

            fig.set_box_aspect(aspect=None, zoom=0.88)
            fig.ticklabel_format(
                axis="z", style="sci", useMathText=True, scilimits=(0, 0)
            )
            fig.grid(color="#FF0000", linewidth=0.8)
            fig.xaxis.pane.fill = False
            fig.yaxis.pane.fill = False
            fig.zaxis.pane.fill = False
            fig.view_init(elev=25, azim=-45, roll=0)
        return fig

    def _update_plot_aes(self, ax):
        """
        Update the plot aesthetics.

        Args:
            ax: The axes object.
            **kwargs: Additional keyword arguments.
        """
        ax.grid(self.grid)
        # Update the title, xlabel, and ylabel
        ax.set_title(self.title, fontsize=self.title_font_size)
        ax.set_xlabel(self.xlabel, fontsize=self.xaxis_label_font_size)
        ax.set_ylabel(self.ylabel, fontsize=self.yaxis_label_font_size)
        # Update axis tick labels
        ax.tick_params(axis="x", labelsize=self.xaxis_tick_font_size)
        ax.tick_params(axis="y", labelsize=self.yaxis_tick_font_size)
        if self.plot_3d:
            ax.set_zlabel(self.zlabel, fontsize=self.yaxis_label_font_size)
            ax.tick_params(axis="z", labelsize=self.yaxis_tick_font_size)
        return ax

    def _add_legend(self, ax, legend):
        """
        Add a legend to the plot.

        Args:
            ax: The axes object.
            legend: The legend configuration.
        """
        if legend is not None and self.legend_config.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.legend_config.loc
            )
            if self.legend_config.orientation == "horizontal":
                ncol = len(legend[0])
            else:
                ncol = 1

            legend = ax.legend(
                *legend,
                loc=matplotlibLegendLoc,
                title=self.legend_config.title,
                prop={"size": self.legend_config.fontsize},
                bbox_to_anchor=self.legend_config.bbox_to_anchor,
                ncol=ncol,
            )

            legend.get_title().set_fontsize(str(self.legend_config.fontsize))
        return ax

    def _modify_x_range(
        self,
        ax,
        x_range: Tuple[float, float],
        padding: Tuple[float, float] | None = None,
    ):
        """
        Modify the x-axis range.

        Args:
            x_range (Tuple[float, float]): The x-axis range.
            padding (Tuple[float, float] | None, optional): The padding for the range. Defaults to None.
        """
        start, end = x_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        ax.set_xlim(start, end)
        return ax

    def _modify_y_range(
        self,
        ax,
        y_range: Tuple[float, float],
        padding: Tuple[float, float] | None = None,
    ):
        """
        Modify the y-axis range.

        Args:
            y_range (Tuple[float, float]): The y-axis range.
            padding (Tuple[float, float] | None, optional): The padding for the range. Defaults to None.
        """
        start, end = y_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        ax.set_ylim(start, end)
        return ax

    # since matplotlib creates static plots, we don't need to implement the following methods
    def _add_tooltips(self, fig, tooltips):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_tooltips'"
        )

    def _add_bounding_box_drawer(self, fig):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_bounding_box_drawer'"
        )

    def _add_bounding_vertical_drawer(self, fig):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_bounding_vertical_drawer'"
        )

    def show_default(self, fig):
        """
        Show the plot.
        """
        if isinstance(self.fig, Axes):
            self.fig.get_figure().tight_layout()
        else:
            self.superFig.tight_layout()
        plt.show()


class MATPLOTLIBLinePlot(MATPLOTLIBPlot, LinePlot):
    """
    Class for assembling a matplotlib line plot
    """

    @APPEND_PLOT_DOC
    def plot(self, ax) -> Tuple[Axes, "Legend"]:
        """
        Plot a line plot
        """

        legend_lines = []
        legend_labels = []

        if self.by is None:
            (line,) = ax.plot(
                self.data[self.x], self.data[self.y], color=self.current_color
            )

            return ax, None
        else:
            for group, df in self.data.groupby(self.by):
                (line,) = ax.plot(df[self.x], df[self.y], color=self.current_color)
                legend_lines.append(line)
                legend_labels.append(group)
            return ax, (legend_lines, legend_labels)


class MATPLOTLIBVLinePlot(MATPLOTLIBPlot, VLinePlot):
    """
    Class for assembling a matplotlib vertical line plot
    """

    @APPEND_PLOT_DOC
    def plot(self, ax) -> Tuple[Axes, "Legend"]:
        """
        Plot a vertical line
        """
        if not self.plot_3d:
            legend_lines = []
            legend_labels = []

            if self.by is None:
                for _, row in self.data.iterrows():
                    if self.direction == "horizontal":
                        x_data = [0, row[self.x]]
                        y_data = [row[self.y], row[self.y]]
                    else:
                        x_data = [row[self.x], row[self.x]]
                        y_data = [0, row[self.y]]
                    (line,) = ax.plot(x_data, y_data, color=self.current_color)

                return ax, None
            else:
                for group, df in self.data.groupby(self.by):
                    for _, row in df.iterrows():
                        if self.direction == "horizontal":
                            x_data = [0, row[self.x]]
                            y_data = [row[self.y], row[self.y]]
                        else:
                            x_data = [row[self.x], row[self.x]]
                            y_data = [0, row[self.y]]
                        (line,) = ax.plot(x_data, y_data, color=self.current_color)
                    legend_lines.append(line)
                    legend_labels.append(group)

                return ax, (legend_lines, legend_labels)
        else:
            if self.by is None:
                for i in range(len(self.data)):
                    (line,) = ax.plot(
                        [self.data[self.y].iloc[i], self.data[self.y].iloc[i]],
                        [self.data[self.z].iloc[i], 0],
                        [self.data[self.x].iloc[i], self.data[self.x].iloc[i]],
                        zdir="x",
                        color=plt.cm.magma_r(
                            (self.data[self.z].iloc[i] / self.data[self.z].max())
                        ),
                    )
                return ax, None
            else:
                legend_lines = []
                legend_labels = []

                for group, df in self.data.groupby(self.by):
                    for i in range(len(df)):
                        (line,) = ax.plot(
                            [df[self.y].iloc[i], df[self.y].iloc[i]],
                            [df[self.z].iloc[i], 0],
                            [df[self.x].iloc[i], df[self.x].iloc[i]],
                            zdir="x",
                            color=self.current_color,
                        )
                    legend_lines.append(line)
                    legend_labels.append(group)

                return ax, (legend_lines, legend_labels)

    def _add_annotations(
        self,
        fig,
        ann_texts: list[list[str]],
        ann_xs: list[float],
        ann_ys: list[float],
        ann_colors: list[str],
    ):
        for text, x, y, color in zip(ann_texts, ann_xs, ann_ys, ann_colors):
            if text is not nan and text != "" and text != "nan":
                # Check if the text contains LaTeX-style expressions
                if is_latex_formatted(text):
                    # Wrap the text in '$' to indicate LaTeX math mode
                    text = r"${}$".format(text)
                fig.annotate(
                    text,
                    xy=(x, y),
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=self.annotation_font_size,
                    color=color,
                )


class MATPLOTLIBScatterPlot(MATPLOTLIBPlot, ScatterPlot):
    """
    Class for assembling a matplotlib scatter plot
    """

    def __post_init__(self):
        super().__post_init__()
        if self.marker is None:
            self.marker = MarkerShapeGenerator(engine="MATPLOTLIB")

    @APPEND_PLOT_DOC
    def plot(self, ax) -> Tuple[Axes, "Legend"]:
        """
        Plot a scatter plot
        """

        kwargs = dict(s=self.marker_size, edgecolors="none", cmap="magma_r", zorder=2)

        legend_lines = []
        legend_labels = []

        if self.by is None:
            use_color = self.current_color if self.z is None else self.data[self.z]
            scatter = ax.scatter(
                self.data[self.x], self.data[self.y], c=use_color, **kwargs
            )
            return ax, None
        else:
            if self.z is not None:
                vmin, vmax = self.data[self.z].min(), self.data[self.z].max()
            for group, df in self.data.groupby(self.by):
                use_color = self.current_color if self.z is None else self.data[self.z]
                # Normalize colors if z is specified
                if self.z is not None:
                    normalize = plt.Normalize(vmin=vmin, vmax=vmax)
                    scatter = ax.scatter(
                        df[self.x],
                        df[self.y],
                        c=use_color,
                        norm=normalize,
                        marker=self.current_marker,
                        **kwargs,
                    )
                else:
                    scatter = ax.scatter(df[self.x], df[self.y], c=use_color, **kwargs)
                legend_lines.append(scatter)
                legend_labels.append(group)
            return ax, (legend_lines, legend_labels)


class MATPLOTLIB_MSPlot(BaseMSPlot, MATPLOTLIBPlot, ABC):

    def get_line_renderer(self, **kwargs) -> None:
        return MATPLOTLIBLinePlot(**kwargs)

    def get_vline_renderer(self, **kwargs) -> None:
        return MATPLOTLIBVLinePlot(**kwargs)

    def get_scatter_renderer(self, **kwargs) -> None:
        return MATPLOTLIBScatterPlot(**kwargs)

    def plot_x_axis_line(self, fig):
        fig.plot(fig.get_xlim(), [0, 0], color="#EEEEEE", linewidth=1.5)

    def _create_tooltips(self, entries, index=True):
        # No tooltips for MATPLOTLIB because it is not interactive
        return None, None


@APPEND_PLOT_DOC
class MATPLOTLIBChromatogramPlot(MATPLOTLIB_MSPlot, ChromatogramPlot):
    """
    Class for assembling a matplotlib extracted ion chromatogram plot
    """

    def _add_peak_boundaries(self, fig, annotation_data):
        """
        Add peak boundaries to the plot.

        Args:
            annotation_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        if self.by is not None and self.legend.show:
            legend = fig.get_legend()
            fig.add_artist(legend)

        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=annotation_data.shape[0]
        )

        legend_items = []
        legend_labels = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            use_color = next(color_gen)
            left_vlne = fig.vlines(
                x=feature["leftWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.feature_config.line_width,
                color=use_color,
                ls=self.feature_config.line_type,
            )
            fig.vlines(
                x=feature["rightWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.feature_config.line_width,
                color=use_color,
                ls=self.feature_config.line_type,
            )
            legend_items.append(left_vlne)

            if "name" in annotation_data.columns:
                use_name = feature["name"]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                cur_legend_labels = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                cur_legend_labels = f"{use_name}"
            legend_labels.append(cur_legend_labels)

        if self.annotation_legend_config.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.annotation_legend_config.loc
            )
            fig.legend(
                legend_items,
                legend_labels,
                loc=matplotlibLegendLoc,
                title=self.annotation_legend_config.title,
                prop={"size": self.annotation_legend_config.fontsize},
                bbox_to_anchor=self.annotation_legend_config.bbox_to_anchor,
            )

        # since matplotlib is not interactive cannot implement the following methods
        def get_manual_bounding_box_coords(self):
            pass


class MATPLOTLIBMobilogramPlot(MATPLOTLIBChromatogramPlot, MobilogramPlot):
    """
    Class for assembling a matplotlib mobilogram plot
    """

    pass


@APPEND_PLOT_DOC
class MATPLOTLIBSpectrumPlot(MATPLOTLIB_MSPlot, SpectrumPlot):
    """
    Class for assembling a matplotlib spectrum plot
    """

    pass


class MATPLOTLIBPeakMapPlot(MATPLOTLIB_MSPlot, PeakMapPlot):
    """
    Class for assembling a matplotlib feature heatmap plot
    """

    # override creating figure because create a 2 by 2 figure
    def _create_figure(self):
        # Create a 2 by 2 figure and axis for marginal plots
        if self.add_marginals:
            superFig, ax_grid = plt.subplots(
                2, 2, figsize=(self.width / 100, self.height / 100), dpi=200
            )
            return superFig, ax_grid
        else:
            super()._create_figure()

    def plot(self):
        """
        Create the plot
        """

        if self.add_marginals:
            superFig, ax_grid = self._create_figure()
            ax_grid[0, 0].remove()
            ax_grid[0, 0].axis("off")
            superFig.set_size_inches(self.width / 100, self.height / 100)
            superFig.subplots_adjust(wspace=0, hspace=0)

            self.create_main_plot_marginals(ax=ax_grid[1, 1])
            self.create_x_axis_plot(ax=ax_grid[0, 1])
            self.create_y_axis_plot(ax=ax_grid[1, 0])
            return ax_grid

        else:
            return super().plot()

    def combine_plots(
        self, main_fig, x_fig, y_fig
    ):  # plots all plotted on same figure do not need to combine
        pass

    def create_x_axis_plot(self, main_plot=None, ax=None) -> "figure":
        super().create_x_axis_plot(ax=ax)

        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_ylabel(self.zlabel)
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.legend_ = None

        return ax

    def create_y_axis_plot(self, ax=None) -> "figure":
        # Note y_config is different so we cannot use the base class methods
        group_cols = [self.y]
        if self.by is not None:
            group_cols.append(self.by)

        y_data = self._integrate_data_along_dim(self.data, group_cols, self.z)

        if self.y_kind in ["chromatogram", "mobilogram"]:
            y_plot_obj = self.get_line_renderer(
                data=y_data, x=self.z, y=self.y, by=self.by, _config=self.y_plot_config
            )
        elif self.y_kind == "spectrum":
            y_plot_obj = self.get_vline_renderer(
                data=y_data, x=self.z, y=self.y, by=self.by, _config=self.y_plot_config
            )
        else:
            raise ValueError(f"Invalid y_kind: {self.y_kind}")
        y_fig = y_plot_obj.generate(None, None, fig=ax)

        self.plot_x_axis_line(y_fig)

        ax.set_xlim((0, y_data[self.z].max() + y_data[self.z].max() * 0.1))
        ax.invert_xaxis()
        ax.set_title(None)
        ax.set_xlabel(self.y_plot_config.xlabel)
        ax.set_ylabel(self.y_plot_config.ylabel)
        ax.set_ylim(ax.get_ylim())

        return ax

    def create_main_plot(self):
        if not self.plot_3d:
            scatterPlot = self.get_scatter_renderer(
                data=self.data,
                x=self.x,
                y=self.y,
                z=self.z,
                _config=self._config,
            )
            fig = scatterPlot.generate(None, None)

            if self.annotation_data is not None:
                self._add_box_boundaries(fig, self.annotation_data)

            return fig
        else:
            vlinePlot = self.get_vline_renderer(
                data=self.data, x=self.x, y=self.y, _config=self._config
            )
            return vlinePlot.generate(None, None)

    def create_main_plot_marginals(self, ax=None):
        scatterPlot = self.get_scatter_renderer(
            data=self.data, x=self.x, y=self.y, z=self.z, _config=self._config
        )
        scatterPlot.generate(None, None, fig=ax)

        ax.set_title(None)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(None)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.legend_ = None

    def _add_box_boundaries(self, fig, annotation_data):
        if self.by is not None:
            legend = fig.get_legend()
            fig.add_artist(legend)

        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=annotation_data.shape[0]
        )
        legend_items = []

        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            x0 = feature["leftWidth"]
            x1 = feature["rightWidth"]
            y0 = feature["IM_leftWidth"]
            y1 = feature["IM_rightWidth"]

            # Calculate center points and dimensions
            width = abs(x1 - x0)
            height = abs(y1 - y0)

            color = next(color_gen)
            custom_lines = Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=color,
                linestyle=self.feature_config.line_type,
                linewidth=self.feature_config.line_width,
            )
            fig.add_patch(custom_lines)

            if "name" in annotation_data.columns:
                use_name = feature["name"]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_labels = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_labels = f"{use_name}"

        # Add legend
        if self.feature_config.legend.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.feature_config.legend.loc
            )
            fig.legend(
                [custom_lines],
                [legend_labels],
                loc=matplotlibLegendLoc,
                title=self.feature_config.legend.title,
                prop={"size": self.feature_config.legend.fontsize},
                bbox_to_anchor=self.feature_config.legend.bbox_to_anchor,
            )

    # since matplotlib is not interactive cannot implement the following methods
    def get_manual_bounding_box_coords(self):
        pass
