from __future__ import annotations

from abc import ABC

from typing import Tuple

from bokeh.plotting import figure
from bokeh.palettes import Plasma256
from bokeh.transform import linear_cmap
from bokeh.models import (
    ColumnDataSource,
    Legend,
    Range1d,
    BoxEditTool,
    Span,
    VStrip,
    GlyphRenderer,
)

from pandas.core.frame import DataFrame

# pyopenms_viz imports
from .._core import (
    BasePlot,
    LinePlot,
    VLinePlot,
    ScatterPlot,
    BaseMSPlot,
    ChromatogramPlot,
    MobilogramPlot,
    FeatureHeatmapPlot,
    SpectrumPlot,
    APPEND_PLOT_DOC,
)
from .._misc import ColorGenerator
from ..constants import PEAK_BOUNDARY_ICON, FEATURE_BOUNDARY_ICON


class BOKEHPlot(BasePlot, ABC):
    """
    Base class for assembling a Bokeh plot
    """

    def _interactive(self):
        return False

    def _load_extension(self) -> None:
        try:
            from bokeh.plotting import figure, show
            from bokeh.models import ColumnDataSource, Legend
        except ImportError:
            raise ImportError(
                f"bokeh is not installed. Please install using `pip install bokeh` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self) -> None:
        if self.fig is None:
            self.fig = figure(
                title=self.title,
                x_axis_label=self.xlabel,
                y_axis_label=self.ylabel,
                width=self.width,
                height=self.height,
            )

    def _update_plot_aes(self, fig, **kwargs):
        """
        Update the aesthetics of the plot
        """
        fig.grid.visible = self.grid
        fig.toolbar_location = self.toolbar_location

    def _add_legend(self, fig, legend):
        """
        Add the legend
        """
        if self.legend_config.show:
            fig.add_layout(legend, self.legend_config.loc)
            fig.legend.orientation = self.legend_config.orientation
            fig.legend.click_policy = self.legend_config.onClick
            fig.legend.title = self.legend_config.title
            fig.legend.label_text_font_size = str(self.legend_config.fontsize) + "pt"

    def _add_tooltips(self, fig, tooltips, custom_hover_data=None):
        """
        Add tooltips to the plot
        """
        from bokeh.models import HoverTool

        hover = HoverTool()
        hover.tooltips = tooltips
        fig.add_tools(hover)

    def _add_bounding_box_drawer(self, fig, **kwargs):
        """
        Add a BoxEditTool to the figure for drawing bounding boxes.

        Args:
            fig (figure): The Bokeh figure object to add the BoxEditTool to.

        Returns:
            The renderer object that is used to draw the bounding box.
        """
        r = fig.rect(
            [],
            [],
            [],
            [],
            fill_alpha=0,
            line_dash="dashed",
            line_width=3,
            line_color="#F02D1A",
        )
        draw_tool = BoxEditTool(renderers=[r], empty_value=0)
        # TODO: change how icon path is defined
        draw_tool.icon = FEATURE_BOUNDARY_ICON
        draw_tool.name = "Draw Bounding Box"

        # Add the tool to the figure
        fig.add_tools(draw_tool)
        return r

    def _add_bounding_vertical_drawer(self, fig, **kwargs):
        """
        Add a BoxEditTool to the figure for drawing bounding vertical strips.

        Args:
            fig (figure): The Bokeh figure object to add the BoxEditTool to.

        Returns:
            The renderer object that is used to draw the bounding box.
        """
        # Create empty source data
        source = ColumnDataSource(data=dict(x0=[], x1=[]))

        # Create the VStrip glyph
        glyph = VStrip(
            x0="x0",
            x1="x1",
            fill_alpha=0,
            line_dash="dashed",
            line_width=3,
            line_color="#F02D1A",
        )

        # Create a GlyphRenderer for the VStrip glyph using the same data source
        renderer = GlyphRenderer(data_source=source, glyph=glyph)

        # Add the GlyphRenderer to the fig object's renderers list
        fig.renderers.append(renderer)

        draw_tool = BoxEditTool(renderers=[renderer], empty_value=0)
        # TODO: change how icon path is defined
        draw_tool.icon = PEAK_BOUNDARY_ICON
        draw_tool.name = "Draw Peak Boundary Strip"

        # Add the tool to the figure
        fig.add_tools(draw_tool)
        return renderer

    def _modify_x_range(
        self, x_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the x-axis range.

        Args:
            x_range (Tuple[float, float]): The desired x-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the x-axis range, in decimal percent. Defaults to None.
        """
        start, end = x_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.x_range = Range1d(start=start, end=end)

    def _modify_y_range(
        self, y_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modifies the y-axis range of the plot.

        Args:
            y_range (Tuple[float, float]): The desired y-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the y-axis range, in decimal percent. Defaults to None.
        """
        start, end = y_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.y_range = Range1d(start=start, end=end)

    def show(self):
        from bokeh.io import show

        def app(doc):
            doc.add_root(self.fig)

        show(app)


class BOKEHLinePlot(BOKEHPlot, LinePlot):
    """
    Class for assembling a collection of Bokeh line plots
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a line plot
        """
        color_gen = kwargs.pop("line_color", None)

        if by is None:
            source = ColumnDataSource(data)
            if color_gen is not None:
                kwargs["line_color"] = next(color_gen)
            line = fig.line(x=x, y=y, source=source, **kwargs)

            return fig, None
        else:

            legend_items = []
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                if color_gen is not None:
                    kwargs["line_color"] = next(color_gen)
                line = fig.line(x=x, y=y, source=source, **kwargs)
                legend_items.append((group, [line]))

            legend = Legend(items=legend_items)

            return fig, legend


class BOKEHVLinePlot(BOKEHPlot, VLinePlot):
    """
    Class for assembling a series of vertical line plots in Bokeh
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a set of vertical lines
        """

        if by is None:
            color_gen = kwargs.pop("line_color", None)
            source = ColumnDataSource(data)
            if color_gen is not None:
                kwargs["line_color"] = next(color_gen)
            line = fig.segment(x0=x, y0=0, x1=x, y1=y, source=source, **kwargs)
            return fig, None
        else:
            color_gen = kwargs.pop("line_color", None)
            legend_items = []
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                if color_gen is not None:
                    kwargs["line_color"] = next(color_gen)
                line = fig.segment(x0=x, y0=0, x1=x, y1=y, source=source, **kwargs)
                legend_items.append((group, [line]))

            legend = Legend(items=legend_items)

            return fig, legend

    def _add_annotation(self, fig, data, x, y, **kwargs):
        # TODO: Implement text label annotations
        pass


class BOKEHScatterPlot(BOKEHPlot, ScatterPlot):
    """
    Class for assembling a Bokeh scatter plot
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a scatter plot
        """

        print(kwargs)
        print(cls.z)
        if by is None:
            source = ColumnDataSource(data)
            line = fig.scatter(x=x, y=y, source=source, **kwargs)
            return fig, None
        else:
            legend_items = []
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                line = fig.scatter(x=x, y=y, source=source, **kwargs)
                legend_items.append((group, [line]))
            legend = Legend(items=legend_items)

            return fig, legend


class BOKEH_MSPlot(BaseMSPlot, BOKEHPlot, ABC):

    def get_line_renderer(self, data, x, y, **kwargs) -> None:
        return BOKEHLinePlot(data, x, y, **kwargs)

    def get_vline_renderer(self, data, x, y, **kwargs) -> None:
        return BOKEHVLinePlot(data, x, y, **kwargs)

    def get_scatter_renderer(self, data, x, y, **kwargs) -> None:
        return BOKEHScatterPlot(data, x, y, **kwargs)

    def plot_x_axis_line(self, fig):
        zero_line = Span(
            location=0, dimension="width", line_color="#EEEEEE", line_width=1.5
        )
        fig.add_layout(zero_line)

    def _create_tooltips(self):
        # Tooltips for interactive information
        TOOLTIPS = [
            ("index", "$index"),
            ("Retention Time", "@rt{0.2f}"),
            ("Intensity", "@int{0.2f}"),
            ("m/z", "@mz{0.4f}"),
        ]
        if "Annotation" in self.data.columns:
            TOOLTIPS.append(("Annotation", "@Annotation"))
        if "product_mz" in self.data.columns:
            TOOLTIPS.append(("Target m/z", "@product_mz{0.4f}"))
        return TOOLTIPS, None


class BOKEHChromatogramPlot(BOKEH_MSPlot, ChromatogramPlot):
    """
    Class for assembling a Bokeh extracted ion chromatogram plot
    """

    def _add_peak_boundaries(self, annotation_data):
        """
        Add peak boundaries to the plot.

        Args:
            annotation_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        color_gen = ColorGenerator(
            colormap=self.annotation_config.colormap, n=annotation_data.shape[0]
        )
        legend_items = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            peak_boundary_lines = self.fig.segment(
                x0=[feature["leftWidth"], feature["rightWidth"]],
                y0=[0, 0],
                x1=[feature["leftWidth"], feature["rightWidth"]],
                y1=[feature["apexIntensity"], feature["apexIntensity"]],
                color=next(color_gen),
                line_dash=self.annotation_config.line_type,
                line_width=self.annotation_config.line_width,
            )
            if "name" in annotation_data.columns:
                use_name = feature["name"]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_label = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"{use_name}"
            legend_items.append((legend_label, [peak_boundary_lines]))

        if self.annotation_config.legend_config.show:
            legend = Legend(items=legend_items)
            legend.click_policy = self.annotation_config.legend_config.onClick
            legend.title = self.annotation_config.legend_config.title
            legend.orientation = self.annotation_config.legend_config.orientation
            legend.label_text_font_size = (
                str(self.annotation_config.legend_config.fontsize) + "pt"
            )
            self.fig.add_layout(legend, self.annotation_config.legend_config.loc)

    def get_manual_bounding_box_coords(self):
        # Get the original data source
        data_source = self.manual_boundary_renderer.data_source

        # Make a copy of the data since we don't want to modify the original active documents data source
        bbox_data = data_source.data.copy()

        # Return the modified copy
        return DataFrame(bbox_data).rename(
            columns={"x0": "leftWidth", "x1": "rightWidth"}
        )


class BOKEHMobilogramPlot(BOKEHChromatogramPlot, MobilogramPlot):
    """
    Class for assembling a Bokeh mobilogram plot
    """

    pass


class BOKEHSpectrumPlot(BOKEH_MSPlot, SpectrumPlot):
    """
    Class for assembling a Bokeh spectrum plot
    """

    pass


class BOKEHFeatureHeatmapPlot(BOKEH_MSPlot, FeatureHeatmapPlot):
    """
    Class for assembling a Bokeh feature heatmap plot
    """

    def create_main_plot(self, x, y, z, by):
        scatterPlot = self.get_scatter_renderer(
            self.data, x, y, by=by, _config=self._config
        )
        mapper = linear_cmap(
            field_name=z,
            palette=Plasma256[::-1],
            low=min(self.data[z]),
            high=max(self.data[z]),
        )

        self.fig = scatterPlot.generate(
            marker="square", line_color=mapper, fill_color=mapper
        )

        if self.annotation_data is not None:
            self._add_box_boundaries(self.annotation_data)

    def create_x_axis_plot(self, x, z, by):
        x_fig = super().create_x_axis_plot(x, z, by)

        # Modify plot
        x_fig.x_range = self.fig.x_range
        x_fig.width = self.fig.width
        x_fig.xaxis.visible = False
        x_fig.min_border = 0
        return x_fig

    def create_y_axis_plot(self, y, z, by):
        y_fig = super().create_y_axis_plot(y, z, by)

        # Modify plot
        y_fig.y_range = self.fig.y_range
        y_fig.height = self.fig.height
        y_fig.legend.orientation = "horizontal"
        y_fig.x_range.flipped = True
        y_fig.min_border = 0
        return y_fig

    def combine_plots(self, x_fig, y_fig):
        # Modify the main plot
        self.fig.yaxis.visible = False

        from bokeh.layouts import gridplot

        self.fig = gridplot([[None, x_fig], [y_fig, self.fig]])

    def get_manual_bounding_box_coords(self):
        # Get the original data source
        data_source = self.manual_bbox_renderer.data_source

        # Make a copy of the data since we don't want to modify the original active documents data source
        bbox_data = data_source.data.copy()

        x1 = []
        y1 = []
        for i in range(0, len(bbox_data["x"])):
            x1.append(bbox_data["x"][i] + bbox_data["width"][i])
            y1.append(bbox_data["y"][i] + bbox_data["height"][i])

        # Return the modified copy
        return DataFrame(
            {"x0": bbox_data["x"], "x1": x1, "y0": bbox_data["y"], "y1": y1}
        ).rename(
            columns={
                "x0": f"{self.x}_0",
                "x1": f"{self.x}_1",
                "y0": f"{self.y}_0",
                "y1": f"{self.y}_1",
            }
        )

    def _add_box_boundaries(self, annotation_data):
        color_gen = ColorGenerator(
            colormap=self.annotation_config.colormap, n=annotation_data.shape[0]
        )
        legend_items = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            x0 = feature["leftWidth"]
            x1 = feature["rightWidth"]
            y0 = feature["IM_leftWidth"]
            y1 = feature["IM_rightWidth"]

            # Calculate center points and dimensions
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            width = abs(x1 - x0)
            height = abs(y1 - y0)

            box_boundary_lines = self.fig.rect(
                x=center_x,
                y=center_y,
                width=width,
                height=height,
                color=next(color_gen),
                line_dash=self.annotation_config.line_type,
                line_width=self.annotation_config.line_width,
                fill_alpha=0,
            )
            if "name" in annotation_data.columns:
                use_name = feature["name"]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_label = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"{use_name}"
            legend_items.append((legend_label, [box_boundary_lines]))

        if self.annotation_config.legend.show:
            legend = Legend(items=legend_items)
            legend.click_policy = self.annotation_config.legend_config.onClick
            legend.title = self.annotation_config.legend_config.title
            legend.orientation = self.annotation_config.legend_config.orientation
            legend.label_text_font_size = (
                str(self.annotation_config.legend_config.fontsize) + "pt"
            )
            self.fig.add_layout(legend, self.annotation_config.legend_config.loc)
