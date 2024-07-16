from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal, Union, List, Dict
import importlib
import types

from pandas.core.frame import DataFrame
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.common import is_integer
from pandas.util._decorators import Appender

from ._config import LegendConfig, AnnotationConfig, _BasePlotConfig
from ._misc import ColorGenerator
from dataclasses import dataclass, asdict

_common_kinds = ("line", "vline", "scatter")
_msdata_kinds = ("chromatogram", "mobilogram", "spectrum", "feature_heatmap")
_all_kinds = _common_kinds + _msdata_kinds
_entrypoint_backends = ("pomsvim", "pomsvib", "pomsvip")

_baseplot_doc = f"""
    Plot method for creating plots from a Pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        The data to be plotted.
    x : str or None, optional
        The column name for the x-axis data.
    y : str or None, optional
        The column name for the y-axis data.
    z : str or None, optional
        The column name for the z-axis data (for 3D plots).
    kind : str, optional
        The kind of plot to create. One of: {_all_kinds}
    by : str or None, optional
        Column in the DataFrame to group by.
    relative_intensity : bool, default False
        Whether to use relative intensity for the y-axis.
    subplots : bool or None, optional
        Whether to create separate subplots for each column.
    sharex, sharey : bool or None, optional
        Whether to share x or y axes among subplots.
    height, width : int or None, optional
        The height and width of the figure in pixels.
    grid : bool or None, optional
        Whether to show the grid on the plot.
    toolbar_location : str or None, optional
        The location of the toolbar (e.g., 'above', 'below', 'left', 'right').
    fig : figure or None, optional
        An existing figure object to plot on.
    title : str or None, optional
        The title of the plot.
    xlabel, ylabel : str or None, optional
        Labels for the x and y axes.
    line_type : str or None, optional
        The type of line to use (e.g., 'solid', 'dashed', 'dotted').
    line_width : float or None, optional
        The width of the lines in the plot.
    min_border : int or None, optional
        The minimum border size around the plot.
    show_plot : bool or None, optional
        Whether to display the plot immediately after creation.
    legend : LegendConfig or dict or None, optional
        Configuration for the plot legend.
    feature_config : FeatureConfig or dict or None, optional
        Configuration for additional plot features.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For pyopenms_viz, options are one of {_entrypoint_backends} Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    **kwargs
        Additional keyword arguments to be passed to the plotting function.

    Returns
    -------
    None

    Examples
    --------
    >>> import pandas as pd
    >>> 
    >>> data = pd.DataFrame(dict'x': [1, 2, 3], 'y': [4, 5, 6]))
    >>> data.plot(x='x', y='y', kind='spectrum', backend='pomsvim')
    """

APPEND_PLOT_DOC = Appender(_baseplot_doc)


@dataclass
class BasePlot(ABC):
    """
    This class shows functions which must be implemented by all backends
    """

    # Data Attributes
    data: DataFrame
    x: str | None = None
    y: str | None = None
    z: str | None = None
    kind: (
        Literal[
            "line",
            "vline",
            "scatter",
            "chromatogram",
            "mobilogram",
            "spectrum",
            "feature_heatmap",
            "complex",
        ]
        | None
    ) = None
    by: str | None = None
    relative_intensity: bool = False

    # Plotting Attributes
    height: int | None = None
    width: int | None = None
    grid: bool | None = None
    toolbar_location: str | None = None
    fig: "figure" | None = None
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    line_type: str | None = None
    line_width: float | None = None
    show_plot: bool | None = None

    # Configurations
    legend_config: LegendConfig | Dict | None = None
    annotation_config: AnnotationConfig | Dict | None = None
    _config: _BasePlotConfig | None = None

    # Note priority is keyword arguments > config > default values
    # This allows for plots to have their own default configurations which can be overridden by the user
    def __post_init__(self):
        self.data = self.data.copy()
        if self._config is not None:
            self._update_from_config(self._config)

        if self.legend_config is not None:
            if isinstance(self.legend_config, dict):
                self.legend_config = LegendConfig.from_dict(self.legend_config)
        else:
            self.legend_config = LegendConfig()

        if self.annotation_config is not None:
            if isinstance(self.annotation_config, dict):
                self.annotation_config = AnnotationConfig.from_dict(
                    self.annotation_config
                )
        else:
            self.annotation_config = AnnotationConfig()

        self.update_config()  # update config based on kwargs

        ### get x and y data
        if self._kind in {
            "line",
            "vline",
            "scatter",
            "chromatogram",
            "mobilogram",
            "spectrum",
            "feature_heatmap",
            "complex",
        }:
            self.x = self._verify_column(self.x, "x")
            self.y = self._verify_column(self.y, "y")

        if self._kind in {"feature_heatmap"}:
            self.z = self._verify_column(self.z, "z")

        if self.by is not None:
            # Ensure by column data is string
            self.by = self._verify_column(self.by, "by")
            self.data[self.by] = self.data[self.by].astype(str)

        self._load_extension()
        self._create_figure()

    def _verify_column(self, colname: str | int, name: str) -> str:
        """fetch data from column name

        Args:
            colname (str | int): column name of data to fetch or the index of the column to fetch
            name (str): name of the column e.g. x, y, z for error message

        Returns:
            pd.Series: pandas series or None

        Raises:
            ValueError: if colname is None
            KeyError: if colname is not in data
            ValueError: if colname is not numeric
        """

        def holds_integer(column) -> bool:
            return column.inferred_type in {"integer", "mixed-integer"}

        if colname is None:
            raise ValueError(f"For `{self.kind}` plot, `{name}` must be set")

        # if integer is supplied get the corresponding column associated with that index
        if is_integer(colname) and not holds_integer(self.data.columns):
            if colname >= len(self.data.columns):
                raise ValueError(
                    f"Column index `{colname}` out of range, `{name}` could not be set"
                )
            else:
                colname = self.data.columns[colname]
        else:  # assume column name is supplied
            if colname not in self.data.columns:
                raise KeyError(
                    f"Column `{colname}` not in data, `{name}` could not be set"
                )

        # checks passed return column name
        return colname

    def __repr__(self):
        return f"{self.__class__.__name__}(kind={self._kind}, data=DataFrame({self.data.shape[0]} rows {self.data.shape[1]} columns), x={self.x}, y={self.y}, by={self.by})"

    @property
    @abstractmethod
    def _kind(self) -> str:
        """
        The kind of plot to assemble. Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def _interactive(self) -> bool:
        """
        Whether the plot is interactive. Must be overridden by subclasses
        """
        return NotImplementedError

    def _update_from_config(self, config) -> None:
        """
        Updates the plot configuration based on the provided `config` object.

        Args:
            config (Config): The configuration object containing the plot settings.

        Returns:
            None
        """
        for attr, value in config.__dict__.items():
            if (
                value is not None
                and hasattr(self, attr)
                and self.__dict__[attr] is None
            ):
                setattr(self, attr, value)

    def update_config(self) -> None:
        """
        Update the _config object based on the provided kwargs. This means that the _config will store an accurate representation of the parameters
        """
        for attr in self._config.__dict__.keys():
            setattr(self._config, attr, self.__dict__[attr])

    def _separate_class_kwargs(self, **kwargs):
        """
        Separates the keyword arguments into class-specific arguments and other arguments.

        Parameters:
            **kwargs: Keyword arguments passed to the method.

        Returns:
            class_kwargs: A dictionary containing the class-specific keyword arguments.
            other_kwargs: A dictionary containing the remaining keyword arguments.

        """
        class_kwargs = {k: v for k, v in kwargs.items() if k in dir(self)}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in dir(self)}
        return class_kwargs, other_kwargs

    @abstractmethod
    def _load_extension(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_figure(self) -> None:
        raise NotImplementedError

    def _make_plot(self, fig, **kwargs) -> None:
        # Check for tooltips in kwargs and pop
        tooltips = kwargs.pop("tooltips", None)
        custom_hover_data = kwargs.pop("custom_hover_data", None)

        newlines, legend = self.plot(fig, self.data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

        if tooltips is not None and self._interactive:
            self._add_tooltips(newlines, tooltips, custom_hover_data)

    @abstractmethod
    def plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Create the plot
        """
        pass

    @abstractmethod
    def _update_plot_aes(self, fig, **kwargs):
        pass

    @abstractmethod
    def _add_legend(self, fig, legend):
        pass

    @abstractmethod
    def _modify_x_range(
        self, x_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the x-axis range.

        Args:
            x_range (Tuple[float, float]): The desired x-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the x-axis range, in decimal percent. Defaults to None.
        """
        pass

    @abstractmethod
    def _modify_y_range(
        self, y_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the y-axis range.

        Args:
            y_range (Tuple[float, float]): The desired y-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the x-axis range, in decimal percent. Defaults to None.
        """
        pass

    def generate(self, **kwargs):
        """
        Generate the plot
        """
        self._make_plot(self.fig, **kwargs)
        return self.fig

    @abstractmethod
    def show(self):
        pass

    # methods only for interactive plotting
    @abstractmethod
    def _add_tooltips(self, fig, tooltips):
        pass

    @abstractmethod
    def _add_bounding_box_drawer(self, fig, **kwargs):
        pass

    def _add_bounding_vertical_drawer(self, fig, **kwargs):
        pass


class LinePlot(BasePlot, ABC):
    @property
    def _kind(self):
        return "line"


class VLinePlot(BasePlot, ABC):
    @property
    def _kind(self):
        return "vline"


class ScatterPlot(BasePlot, ABC):
    @property
    def _kind(self):
        return "scatter"


class BaseMSPlot(BasePlot, ABC):
    """
    Abstract class for complex plots, such as chromatograms and mobilograms which are made up of simple plots such as ScatterPlots, VLines and LinePlots.

    Args:
        BasePlot (_type_): _description_
        ABC (_type_): _description_
    """

    @abstractmethod
    def get_line_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def get_vline_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def get_scatter_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def plot_x_axis_line(self, fig):
        """
        plot line across x axis
        """
        pass

    @abstractmethod
    def _create_tooltips(self):
        pass


class ChromatogramPlot(BaseMSPlot, ABC):
    @property
    def _kind(self):
        return "chromatogram"

    def __init__(
        self, data, x, y, annotation_data: DataFrame | None = None, **kwargs
    ) -> None:

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        super().__init__(data, x, y, **kwargs)

        if annotation_data is not None:
            self.annotation_data = annotation_data.copy()
        else:
            self.annotation_data = None
        self.label_suffix = self.x  # set label suffix for bounding box

        self.plot(self.data, self.x, self.y, **kwargs)
        if self.show_plot:
            self.show()

    def plot(self, data, x, y, **kwargs):
        """
        Create the plot
        """
        color_gen = ColorGenerator()
        TOOLTIPS, custom_hover_data = self._create_tooltips()
        kwargs.pop(
            "fig", None
        )  # remove figure from **kwargs if exists, use the ChromatogramPlot figure object instead of creating a new figure
        linePlot = self.get_line_renderer(data, x, y, fig=self.fig, **kwargs)
        self.fig = linePlot.generate(
            line_color=color_gen, tooltips=TOOLTIPS, custom_hover_data=custom_hover_data
        )

        self._modify_y_range((0, self.data[y].max()), (0, 0.1))

        self.manual_boundary_renderer = (
            self._add_bounding_vertical_drawer(self.fig) if self._interactive else None
        )

        if self.annotation_data is not None:
            self._add_peak_boundaries(self.annotation_data)

    @abstractmethod
    def _add_peak_boundaries(self, annotation_data):
        """
        Prepare data for adding peak boundaries to the plot.

        Args:
            annotation_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        pass


class MobilogramPlot(ChromatogramPlot, ABC):

    @property
    def _kind(self):
        return "mobilogram"

    def __init__(
        self, data, x, y, annotation_data: DataFrame | None = None, **kwargs
    ) -> None:
        super().__init__(data, x, y, annotation_data=annotation_data, **kwargs)

    def plot(self, data, x, y, **kwargs):
        super().plot(data, x, y, **kwargs)
        self._modify_y_range((0, self.data[y].max()), (0, 0.1))


class SpectrumPlot(BaseMSPlot, ABC):
    @property
    def _kind(self):
        return "spectrum"

    def __init__(
        self,
        data,
        x,
        y,
        by,
        reference_spectrum: DataFrame | None = None,
        mirror_spectrum: bool = False,
        **kwargs,
    ) -> None:

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        super().__init__(data, x, y, **kwargs)

        self.reference_spectrum = reference_spectrum
        self.mirror_spectrum = mirror_spectrum

        self.plot(x, y, by)
        if self.show_plot:
            self.show()

    def plot(self, x, y, by=None):

        spectrum, reference_spectrum = self._prepare_data(
            self.data, y, self.reference_spectrum
        )

        color_gen = ColorGenerator()

        TOOLTIPS, custom_hover_data = self._create_tooltips()

        spectrumPlot = self.get_vline_renderer(
            spectrum, x, y, by=by, fig=self.fig, _config=self._config
        )
        self.fig = spectrumPlot.generate(
            line_color=color_gen, tooltips=TOOLTIPS, custom_hover_data=custom_hover_data
        )

        if self.mirror_spectrum and reference_spectrum is not None:
            ## create a mirror spectrum
            color_gen_mirror = ColorGenerator()
            reference_spectrum[y] = reference_spectrum[y] * -1

            mirror_spectrum = self.get_vline_renderer(
                reference_spectrum, x, y, by=by, fig=self.fig, _config=self._config
            )
            mirror_spectrum.generate(line_color=color_gen_mirror)
            self.plot_x_axis_line(self.fig)

    def _prepare_data(
        self,
        spectrum: DataFrame,
        y: str,
        reference_spectrum: Union[DataFrame, None],
    ) -> tuple[list, list]:
        """Prepares data for plotting based on configuration (ensures list format for input spectra, relative intensity, hover text)."""

        # copy spectrum data to not modify the original
        spectrum = spectrum.copy()
        reference_spectrum = (
            self.reference_spectrum.copy() if reference_spectrum is not None else None
        )

        # Convert to relative intensity if required
        if self.relative_intensity or self.mirror_spectrum:
            spectrum[y] = spectrum[y] / spectrum[y].max() * 100
            if reference_spectrum is not None:
                reference_spectrum[y] = (
                    reference_spectrum[y] / reference_spectrum[y].max() * 100
                )

        return spectrum, reference_spectrum


class FeatureHeatmapPlot(BaseMSPlot, ABC):
    # need to inherit from ChromatogramPlot and SpectrumPlot for get_line_renderer and get_vline_renderer methods respectively
    @property
    def _kind(self):
        return "feature_heatmap"

    def __init__(
        self,
        data,
        x,
        y,
        z,
        zlabel=None,
        add_marginals=False,
        annotation_data: DataFrame | None = None,
        **kwargs,
    ) -> None:

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        if add_marginals:
            kwargs["_config"].title = None

        self.zlabel = zlabel
        self.add_marginals = add_marginals

        if annotation_data is not None:
            self.annotation_data = annotation_data.copy()
        else:
            self.annotation_data = None
        super().__init__(data, x, y, z=z, **kwargs)

        self.plot(x, y, z, **kwargs)
        if self.show_plot:
            self.show()

    def plot(self, x, y, z, **kwargs):
        class_kwargs, other_kwargs = self._separate_class_kwargs(**kwargs)

        if self.add_marginals:
            self.create_main_plot_marginals(x, y, z, class_kwargs, other_kwargs)
        else:
            self.create_main_plot(x, y, z, class_kwargs, other_kwargs)

        self.manual_bbox_renderer = (
            self._add_bounding_box_drawer(self.fig) if self._interactive else None
        )

        if self.add_marginals:
            # remove 'config' from class_kwargs
            class_kwargs_copy = class_kwargs.copy()
            class_kwargs_copy.pop("_config", None)
            class_kwargs_copy.pop("by", None)

            x_fig = self.create_x_axis_plot(x, z, class_kwargs_copy)

            y_fig = self.create_y_axis_plot(y, z, class_kwargs_copy)

            self.combine_plots(x_fig, y_fig)

    @staticmethod
    def _integrate_data_along_dim(
        data: DataFrame, group_cols: List[str] | str, integrate_col: str
    ) -> DataFrame:
        # First fill NaNs with 0s for numerical columns and '.' for categorical columns
        grouped = (
            data.apply(
                lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
            )
            .groupby(group_cols)[integrate_col]
            .sum()
            .reset_index()
        )
        return grouped

    @abstractmethod
    def create_main_plot(self, x, y, z, class_kwargs, other_kwargs):
        pass

    # by default the main plot with marginals is plotted the same way as the main plot unless otherwise specified
    def create_main_plot_marginals(self, x, y, z, class_kwargs, other_kwargs):
        self.create_main_plot(x, y, z, class_kwargs, other_kwargs)

    @abstractmethod
    def create_x_axis_plot(self, x, z, class_kwargs) -> "figure":
        # get cols to integrate over and exclude y and z
        group_cols = [x]
        if self.by is not None:
            group_cols.append(self.by)

        x_data = self._integrate_data_along_dim(self.data, group_cols, z)

        x_config = self._config.copy()
        x_config.ylabel = self.zlabel
        x_config.y_axis_location = "right"
        x_config.legend_config.show = True
        x_config.legend_config.loc = "right"

        color_gen = ColorGenerator()

        # remove legend from class_kwargs to update legend args for x axis plot
        class_kwargs.pop("legend", None)
        class_kwargs.pop("ylabel", None)

        x_plot_obj = self.get_line_renderer(
            x_data, x, z, by=self.by, _config=x_config, **class_kwargs
        )
        x_fig = x_plot_obj.generate(line_color=color_gen)
        self.plot_x_axis_line(x_fig)

        return x_fig

    @abstractmethod
    def create_y_axis_plot(self, y, z, class_kwargs) -> "figure":
        group_cols = [y]
        if self.by is not None:
            group_cols.append(self.by)

        y_data = self._integrate_data_along_dim(self.data, group_cols, z)

        y_config = self._config.copy()
        y_config.xlabel = self.zlabel
        y_config.ylabel = self.ylabel
        y_config.y_axis_location = "left"
        y_config.legend_config.show = True
        y_config.legend_config.loc = "below"

        # remove legend from class_kwargs to update legend args for y axis plot
        class_kwargs.pop("legend", None)
        class_kwargs.pop("xlabel", None)

        color_gen = ColorGenerator()

        y_plot_obj = self.get_line_renderer(
            y_data, z, y, by=self.by, _config=y_config, **class_kwargs
        )
        y_fig = y_plot_obj.generate(line_color=color_gen)
        self.plot_x_axis_line(y_fig)

        return y_fig

    @abstractmethod
    def combine_plots(self, x_fig, y_fig):
        pass

    @abstractmethod
    def _add_box_boundaries(self, annotation_data):
        """
        Prepare data for adding box boundaries to the plot.

        Args:
            annotation_data (DataFrame): The feature data containing the box boundaries.

        Returns:
            None
        """
        pass
