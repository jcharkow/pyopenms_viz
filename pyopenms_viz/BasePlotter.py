from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Engine(Enum):
    PLOTLY = 1
    BOKEH = 2


# A colorset suitable for color blindness
class Colors(str, Enum):
    BLUE = "#4575B4"
    RED = "#D73027"
    LIGHTBLUE = "#91BFDB"
    ORANGE = "#FC8D59"
    PURPLE = "#7B2C65"
    YELLOW = "#FCCF53"
    DARKGRAY = "#555555"
    LIGHTGRAY = "#DDDDDD"


@dataclass(kw_only=True)
class _BasePlotterConfig(ABC):
    title: str = "1D Plot"
    xlabel: str = "X-axis"
    ylabel: str = "Y-axis"
    engine: Literal["PLOTLY", "BOKEH"] = "PLOTLY"
    height: int = 500
    width: int = 500
    relative_intensity: bool = False
    show_legend: bool = True

    @property
    def engine_enum(self):
        return Engine[self.engine]


# Abstract Class for Plotting
class _BasePlotter(ABC):
    def __init__(self, config: _BasePlotterConfig) -> None:
        self.config = config
        self.fig = None  # holds the figure object

    def updateConfig(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid config setting: {key}")

    def plot(self, data, **kwargs):
        if self.config.engine_enum == Engine.PLOTLY:
            return self._plotPlotly(data, **kwargs)
        else:  # self.config.engine_enum == Engine.BOKEH:
            return self._plotBokeh(data, **kwargs)

    @abstractmethod
    def _plotBokeh(self, data, **kwargs):
        pass

    @abstractmethod
    def _plotPlotly(self, data, **kwargs):
        pass
