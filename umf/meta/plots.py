"""Abstract class for plotting functions.

!!! info "About"

    This module defines an abstract class `Plot` and a named tuple `GraphSettings` for
    plotting functions.

    - `GraphSettings`: A class for named tuples which defines the settings for
        the graph, such as size, dpi, axis labels, title, color, colormap, and
        alpha value.
    - `Plot`: An abstract class that defines methods for plotting data in 2D, 3D,
        contour, surface, and dashboard formats, as well as animating the plot. It also
        defines abstract methods for showing and saving the plot.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any


if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    from umf.types.static_types import UniversalArray

from typing import NamedTuple


class GraphSettings(NamedTuple):
    """Settings for the graph.

    Attributes:
        size (tuple[int, int]): The size of the plot. Defaults to (5, 5).
        dpi (int, optional): The DPI of the plot. Defaults to 200.
        axis (list[str], optional): The labels for the axes of the plot. Defaults to
             None.
        title (str, optional): The title of the plot. Defaults to None.
        color (str, optional): The color of the plot. Defaults to None.
        cmap (str, optional): The colormap of the plot. Defaults to "YlGnBu_r".
        alpha (float, optional): The alpha value of the plot. Defaults to None.
    """

    size: tuple[int, int] = (5, 5)
    dpi: int = 200
    axis: list[str] | None = None
    title: str | None = None
    color: str | None = None
    cmap: str = "viridis_r"
    alpha: float | None = None


class GIFSettings(NamedTuple):
    """A named tuple representing the settings for the GIF animation of a 3D plot.

    Args:
        dpi (int, optional): The resolution of the output GIF in dots per inch.
            Defaults to 100.
        zoom (bool, optional): Whether or not to include a zoom effect in the animation.
            Defaults to True.
        zoom_start (float, optional): The starting zoom level for the zoom effect.
            Defaults to 0.5.
        zoom_stop (float, optional): The ending zoom level for the zoom effect.
            Defaults to 1.5.
        rotate (bool, optional): Whether or not to include a rotation effect in the
            animation. Defaults to True.
        elev (int, optional): The elevation angle of the plot in degrees. Defaults to
            30.
        azim (int, optional): The azimuth angle of the plot in degrees. Defaults to 5.
        frames (int, optional): The number of frames in the animation. Defaults to 72.
        interval (int, optional): The delay between frames in milliseconds. Defaults to
             50.
    """

    dpi: int = 100
    zoom: bool = True
    zoom_start: float = 0.5
    zoom_stop: float = 1.5
    rotate: bool = True
    elev: int = 30
    azim: int = 5
    frames: int = 72
    interval: int = 50


class AnimationSettings(NamedTuple):
    """A named tuple representing the settings for the animation of a 2D and 3D plot.

    Args:
        frames (int, optional): The number of frames in the animation. Defaults to 40.
        interval (int, optional): The delay between frames in milliseconds. Defaults
            to 30.
        dpi (int, optional): The resolution of the output animation in dots per inch.
            Defaults to 100.
    """

    frames: int = 40
    interval: int = 30
    dpi: int = 100
    steps: int = 100


class Plot(ABC):
    """Abstract class for plotting functions.

    Args:
        *x (UniversalArray):  Input data, which can be one, two, three, or higher
             dimensional.

        **kwargs (dict[str, Any]): Additional keyword arguments to pass to the plot
             function.
    """

    def __init__(
        self,
        *x: UniversalArray,
        settings: GraphSettings = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the function."""
        if x is None:
            msg = "x has to be specified."
            raise ValueError(msg)

        if settings is None:
            settings = GraphSettings()

        self._x = x
        self.dimension = len(x)
        self.title = settings.title
        self.axis = (
            settings.axis
            if settings.axis is not None
            else [f"x_{i}" for i in range(self.dimension)]
        )
        self.color = settings.color
        self.cmap = settings.cmap
        self.alpha = settings.alpha
        self.size = settings.size
        self.dpi = settings.dpi
        self._kwargs = kwargs

    def plot_2d(self) -> None:
        """Plot the data in 2D."""
        raise NotImplementedError

    def plot_series(self) -> None:
        """Plot the data in 2D as a series."""
        raise NotImplementedError

    def plot_3d(self) -> None:
        """Plot the data in 3D."""
        raise NotImplementedError

    def plot_contour(self) -> None:
        """Plot the data as a contour plot."""
        raise NotImplementedError

    def plot_surface(self) -> None:
        """Plot the data as a surface plot."""
        raise NotImplementedError

    def plot_dashboard(self) -> None:
        """Plot the data as a dashboard."""
        raise NotImplementedError

    def animate(self) -> None:
        """Animate the plot."""
        raise NotImplementedError

    @abstractmethod
    def plot_show(self) -> None:
        """Show the plot."""

    @property
    @abstractmethod
    def plot_return(self) -> plt.figure | go.Figure:
        """Return the plot."""

    @staticmethod
    def ax_return() -> plt.Figure:
        """Return the Figure."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def plot_save(
        fig: plt.Figure | go.Figure,
        fname: Path,
        fformat: str = "png",
        **kwargs: dict[str, Any],
    ) -> None:
        """Saves the given plot to a file.

        Args:
            fig (plt.Figure | go.Figure): The figure to save as an image file.
            fname (Path): The file path to save the plot to.
            fformat (str, optional): The format to save the plot as. Defaults to "png".
            **kwargs (dict[str, Any]): Additional keyword arguments to pass to the save
                 function.
        """

    @staticmethod
    def plot_save_gif(
        fig: plt.Figure,
        ax_fig: plt.Figure,
        fname: Path,
        settings: GIFSettings,
        **kwargs: dict[str, Any],
    ) -> None:
        """Saves the given plot to a file.

        Notes:
            This function can currently only be implemented for matplotlib plots and
            not for plotly plots.

        Args:
            fig (plt.figure): The figure to save as an image file.
            ax_fig (plt.Figure): The figure to save as an image file.
            fname (Path): The file path to save the plot to.
            settings (GIFSettings): The settings for the GIF animation.
            **kwargs (dict[str, Any]): Additional keyword arguments to pass to the save
                 function.

        Raises:
            NotImplementedError: This function is not yet implemented.
        """
        raise NotImplementedError
