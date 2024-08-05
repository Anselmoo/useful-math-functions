"""Create math function and plot it."""

from __future__ import annotations

import json

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import numpy as np

from umf import core
from umf.constants.dimensions import __2d__
from umf.constants.dimensions import __3d__
from umf.constants.exceptions import PlotAttributeError
from umf.constants.exceptions import TooHighDimensionError
from umf.constants.exceptions import TooLowDimensionError
from umf.images.diagrams import ClassicPlot
from umf.images.diagrams import PlotlyPlot
from umf.meta.plots import GIFSettings
from umf.meta.plots import GraphSettings


if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.pyplot import Figure as FigureTypeMatplotlib
    from plotly.graph_objects import Figure as FigureTypePlotly

    from umf.types.static_types import MeshArray
    from umf.types.static_types import UniversalArray

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


class Borg:
    """Borg class for functions.

    !!! info "About Borg"

        This class is used to create shared states for math functions. It uses the
        Borg design pattern to ensure that all instances of a math function share the
        same state. The shared state is stored in the `shared_state` dictionary, while
        the shared result and plot are stored in the `shared_result` and `shared_plot`
        dictionaries, respectively.
    """

    shared_state: ClassVar[dict[str, Any]] = {}
    shared_result: ClassVar[dict[str, Any]] = {}
    shared_plot: ClassVar[dict[str, Any]] = {}

    def __init__(self) -> None:
        """Initialize shared states of math function."""
        self.__dict__ = self.shared_state

    def auto_cleanup(self) -> None:
        """Auto clean up shared states of math function."""
        Borg.shared_state = {} if Borg.shared_state else Borg.shared_state
        Borg.shared_result = {} if Borg.shared_result else Borg.shared_result
        Borg.shared_plot = {} if Borg.shared_plot else Borg.shared_plot


class FunctionBorg(Borg):
    """A class that initializes shared states of math functions.

    Attributes:
    _x (UniversalArray): Arguments to be passed to the math function.
    func_name (list[str] | str): Name of the math function(s) to be called.
    func_args (list[dict[str, Any]] | dict[str, Any] | None, optional): Arguments to be
        passed to the math function(s), by default None.
    shared_state (dict[str, Any]): Shared states of math function.

    Methods:
    __init__(*x: UniversalArray, func_name: list[str] | str, func_args: list[dict[str,
    Any]] | dict[str, Any] | None = None) -> None:
        Initializes shared states of math functions.
    initialize_borg() -> None:
        Initialize shared states of math function.
    return_borg() -> dict[str, Any]:
        Return shared states of math function.
    """

    def __init__(
        self,
        *x: UniversalArray,
        func_name: list[str] | str,
        func_args: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> None:
        """Initializes shared states of math functions.

        Args:
        *x: UniversalArray
            Arguments to be passed to the math function.
        func_name: list[str] | str
            Name of the math function(s) to be called.
        func_args: list[dict[str, Any]] | dict[str, Any] | None, optional
            Arguments to be passed to the math function(s), by default None.
        """
        super().__init__()

        self._x: tuple[UniversalArray, ...] = x
        func_args = func_args or [{} for _ in range(len(func_name))]
        self.func_args = [func_args] if isinstance(func_args, dict) else func_args
        self.func_name = [func_name] if isinstance(func_name, str) else func_name
        self.initialize_borg()

    def initialize_borg(self) -> None:
        """Initialize shared states of math function."""
        for name, args in zip(self.func_name, self.func_args, strict=True):
            try:
                func: Callable[[Any], Any] = getattr(core, name)
            except AttributeError as exc:
                msg = f"Unknown function '{name}'"
                raise AttributeError(msg) from exc
            try:
                self.shared_state[name] = func(*self._x, **args)()
            except TypeError as exc:
                msg = f"Function '{name}' has invalid arguments"
                raise TypeError(msg) from exc

    @property
    def return_borg(self) -> dict[str, Any]:
        """Return shared states of math function."""
        return self.shared_state


class PlotBorg(Borg):
    """A singleton class that manages shared settings for plotting math functions.

    !!! info "About PlotBorg"

        This class inherits from the `Borg` class, which allows it to share state
        across multiple instances. It defines various properties and methods for
        managing settings related to plotting, such as the size and resolution of
        the plot, the axis labels, the color scheme, and the plot style (either
        Matplotlib or Plotly). It also defines a `plot` method that plots the
        shared states of a math function using either Matplotlib or Plotly,
        depending on the value of the `_plot_style` attribute.

    Attributes:
        size (tuple[int, int]): The size of the plot in inches.
        dpi (int): The resolution of the plot in dots per inch.
        axis (list[str] | None): The labels for the x and y axes.
        color (str | None): The color scheme for the plot.
        cmap (str): The colormap for the plot.
        alpha (float | None): The transparency of the plot.
        plot_type_3d (str): The type of 3D plot to use (if any).
        plot_style (str): The style of the plot (either "matplot" or "plotly").
        show (bool): Whether to show the plot after it is generated.
    """

    def __init__(self) -> None:
        """Initialize shared settings for plotting math functions."""
        super().__init__()

        self._dpi = GraphSettings().dpi
        self._size = GraphSettings().size
        self._dpi = GraphSettings().dpi
        self._axis = GraphSettings().axis
        self._color = GraphSettings().color
        self._cmap = GraphSettings().cmap
        self._alpha = GraphSettings().alpha
        self._plot_type_3d = "plot_3d"
        self._plot_style = "matplot"
        self._show = True

    @property
    def size(self) -> tuple[int, int]:
        """Return the size of the plot."""
        return self._size

    @size.setter
    def size(self, value: tuple[int, int]) -> None:
        """Set the size of the plot."""
        self._size = value

    @property
    def dpi(self) -> int:
        """Return the resolution of the plot."""
        return self._dpi

    @dpi.setter
    def dpi(self, value: int) -> None:
        """Set the resolution of the plot."""
        self._dpi = value

    @property
    def axis(self) -> list[str] | None:
        """Return the labels for the x and y axes."""
        return self._axis

    @axis.setter
    def axis(self, value: list[str] | None) -> None:
        """Set the labels for the x and y axes."""
        self._axis = value

    @property
    def color(self) -> str | None:
        """Return the color scheme for the plot."""
        return self._color

    @color.setter
    def color(self, value: str | None) -> None:
        """Set the color scheme for the plot."""
        self._color = value

    @property
    def cmap(self) -> str:
        """Return the colormap for the plot."""
        return self._cmap

    @cmap.setter
    def cmap(self, value: str) -> None:
        """Set the colormap for the plot."""
        self._cmap = value

    @property
    def alpha(self) -> float | None:
        """Return the transparency of the plot."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float | None) -> None:
        """Set the transparency of the plot."""
        self._alpha = value

    @property
    def plot_type_3d(self) -> str:
        """Return the type of 3D plot to use."""
        return self._plot_type_3d

    @plot_type_3d.setter
    def plot_type_3d(self, value: str) -> None:
        """Set the type of 3D plot to use."""
        self._plot_type_3d = value

    @property
    def plot_style(self) -> str:
        """Return the style of the plot."""
        return self._plot_style

    @plot_style.setter
    def plot_style(self, value: str) -> None:
        """Set the style of the plot."""
        self._plot_style = value

    @property
    def show(self) -> bool:
        """Return whether to show the plot after it is generated."""
        return self._show

    @show.setter
    def show(self, value: bool) -> None:
        """Set whether to show the plot after it is generated."""
        self._show = value

    def plot(self, **kwargs: dict[str, Any]) -> None:
        """Plot shared states of math function.

        Info "About plot"
            This method plots the shared states of a math function using either
            Matplotlib or Plotly, depending on the value of the `_plot_style`
            attribute. The settings for the plot are determined by the values of various
            other attributes, such as `_size`, `_dpi`, `_axis`, `_color`, `_cmap`,
            and `_alpha`.

        Args:
            **kwargs (dict[str, Any]): Additional arguments for plot_2d, plot_3d,
        """
        settings = GraphSettings(
            size=self._size,
            dpi=self._dpi,
            axis=self._axis,
            color=self._color,
            cmap=self._cmap,
            alpha=self._alpha,
        )
        for name, x in self.shared_result.items():
            if self._plot_style == "matplot":
                self.matplot(*x, title=name, settings=settings, **kwargs)
            elif self._plot_style == "plotly":
                self.plotly(*x, title=name, settings=settings, **kwargs)
            else:
                raise PlotAttributeError(
                    choose=self._plot_style,
                    modes={"matplot", "plotly"},
                    error_type="style",
                )

    def plot_series(self, **kwargs: dict[str, Any]) -> None:
        """Plot shared states of math function as series.

        Info "About plot_series"
            This method plots the shared states of a math function as series using
            either Matplotlib or Plotly, depending on the value of the `_plot_style`
            attribute. The settings for the plot are determined by the values of various
            other attributes, such as `_size`, `_dpi`, `_axis`, `_color`, `_cmap`,
            and `_alpha`.

        Args:
            **kwargs (dict[str, Any]): Additional arguments for plot_2d, plot_3d,
        """
        settings = GraphSettings(
            size=self._size,
            dpi=self._dpi,
            axis=self._axis,
            color=self._color,
            cmap=self._cmap,
            alpha=self._alpha,
        )

        label = []
        values: list[UniversalArray] = []
        for name, x in self.shared_result.items():
            label.append(name)
            values.append(x)

        if self._plot_style == "matplot":
            plot = ClassicPlot(*values, settings=settings, **kwargs)
            plot.plot_series(label=label)
            if self._show:
                plot.plot_show()
        elif self._plot_style == "plotly":
            plot = PlotlyPlot(*values, settings=settings, **kwargs)
            plot.plot_series(label=label)
            if self._show:
                plot.plot_show()
        else:
            raise PlotAttributeError(
                choose=self._plot_style,
                modes={"matplot", "plotly"},
                error_type="style",
            )
        self.shared_plot[f"Series_of_{'-'.join(label)}"] = plot.plot_return

    def matplot(
        self,
        *x: UniversalArray,
        title: str,
        settings: GraphSettings,
        **kwargs: dict[str, Any],
    ) -> None:
        """Plot shared states of math function via matplotlib.

        Args:
            *x (UniversalArray): Arrays to plot.
            title (str): Title of plots (used as key in shared plot).
            settings (GraphSettings): Settings of plot via `GraphSettings`.
            **kwargs (dict[str, Any]): Additional arguments for plot_2d, plot_3d,
                plot_contour, plot_surface, plot_dashboard.
        """
        plot = ClassicPlot(*x, settings=settings, **kwargs)
        dim = len(x)

        modes: set[str] = {"plot_3d", "plot_contour", "plot_surface", "plot_dashboard"}
        if self._plot_type_3d not in modes:
            raise PlotAttributeError(choose=self._plot_type_3d, modes=modes)

        if dim == __2d__:
            plot.plot_2d()
        elif dim == __3d__:
            getattr(plot, self._plot_type_3d)()
        else:
            raise TooHighDimensionError(max_dimension=__3d__, current_dimension=dim)
        self.shared_plot[title] = plot.plot_return
        if self._show:
            plot.plot_show()

    def plotly(
        self,
        *x: UniversalArray,
        title: str,
        settings: GraphSettings,
        **kwargs: dict[str, Any],
    ) -> None:
        """Plot shared states of math function via plotly.

        Args:
            *x (UniversalArray): Arrays to plot.
            title (str): Title of plots (used as key in shared plot).
            settings (GraphSettings): Settings of plot via `GraphSettings`.
            **kwargs (dict[str, Any]): Additional arguments for plot_2d, plot_3d,
        """
        plot = PlotlyPlot(*x, settings=settings, **kwargs)
        dim = len(x)
        modes: set[str] = {"plot_3d", "plot_contour", "plot_surface", "plot_dashboard"}
        if self._plot_type_3d not in modes:
            raise PlotAttributeError(choose=self._plot_type_3d, modes=modes)
        if dim == __2d__:
            plot.plot_2d()
        elif dim == __3d__:
            getattr(plot, self._plot_type_3d)()
        else:
            raise TooHighDimensionError(max_dimension=__3d__, current_dimension=dim)
        self.shared_plot[title] = plot.plot_return
        if self._show:
            plot.plot_show()


class SaveBorg(Borg):
    """Class for saving shared states of math function."""

    def __init__(self) -> None:
        """Initialize shared states of math function."""
        super().__init__()

        self.dpi = GIFSettings().dpi
        self._zoom_start = GIFSettings().zoom_start
        self._zoom_stop = GIFSettings().zoom_stop
        self._elev = GIFSettings().elev
        self._azim = GIFSettings().azim
        self._frames = GIFSettings().frames
        self._interval = GIFSettings().interval

    @property
    def dpi(self) -> int:
        """Return dpi of gif."""
        return self._dpi

    @dpi.setter
    def dpi(self, value: int) -> None:
        """Set dpi of gif."""
        self._dpi = value

    @property
    def zoom_start(self) -> float:
        """Return zoom in of gif."""
        return self._zoom_start

    @zoom_start.setter
    def zoom_start(self, value: float) -> None:
        """Set zoom in of gif."""
        self._zoom_start = value

    @property
    def zoom_stop(self) -> float:
        """Return zoom out of gif."""
        return self._zoom_stop

    @zoom_stop.setter
    def zoom_stop(self, value: float) -> None:
        """Set zoom out of gif."""
        self._zoom_stop = value

    @property
    def elev(self) -> int:
        """Return elevation of gif."""
        return self._elev

    @elev.setter
    def elev(self, value: int) -> None:
        """Set elevation of gif."""
        self._elev = value

    @property
    def azim(self) -> int:
        """Return azimuth of gif."""
        return self._azim

    @azim.setter
    def azim(self, value: int) -> None:
        """Set azimuth of gif."""
        self._azim = value

    @property
    def frames(self) -> int:
        """Return frames of gif."""
        return self._frames

    @frames.setter
    def frames(self, value: int) -> None:
        """Set frames of gif."""
        self._frames = value

    @property
    def interval(self) -> int:
        """Return interval of gif."""
        return self._interval

    def save_as_csv(self, dir_name: str = ".", **kwargs: dict[str, Any]) -> None:
        """Save shared states of math function as csv file.

        Args:
            dir_name (str, optional): Directory name to save file. Defaults to '.'.
            **kwargs (Any): Additional arguments for numpy.savetxt.
        """
        self.if_dir_not_exist(dir_name)
        for name, values in self.shared_result.items():
            self.to_csv(*values, fname=Path(dir_name) / f"{name}.csv", **kwargs)

    def save_as_json(self, dir_name: str = ".", **kwargs: dict[str, Any]) -> None:
        """Save shared states of math function as json file.

        Args:
            dir_name (str, optional): Directory name to save file. Defaults to '.'.
            **kwargs (Any): Additional arguments for json.dump.
        """
        self.if_dir_not_exist(dir_name)
        for name, values in self.shared_result.items():
            self.to_json(*values, fname=Path(dir_name) / f"{name}.json", **kwargs)

    def save_as_pickle(self, dir_name: str = ".", **kwargs: dict[str, Any]) -> None:
        """Save shared states of math function as pickle file.

        Args:
            dir_name (str, optional): Directory name to save file. Defaults to '.'.
            **kwargs (Any): Additional arguments for numpy.savez.
        """
        self.if_dir_not_exist(dir_name)
        for name, values in self.shared_result.items():
            self.to_pickle(*values, fname=Path(dir_name) / f"{name}.npz", **kwargs)

    def save_as_clipboard(self) -> None:
        """Save shared states of math function to clipboard."""
        raise NotImplementedError

    def save_as_image(
        self,
        dir_name: str = ".",
        ffomat: str = "png",
        **kwargs: dict[str, Any],
    ) -> None:
        """Save shared states of math function as image file."""
        self.if_dir_not_exist(dir_name)
        for name, fig in self.shared_plot.items():
            self.to_image(fig, fname=Path(dir_name) / name, fformat=ffomat, **kwargs)

    def save_as_gif(
        self,
        *,
        dir_name: str = ".",
        zoom: bool = True,
        rotate: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """Save shared states of math function as gif file.

        Args:
            dir_name (str, optional): Directory name to save file. Defaults to '.'.
            zoom (bool, optional): Whether to save zoom gif. Defaults to True.
            rotate (bool, optional): Whether to save rotate gif. Defaults to True.
            **kwargs (dict[str, Any]): Additional arguments for plot_save.
        """
        settings = GIFSettings(
            dpi=self._dpi,
            zoom_start=self._zoom_start,
            zoom_stop=self._zoom_stop,
            elev=self._elev,
            azim=self._azim,
            frames=self._frames,
            interval=self._interval,
            zoom=zoom,
            rotate=rotate,
        )
        self.if_dir_not_exist(dir_name)
        for name, fig in self.shared_plot.items():
            self.to_gif(
                fig,
                fname=Path(dir_name) / f"{name}.gif",
                settings=settings,
                **kwargs,
            )

    @staticmethod
    def if_dir_not_exist(dir_name: str) -> None:
        """Check if directory exist, if not create it."""
        if not Path(dir_name).exists():
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def to_csv(
        *x: UniversalArray,
        fname: Path,
        fmt: str = "%f",
        delimiter: str = ",",
        comments: str = "",
        **kwargs: dict[str, Any],
    ) -> None:
        """Save shared states of math function as csv file.

        Args:
            *x (UniversalArray): Arrays to save.
            fname (Path): Path to save file.
            fmt (str, optional): Format of file. Defaults to '%f'.
            delimiter (str, optional): Delimiter of file. Defaults to ','.
            comments (str, optional): Comments of file. Defaults to ''.
            **kwargs (Any): Additional arguments for numpy.savetxt.
        """
        length = len(x)
        np.savetxt(
            fname,
            np.asarray(x).T.reshape(-1, length),
            fmt=fmt,
            delimiter=delimiter,
            header=delimiter.join([f"x_{i}" for i in range(length)]),
            comments=comments,
            **kwargs,
        )

    @staticmethod
    def to_json(
        *x: UniversalArray,
        fname: Path,
        mode: str = "w+",
        encoding: str = "UTF-8",
        indent: int = 4,
        **kwargs: dict[str, Any],
    ) -> None:
        """Save shared states of math function as json file.

        Args:
            *x (UniversalArray): Arrays to save.
            mode (str, optional): Mode of file. Defaults to 'w+'.
            encoding (str, optional): Encoding of file. Defaults to 'UTF-8'.
            indent (int, optional): Indent of file. Defaults to 4.
            fname (Path): Path to save file.
            **kwargs (dict[str, Any]): Additional arguments for json.dump.
        """
        with Path(fname).open(mode=mode, encoding=encoding) as f:
            json.dump(
                {f"x_{i}": x[i].tolist() for i in range(len(x))},
                f,
                indent=indent,
                **kwargs,
            )

    @staticmethod
    def to_pickle(*x: UniversalArray, fname: Path, **kwargs: dict[str, Any]) -> None:
        """Save shared states of math function as pickle file.

        Args:
            *x (UniversalArray): Arrays to save.
            fname (Path): Path to save file.
            **kwargs (Any): Additional arguments for numpy.savez.
        """
        np.savez(fname, *x, **kwargs)

    def to_clipboard(self) -> None:
        """Save shared states of math function to clipboard."""
        raise NotImplementedError

    @staticmethod
    def to_image(
        fig: FigureTypePlotly | FigureTypeMatplotlib,
        fname: Path,
        fformat: str = "png",
        **kwargs: dict[str, Any],
    ) -> None:
        """Save shared states of math function as image file.

        Args:
            fig (FigureTypePlotly | FigureTypeMatplotlib): Figure to save.
            fname (Path): Path to save file.
            fformat (str, optional): Format of file. Defaults to 'png'.
            **kwargs (dict[str, Any]): Additional arguments for plot_save.
        """
        plot_classes = {
            plt.Figure: ClassicPlot,
            go.Figure: PlotlyPlot,
        }

        plot_class = plot_classes.get(type(fig))
        if plot_class is not None:
            plot_class.plot_save(fig=fig, fname=fname, fformat=fformat, **kwargs)

    @staticmethod
    def to_gif(
        fig: FigureTypePlotly | FigureTypeMatplotlib,
        fname: Path,
        ax_fig: plt.Figure,
        settings: GIFSettings,
        **kwargs: dict[str, Any],
    ) -> None:
        """Save shared states of math function as gif file.

        Notes:
            This method is only supported for Matplotlib plots.

        Args:
            fig (FigureTypePlotly | FigureTypeMatplotlib): Figure to save.
            fname (Path): Path to save file.
            ax_fig (plt.Figure): Figure to save.
            settings (GIFSettings): Settings of gif via `GIFSettings`.
            **kwargs (dict[str, Any]): Additional arguments for plot_save_gif.
        """
        plot_classes = {
            plt.Figure: ClassicPlot,
            go.Figure: PlotlyPlot,
        }

        plot_class = plot_classes.get(type(fig))
        if plot_class is not None:
            plot_class.plot_save_gif(
                fig=fig,
                ax_fig=ax_fig,
                fname=fname,
                settings=settings,
                **kwargs,
            )


class OptBench(FunctionBorg, PlotBorg, SaveBorg):
    """A class for optimizing and benchmarking mathematical functions.

    !!! info "About OptBench"

        This class inherits from the `FunctionBorg`, `PlotBorg`, and `SaveBorg` classes,
        which allows it to share state across multiple instances. It defines various
        properties and methods for managing settings related to plotting, such as the
        size and resolution of the plot, the axis labels, the color scheme, and the
        plot style (either Matplotlib or Plotly). It also defines a `plot` method that
        plots the shared states of a math function using either Matplotlib or Plotly,
        depending on the value of the `_plot_style` attribute.

    Examples:
        >>> # 3D examples
        >>> from umf.core.create import OptBench
        >>> res = OptBench(["DeJongN5Function", "AckleyFunction"], dim=3)
        >>> res.show = False
        >>> res.plot_type_3d = "plot_surface"
        >>> res.plot_style = "matplot"
        >>> res.plot()
        >>> res.save_as_image(dir_name="matplot")
        >>> # As plotly
        >>> res.plot_style = "plotly"
        >>> res.plot()
        >>> res.save_as_image(dir_name="plotly")
        >>> res.save_as_csv(dir_name="csv")
        >>> res.save_as_json(dir_name="json")

        >>> # 2D examples
        >>> from umf.core.create import OptBench
        >>> res = OptBench(
        ...     [
        ...         "CrystalBallDistribution",
        ...         "PseudoVoigtDistribution",
        ...         "AsymmetricRamanLineshape",
        ...     ],
        ...     [
        ...         {"mu": 1.0, "sigma":2, "alpha":0.5, "n":1.0},
        ...         {"mu": 1.0, "sigma":2, "eta":0.5},
        ...         {"mu": 1.0, "sigma":2, "eta":0.5, "gamma":0.5},
        ...     ],
        ...     start=-10,
        ...     stop=10,
        ...     step_size=0.1,
        ...     dim=2,
        ... )
        >>> res.show = False
        >>> res.plot_style = "matplot"
        >>> res.plot_series()
        >>> res.save_as_image(dir_name="matplot")
        >>> # As plotly
        >>> res.plot_style = "plotly"
        >>> res.plot_series()
        >>> res.save_as_image(dir_name="plotly")

    Args:
        func_name (list[str] | str): Name of the function(s) to optimize and benchmark.
        func_args (list[dict[str, Any]] | dict[str, Any] | None, optional): Arguments
             for the function(s). Defaults to None.
        start (float, optional): Start value of the range. Defaults to -5.0.
        stop (float, optional): Stop value of the range. Defaults to 5.0.
        step_size (float | int, optional): Step value of the range or number of maximum
             steps. Defaults to 0.1.
        dim (int, optional): Number of dimensions. Defaults to 2.
        mode (str, optional): Mode for creating the range of values. Must be either
             'arange' or 'linspace'. Defaults to "arange".

    Raises:
        ValueError: Mode must be either 'arange' or 'linspace'.
        ValueError: Number of dimensions must be greater than 1.

    Attributes:
        dim (int): Number of dimensions.
        shared_result (dict): Dictionary containing arrays for plotting and saving.

    Methods:
        initializeshared_result(self) -> None:
            Make arrays for plotting, saving and add them to share result.
        create_range(start: float, stop: float,
        step_size: float, dim: int = 2, mode: str = "arange")
        -> list[UniversalArray] | MeshArray:
            Create range of values for each dimension.
    """

    def __init__(  # noqa: PLR0913
        self,
        func_name: list[str] | str,
        func_args: list[dict[str, Any]] | dict[str, Any] | None = None,
        start: float = -5.0,
        stop: float = 5.0,
        step_size: float = 0.1,
        dim: int = 2,
        mode: str = "arange",
    ) -> None:
        """Initialize shared states of math functions."""
        self.auto_cleanup()
        super().__init__(
            *self.create_range(
                start=start,
                stop=stop,
                step_size=step_size,
                dim=dim,
                mode=mode,
            ),
            func_name=func_name,
            func_args=func_args,
        )
        self.initializeshared_result()
        super(PlotBorg, self).__init__()
        super(SaveBorg, self).__init__()
        self.dim = dim

    def initializeshared_result(self) -> None:
        """Make arrays for plotting, saving and add them to share result."""
        for name in self.func_name:
            self.shared_result[name] = np.asarray(
                [*self.shared_state[name].x, self.shared_state[name].result],
                dtype=np.float64,
            )

    @staticmethod
    def create_range(
        start: float,
        stop: float,
        step_size: float,
        dim: int = 2,
        mode: str = "arange",
    ) -> list[UniversalArray] | MeshArray:
        """Create range of values for each dimension.

        Args:
            start (float): Start value of range.
            stop (float): Stop value of range.
            step_size (float): Step value of range or number of maximum steps.
            dim (int, optional): Number of dimensions. Defaults to 2.
            mode (str, optional): Mode for creating the range of values. Must be either
                'arange' or 'linspace'. Defaults to "arange".

        Raises:
            ValueError: Mode must be either 'arange' or 'linspace'.
            TooSmallDimensionError: Number of dimensions must be greater than 1.

        Returns:
            list[UniversalArray] | MeshArray: Range of values for each dimension. In
                case of 2 dimensions, a 1D array is returned. In case of more than 2
                dimensions, a meshgrid is returned.
        """
        if mode not in {"arange", "linspace"}:
            msg = "Mode must be either 'arange' or 'linspace'"
            raise ValueError(msg)

        if dim < __2d__:
            raise TooLowDimensionError(min_dimension=__2d__, current_dimension=dim)

        if dim == __2d__:
            return [getattr(np, mode)(start, stop, step_size, dtype=np.float64)]

        return np.meshgrid(
            *[
                getattr(np, mode)(start, stop, step_size, dtype=np.float64)
                for _ in range(dim - 1)
            ],
        )
