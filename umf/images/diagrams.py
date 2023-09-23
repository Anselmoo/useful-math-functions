"""Plotting functions for 2D and 3D functions."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from matplotlib.animation import FuncAnimation

from umf.constants.exceptions import PlotAttributeError
from umf.meta.plots import GIFSettings
from umf.meta.plots import Plot


if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.pyplot import Figure as FigureTypeMatplotlib
    from plotly.graph_objects import Figure as FigureTypePlotly

    from umf.types.static_types import PlotlyScatterParameters


class ClassicPlot(Plot):
    r"""Plotting functions using via matplotlib.

    Examples:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from umf.images.diagrams import ClassicPlot
        >>> from umf.meta.plots import GraphSettings
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = X ** 2 + Y ** 2
        >>> plot = ClassicPlot(X, Y, Z, settings=GraphSettings(axis=["x", "y", "z"]))
        >>> plot.plot_3d()
        >>> ClassicPlot.plot_save(plot.plot_return, Path("ClassicPlot_3d.png"))
        >>> plot.plot_contour()
        >>> ClassicPlot.plot_save(plot.plot_return, Path("ClassicPlot_contour.png"))
        >>> plot.plot_surface()
        >>> ClassicPlot.plot_save(plot.plot_return, Path("ClassicPlot_surface.png"))
        >>> plot.plot_dashboard()
        >>> ClassicPlot.plot_save(plot.plot_return, Path("ClassicPlot_dashboard.png"))
        >>> plot.plot_close()

    Examples:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from umf.images.diagrams import ClassicPlot
        >>> from umf.meta.plots import GraphSettings
        >>> from umf.meta.plots import GIFSettings
        >>> from umf.functions.optimization.special import GoldsteinPriceFunction
        >>> # Start with a simple plot
        >>> x = np.linspace(-2, 2, 100)
        >>> y = np.linspace(-2, 2, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = GoldsteinPriceFunction(X, Y).__eval__
        >>> plot = ClassicPlot(
        ...     X,
        ...     Y,
        ...     Z,
        ...     settings=GraphSettings(axis=["x", "y", "z"]),
        ... )
        >>> plot.plot_surface()
        >>> # Now only zoom
        >>> plot.plot_save_gif(
        ...     fig=plot.plot_return,
        ...     ax_fig=plot.ax_return,
        ...     fname=Path("GoldsteinPriceFunction_zoom.gif"),
        ...     settings=GIFSettings(rotate=False),
        ...     savefig_kwargs={"transparent": True},
        ... )
        >>> # Now only rotate
        >>> plot.plot_save_gif(
        ...     fig=plot.plot_return,
        ...     ax_fig=plot.ax_return,
        ...     fname=Path("GoldsteinPriceFunction_rotate.gif"),
        ...     settings=GIFSettings(zoom=False),
        ...     savefig_kwargs={"transparent": True},
        ... )
        >>> # Now only zoom and rotate
        >>> plot.plot_save_gif(
        ...     fig=plot.plot_return,
        ...     ax_fig=plot.ax_return,
        ...     fname=Path("GoldsteinPriceFunction_all.gif"),
        ...     settings=GIFSettings(),
        ...     savefig_kwargs={"transparent": True},
        ... )
        >>> plot.plot_close()

    Examples:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from umf.images.diagrams import ClassicPlot
        >>> from umf.meta.plots import GraphSettings
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ...    RayleighDistribution,
        ... )
        >>> x = np.linspace(0, 10, 100)
        >>> y_sigma_1 = RayleighDistribution(x, sigma=1).__eval__
        >>> y_sigma_2 = RayleighDistribution(x, sigma=2).__eval__
        >>> y_sigma_3 = RayleighDistribution(x, sigma=3).__eval__
        >>> plot = ClassicPlot(
        ...     np.array([x, y_sigma_1]),
        ...     np.array([x, y_sigma_2]),
        ...     np.array([x, y_sigma_3]),
        ...     settings=GraphSettings(
        ...         axis=["x", r"$f(x)$"],
        ...         title="Rayleigh Distribution",
        ...     ),
        ... )
        >>> plot.plot_series(label=[r"$\sigma=1$", r"$\sigma=2$", r"$\sigma=3$"])
        >>> ClassicPlot.plot_save(plot.plot_return, Path("ClassicPlot_series.png"))

    Args:
        *x (UniversalArray): Input data, which can be one, two, three, or higher
             dimensional.
        settings (GraphSettings, optional): Settings for the graph. Defaults to None.
        **kwargs (dict[str, Any]): Additional keyword arguments to pass to the plot
             function.
    """

    fig: plt.figure
    ax: plt.Figure

    def plot_2d(self, ax: FigureTypeMatplotlib | None = None) -> None:
        """Plot a 2D function.

        Args:
            ax (FigureTypeMatplotlib, optional): Figure object to plot the data.
                 Defaults to None.
        """
        if ax is None:
            self.fig = plt.figure(figsize=self.size)
            self.ax = self.fig.add_subplot(111)
        self.ax.plot(*self._x, color=self.color, **self._kwargs)
        self.label_settings()

    def plot_series(
        self,
        ax: FigureTypeMatplotlib | None = None,
        label: list[str | None] | None = None,
    ) -> None:
        """Plot a 2D function as a series.

        Args:
            ax (FigureTypeMatplotlib, optional): Figure object to plot the data.
                 Defaults to None.
            label (list[str | None], optional): The label of each line. Defaults to
                     None.
        """
        if ax is None:
            self.fig = plt.figure(figsize=self.size)
            self.ax = self.fig.add_subplot(111)
        if label is None:
            label = [None for _ in self._x]
        for i, _x in enumerate(self._x):
            self.ax.plot(*_x, label=label[i], **self._kwargs)
        self.label_settings(legend=True)

    def plot_3d(self, ax: plt.Figure | None = None) -> None:
        """Plot a 3D function.

        Args:
            ax (FigureTypeMatplotlib, optional): Figure object to plot the data.
                 Defaults to None.
        """
        if ax is None:
            self.fig = plt.figure(figsize=self.size, dpi=self.dpi)
            self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.plot_wireframe(
            *self._x,
            edgecolor=plt.cm.get_cmap(self.color).colors,
            alpha=self.alpha,
            **self._kwargs,
        )
        self.label_settings(dim3=True)

    def plot_contour(self, ax: FigureTypeMatplotlib | None = None) -> None:
        """Plot a contour plot."""
        if ax is None:
            self.fig = plt.figure(figsize=self.size, dpi=self.dpi)
            self.ax = self.fig.add_subplot(111)
        self.ax.contour(
            *self._x,
            cmap=plt.cm.get_cmap(self.cmap),
            alpha=self.alpha,
            **self._kwargs,
        )
        self.label_settings()

    def plot_surface(self, ax: FigureTypeMatplotlib | None = None) -> None:
        """Plot a 3D function."""
        if ax is None:
            self.fig = plt.figure(figsize=self.size, dpi=self.dpi)
            self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.plot_surface(
            *self._x,
            cmap=plt.cm.get_cmap(self.cmap),
            alpha=self.alpha,
            **self._kwargs,
        )
        self.label_settings(dim3=True)

    def label_settings(self, *, dim3: bool = False, legend: bool = False) -> None:
        """Set the labels for a 2D or 3D plot.

        Args:
            ax (plt.Figure): Figure objects to set the labels and title.
            dim3 (bool, optional): Whether the plot is 3D. Defaults to False.
            legend (bool, optional): Whether to show the legend. Defaults to False.
        """
        self.ax.set_xlabel(self.axis[0])
        self.ax.set_ylabel(self.axis[1])
        if dim3:
            self.ax.set_zlabel(self.axis[2])
        self.ax.set_title(self.title)
        if legend is not None:
            self.ax.legend()

    def plot_dashboard(self) -> None:
        """Plot a dashboard."""
        from matplotlib import gridspec

        self.fig = plt.figure(tight_layout=True, figsize=self.size, dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2)

        self.plot_contour(self.fig.add_subplot(gs[0, :]))
        self.plot_3d(self.fig.add_subplot(gs[1, 0], projection="3d"))
        self.plot_surface(self.fig.add_subplot(gs[1, 1], projection="3d"))
        self.fig.align_labels()

    def plot_show(self) -> None:
        """Show the plot."""
        plt.show()

    @property
    def plot_return(self) -> plt.figure:
        """Return the plot."""
        return self.fig

    @property
    def ax_return(self) -> plt.Figure:
        """Return the Figure."""
        return self.ax

    @staticmethod
    def plot_save(
        fig: plt.figure,
        fname: Path,
        fformat: str = "png",
        **kwargs: dict[str, Any],
    ) -> None:
        """Save the plot.

        Args:
            fig (plt.figure): The figure to save.
            fname (Path): The filename to save the figure to.
            fformat (str, optional): The format to save the plot as. Defaults to "png".
            **kwargs (dict[str, Any]): Additional keyword arguments to pass to the
                 save function.
        """
        fig.savefig(fname.with_suffix(f".{fformat}"), **kwargs)

    @staticmethod
    def plot_save_gif(
        *,
        fig: plt.figure,
        ax_fig: plt.Figure,
        fname: Path,
        settings: GIFSettings,
        **kwargs: dict[str, Any],
    ) -> None:
        """Saves the given plot to a file.

        Note:
            For gnerating GIFs, the a subfunction is used to update the plot for each
            frame of the animation. This subfunction is defined in the function
            `update`.

        Args:
            fig (plt.figure): The figure to save.
            ax_fig (plt.Figure): The figure to save.
            fname (Path): The filename to save the figure to.
            settings (GIFSettings): The settings for the GIF.
            **kwargs (dict[str, Any]): Additional keyword arguments to pass to the
                    save function.
        """

        def update(frame: int, settings: GIFSettings) -> list[plt.Figure]:
            """Updates the plot for each frame of the animation.

            Args:
                fig (plt.figure): The figure to update.
                ax_fig (plt.Figure): The figure to update.
                frame (int): The current frame number.
                settings (GIFSettings): The settings for the GIF.
                **kwargs (dict[str, Any]): Additional keyword arguments to pass to the
                    save function.

            Returns:
                list[plt.Figure]: A list of the updated plot elements.
            """
            surf = ax_fig
            if settings.zoom:
                ax_fig.set_box_aspect(
                    None,
                    zoom=np.linspace(
                        settings.zoom_start,
                        settings.zoom_stop,
                        settings.frames,
                    )[frame],
                )
            if settings.rotate:
                ax_fig.view_init(
                    elev=settings.elev,
                    azim=frame * settings.azim % 360,
                )
            return [surf]

        anim = FuncAnimation(
            fig,
            update,
            frames=settings.frames,
            interval=settings.interval,
            fargs=(settings,),
        )
        anim.save(fname, writer="imagemagick", dpi=settings.dpi, **kwargs)

    @staticmethod
    def plot_close() -> None:
        """Close all plots."""
        plt.close("all")


class PlotlyPlot(Plot):
    r"""Plotting functions using via plotly.

    Examples:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from umf.images.diagrams import PlotlyPlot
        >>> from umf.meta.plots import GraphSettings
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = X ** 2 + Y ** 2
        >>> plot = PlotlyPlot(X, Y, Z, settings=GraphSettings(axis=["x", "y", "z"]))
        >>> plot.plot_3d()
        >>> PlotlyPlot.plot_save(plot.plot_return, Path("PlotlyPlot_3d.png"))
        >>> plot.plot_contour()
        >>> PlotlyPlot.plot_save(plot.plot_return, Path("PlotlyPlot_contour.png"))
        >>> plot.plot_surface()
        >>> PlotlyPlot.plot_save(plot.plot_return, Path("PlotlyPlot_surface.png"))

    Examples:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from umf.images.diagrams import PlotlyPlot
        >>> from umf.meta.plots import GraphSettings
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ...    RayleighDistribution,
        ... )
        >>> x = np.linspace(0, 10, 100)
        >>> y_sigma_1 = RayleighDistribution(x, sigma=1).__eval__
        >>> y_sigma_2 = RayleighDistribution(x, sigma=2).__eval__
        >>> y_sigma_3 = RayleighDistribution(x, sigma=3).__eval__
        >>> plot = PlotlyPlot(
        ...     np.array([x, y_sigma_1]),
        ...     np.array([x, y_sigma_2]),
        ...     np.array([x, y_sigma_3]),
        ...     settings=GraphSettings(
        ...         axis=["x", r"$f(x)$"],
        ...         title="Rayleigh Distribution",
        ...     ),
        ... )
        >>> plot.plot_series(label=[r"$\sigma=1$", r"$\sigma=2$", r"$\sigma=3$"])
        >>> PlotlyPlot.plot_save(plot.plot_return, Path("PlotlyPlot_series.png"))

    Args:
        *x (UniversalArray): Input data, which can be one, two, three, or higher
             dimensional.
        settings (GraphSettings, optional): Settings for the graph. Defaults to None.
        **kwargs (dict[str, Any]): Keyword arguments for the plot.
    """

    fig: FigureTypePlotly

    def plot_2d(self, *, mode: str = "lines", width: int = 2) -> None:
        """Plot a 2D function.

        Args:
            mode (str, optional): The mode of the plot. Defaults to "lines".
            width (int, optional): The width of the line. Defaults to 2.
        """
        self.check_mode(mode)

        self.fig = go.Figure(
            data=go.Scatter(
                x=self._x[0],
                y=self._x[1],
                mode=mode,
                marker_color=self.color,
                line={"color": self.color, "width": width},
                **self._kwargs,
            ),
        )
        self.label_settings()

    def plot_series(
        self,
        mode: str = "lines",
        width: int = 2,
        label: list[str | None] | None = None,
    ) -> None:
        """Plot a 2D function as a series."""
        self.check_mode(mode)

        self.fig = go.Figure()
        if label is None:
            label = [None for _ in self._x]
        for i, _x in enumerate(self._x):
            self.fig.add_trace(
                go.Scatter(
                    x=_x[0],
                    y=_x[1],
                    mode=mode,
                    marker_color=self.color,
                    line={"color": self.color, "width": width},
                    name=label[i],
                    **self._kwargs,
                ),
            )
        self.label_settings(legend=True)

    def check_mode(self, mode: str) -> None:
        """Check if the mode is valid."""
        if mode not in {"lines", "markers", "lines+markers"}:
            raise PlotAttributeError(
                choose=mode,
                modes={"lines", "markers", "lines+markers"},
            )

    # 3D plots as mesh
    def plot_3d(self, width: int = 2) -> None:
        """Plot a 3D function as meshgrid.

        Args:
            width (int, optional): The width of the line. Defaults to 2.
        """
        lines = []

        def _scatter3d(
            *,
            width: int,
            i: int,
            twist: bool = False,
        ) -> PlotlyScatterParameters:
            """Return the function parameter for a Plotly scatter3d plot.

            Args:
                width (int): The width of the line.
                i (int): The index of the line.
                twist (bool, optional): Whether to twist the line. Defaults to False.

            Returns:
                PlotlyScatterParameters: The function parameter for a Plotly scatter3d
                     plot.
            """
            x, y, z = self._x
            x, y, z = (
                (x[:, i], y[:, i], z[:, i]) if twist else (x[i, :], y[i, :], z[i, :])
            )
            return {
                "x": x,
                "y": y,
                "z": z,
                "mode": "lines",
                "line": {"color": z, "width": width, "colorscale": self.cmap},
            }

        for i in range(len(self._x[0])):
            lines.extend(
                (
                    go.Scatter3d(**_scatter3d(width=width, i=i)),
                    go.Scatter3d(**_scatter3d(width=width, i=i, twist=True)),
                ),
            )
        self.fig = go.Figure(data=lines)
        self.label_settings()

    def plot_contour(
        self,
        *,
        contours_coloring: str | None = None,
        showscale: bool = False,
    ) -> None:
        """Plot a contour plot.

        Args:
            contours_coloring (str, optional): The color of the contours. Defaults to
                     None.
            showscale (bool, optional): Whether to show the color scale. Defaults to
                     False.

        Raises:
            ValueError: If contours_coloring is not one of "fill", "heatmap", "lines",
                    or "none".
        """
        if (
            contours_coloring not in {"fill", "heatmap", "lines", "none"}
            and contours_coloring is not None
        ):
            raise PlotAttributeError(
                choose=contours_coloring,
                modes={"fill", "heatmap", "lines", "none"},
            )
        self.fig = go.Figure(
            data=go.Contour(
                x=self._x[0].flatten(),
                y=self._x[1].flatten(),
                z=self._x[2].flatten(),
                colorscale=self.cmap,
                opacity=self.alpha,
                showscale=showscale,
                contours_coloring=contours_coloring,
                **self._kwargs,
            ),
        )
        self.label_settings()

    def plot_surface(
        self,
        *,
        color: str | None = None,
        showscale: bool = False,
    ) -> None:
        """Plot a 3D function as surface.

        Args:
            color (str, optional): The color of the plot. Defaults to None.
            showscale (bool, optional): Whether to show the color scale. Defaults to
                     False.
        """
        self.fig = go.Figure(
            data=go.Surface(
                x=self._x[0],
                y=self._x[1],
                z=self._x[2],
                colorscale=self.cmap,
                opacity=self.alpha,
                showscale=showscale,
                surfacecolor=color,
                **self._kwargs,
            ),
        )
        self.label_settings()

    def label_settings(self, *, legend: bool = False) -> None:
        """Set the labels for a 3D plot."""
        axis_titles = {
            2: {"xaxis_title", "yaxis_title"},
            3: {"xaxis_title", "yaxis_title", "zaxis_title"},
        }
        if len(self.axis) in axis_titles:
            scene = {
                title: self.axis[i]
                for i, title in enumerate(axis_titles[len(self.axis)])
            }
            self.fig.update_layout(
                title=self.title,
                scene=scene,
                showlegend=legend,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                }
                if legend
                else None,
            )

    def plot_show(self) -> None:
        """Show the plot."""
        self.fig.show()

    @property
    def plot_return(self) -> FigureTypePlotly:
        """Return the plot."""
        return self.fig

    @staticmethod
    def plot_save(
        fig: go.Figure,
        fname: Path,
        fformat: str = "png",
        scale: int = 3,
        **kwargs: dict[str, Any],
    ) -> None:
        """Save the plot.

        Args:
            fig (go.Figure): The figure to save.
            fname (Path): The filename to save the figure to.
            fformat (str, optional): The format to save the plot as. Defaults to "png".
            scale (int, optional): The scale of the plot. Defaults to 3.
            **kwargs (dict[str, Any]): Additional keyword arguments to pass to the
                 save function.
        """
        fig.write_image(fname.with_suffix(f".{fformat}"), scale=scale, **kwargs)
