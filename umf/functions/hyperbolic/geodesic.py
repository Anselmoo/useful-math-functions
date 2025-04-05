"""Hyperbolic geodesic function for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.meta.functions import HyperbolicFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "GeodesicFunction",
]


class GeodesicFunction(HyperbolicFunction):
    r"""Determine the shortest path between two points in the hyperbolic plane.

    The hyperbolic geodesic function determines the shortest path between two points
    in the hyperbolic plane.

    Examples:
        >>> from umf.functions.hyperbolic.geodesic import GeodesicFunction
        >>> point1 = np.array([0.1, 0.1])
        >>> point2 = np.array([1, 1])
        >>> hgf = GeodesicFunction(point1, point2)()
        >>> hgf.result
        array(2.89838887)

        >>> # Visualization Example
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.hyperbolic.geodesic import GeodesicFunction
        >>> point1 = np.array([0.1, 0.1])
        >>> point2 = np.array([1, 1])
        >>> hgf = GeodesicFunction(point1, point2)()
        >>> distance = hgf.result
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-')
        >>> _ = ax.set_xlim(-0.5, 1.5)
        >>> _ = ax.set_ylim(-0.5, 1.5)
        >>> _ = ax.set_aspect('equal')
        >>> _ = plt.title(f'Distance: {distance:.2f}')
        >>> plt.grid()
        >>> plt.savefig("GeodesicFunction.png", dpi=300, transparent=True)

    Notes:
        The hyperbolic geodesic between two points $(x_1, y_1)$ and $(x_2, y_2)$ in the
        hyperbolic plane is given by:

        $$
        d = \cosh^{-1}\left(1 + \frac{(x_2 - x_1)^2 + (y_2 - y_1)^2}{2 y_1 y_2}\right)
        $$

        > Reference: https://en.wikipedia.org/wiki/Hyperbolic_geodesic

    Args:
        *points (UniversalArray): The coordinates of the two points in the hyperbolic
            plane.
    """

    def __init__(self, *points: UniversalArray) -> None:
        """Initialize the hyperbolic geodesic function."""
        super().__init__(*points)

    @property
    def __eval__(self) -> float:
        """Determine the shortest path between two points in the hyperbolic plane.

        Returns:
            float: The length of the geodesic between the two points.
        """
        x1, y1 = self._x[0]
        x2, y2 = self._x[1]
        return np.arccosh(1 + ((x2 - x1) ** 2 + (y2 - y1) ** 2) / (2 * y1 * y2))
