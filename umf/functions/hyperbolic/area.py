"""Hyperbolic area function for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.meta.functions import HyperbolicFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "AreaFunction",
]


class AreaFunction(HyperbolicFunction):
    r"""Calculate the area of a polygon in the hyperbolic plane.

    The hyperbolic area function calculates the area of a polygon in the hyperbolic
    plane.

    Examples:
        >>> from umf.functions.hyperbolic.area import AreaFunction
        >>> vertices = np.array([(0, 0), (1, 0), (0, 1)])
        >>> haf = AreaFunction(*vertices)()
        >>> haf.result
        array(0.5)

        >>> # Visualization Example
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.hyperbolic.area import AreaFunction
        >>> vertices = np.array([(0, 0), (1, 0), (0, 1)])
        >>> haf = AreaFunction(*vertices)()
        >>> area = haf.result
        >>> fig, ax = plt.subplots()
        >>> polygon = plt.Polygon(vertices, closed=True, fill=None, edgecolor='r')
        >>> _ = ax.add_patch(polygon)
        >>> _ = ax.set_xlim(-0.5, 1.5)
        >>> _ = ax.set_ylim(-0.5, 1.5)
        >>> _ = ax.set_aspect('equal')
        >>> _ = plt.title(f'Area: {area:.2f}')
        >>> plt.grid()
        >>> plt.savefig("AreaFunction.png", dpi=300, transparent=True)

    Notes:
        The area of a polygon in the hyperbolic plane is given by:

        $$
        A = \frac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right|
        $$

        > Reference: https://en.wikipedia.org/wiki/Hyperbolic_area

    Args:
        *vertices (UniversalArray): The coordinates of the vertices of the polygon in
            the hyperbolic plane.
    """

    def __init__(self, *vertices: UniversalArray) -> None:
        """Initialize the hyperbolic area function."""
        super().__init__(*vertices)

    @property
    def __eval__(self) -> float:
        """Calculate the area of a polygon in the hyperbolic plane.

        Returns:
            float: The area of the polygon in the hyperbolic plane.
        """
        n = len(self._x)
        area = 0.0
        for i in range(n):
            x1, y1 = self._x[i]
            x2, y2 = self._x[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return 0.5 * np.abs(area)
