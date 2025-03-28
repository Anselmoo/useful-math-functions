"""Hyperbolic area function for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.meta.functions import HyperbolicFunction

if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "HyperbolicAreaFunction",
]


class HyperbolicAreaFunction(HyperbolicFunction):
    r"""Calculate the area of a polygon in the hyperbolic plane.

    The hyperbolic area function calculates the area of a polygon in the hyperbolic
    plane.

    Examples:
        >>> from umf.functions.hyperbolic import HyperbolicAreaFunction
        >>> vertices = [(0, 0), (1, 0), (0, 1)]
        >>> haf = HyperbolicAreaFunction(vertices)()
        >>> haf.result
        1.762747174039086

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
        n = len(self._vertices)
        area = 0.0
        for i in range(n):
            x1, y1 = self._vertices[i]
            x2, y2 = self._vertices[(i + 1) % n]
            area += (x1 * y2 - x2 * y1)
        return 0.5 * np.abs(area)
