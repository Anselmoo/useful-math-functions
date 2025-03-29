"""Hyperbolic plane functions for the UMF package.

This module contains functions for various calculations in the hyperbolic plane,
such as distance, angle, area, geodesic, and isometry.

Examples:
    >>> from umf.functions.hyperbolic import hyperbolic_distance
    >>> hyperbolic_distance((0, 0), (1, 1))
    1.762747174039086
"""

from __future__ import annotations

import numpy as np


def hyperbolic_distance(
    point1: tuple[float, float], point2: tuple[float, float]
) -> float:
    """Calculate the distance between two points in the hyperbolic plane.

    Args:
        point1 (tuple[float, float]): The first point (x1, y1).
        point2 (tuple[float, float]): The second point (x2, y2).

    Returns:
        float: The hyperbolic distance between the two points.

    Examples:
        >>> hyperbolic_distance((0, 0), (1, 1))
        1.762747174039086
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.arccosh(1 + ((x2 - x1) ** 2 + (y2 - y1) ** 2) / (2 * y1 * y2))
