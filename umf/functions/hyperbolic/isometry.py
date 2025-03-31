"""Hyperbolic isometry function for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.meta.functions import HyperbolicFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "HyperbolicIsometryFunction",
]


class HyperbolicIsometryFunction(HyperbolicFunction):
    r"""Apply an isometry transformation to a point in the hyperbolic plane.

    The hyperbolic isometry function applies isometries (transformations that preserve
    distances) in the hyperbolic plane.

    Examples:
        >>> from umf.functions.hyperbolic.isometry import HyperbolicIsometryFunction
        >>> point = np.array([1, 1], dtype=float)
        >>> matrix = np.array([[1, 1], [1, 1]], dtype=float)
        >>> hif = HyperbolicIsometryFunction(point, matrix)()
        >>> hif.result
        array([2., 2.])

        >>> # Visualization Example
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.hyperbolic.isometry import HyperbolicIsometryFunction
        >>> point = np.array([1, 1])
        >>> matrix = np.array([[1, 1], [1, 1]])
        >>> hif = HyperbolicIsometryFunction(point, matrix)()
        >>> transformed_point = hif.result
        >>> fig, ax = plt.subplots()
        >>> _  = ax.quiver(
        ...     0, 0, point[0], point[1], angles='xy', scale_units='xy', scale=1,
        ...     color='r', label='Original Point'
        ... )
        >>> _  = ax.quiver(
        ...     0, 0, transformed_point[0], transformed_point[1], angles='xy',
        ...     scale_units='xy', scale=1, color='b', label='Transformed Point'
        ... )

        >>> _  = ax.set_xlim(-1.5, 2.5)
        >>> _  = ax.set_ylim(-1.5, 2.5)
        >>> ax.set_aspect('equal')
        >>> _  = ax.legend()
        >>> _  = plt.title('Hyperbolic Isometry Transformation')
        >>> plt.grid()
        >>> plt.savefig("HyperbolicIsometryFunction.png", dpi=300, transparent=True)

    Notes:
        An isometry transformation in the hyperbolic plane is represented by a 2x2
        matrix. The transformation is applied to a point $(x, y)$ in the hyperbolic
        plane to obtain a new point $(x', y')$.

        > Reference: https://en.wikipedia.org/wiki/Isometry

    Args:
        *args (UniversalArray): The point (x, y) to be transformed and the 2x2 isometry
            matrix.
    """

    def __init__(self, *args: UniversalArray) -> None:
        """Initialize the hyperbolic isometry function."""
        super().__init__(*args)

    @property
    def __eval__(self) -> np.ndarray:
        """Apply an isometry transformation to a point in the hyperbolic plane.

        Returns:
            np.ndarray: The transformed point (x', y').
        """
        point = self._x[0].astype(np.float64)
        matrix = self._x[1].astype(np.float64)
        return np.dot(matrix, point)
