"""Hyperbolic angle function for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.meta.functions import HyperbolicFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "AngleFunction",
]


class AngleFunction(HyperbolicFunction):
    r"""Compute the angle between two vectors in the hyperbolic plane.

    The hyperbolic angle function computes the angle between two vectors in the
    hyperbolic plane.

    Examples:
        >>> from umf.functions.hyperbolic.angle import AngleFunction
        >>> vector1 = np.array([1, 0])
        >>> vector2 = np.array([0, 1])
        >>> haf = AngleFunction(vector1, vector2)()
        >>> haf.result
        array(1.57079633)

        >>> # Visualization Example
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.hyperbolic.angle import AngleFunction
        >>> vector1 = np.array([1, 0])
        >>> vector2 = np.array([0, 1])
        >>> haf = AngleFunction(vector1, vector2)()
        >>> angle = haf.result
        >>> fig, ax = plt.subplots()
        >>> _ = ax.quiver(
        ...     0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1,
        ...     color='r', label='Vector 1'
        ... )
        >>> _ = ax.quiver(
        ...     0, 0, vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1,
        ...     color='b', label='Vector 2'
        ... )
        >>> _ = ax.set_xlim(-1.5, 1.5)
        >>> _ = ax.set_ylim(-1.5, 1.5)
        >>> _ = ax.set_aspect('equal')
        >>> _ = ax.legend()
        >>> _ = plt.title(f'Angle: {angle:.2f} radians')
        >>> plt.grid()
        >>> plt.savefig("AngleFunction.png", dpi=300, transparent=True)

    Notes:
        The angle between two vectors $(x_1, y_1)$ and $(x_2, y_2)$ in the hyperbolic
        plane is given by:

        $$
        \theta = \cos^{-1}\left(\frac{x_1 x_2 + y_1 y_2}{\sqrt{x_1^2 + y_1^2}
        \sqrt{x_2^2 + y_2^2}}\right)
        $$

        > Reference: https://en.wikipedia.org/wiki/Hyperbolic_angle

    Args:
        *vectors (UniversalArray): The coordinates of the two vectors in the hyperbolic
            plane.
    """

    def __init__(self, *vectors: UniversalArray) -> None:
        """Initialize the hyperbolic angle function."""
        super().__init__(*vectors)

    @property
    def __eval__(self) -> float:
        """Compute the angle between two vectors in the hyperbolic plane.

        Returns:
            float: The angle between the two vectors in radians.
        """
        x1, y1 = self._x[0]
        x2, y2 = self._x[1]
        dot_product = x1 * x2 + y1 * y2
        norm1 = np.sqrt(x1**2 + y1**2)
        norm2 = np.sqrt(x2**2 + y2**2)
        return np.arccos(dot_product / (norm1 * norm2))
