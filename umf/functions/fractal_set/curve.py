"""Curve-based fractals for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.meta.functions import CurveFractalFunction

if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__ = [
    "CantorSet",
    "DragonCurve",
    "HilbertCurve",
    "SpaceFillingCurve",
]


class CantorSet(CurveFractalFunction):
    r"""Implementation of the Cantor set fractal.

    The Cantor set is created by repeatedly removing the middle third of line segments.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.curve import CantorSet
        >>> # Generate Cantor set with 5 iterations
        >>> x = np.array([0., 1.])  # Initial interval
        >>> cantor = CantorSet(x, max_iter=5)()
        >>> intervals = cantor.result

        >>> # Visualization
        >>> fig, ax = plt.subplots(figsize=(10, 4))
        >>> for level, points in enumerate(intervals):
        ...     y = cantor.max_iter - level
        ...     for start, end in points:
        ...         _ = plt.plot([start, end], [y, y], 'k', linewidth=2)
        >>> _ = plt.ylim(-0.5, cantor.max_iter + 0.5)
        >>> _ = plt.title("Cantor Set")
        >>> plt.savefig("CantorSet.png", dpi=300, transparent=True)

    Notes:
        The Cantor set has dimension log(2)/log(3) ≈ 0.6309

    Args:
        x (UniversalArray): Initial interval [start, end]
        max_iter (int, optional): Maximum iterations. Defaults to 5.
    """

    def __init__(self, x: UniversalArray, max_iter: int = 5) -> None:
        """Initialize the Cantor set."""
        self.fractal_dimension = np.log(2) / np.log(3)  # Exact dimension
        super().__init__(x, max_iter=max_iter, scale_factor=1 / 3)

    @property
    def __eval__(self) -> list[list[tuple[float, float]]]:
        """Generate the Cantor set intervals.

        Returns:
            list[list[tuple[float, float]]]: List of intervals at each iteration
        """
        intervals = [[(self._x[0], self._x[1])]]

        for _ in range(self.max_iter):
            new_intervals = []
            for start, end in intervals[-1]:
                length = end - start
                third = length * self.scale_factor
                # Add left interval
                new_intervals.append((start, start + third))
                # Add right interval
                new_intervals.append((end - third, end))
            intervals.append(new_intervals)

        return intervals


class DragonCurve(CurveFractalFunction):
    r"""Implementation of the Dragon curve fractal.

    The Dragon curve is formed by repeatedly folding a strip of paper in half.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.curve import DragonCurve
        >>> # Generate Dragon curve with 10 iterations
        >>> start = np.array([0., 0.])
        >>> end = np.array([1., 0.])
        >>> dragon = DragonCurve(start, end, max_iter=10)()
        >>> points = dragon.result

        >>> # Visualization
        >>> _ = plt.figure(figsize=(10, 10))
        >>> _ = plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=1)
        >>> _ = plt.axis('equal')
        >>> _ = plt.title("Dragon Curve")
        >>> plt.savefig("DragonCurve.png", dpi=300, transparent=True)

    Args:
        start (UniversalArray): Starting point coordinates
        end (UniversalArray): Ending point coordinates
        max_iter (int, optional): Maximum iterations. Defaults to 10.
    """

    def __init__(
        self, start: UniversalArray, end: UniversalArray, max_iter: int = 10
    ) -> None:
        """Initialize the Dragon curve."""
        self.fractal_dimension = 2.0  # Approximate dimension
        super().__init__(start, end, max_iter=max_iter)

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the Dragon curve points.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        # Start with a single line segment
        points = [self._x[0], self._x[1]]

        # Iterate to create the dragon curve
        for _ in range(self.max_iter):
            new_points = []
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                # Find midpoint
                mid = (p1 + p2) / 2
                # Rotate 45 degrees clockwise or counterclockwise
                direction = 1 if i % 2 == 0 else -1
                angle = direction * np.pi / 4
                rot_matrix = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                # Add rotated point
                new_points.extend([p1, mid + np.dot(rot_matrix, (mid - p1))])
            new_points.append(points[-1])
            points = new_points

        return np.array(points)


class HilbertCurve(CurveFractalFunction):
    r"""Implementation of the Hilbert curve fractal.

    The Hilbert curve is a continuous space-filling curve that visits every point
    in a square grid.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.curve import HilbertCurve
        >>> # Generate Hilbert curve with order 5
        >>> size = 32  # Grid size (power of 2)
        >>> hilbert = HilbertCurve(np.array([size, size]), max_iter=5)()
        >>> points = hilbert.result

        >>> # Visualization
        >>> plt.figure(figsize=(10, 10))
        >>> plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=1)
        >>> plt.axis('equal')
        >>> plt.title("Hilbert Curve")
        >>> plt.savefig("HilbertCurve.png", dpi=300, transparent=True)

    Args:
        size (UniversalArray): Grid size [width, height]
        max_iter (int, optional): Order of the curve. Defaults to 5.
    """

    def __init__(self, size: UniversalArray, max_iter: int = 5) -> None:
        """Initialize the Hilbert curve."""
        self.fractal_dimension = 2.0  # Space-filling curve
        super().__init__(size, max_iter=max_iter)

    def _hilbert_to_xy(self, h: int, size: int) -> tuple[float, float]:
        """Convert Hilbert curve index to x,y coordinates.

        Args:
            h (int): Hilbert curve index
            size (int): Grid size

        Returns:
            tuple[float, float]: (x, y) coordinates
        """
        positions = [(0, 0), (0, 1), (1, 1), (1, 0)]
        temp = [(0, 0)]

        for i in range(size):
            new_temp = []
            for j in range(len(temp)):
                for k in range(4):
                    x = temp[j][0] * 2 + positions[k][0]
                    y = temp[j][1] * 2 + positions[k][1]
                    new_temp.append((x, y))
            temp = new_temp

        return temp[h]

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the Hilbert curve points.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        size = 2**self.max_iter
        points = []

        for i in range(size * size):
            x, y = self._hilbert_to_xy(i, self.max_iter)
            points.append([x * self._x[0] / size, y * self._x[1] / size])

        return np.array(points)


class SpaceFillingCurve(CurveFractalFunction):
    r"""Implementation of a generic space-filling curve.

    A space-filling curve is a continuous curve whose range contains the entire
    2-dimensional unit square.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.curve import SpaceFillingCurve
        >>> # Generate space-filling curve with 5 iterations
        >>> size = np.array([1., 1.])  # Unit square
        >>> curve = SpaceFillingCurve(size, max_iter=5)()
        >>> points = curve.result

        >>> # Visualization
        >>> plt.figure(figsize=(10, 10))
        >>> plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=1)
        >>> plt.axis('equal')
        >>> plt.title("Space-Filling Curve")
        >>> plt.savefig("SpaceFillingCurve.png", dpi=300, transparent=True)

    Args:
        size (UniversalArray): Size of the square [width, height]
        max_iter (int, optional): Number of iterations. Defaults to 5.
        curve_type (str, optional): Type of space-filling curve. Defaults to "peano".
    """

    def __init__(
        self, size: UniversalArray, max_iter: int = 5, curve_type: str = "peano"
    ) -> None:
        """Initialize the space-filling curve."""
        self.curve_type = curve_type
        self.fractal_dimension = 2.0  # All space-filling curves have dimension 2
        super().__init__(size, max_iter=max_iter)

    def _peano_helper(
        self, x: float, y: float, size: float, dir_x: int, dir_y: int
    ) -> list:
        """Helper function for generating Peano curve points.

        Args:
            x, y (float): Current position
            size (float): Current segment size
            dir_x, dir_y (int): Direction multipliers

        Returns:
            list: List of points
        """
        if self.max_iter == 0:
            return [[x, y]]

        points = []
        new_size = size / 3

        # Pattern for Peano curve
        pattern = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 1),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 2),
        ]

        for px, py in pattern:
            new_x = x + dir_x * px * new_size
            new_y = y + dir_y * py * new_size
            points.extend(self._peano_helper(new_x, new_y, new_size, dir_x, dir_y))

        return points

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the space-filling curve points.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        if self.curve_type == "peano":
            points = self._peano_helper(0, 0, self._x[0], 1, 1)
            return np.array(points)
        raise ValueError(f"Unknown curve type: {self.curve_type}")
