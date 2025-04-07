"""Curve-based fractals for the UMF package."""

from __future__ import annotations

import itertools

from typing import TYPE_CHECKING

import numpy as np

from umf.meta.functions import CurveFractalFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__ = [
    "CantorSet",
    "DragonCurve",
    "GosperCurve",
    "HilbertCurve",
    "KochCurve",
    "SpaceFillingCurve",
]


class CantorSet(CurveFractalFunction):
    r"""Implementation of the Cantor set fractal.

    The Cantor set is created by repeatedly removing the middle third of line segments.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.curve import CantorSet
        >>> # Generate Cantor set with 5 iterations
        >>> x0, x1 = np.array([0.]), np.array([1.])  # Initial interval
        >>> cantor = CantorSet(x0, x1, max_iter=15)()
        >>> intervals = cantor.result
        >>>
        >>> # Visualization with improved coloring
        >>> fig, ax = plt.subplots(figsize=(10, 4))
        >>> # Create a blue-purple color gradient
        >>> colors = [(0.2, 0.2, 0.8), (0.5, 0.0, 0.9), (0.8, 0.0, 0.7)]
        >>> cm = LinearSegmentedColormap.from_list('cantor_colors', colors, N=256)
        >>>
        >>> for level, points in enumerate(intervals):
        ...     y = cantor.parameters["max_iter"] - level
        ...     # Color based on iteration level
        ...     color = cm(level / cantor.parameters["max_iter"])
        ...     for start, end in points:
        ...         _ = plt.plot([start, end], [y, y],
        ...                      color=color,
        ...                      linewidth=(
        ...                         5 - 3 * level
        ...                          / cantor.parameters["max_iter"]
        ...                      ),
        ...                      solid_capstyle='butt')
        >>>
        >>> _ = plt.ylim(-0.5, cantor.parameters["max_iter"] + 0.5)
        >>> _ = plt.title("Cantor Set")
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> plt.savefig("CantorSet.png", dpi=300, transparent=True)

    Notes:
        The Cantor set has a fractal dimension of:

        $$
        D = \frac{\log(2)}{\log(3)} \approx 0.6309
        $$

        For the recursive construction of the Cantor set, we can define:

        $$
        C_0 = [0,1]
        $$

        $$
        C_{n+1} = \frac{1}{3}C_n \cup \left(\frac{2}{3} + \frac{1}{3}C_n\right)
        $$

        where each iteration removes the middle third of all remaining segments.
        The Cantor set is the limit:

        $$
        C = \bigcap_{n=0}^{\infty} C_n
        $$

        > Reference: Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.

    Args:
        *x (UniversalArray): Initial interval [start, end]
        max_iter (int, optional): Maximum iterations. Defaults to 5.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 5,
        scale_factor: float = 1 / 3,
        fractal_dimension: float = np.log(2) / np.log(3),
    ) -> None:
        """Initialize the Cantor set."""
        super().__init__(
            *x,
            max_iter=max_iter,
            scale_factor=scale_factor,
            fractal_dimension=fractal_dimension,
        )

    @property
    def __eval__(self) -> list[list[tuple[UniversalArray, UniversalArray]]]:
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
                new_intervals.extend(((start, start + third), (end - third, end)))
            intervals.append(new_intervals)

        return intervals


class DragonCurve(CurveFractalFunction):
    r"""Implementation of the Dragon curve fractal.

    The Dragon curve is formed by repeatedly folding a strip of paper in half.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.curve import DragonCurve
        >>> # Generate Dragon curve with 10 iterations
        >>> start = np.array([0., 0.])
        >>> end = np.array([1., 0.])
        >>> dragon = DragonCurve(start, end, max_iter=10)()
        >>> points = dragon.result
        >>>
        >>> # Visualization with improved coloring
        >>> fig = plt.figure(figsize=(10, 10))
        >>> # Create a colormap from red to blue-ish
        >>> colors = [(0.8, 0.0, 0.0), (0.5, 0.0, 0.8), (0.0, 0.5, 0.8)]
        >>> cm = LinearSegmentedColormap.from_list('dragon_colors', colors, N=256)
        >>>
        >>> # Color segments based on their position in the sequence
        >>> for i in range(len(points) - 1):
        ...     x = [points[i][0], points[i+1][0]]
        ...     y = [points[i][1], points[i+1][1]]
        ...     # Normalize position for color mapping
        ...     position = i / (len(points) - 2)
        ...     color = cm(position)
        ...     _ = plt.plot(x, y, color=color, linewidth=1.5)
        >>>
        >>> _ = plt.axis('equal')
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> _ = plt.title("Dragon Curve")
        >>> plt.tight_layout()
        >>> plt.savefig("DragonCurve.png", dpi=300, transparent=True)

    Notes:
        The Dragon curve has an approximate fractal dimension of 2.0 and follows
        a recursive construction pattern. It can be defined using an L-system with:

        $$
        \text{Variables: } X, Y
        $$

        $$
        \text{Constants: } F, +, -
        $$

        $$
        \text{Start: } F X
        $$

        $$
        \text{Rules: } X \rightarrow X + Y F +, \quad Y \rightarrow - F X - Y
        $$

        Where $F$ means "draw forward," $+$ means "turn right 90°," and $-$ means
        "turn left 90°". The resulting curve is self-avoiding and fills a portion
        of the plane.

        Each fold in the Dragon curve can be represented by the transformation:

        $$
        p_{\text{new}} = p_{\text{mid}} + R_{\theta} \cdot (p_{\text{mid}} - p_1)
        $$

        where $p_{\text{mid}}$ is the midpoint between consecutive points, and
        $R_{\theta}$ is a rotation matrix for angle $\theta = \pm 45°$.

        > Reference: Davis, C., & Knuth, D. E. (1970). Number Representations and

    Args:
        *x (UniversalArray): Coordinates [start_point, end_point]
        max_iter (int, optional): Maximum iterations. Defaults to 10.
    """

    def __init__(
        self, *x: UniversalArray, max_iter: int = 10, fractal_dimension: float = 2.0
    ) -> None:
        """Initialize the Dragon curve."""
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

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
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.curve import HilbertCurve
        >>> # Generate Hilbert curve with order 5
        >>> size = np.array([32])  # Grid size (power of 2)
        >>> hilbert = HilbertCurve(size, size, max_iter=5)()
        >>> points = hilbert.result
        >>>
        >>> # Visualization with improved coloring
        >>> fig = plt.figure(figsize=(10, 10))
        >>> # Create custom colormap
        >>> colors = [
        ...     (0.0, 0.5, 0.0),
        ...     (0.3, 0.7, 0.0),
        ...     (0.9, 0.9, 0.0),
        ...     (1.0, 0.5, 0.0),
        ... ]
        >>> cm = LinearSegmentedColormap.from_list('hilbert_colors', colors, N=256)
        >>>
        >>> # Color segments based on their position
        >>> for i in range(len(points) - 1):
        ...     x = [points[i][0], points[i+1][0]]
        ...     y = [points[i][1], points[i+1][1]]
        ...     # Normalize position for color mapping
        ...     position = i / (len(points) - 2)
        ...     color = cm(position)
        ...     _ = plt.plot(x, y, color=color, linewidth=1.5)
        >>>
        >>> _ = plt.axis('equal')
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> _ = plt.title("Hilbert Curve")
        >>> plt.tight_layout()
        >>> plt.savefig("HilbertCurve.png", dpi=300, transparent=True)

    Notes:
        The Hilbert curve has a fractal dimension of exactly 2 and completely fills
        the 2D space. Mathematically, the nth-order Hilbert curve can be constructed
        recursively using the following transformations:

        The mapping from 1D Hilbert curve index $h$ to 2D coordinates $(x,y)$ can be
        expressed through a recursive algorithm:

        $$
        H_1 = \begin{pmatrix} 0,0 & 0,1 & 1,1 & 1,0 \end{pmatrix}
        $$

        For higher orders $n > 1$:

        $$
        H_n = \begin{pmatrix}
        H_{n-1}^{\text{rot270}} \\
        H_{n-1} + (0,2^{n-1}) \\
        H_{n-1} + (2^{n-1},2^{n-1}) \\
        H_{n-1}^{\text{rot90}} + (2^{n-1}-1,2^{n-1}-1)
        \end{pmatrix}
        $$

        where $H_{n-1}^{\text{rot90}}$ and $H_{n-1}^{\text{rot270}}$ represent the
        $(n-1)$-order Hilbert curve rotated by 90° and 270° respectively.

        The Hilbert curve maintains locality: points close on the 1D curve are
        generally close in the 2D mapping, making it useful for dimensionality
        reduction.

        > Reference: Hilbert, D. (1891). Über die stetige Abbildung einer Linie auf ein
        > Flächenstück. Mathematische Annalen, 38(3), 459-460.

    Args:
        *x (UniversalArray): Grid size [width, height]
        max_iter (int, optional): Order of the curve. Defaults to 5.
    """

    def __init__(
        self, *x: UniversalArray, max_iter: int = 5, fractal_dimension: float = 2.0
    ) -> None:
        """Initialize the Hilbert curve."""
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

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

        for _ in range(size):
            new_temp = []
            for j, k in itertools.product(range(len(temp)), range(4)):
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
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.curve import SpaceFillingCurve
        >>> # Generate space-filling curve with 5 iterations
        >>> height, width = np.array([32]), np.array([32])  # Grid size (power of 2)
        >>> curve = SpaceFillingCurve(height, width, max_iter=5)()
        >>> points = curve.result
        >>>
        >>> # Visualization with enhanced coloring
        >>> fig = plt.figure(figsize=(10, 10))
        >>>
        >>> # Create custom colormap for better visualization
        >>> colors = [(0.0, 0.2, 0.6), (0.2, 0.6, 1.0), (0.8, 0.9, 1.0)]
        >>> cm = LinearSegmentedColormap.from_list('curve_colors', colors, N=256)
        >>>
        >>> # Color points based on their position in the sequence
        >>> points_count = len(points)
        >>> colors = cm(np.linspace(0, 1, points_count-1))
        >>>
        >>> for i in range(points_count-1):
        ...     x = [points[i, 0], points[i+1, 0]]
        ...     y = [points[i, 1], points[i+1, 1]]
        ...     _ = plt.plot(x, y, color=colors[i], linewidth=1)
        >>>
        >>> _ = plt.axis('equal')
        >>> _ = plt.title("Space-Filling Curve")
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> plt.tight_layout()
        >>> plt.savefig("SpaceFillingCurve.png", dpi=300, transparent=True)

    Notes:
        Space-filling curves create a continuous mapping from a 1-dimensional space
        to an n-dimensional space. For the Peano curve (implemented here), the
        construction follows a recursive pattern that divides the square into 9
        equal sub-squares and connects their centers in a specific pattern.

        Mathematically, a space-filling curve is a surjective continuous function:

        $$
        f: [0,1] \rightarrow [0,1]^n
        $$

        The Peano curve specifically has the following recursive construction pattern:

        $$
        P_1(t) = \begin{cases}
        (3t, 0) & \text{if } 0 \leq t < \frac{1}{9} \\
        (1, 3t - \frac{1}{3}) & \text{if } \frac{1}{9} \leq t < \frac{2}{9} \\
        (2, 3t - \frac{2}{3}) & \text{if } \frac{2}{9} \leq t < \frac{1}{3} \\
        \ldots \\
        (3t - \frac{8}{3}, 2) & \text{if } \frac{8}{9} \leq t \leq 1
        \end{cases}
        $$

        For higher-order curves, each segment is replaced with a scaled, potentially
        rotated, version of the base pattern, creating a self-similar structure that
        progressively fills the square more completely.

        > Reference: https://en.wikipedia.org/wiki/Space-filling_curve

    Args:
        *x (UniversalArray): Size of the square [width, height]
        max_iter (int, optional): Number of iterations. Defaults to 5.
        curve_type (str, optional): Type of space-filling curve. Defaults to "peano".
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 5,
        curve_type: str = "peano",
        fractal_dimension: float = 2.0,
    ) -> None:
        """Initialize the space-filling curve."""
        self.curve_type = curve_type
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

    def _peano_helper(  # noqa: PLR0913
        self, x: float, y: float, size: float, dir_x: int, dir_y: int, depth: int
    ) -> list:
        """Helper function for generating Peano curve points.

        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            size (float): Size of the current segment
            dir_x (int): Direction in x-axis
            dir_y (int): Direction in y-axis
            depth (int): Current recursion depth

        Returns:
            list: List of points
        """
        if depth >= self.max_iter:
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
            points.extend(
                self._peano_helper(new_x, new_y, new_size, dir_x, dir_y, depth + 1)
            )

        return points

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the space-filling curve points.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        if self.curve_type == "peano":
            # Extract the float value from the NumPy array
            size = float(self._x[0][0])
            points = self._peano_helper(0, 0, size, 1, 1, 0)
            return np.array(points)
        msg = f"Unknown curve type: {self.curve_type}"
        raise ValueError(msg)


class GosperCurve(CurveFractalFunction):
    r"""Implementation of the Gosper curve fractal.

    The Gosper curve, also known as the flowsnake, is a space-filling curve
    that follows a hexagonal pattern.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.curve import GosperCurve
        >>> # Generate Gosper curve
        >>> start = np.array([0., 0.])
        >>> end = np.array([1., 1.])
        >>> gosper = GosperCurve(start, end, max_iter=5)()
        >>> points = gosper.result
        >>>
        >>> # Visualization with enhanced coloring
        >>> fig = plt.figure(figsize=(10, 10))
        >>> # Create custom colormap
        >>> colors = [(0.2, 0.0, 0.5), (0.5, 0.0, 0.8), (0.8, 0.5, 1.0)]
        >>> cm = LinearSegmentedColormap.from_list('gosper_colors', colors, N=256)
        >>>
        >>> # Color segments based on their position
        >>> for i in range(len(points) - 1):
        ...     x = [points[i][0], points[i+1][0]]
        ...     y = [points[i][1], points[i+1][1]]
        ...     # Normalize position for color mapping
        ...     position = i / (len(points) - 2)
        ...     color = cm(position)
        ...     _ = plt.plot(x, y, color=color, linewidth=1.5)
        >>>
        >>> _ = plt.axis('equal')
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> _ = plt.title("Gosper Curve")
        >>> plt.tight_layout()
        >>> plt.savefig("GosperCurve.png", dpi=300, transparent=True)

    Notes:
        The Gosper curve has a fractal dimension of 2.0 and can be constructed
        using an L-system approach. It is defined with the following rules:

        $$
        \text{Variables: } A, B
        $$

        $$
        \text{Start: } A
        $$

        $$
        \text{Rules: } A \rightarrow A-B--B+A++AA+B-,
        $$

        $$
        B \rightarrow +A-BB--B-A++A+B
        $$

        Where $+$ means "turn left 60°" and $-$ means "turn right 60°".

        The Gosper curve can also be constructed by repeatedly replacing each
        straight line segment with seven segments arranged in a pattern that
        resembles the letter "Z" with an extra segment on one side. The curve
        follows paths along a hexagonal grid.

        The resulting curve is self-similar and has the property that it
        completely fills a plane without crossing itself.

        > Reference: Gosper, B. (1976). Exploiting regularities in large cellular
        > spaces. Physica D: Nonlinear Phenomena, 10(1-2), 75-80.

    Args:
        *x (UniversalArray): Coordinates [start_point, end_point]
        max_iter (int, optional): Number of iterations. Defaults to 4.
    """

    def __init__(
        self, *x: UniversalArray, max_iter: int = 4, fractal_dimension: float = 2.0
    ) -> None:
        """Initialize the Gosper curve."""
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

    def _apply_l_system(self, commands: str) -> list[tuple[float, float]]:
        """Apply L-system rules to generate the Gosper curve.

        Args:
            commands (str): String of L-system commands

        Returns:
            list[tuple[float, float]]: List of points defining the curve
        """
        # Starting position and direction
        x, y = 0.0, 0.0
        angle = 0.0
        points = [(x, y)]

        # Step size
        step = 1.0

        # Interpret L-system commands
        for cmd in commands:
            if cmd in {"A", "B"}:  # Using set membership for better performance
                # Move forward
                x += step * np.cos(angle)
                y += step * np.sin(angle)
                points.append((x, y))
            elif cmd == "+":
                # Turn left 60 degrees
                angle += np.pi / 3
            elif cmd == "-":
                # Turn right 60 degrees
                angle -= np.pi / 3

        return points

    def _expand_l_system(self, axiom: str, iterations: int) -> str:
        """Expand L-system rules for Gosper curve.

        Args:
            axiom (str): Starting axiom
            iterations (int): Number of iterations

        Returns:
            str: Expanded L-system command string
        """
        # L-system rules for Gosper curve (corrected according to Wikipedia)
        rules = {"A": "A-B--B+A++AA+B-", "B": "+A-BB--B-A++A+B"}

        current = axiom
        for _ in range(iterations):
            # Using string join for better performance
            next_gen = "".join(rules.get(char, char) for char in current)
            current = next_gen

        return current

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the Gosper curve points.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        # Generate L-system commands
        commands = self._expand_l_system("A", self.max_iter)

        # Apply L-system rules to generate points
        raw_points = self._apply_l_system(commands)

        # Normalize and scale points to fit desired dimensions
        points = np.array(raw_points)
        if len(points) > 1:
            # Calculate scaling factors
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            x_scale = (
                np.abs(self._x[1][0] - self._x[0][0]) / (x_max - x_min)
                if x_max > x_min
                else 1.0
            )
            y_scale = (
                np.abs(self._x[1][1] - self._x[0][1]) / (y_max - y_min)
                if y_max > y_min
                else 1.0
            )

            # Apply scaling and translation
            points = (points - np.array([x_min, y_min])) * np.array(
                [x_scale, y_scale]
            ) + self._x[0]

        return points


class KochCurve(CurveFractalFunction):
    r"""Implementation of the Koch curve fractal.

    The Koch curve is created by repeatedly replacing each line segment with four
    segments that form an equilateral triangle.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.curve import KochCurve
        >>> # Generate Koch curve
        >>> start = np.array([0., 0.])
        >>> end = np.array([1., 0.])
        >>> koch = KochCurve(start, end, max_iter=5)()
        >>> points = koch.result
        >>>
        >>> # Visualization with enhanced coloring
        >>> fig = plt.figure(figsize=(10, 5))
        >>> # Create a blue-to-purple color gradient
        >>> colors = [(0.0, 0.4, 0.8), (0.4, 0.2, 0.8), (0.8, 0.0, 0.7)]
        >>> cm = LinearSegmentedColormap.from_list('koch_colors', colors, N=256)
        >>>
        >>> # Color segments based on their position
        >>> for i in range(len(points) - 1):
        ...     x = [points[i][0], points[i+1][0]]
        ...     y = [points[i][1], points[i+1][1]]
        ...     # Normalize position for color mapping
        ...     position = i / (len(points) - 2)
        ...     color = cm(position)
        ...     _ = plt.plot(x, y, color=color, linewidth=1.3)
        >>>
        >>> _ = plt.axis('equal')
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> _ = plt.title("Koch Curve")
        >>> plt.tight_layout()
        >>> plt.savefig("KochCurve.png", dpi=300, transparent=True)

    Notes:
        The Koch curve has a fractal dimension of:

        $$
        D = \frac{\log(4)}{\log(3)} \approx 1.2619
        $$

        The construction of the Koch curve follows these steps for each iteration:

        1. Divide each line segment into three equal parts.
        2. Replace the middle part with two sides of an equilateral triangle.

        Mathematically, this can be expressed as a geometric transformation:

        $$
        K_{n+1} = T_1(K_n) \cup T_2(K_n) \cup T_3(K_n) \cup T_4(K_n)
        $$

        where $T_i$ are affine transformations that scale by $1/3$ and rotate/translate
        to create the characteristic "bump" in the curve.

        The Koch curve is everywhere continuous but nowhere differentiable, making it
        one of the earliest examples of a fractal with these properties.


    Args:
        *x (UniversalArray): Coordinates [start_point, end_point]
        max_iter (int, optional): Number of iterations. Defaults to 5.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 5,
        fractal_dimension: float = np.log(4) / np.log(3),
    ) -> None:
        """Initialize the Koch curve."""
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the Koch curve points.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        # Start with a single line segment
        points = [self._x[0], self._x[1]]

        # Create the Koch curve through iterations
        for _ in range(self.max_iter):
            new_points = []
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]

                # Calculate the four points that replace the current segment
                vector = p2 - p1
                third = vector / 3

                # First point
                new_points.append(p1)

                # Second point (1/3 of the way)
                p_1_3 = p1 + third
                new_points.append(p_1_3)

                # Middle point (equilateral triangle peak)
                # Rotate the vector by 60 degrees to create the peak
                angle = np.pi / 3  # 60 degrees
                rot_matrix = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                peak = p_1_3 + np.dot(rot_matrix, third)
                new_points.append(peak)

                # Fourth point (2/3 of the way)
                p_2_3 = p1 + 2 * third
                new_points.append(p_2_3)

                # We don't add p2 here as it will be the first point of the next segment
                # Except for the last segment
                if i == len(points) - 2:
                    new_points.append(p2)

            points = new_points

        return np.array(points)
