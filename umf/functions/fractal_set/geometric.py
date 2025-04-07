"""Geometric fractals for the UMF package."""

from __future__ import annotations

import itertools

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.dimensions import __2d__
from umf.meta.functions import GeometricFractalFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__ = [
    "MengerSponge",
    "PythagorasTree",
    "SierpinskiCarpet",
    "SierpinskiTriangle",
    "UniformMassCenterTriangle",
]


class KochCurve(GeometricFractalFunction):
    r"""Implementation of the Koch snowflake curve.

    The Koch snowflake is built by repeatedly replacing each line segment with four
    segments that form an equilateral bump.

    Notes:
        The Koch curve has a fractal dimension of $\log 4 /\log(3) \approx 1.2619$,
        making in a curve with infinite length but enclosing a finite area. It was
        one of the earliest fractals described, introduced by Helge von Koch in 1904.

        At each iteration, each line segment is divided into four segments of equal
        length, creating an equilateral triangle bump in the middle third.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.geometric import KochCurve
        >>> # Generate Koch snowflake
        >>> points = np.array([[0, 0], [1, 0]])  # Initial line
        >>> koch = KochCurve(points, max_iter=5)()
        >>> curve = koch.result

        >>> # Visualization
        >>> _ = plt.figure(figsize=(10, 10))
        >>> _ = plt.plot(curve[:, 0], curve[:, 1], 'b-')
        >>> _ = plt.axis('equal')
        >>> _ = plt.title("Koch Snowflake")
        >>> plt.savefig("KochCurve.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Initial line segment points
        max_iter (int, optional): Number of iterations. Defaults to 5.
    """

    def __init__(self, *x: UniversalArray, max_iter: int = 5) -> None:
        """Initialize the Koch curve."""
        self.fractal_dimension = np.log(4) / np.log(3)  # Exact dimension
        super().__init__(*x, max_iter=max_iter)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply Koch curve transformation to line segments.

        Args:
            points (np.ndarray): Array of points defining line segments

        Returns:
            np.ndarray: New set of points after transformation
        """
        new_points = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            # Calculate segment points
            v = p2 - p1
            p3 = p1 + v / 3
            p5 = p1 + 2 * v / 3
            # Calculate peak point
            angle = np.pi / 3  # 60 degrees
            rot = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            p4 = p3 + np.dot(rot, (p5 - p3))

            new_points.extend([p1, p3, p4, p5])
        new_points.append(points[-1])
        return np.array(new_points)

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the Koch curve points.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        points = np.asarray(self._x).copy()

        for _ in range(self.max_iter):
            points = self.transform_points(points)

        return points


class SierpinskiTriangle(GeometricFractalFunction):
    r"""Implementation of the Sierpinski triangle fractal.

    The Sierpinski triangle is formed by repeatedly removing the central triangle
    from a triangular array.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.geometric import SierpinskiTriangle
        >>> # Generate Sierpinski triangle
        >>> points = (
        ...     np.array([0, 0]),
        ...     np.array([1, 0]),
        ...     np.array([0.5, np.sqrt(0.75)])
        ... )
        >>> sierpinski = SierpinskiTriangle(*points, max_iter=7)()
        >>> triangles = sierpinski.result
        >>>
        >>> # Visualization with gradient colors
        >>> fig = plt.figure(figsize=(10, 10))
        >>> # Create a custom colormap
        >>> colors = [(0.8, 0.0, 0.0), (0.5, 0.0, 0.5), (0.0, 0.0, 0.8)]
        >>> cm = LinearSegmentedColormap.from_list('triangle_colors', colors, N=256)
        >>> # Plot triangles with color based on size/level
        >>> for i, triangle in enumerate(triangles):
        ...     # Color based on triangle area (smaller triangles = later iterations)
        ...     area = 0.5 * np.abs(np.cross(
        ...         triangle[1] - triangle[0],
        ...         triangle[2] - triangle[0]
        ...     ))
        ...     # Normalize area for color mapping (log scale for better distribution)
        ...     norm_area = np.log(area + 1e-10) / np.log(1)
        ...     color = cm(max(0, min(1, 1 + norm_area)))
        ...     _ = plt.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=0.8)
        >>> _ = plt.axis('equal')
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> _ = plt.title("Sierpinski Triangle")
        >>> plt.savefig("SierpinskiTriangle.png", dpi=300, transparent=True)

    Notes:
        The Sierpinski triangle has a fractal dimension of:

        $$
        D = \frac{\log(3)}{\log(2)} \approx 1.585
        $$

        suggesting it has more complexity than a line (dimension 1) but less than
        a filled area (dimension 2). It was described by Wacław Sierpiński in 1915.

        The triangle is constructed recursively by removing the central triangle
        from each remaining sub-triangle. Each iteration produces three new
        triangles at half the scale of the original.

    Args:
        *x (UniversalArray): Initial triangle vertices
        max_iter (int, optional): Number of iterations. Defaults to 7.
    """

    def __init__(self, *x: UniversalArray, max_iter: int = 7) -> None:
        """Initialize the Sierpinski triangle."""
        self.fractal_dimension = np.log(3) / np.log(2)  # Exact dimension
        super().__init__(*x, max_iter=max_iter)

    def transform_points(self, points: list[np.ndarray]) -> list[np.ndarray]:
        """Subdivide triangles according to Sierpinski pattern.

        Args:
            points: List of triangle vertex arrays

        Returns:
            list[np.ndarray]: New set of triangles after subdivision
        """
        new_triangles = []
        for triangle in points:
            # Get midpoints
            midpoints = [(triangle[i] + triangle[(i + 1) % 3]) / 2 for i in range(3)]
            # Add three corner triangles
            new_triangles.extend(
                [
                    np.array([triangle[i], midpoints[i], midpoints[(i - 1) % 3]])
                    for i in range(3)
                ]
            )
        return new_triangles

    @property
    def __eval__(self) -> list[np.ndarray]:
        """Generate the Sierpinski triangle points.

        Returns:
            list[np.ndarray]: List of triangle vertex arrays
        """
        triangles = [np.array(self._x)]

        for _ in range(self.max_iter):
            triangles = self.transform_points(triangles)

        return triangles


class SierpinskiCarpet(GeometricFractalFunction):
    r"""Implementation of the Sierpinski carpet fractal.

    The Sierpinski carpet is created by repeatedly removing the central square
    from a grid of squares.

    Notes:
        The Sierpinski carpet has a fractal dimension of:

        $$
        D = \frac{\log(8)}{\log(3)} \approx 1.893
        $$

        approaching but not reaching dimension 2. It's a two-dimensional analog
        of the Cantor set, created by recursively removing the central ninth
        from each remaining square.

        Each iteration subdivides each square into 9 congruent sub-squares and
        removes the central one, resulting in 8 remaining squares per subdivision.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.geometric import SierpinskiCarpet
        >>> # Generate Sierpinski carpet
        >>> width = np.array([1.0])
        >>> height = np.array([1.0])
        >>> carpet = SierpinskiCarpet(width, height, max_iter=5)()
        >>> squares = carpet.result
        >>>
        >>> # Visualization with gradient colors
        >>> fig = plt.figure(figsize=(10, 10))
        >>> # Create a custom colormap
        >>> colors = [(0.0, 0.0, 0.6), (0.4, 0.0, 0.8), (0.8, 0.0, 0.8)]
        >>> cm = LinearSegmentedColormap.from_list('carpet_colors', colors, N=256)
        >>> # Plot squares with color based on size
        >>> for i, square in enumerate(squares):
        ...     # Calculate square size
        ...     size = np.abs(square[1][0] - square[0][0])
        ...     # Normalize size for color mapping (log scale)
        ...     norm_size = np.log(size + 1e-10) / np.log(1)
        ...     color = cm(max(0, min(1, 1 + norm_size)))
        ...     # Draw square with properly shaped corners
        ...     x = [square[0][0], square[1][0], square[1][0], square[0][0]]
        ...     y = [square[0][1], square[0][1], square[1][1], square[1][1]]
        ...     _ = plt.fill(x, y, color=color, alpha=0.8)
        >>> _ = plt.axis('equal')
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> _ = plt.title("Sierpinski Carpet")
        >>> plt.savefig("SierpinskiCarpet.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Size of the initial square [width, height]
        max_iter (int, optional): Number of iterations. Defaults to 5.
        fractal_dimension (float, optional): Fractal dimension. Defaults to
            $\log 8 \/ \log 3$.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 5,
        fractal_dimension: float = np.log(8) / np.log(3),
    ) -> None:
        """Initialize the Sierpinski carpet."""
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

    def transform_points(self, points: list[np.ndarray]) -> list[np.ndarray]:
        """Subdivide squares according to Sierpinski carpet pattern.

        Args:
            points: List of square vertex arrays

        Returns:
            list[np.ndarray]: New set of squares after subdivision
        """
        new_squares = []
        for square in points:
            size = np.abs(square[1] - square[0]) / 3
            # Add eight outer squares (skip center square)
            for i, j in itertools.product(range(3), range(3)):
                if i != 1 or j != 1:  # Skip center square
                    corner = square[0] + np.array([i, j]) * size
                    new_squares.append(np.array([corner, corner + size]))
        return new_squares

    @property
    def __eval__(self) -> list[np.ndarray]:
        """Generate the Sierpinski carpet points.

        Returns:
            list[np.ndarray]: List of square vertex arrays
        """
        # Initial square
        width = self._x[0][0]
        height = self._x[1][0]
        squares = [np.array([[0, 0], [width, height]])]

        for _ in range(self.max_iter):
            squares = self.transform_points(squares)

        return squares


class MengerSponge(GeometricFractalFunction):
    r"""Implementation of the Menger sponge fractal.

    The Menger sponge is a three-dimensional analog of the Sierpinski carpet.

    Notes:
        The Menger sponge has a fractal dimension of:

        $$
        D = \frac{\log(20)}{\log(3)} \approx 2.727
        $$

        making it a complex structure between a surface (dimension 2) and a
        volume (dimension 3). It was first described by Karl Menger in 1926.

        At each iteration, each cube is divided into 27 smaller cubes, and 7
        of these cubes are removed - the central cube and the six cubes sharing
        faces with the central cube. This leaves 20 cubes per subdivision.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from mpl_toolkits.mplot3d import Axes3D
        >>> from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.geometric import MengerSponge
        >>> # Generate Menger sponge
        >>> length, width, height = np.array([1.0]), np.array([1.0]), np.array([1.0])
        >>> sponge = MengerSponge(length, width, height, max_iter=2)()
        >>> cubes = sponge.result
        >>>
        >>> # Visualization with enhanced coloring
        >>> fig = plt.figure(figsize=(10, 10))
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> # Create custom colormap for better visualization
        >>> colors = [(0.1, 0.1, 0.5), (0.3, 0.2, 0.7), (0.8, 0.3, 0.6)]
        >>> cm = LinearSegmentedColormap.from_list('sponge_colors', colors, N=256)
        >>> # Draw each cube as a collection of faces
        >>> for i, cube in enumerate(cubes):
        ...     # Define the vertices of the cube
        ...     x0, y0, z0 = cube[0]
        ...     x1, y1, z1 = cube[1]
        ...     vertices = np.array([
        ...         [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        ...         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ...     ])
        ...     # Define the faces using indices into the vertices array
        ...     faces = [
        ...         [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        ...         [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        ...         [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        ...         [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        ...         [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        ...         [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
        ...     ]
        ...     # Choose color based on position and size
        ...     center = (cube[0] + cube[1]) / 2
        ...     size = np.linalg.norm(cube[1] - cube[0])
        ...     # Combine position and size for interesting color effects
        ...     color_val = (center[0] + center[1] + center[2])/3 + size/2
        ...     color = cm(min(1.0, max(0.0, color_val)))
        ...     # Add faces to plot with better styling
        ...     collection = Poly3DCollection(
        ...         faces,
        ...         alpha=0.8,
        ...         linewidths=0.2,
        ...         edgecolor='black'
        ...     )
        ...     collection.set_facecolor(color)
        ...     _ = ax.add_collection3d(collection)
        >>>
        >>> # Set equal aspect ratio and labels
        >>> _ = ax.set_box_aspect([1, 1, 1])
        >>> _ = ax.set_xlabel('X')
        >>> _ = ax.set_ylabel('Y')
        >>> _ = ax.set_zlabel('Z')
        >>> _ = ax.set_title("Menger Sponge")
        >>> # Set optimal viewing angle
        >>> _ = ax.view_init(elev=30, azim=45)
        >>> plt.savefig("MengerSponge.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Size of the initial cube [length, width, height]
        max_iter (int, optional): Number of iterations. Defaults to 3.
        fractal_dimension (float, optional): Fractal dimension. Defaults to
            $\log 20 \/ \log 3$.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 3,
        fractal_dimension: float = np.log(20) / np.log(3),
    ) -> None:
        """Initialize the Menger sponge."""
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

    def transform_points(self, points: list[np.ndarray]) -> list[np.ndarray]:
        """Subdivide cubes according to Menger sponge pattern.

        Args:
            points: List of cube vertex arrays

        Returns:
            list[np.ndarray]: New set of cubes after subdivision
        """
        new_cubes = []
        for cube in points:
            sub_size = np.abs(cube[1] - cube[0]) / 3
            # Check which subcubes to keep
            for i, j, k in itertools.product(range(3), range(3), range(3)):
                # Keep cube if at most one coordinate is in the middle
                # This removes center cube and "cross pieces"
                axes_in_middle = sum(coord == 1 for coord in (i, j, k))
                if axes_in_middle < __2d__:
                    corner = cube[0] + np.array([i, j, k]) * sub_size
                    new_cubes.append(np.array([corner, corner + sub_size]))
        return new_cubes

    @property
    def __eval__(self) -> list[np.ndarray]:
        """Generate the Menger sponge points.

        Returns:
            list[np.ndarray]: List of cube vertex arrays
        """
        # Extract dimensions from inputs
        length = self._x[0][0]
        width = self._x[1][0]
        height = self._x[2][0]

        # Initial cube
        cubes = [np.array([[0, 0, 0], [length, width, height]])]

        for _ in range(self.max_iter):
            cubes = self.transform_points(cubes)

        return cubes


class PythagorasTree(GeometricFractalFunction):
    r"""Implementation of the Pythagoras tree fractal.

    The Pythagoras tree is constructed by recursively adding squares and
    right triangles.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.geometric import PythagorasTree
        >>> # Generate Pythagoras tree
        >>> base_0, base_1 = np.array([0, 0]), np.array([1, 0])
        >>> tree = PythagorasTree(base_0, base_1, max_iter=10)()
        >>> squares = tree.result
        >>>
        >>> # Visualization with improved 3D-like effect
        >>> fig = plt.figure(figsize=(10, 10))
        >>> ax = plt.gca()
        >>> # Create custom colormap for tree-like appearance
        >>> colors = [(0.0, 0.4, 0.0), (0.2, 0.6, 0.0), (0.4, 0.8, 0.0)]
        >>> cm = LinearSegmentedColormap.from_list('tree_colors', colors, N=256)
        >>> # Get height range for normalization
        >>> y_coords = [np.mean(square[:, 1]) for square in squares]
        >>> y_min, y_max = min(y_coords), max(y_coords)
        >>> # Plot squares from back to front for proper occlusion
        >>> sorted_indices = np.argsort(y_coords)
        >>> for idx in sorted_indices:
        ...     square = squares[idx]
        ...     # Normalize height for color mapping
        ...     y_height = (
        ...         (y_coords[idx] - y_min) / (y_max - y_min)
        ...         if y_max > y_min else 0)
        ...     )
        ...     # Size-based variation for more natural appearance
        ...     size = np.linalg.norm(square[1] - square[0])
        ...     size_factor = np.clip(1.0 - np.log10(size + 1) * 0.2, 0.3, 1.0)
        ...     # Combine factors for final color
        ...     color_idx = min(0.99, max(0.0, y_height * 0.8 + size_factor * 0.2))
        ...     color = cm(color_idx)
        ...     # Add drop shadow for 3D effect
        ...     shadow = plt.Polygon(square - np.array([0.01, -0.01]),
        ...                         color='black', alpha=0.2)
        ...     _ = ax.add_patch(shadow)
        ...     # Draw square with depth-based edge thickness
        ...     _ = plt.fill(square[:, 0], square[:, 1], color=color,
        ...                 alpha=0.9, edgecolor='#004000',
        ...                 linewidth=0.8 * size_factor)
        >>> _ = plt.axis('equal')
        >>> _ = plt.title("Pythagoras Tree")
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> plt.tight_layout()
        >>> plt.savefig("PythagorasTree.png", dpi=300, transparent=True)

    Notes:
        The Pythagoras tree has an approximate fractal dimension of:

        $$
        D \approx 2.0
        $$

        It was introduced by Albert E. Bosman in 1942. The tree grows by placing
        two smaller squares at angles on top of each preceding square, similar
        to the geometric representation of the Pythagorean theorem.

    Args:
        *x (UniversalArray): Base line segment points
        max_iter (int, optional): Number of iterations. Defaults to 10.
        angle (float, optional): Angle of branches in radians. Defaults to np.pi/4.
        scale_factor (float, optional): Scaling factor for branches. Defaults to 0.7.
        fractal_dimension (float, optional): Fractal dimension. Defaults to 2.0.
    """

    # Define constants for clarity
    SQUARE_VERTICES = 4
    BRANCH_POINTS = 2

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 10,
        angle: float = np.pi / 4,
        scale_factor: float = 0.7,
        fractal_dimension: float = 2.0,
    ) -> None:
        """Initialize the Pythagoras tree."""
        self.angle = angle
        super().__init__(
            *x,
            max_iter=max_iter,
            scale_factor=scale_factor,
            fractal_dimension=fractal_dimension,
        )

    def transform_points(self, points: list[np.ndarray]) -> list[np.ndarray]:
        """Generate new squares for the Pythagoras tree.

        Args:
            points: List of branch endpoint arrays

        Returns:
            list[np.ndarray]: New set of squares and branches
        """
        new_branches = []
        for branch in points:
            # Create square from branch
            v = branch[1] - branch[0]
            perpendicular = np.array([-v[1], v[0]])
            square = np.array(
                [
                    branch[0],
                    branch[1],
                    branch[1] + perpendicular,
                    branch[0] + perpendicular,
                ]
            )
            new_branches.append(square)

            # Create two new branches
            rot1 = np.array(
                [
                    [np.cos(self.angle), -np.sin(self.angle)],
                    [np.sin(self.angle), np.cos(self.angle)],
                ]
            )
            rot2 = np.array(
                [
                    [np.cos(-self.angle), -np.sin(-self.angle)],
                    [np.sin(-self.angle), np.cos(-self.angle)],
                ]
            )

            v1 = np.dot(rot1, v) * self.scale_factor
            v2 = np.dot(rot2, v) * self.scale_factor

            new_branches.extend(
                [
                    np.array([branch[1], branch[1] + v1]),
                    np.array([branch[1], branch[1] + v2]),
                ]
            )

        return new_branches

    @property
    def __eval__(self) -> list[np.ndarray]:
        """Generate the Pythagoras tree points.

        Returns:
            list[np.ndarray]: List of square vertex arrays
        """
        # Ensure base branches are properly stored as numpy arrays
        base_branch = np.asarray(self._x)
        branches = [base_branch]
        squares = []

        for _ in range(self.max_iter):
            # Convert each branch to a numpy array to avoid type issues
            converted_branches = [np.asarray(b) for b in branches]
            new_branches = self.transform_points(converted_branches)
            squares.extend([b for b in new_branches if len(b) == self.SQUARE_VERTICES])
            branches = [b for b in new_branches if len(b) == self.BRANCH_POINTS]

        return squares


class UniformMassCenterTriangle(GeometricFractalFunction):
    r"""Implementation of a uniform mass center triangle fractal.

    This fractal is generated by repeatedly selecting random vertices of a triangle
    and moving towards them by a fixed ratio.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.geometric import UniformMassCenterTriangle
        >>> # Generate mass center triangle
        >>> vertices = (
        ...     np.array([0, 0]),
        ...     np.array([1, 0]),
        ...     np.array([0.5, np.sqrt(0.75)])
        ... )
        >>> triangle = UniformMassCenterTriangle(*vertices, max_iter=10000, ratio=0.5)()
        >>> points = triangle.result
        >>>
        >>> # Visualization with enhanced coloring and 3D-like effect
        >>> fig = plt.figure(figsize=(10, 10), facecolor='black')
        >>> ax = plt.gca()
        >>> ax.set_facecolor('black')
        >>> # Create custom colormap for glowing effect
        >>> colors = [
        ...     (0.0, 0.0, 0.3),
        ...     (0.0, 0.3, 0.7),
        ...     (0.5, 0.0, 0.8),
        ...     (0.8, 0.2, 0.0),
        ... ]
        >>> cm = LinearSegmentedColormap.from_list('glow_colors', colors, N=256)
        >>> # Create point clusters for efficiency
        >>> from scipy.stats import binned_statistic_2d
        >>> H, xedges, yedges, binnums = binned_statistic_2d(
        ...     points[:, 0], points[:, 1],
        ...     values=None, statistic='count', bins=200
        ... )
        >>> # Normalize and apply log scaling for better visualization
        >>> H_log = np.log1p(H)  # log(1+x) to avoid log(0)
        >>> H_norm = H_log / np.max(H_log)
        >>> # Plot as a heatmap with custom colormap
        >>> extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        >>> im = ax.imshow(
        ...     H_norm.T,  # Transpose for correct orientation
        ...     origin='lower',
        ...     extent=extent,
        ...     cmap=cm,
        ...     interpolation='gaussian',
        ...     aspect='auto'
        ... )
        >>> # Add triangle outline
        >>> vertices_array = np.array([v for v in vertices])
        >>> vertices_array = np.vstack([vertices_array, vertices_array[0]])
        >>> _ = plt.plot(vertices_array[:, 0], vertices_array[:, 1],
        ...              color='white', alpha=0.5, linewidth=1.0)
        >>> _ = plt.axis('equal')
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> _ = plt.title("Uniform Mass Center Triangle (Chaos Game)", color='white')
        >>> plt.tight_layout()
        >>> plt.savefig("UniformMassCenterTriangle.png", dpi=300, transparent=True)

    Notes:
        The uniform mass center triangle, also known as the Sierpinski gasket or
        chaos game, has a fractal dimension of approximately:

        $$
        D = \frac{\log(3)}{\log(2)} \approx 1.585
        $$

        It was popularized by Michael Barnsley in his 1988 book "Fractals Everywhere".

        This fractal is created through an iterative process where each new point
        is positioned partway between the previous point and a randomly chosen
        vertex of the triangle. The ratio parameter determines how far to move
        toward the selected vertex, with 0.5 producing the standard Sierpinski
        pattern.

    Args:
        *x (UniversalArray): Triangle vertices
        max_iter (int, optional): Number of points to generate. Defaults to 10000.
        ratio (float, optional): Movement ratio towards vertex. Defaults to 0.5.
        fractal_dimension (float, optional): Approximate fractal dimension. Defaults
            to $\log 3 \/ \log 2$.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 10000,
        ratio: float = 0.5,
        fractal_dimension: float = np.log(3) / np.log(2),
    ) -> None:
        """Initialize the mass center triangle."""
        self.ratio = ratio
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the mass center triangle points.

        Returns:
            np.ndarray: Array of points in the fractal
        """
        # Convert input to numpy array if it isn't already
        vertices = np.asarray(self._x)

        # Start at centroid - explicitly calculate to avoid type issues
        centroid = np.sum(vertices, axis=0) / len(vertices)
        points = [centroid]
        point = centroid

        # Create random number generator
        rng = np.random.default_rng()

        for _ in range(self.max_iter):
            # Choose random vertex
            vertex_idx = rng.integers(len(vertices))
            vertex = vertices[vertex_idx]

            # Move towards vertex
            point = point + self.ratio * (vertex - point)
            points.append(point)

        return np.array(points)
