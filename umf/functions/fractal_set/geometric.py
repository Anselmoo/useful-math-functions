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
    "KochCurve",
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

    Notes:
        The Sierpinski triangle has a fractal dimension of log(3)/log(2) ≈ 1.585,
        suggesting it has more complexity than a line (dimension 1) but less than
        a filled area (dimension 2). It was described by Wacław Sierpiński in 1915.

        The triangle is constructed recursively by removing the central triangle
        from each remaining sub-triangle. Each iteration produces three new
        triangles at half the scale of the original.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.geometric import SierpinskiTriangle
        >>> # Generate Sierpinski triangle
        >>> points = np.array([[0, 0], [1, 0], [0.5, np.sqrt(0.75)]])
        >>> sierpinski = SierpinskiTriangle(points, max_iter=7)()
        >>> triangles = sierpinski.result

        >>> # Visualization
        >>> _ = plt.figure(figsize=(10, 10))
        >>> for triangle in triangles:
        ...     _ = plt.fill(triangle[:, 0], triangle[:, 1], 'b', alpha=0.1)
        >>> _ = plt.axis('equal')
        >>> _ = plt.title("Sierpinski Triangle")
        >>> plt.savefig("SierpinskiTriangle.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Initial triangle vertices
        max_iter (int, optional): Number of iterations. Defaults to 7.
    """

    def __init__(self, *x: UniversalArray, max_iter: int = 7) -> None:
        """Initialize the Sierpinski triangle."""
        self.fractal_dimension = np.log(3) / np.log(2)  # Exact dimension
        super().__init__(*x, max_iter=max_iter)

    def transform_points(self, triangles: list[np.ndarray]) -> list[np.ndarray]:
        """Subdivide triangles according to Sierpinski pattern.

        Args:
            triangles (list[np.ndarray]): List of triangle vertex arrays

        Returns:
            list[np.ndarray]: New set of triangles after subdivision
        """
        new_triangles = []
        for triangle in triangles:
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
        The Sierpinski carpet has a fractal dimension of
        $\log(8)/\log(3) \approx 1.893$, approaching but not reaching dimension 2.
        It's a two-dimensional analog of the Cantor set, created by recursively
        removing the central ninth from each remaining square.

        Each iteration subdivides each square into 9 congruent sub-squares and
        removes the central one, resulting in 8 remaining squares per subdivision.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.geometric import SierpinskiCarpet
        >>> # Generate Sierpinski carpet
        >>> width = np.array([1.0])
        >>> height = np.array([1.0])
        >>> carpet = SierpinskiCarpet(width, height, max_iter=5)()
        >>> squares = carpet.result

        >>> # Visualization
        >>> _ = plt.figure(figsize=(10, 10))
        >>> for square in squares:
        ...     _ = plt.fill(square[:, 0], square[:, 1], 'b', alpha=0.3)
        >>> _ = plt.axis('equal')
        >>> _ = plt.title("Sierpinski Carpet")
        >>> plt.savefig("SierpinskiCarpet.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Size of the initial square [width, height]
        max_iter (int, optional): Number of iterations. Defaults to 5.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 5,
        fractal_dimension: float = np.log(8) / np.log(3),
    ) -> None:
        """Initialize the Sierpinski carpet."""
        super().__init__(*x, max_iter=max_iter, fractal_dimension=fractal_dimension)

    def transform_points(self, squares: list[np.ndarray]) -> list[np.ndarray]:
        """Subdivide squares according to Sierpinski carpet pattern.

        Args:
            squares (list[np.ndarray]): List of square vertex arrays

        Returns:
            list[np.ndarray]: New set of squares after subdivision
        """
        new_squares = []
        for square in squares:
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
        width = self._x[0] if isinstance(self._x[0], (int, float)) else self._x[0][0]
        height = self._x[1] if isinstance(self._x[1], (int, float)) else self._x[1][0]
        squares = [np.array([[0, 0], [width, height]])]

        for _ in range(self.max_iter):
            squares = self.transform_points(squares)

        return squares


class MengerSponge(GeometricFractalFunction):
    r"""Implementation of the Menger sponge fractal.

    The Menger sponge is a three-dimensional analog of the Sierpinski carpet.

    Notes:
        The Menger sponge has a fractal dimension of log(20)/log(3) ≈ 2.727,
        making it a complex structure between a surface (dimension 2) and a
        volume (dimension 3). It was first described by Karl Menger in 1926.

        At each iteration, each cube is divided into 27 smaller cubes, and 7
        of these cubes are removed - the central cube and the six cubes sharing
        faces with the central cube. This leaves 20 cubes per subdivision.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from mpl_toolkits.mplot3d import Axes3D
        >>> from umf.functions.fractal_set.geometric import MengerSponge
        >>> # Generate Menger sponge
        >>> size = np.array([1.0, 1.0, 1.0])
        >>> sponge = MengerSponge(size, max_iter=3)()
        >>> cubes = sponge.result

        >>> # Visualization
        >>> fig = plt.figure(figsize=(10, 10))
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> for cube in cubes:
        ...     # Plot cube vertices
        ...     for i in range(2):
        ...         for j in range(2):
        ...             _ = ax.plot3D([cube[0,0], cube[1,0]],
        ...                      [cube[0,1], cube[1,1]],
        ...                      [i, i], 'b-', alpha=0.1)
        >>> _ = plt.title("Menger Sponge")
        >>> plt.savefig("MengerSponge.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Size of the initial cube [width, height, depth]
        max_iter (int, optional): Number of iterations. Defaults to 3.
    """

    def __init__(self, *x: UniversalArray, max_iter: int = 3) -> None:
        """Initialize the Menger sponge."""
        self.fractal_dimension = np.log(20) / np.log(3)  # Exact dimension
        super().__init__(*x, max_iter=max_iter)

    def transform_points(self, cubes: list[np.ndarray]) -> list[np.ndarray]:
        """Subdivide cubes according to Menger sponge pattern.

        Args:
            cubes (list[np.ndarray]): List of cube vertex arrays

        Returns:
            list[np.ndarray]: New set of cubes after subdivision
        """
        new_cubes = []
        for cube in cubes:
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
        # Initial cube
        cubes = [np.array([[0, 0, 0], self._x])]

        for _ in range(self.max_iter):
            cubes = self.transform_points(cubes)

        return cubes


class PythagorasTree(GeometricFractalFunction):
    r"""Implementation of the Pythagoras tree fractal.

    The Pythagoras tree is constructed by recursively adding squares and
    right triangles.

    Notes:
        The Pythagoras tree has an approximate fractal dimension of 2.0 and is
        constructed by recursively adding squares on the sides of triangles.
        It was introduced by Albert E. Bosman in 1942.

        The tree grows by placing two smaller squares at angles on top of each
        preceding square, similar to the geometric representation of the
        Pythagorean theorem. The scaling factor and angles determine the
        shape and density of the resulting tree.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.geometric import PythagorasTree
        >>> # Generate Pythagoras tree
        >>> base = np.array([[0, 0], [1, 0]])
        >>> tree = PythagorasTree(base, max_iter=10)()
        >>> squares = tree.result

        >>> # Visualization
        >>> plt.figure(figsize=(10, 10))
        >>> for square in squares:
        ...     plt.fill(square[:, 0], square[:, 1], 'g', alpha=0.1)
        >>> plt.axis('equal')
        >>> plt.title("Pythagoras Tree")
        >>> plt.savefig("PythagorasTree.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Base line segment points
        max_iter (int, optional): Number of iterations. Defaults to 10.
        angle (float, optional): Angle of branches in radians. Defaults to np.pi/4.
        scale_factor (float, optional): Scaling factor for branches. Defaults to 0.7.
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
    ) -> None:
        """Initialize the Pythagoras tree."""
        self.angle = angle
        self.scale_factor = scale_factor
        self.fractal_dimension = 2.0  # Approximate dimension
        super().__init__(*x, max_iter=max_iter)

    def transform_points(self, branches: list[np.ndarray]) -> list[np.ndarray]:
        """Generate new squares for the Pythagoras tree.

        Args:
            branches (list[np.ndarray]): List of branch endpoint arrays

        Returns:
            list[np.ndarray]: New set of squares
        """
        new_branches = []
        for branch in branches:
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
        branches = [self._x]
        squares = []

        for _ in range(self.max_iter):
            new_branches = self.transform_points(branches)
            squares.extend([b for b in new_branches if len(b) == self.SQUARE_VERTICES])
            branches = [b for b in new_branches if len(b) == self.BRANCH_POINTS]

        return squares


class UniformMassCenterTriangle(GeometricFractalFunction):
    r"""Implementation of a uniform mass center triangle fractal.

    This fractal is generated by repeatedly selecting random vertices of a triangle
    and moving towards them by a fixed ratio.

    Notes:
        The uniform mass center triangle, also known as the Sierpinski gasket or
        chaos game, has a fractal dimension of approximately 1.585. It was
        popularized by Michael Barnsley in his 1988 book "Fractals Everywhere".

        This fractal is created through an iterative process where each new point
        is positioned partway between the previous point and a randomly chosen
        vertex of the triangle. The ratio parameter determines how far to move
        toward the selected vertex, with 0.5 producing the standard Sierpinski
        pattern.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.geometric import UniformMassCenterTriangle
        >>> # Generate mass center triangle
        >>> vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(0.75)]])
        >>> triangle = UniformMassCenterTriangle(vertices, max_iter=10000, ratio=0.5)()
        >>> points = triangle.result

        >>> # Visualization
        >>> plt.figure(figsize=(10, 10))
        >>> plt.plot(points[:, 0], points[:, 1], 'k.', markersize=1)
        >>> plt.axis('equal')
        >>> plt.title("Uniform Mass Center Triangle")
        >>> plt.savefig("UniformMassCenterTriangle.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Triangle vertices
        max_iter (int, optional): Number of points to generate. Defaults to 10000.
        ratio (float, optional): Movement ratio towards vertex. Defaults to 0.5.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 10000,
        ratio: float = 0.5,
    ) -> None:
        """Initialize the mass center triangle."""
        self.ratio = ratio
        self.fractal_dimension = 1.585  # Approximate dimension
        super().__init__(*x, max_iter=max_iter)

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the mass center triangle points.

        Returns:
            np.ndarray: Array of points in the fractal
        """
        # Convert input to numpy array if it isn't already
        vertices = np.asarray(self._x)

        # Start at centroid
        points = [np.mean(vertices, axis=0)]
        point = points[0]

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
