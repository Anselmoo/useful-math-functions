"""Dynamic system fractals for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.dimensions import __2d__
from umf.constants.dimensions import __3d__
from umf.constants.exceptions import OutOfDimensionError
from umf.meta.functions import DynamicFractalFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__ = [
    "CurlicueFractal",
    "LorenzAttractor",
    "PercolationModel",
    "RandomWalkFractal",
]


class LorenzAttractor(DynamicFractalFunction):
    r"""Implementation of the Lorenz attractor fractal.

    The Lorenz attractor is a set of chaotic solutions of the Lorenz system:

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from mpl_toolkits.mplot3d import Axes3D
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.dynamic import LorenzAttractor
        >>> # Generate Lorenz attractor
        >>> x, y, z = np.array([0.0]), np.array([1.0]), np.array([1.05])
        >>> lorenz = LorenzAttractor(x, y, z, max_iter=10000)()
        >>> points = lorenz.result
        >>>
        >>> # Visualization with enhanced quality
        >>> fig = plt.figure(figsize=(12, 10), dpi=300)
        >>> ax = fig.add_subplot(111, projection='3d')
        >>>
        >>> # Create a custom colormap for better visualization
        >>> colors = [(0.0, 0.0, 0.5), (0.0, 0.5, 1.0), (0.7, 0.0, 0.7)]
        >>> cm = LinearSegmentedColormap.from_list('lorenz_colors', colors, N=256)
        >>>
        >>> # Color points based on their z-coordinate for better depth perception
        >>> for i in range(len(points) - 1):
        ...     x = [points[i][0], points[i+1][0]]
        ...     y = [points[i][1], points[i+1][1]]
        ...     z = [points[i][2], points[i+1][2]]
        ...     # Normalize z-coordinate for color mapping
        ...     z_norm = (
        ...         (points[i][2] - points[:, 2].min())
        ...         / (points[:, 2].max() - points[:, 2].min())
        ...         )
        ...     _ = ax.plot(x, y, z, color=cm(z_norm), linewidth=0.8, alpha=0.8)
        >>>
        >>> # Set viewing angle for best presentation
        >>> _ = ax.view_init(elev=30, azim=70)
        >>> _ = ax.set_xlabel('X axis', fontsize=12)
        >>> _ = ax.set_ylabel('Y axis', fontsize=12)
        >>> _ = ax.set_zlabel('Z axis', fontsize=12)
        >>> _ = plt.title("Lorenz Attractor", fontsize=14)
        >>> # Remove background grid for cleaner look
        >>> ax.grid(False)
        >>> ax.xaxis.pane.fill = False
        >>> ax.yaxis.pane.fill = False
        >>> ax.zaxis.pane.fill = False
        >>> plt.tight_layout()
        >>> plt.savefig("LorenzAttractor.png", dpi=300, transparent=True)

    Notes:
        The Lorenz attractor has a fractal dimension of approximately 2.06 (Hausdorff
        dimension). It exhibits chaotic behavior for certain parameter values,
        meaning that small  changes in initial conditions lead to dramatically different
        trajectories,  while still being confined to the same strange attractor.

        $$
            \begin{align}
            \frac{dx}{dt} &= \sigma(y - x) \\
            \frac{dy}{dt} &= x(\rho - z) - y \\
            \frac{dz}{dt} &= xy - \beta z
            \end{align}
        $$

        This is one of the most famous examples of deterministic chaos and was
        discovered by Edward Lorenz in 1963 while studying simplified models of
        atmospheric convection.

    Args:
        initial_state (UniversalArray): Initial point [x, y, z]
        max_iter (int, optional): Number of iterations. Defaults to 10000.
        sigma (float, optional): $\sigma$ parameter. Defaults to 10.0.
        rho (float, optional): $\rho$ parameter. Defaults to 28.0.
        beta (float, optional): $\beta$ parameter. Defaults to 8/3.
    """

    def __init__(
        self,
        *initial_state: UniversalArray,
        max_iter: int = 10000,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
    ) -> None:
        """Initialize the Lorenz attractor."""
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.fractal_dimension = 2.06  # Approximate Hausdorff dimension

        if len(initial_state) != __3d__:
            raise OutOfDimensionError(
                function_name=self.__class__.__name__,
                dimension=__3d__,
            )

        super().__init__(*initial_state, max_iter=max_iter)

    def iterate_system(self, state: np.ndarray) -> np.ndarray:
        """Iterate the Lorenz system.

        Args:
            state (np.ndarray): Current state [x, y, z]

        Returns:
            np.ndarray: Next state
        """
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        dt = 0.01  # Time step
        return state + dt * np.array([dx, dy, dz])

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the Lorenz attractor points.

        Returns:
            np.ndarray: Array of points on the attractor
        """
        state = np.array([self._x[0], self._x[1], self._x[2]])
        points = [state]

        # Skip transient steps
        for _ in range(self.transient_steps):
            state = self.iterate_system(state)

        # Generate attractor points
        for _ in range(self.max_iter):
            state = self.iterate_system(state)
            points.append(state.copy())

        return np.array(points)


class CurlicueFractal(DynamicFractalFunction):
    r"""Implementation of the Curlicue fractal.

    The Curlicue fractal is generated by repeatedly rotating a vector by a fixed angle
    and connecting the endpoints.

    Notes:
        The Curlicue fractal is a visual representation of the sum of unit vectors with
        angles that form an arithmetic sequence. For a given angle $\theta$, the pattern
        is created by the sequence:

        $$
        z_n = \sum_{k=0}^{n} e^{i\theta k^2}
        $$

        When $\theta$ is an irrational multiple of $\pi$, the pattern never repeats and
        creates intricate fractal-like structures. The Golden angle
        $\theta = \pi(3-\sqrt{5})$ produces particularly beautiful patterns.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.dynamic import CurlicueFractal
        >>> # Generate Curlicue fractal
        >>> angle = np.pi * (3 - np.sqrt(5))  # Golden angle
        >>> curlicue = CurlicueFractal(
        ...     np.array([0.]),
        ...     np.array([0.]),
        ...     angle=angle,
        ...     max_iter=2000)()
        >>> points = curlicue.result
        >>>
        >>> # Visualization with enhanced quality
        >>> fig = plt.figure(figsize=(10, 10), dpi=300)
        >>>
        >>> # Create a gradient colormap for better visualization
        >>> colors = [(0.0, 0.4, 0.9), (0.4, 0.0, 0.9), (0.9, 0.0, 0.6)]
        >>> cm = LinearSegmentedColormap.from_list('curlicue_colors', colors, N=256)
        >>>
        >>> # Use a segment-based coloring approach
        >>> for i in range(len(points) - 1):
        ...     x = [points[i][0], points[i+1][0]]
        ...     y = [points[i][1], points[i+1][1]]
        ...     # Normalize position for color mapping
        ...     position = i / (len(points) - 2)
        ...     color = cm(position)
        ...     _ = plt.plot(x, y, color=color, linewidth=0.7, alpha=0.8)
        >>>
        >>> # Set equal aspect ratio and clean up the plot
        >>> _ = plt.axis('equal')
        >>> _ = plt.axis('off')  # Hide axes for cleaner look
        >>> _ = plt.title("Curlicue Fractal", fontsize=14)
        >>>
        >>> # Add a subtle background gradient
        >>> plt.gca().set_facecolor('#f8f8ff')
        >>> plt.tight_layout()
        >>> plt.savefig("CurlicueFractal.png", dpi=300, transparent=True)

    Args:
        start (UniversalArray): Starting point [x, y].
        angle (float): Rotation angle in radians.
        max_iter (int, optional): Number of iterations. Defaults to 1000.
        step_size (float, optional): Length of each step. Defaults to 1.0.
        fractal_dimension (float, optional): Fractal dimension. Defaults to 1.5.
    """

    def __init__(
        self,
        *start: UniversalArray,
        angle: float,
        max_iter: int = 1000,
        step_size: float = 1.0,
        fractal_dimension: float = 1.5,
    ) -> None:
        """Initialize the Curlicue fractal."""
        self.angle = angle
        self.step_size = step_size
        if len(start) != __2d__:
            raise OutOfDimensionError(
                function_name=self.__class__.__name__,
                dimension=__2d__,
            )
        super().__init__(*start, max_iter=max_iter, fractal_dimension=fractal_dimension)

    def iterate_system(
        self, state: UniversalArray, points: list[UniversalArray]
    ) -> np.ndarray:
        """Generate next point in the Curlicue pattern.

        Args:
            state (np.ndarray): Current point and cumulative angle.
            points (list[np.ndarray]): List of points generated so far.

        Returns:
            np.ndarray: Next point and updated angle
        """
        x, y = state
        # Calculate new position
        new_x = x + self.step_size * np.cos(self.angle * len(points))
        new_y = y + self.step_size * np.sin(self.angle * len(points))
        return np.array([new_x, new_y])

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the Curlicue fractal points.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        state = np.array([self._x[0], self._x[1]])
        points = [state]

        for _ in range(self.max_iter):
            state = self.iterate_system(state=state, points=points)
            points.append(state.copy())

        return np.array(points)


class PercolationModel(DynamicFractalFunction):
    r"""Implementation of a percolation model fractal.

    The percolation model creates a fractal pattern by randomly filling cells
    according to certain rules.

    Notes:
        Percolation theory describes the behavior of connected clusters in random
        graphs or lattices. At the critical probability threshold $p_c$ (approximately
        0.59275 for a 2D square lattice), the system undergoes a phase transition
        where an infinite spanning cluster emerges.

        The resulting cluster has a fractal dimension of approximately
        $\frac{91}{48} \approx 1.895$.

        The percolation process follows these mathematical rules:
        1. Each site is independently occupied with probability $p$
        2. Adjacent occupied sites form clusters
        3. At the critical threshold, the largest cluster exhibits fractal properties

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from mpl_toolkits.mplot3d import Axes3D
        >>> from umf.functions.fractal_set.dynamic import PercolationModel
        >>> # Generate percolation model at critical threshold
        >>> size_x, size_y = 50, 50
        >>> percolation = PercolationModel(
        ...     np.array([size_x]),
        ...     np.array([size_y]),
        ...     p=0.592, # Close to critical threshold for 2D square lattice
        ...     max_iter=100)()
        >>> grid = percolation.result
        >>>
        >>> # Create enhanced 3D visualization
        >>> fig = plt.figure(figsize=(12, 10), dpi=300)
        >>> ax = fig.add_subplot(111, projection='3d')
        >>>
        >>> # Create custom colormap for better visualization
        >>> colors = [(0.0, 0.2, 0.5), (0.0, 0.5, 0.8), (0.8, 0.0, 0.2)]
        >>> cmap = LinearSegmentedColormap.from_list(
        ...     'percolation_colors', colors, N=256,
        ... )
        >>>
        >>> # Find occupied cells and visualize as 3D bars
        >>> for i in range(size_x):
        ...     for j in range(size_y):
        ...         if grid[i, j] > 0:
        ...             # Height represents connectivity (higher value = more connected)
        ...             height = 0.2 if grid[i, j] == 1 else grid[i, j] / 5.0
        ...             # Color based on connectivity
        ...             color = cmap(grid[i, j] / 5.0 if grid[i, j] > 1 else 0.1)
        ...             # Draw 3D bar
        ...             _ = ax.bar3d(i, j, 0, 0.9, 0.9, height, color=color, alpha=0.7,
        ...                      shade=True)
        >>>
        >>> # Add a semi-transparent plane at z=0 to show percolation threshold
        >>> x = np.arange(0, size_x, 1)
        >>> y = np.arange(0, size_y, 1)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = np.zeros_like(X)
        >>> _ = ax.plot_surface(X, Y, Z, color='gray', alpha=0.1)
        >>>
        >>> # Set viewing angle for best presentation
        >>> _ = ax.view_init(elev=35, azim=45)
        >>> _ = ax.set_xlabel('X axis', fontsize=12)
        >>> _ = ax.set_ylabel('Y axis', fontsize=12)
        >>> _ = ax.set_zlabel('Connectivity', fontsize=12)
        >>> _ = plt.title(
        ...     "Percolation Model at Critical Threshold (p≈0.592)",
        ...     fontsize=14,
        ... )
        >>>
        >>> # Find and highlight the largest connected cluster
        >>> from scipy.ndimage import label
        >>> labeled_array, num_features = label(grid > 0)
        >>> largest_cluster_id = np.bincount(labeled_array.flatten())[1:].argmax() + 1
        >>> largest_cluster = labeled_array == largest_cluster_id
        >>>
        >>> # Plot the largest cluster with distinct color
        >>> # Use a gradient colormap for the largest cluster to avoid a single color
        >>> cluster_colors = [(1.0, 1.0, 0.5), (1.0, 0.8, 0.0), (0.9, 0.3, 0.0)]
        >>> cluster_cmap = LinearSegmentedColormap.from_list(
        ...     "cluster_cmap",
        ...     cluster_colors,
        ...     N=256
        ... )
        >>> for i in range(size_x):
        ...     for j in range(size_y):
        ...         if largest_cluster[i, j]:
        ...             height = grid[i, j] / 5.0 if grid[i, j] > 1 else 0.3
        ...             # Normalize cluster strength for color mapping
        ...             norm_val = (grid[i, j] - grid.min()) / (grid.max() - grid.min())
        ...             norm_val = max(0.0, min(1.0, norm_val))
        ...             grad_color = cluster_cmap(norm_val)
        ...             _ = ax.bar3d(
        ...                 i,
        ...                 j,
        ...                 0,
        ...                 0.9,
        ...                 0.9,
        ...                 height,
        ...                 color=grad_color,
        ...                 alpha=0.8,
        ...                 shade=True
        ...             )
        >>>
        >>> # Add text annotation to explain percolation
        >>> _ = ax.text(size_x/2, -5, 2,
        ...         "Percolation at critical threshold\nshowing fractal clusters",
        ...         fontsize=10, color='black')
        >>>
        >>> # Remove grid lines for cleaner look
        >>> ax.grid(False)
        >>> # Set axis limits
        >>> _ = ax.set_xlim(0, size_x)
        >>> _ = ax.set_ylim(0, size_y)
        >>> _ = ax.set_zlim(0, 2)
        >>>
        >>> _ = plt.tight_layout()
        >>> plt.savefig("PercolationModel3D.png", dpi=300, transparent=True)


    Args:
        size (UniversalArray): Grid dimensions [height, width]
        p (float, optional): Occupation probability. Defaults to 0.59.
        max_iter (int, optional): Number of iterations. Defaults to 1000.
        fractal_dimension (float, optional): Fractal dimension. Defaults to
            $91 / 48$.
    """

    def __init__(
        self,
        *size: UniversalArray,
        p: float = 0.59,
        max_iter: int = 1000,
        fractal_dimension: float = 91 / 48,
    ) -> None:
        """Initialize the percolation model."""
        self.p = p
        self.rng = np.random.default_rng()
        if len(size) != __2d__:
            raise OutOfDimensionError(
                function_name=self.__class__.__name__,
                dimension=__2d__,
            )
        super().__init__(*size, max_iter=max_iter, fractal_dimension=fractal_dimension)

    def iterate_system(self, initial_state: np.ndarray) -> np.ndarray:
        """Update the percolation grid.

        Args:
            initial_state: Current grid state

        Returns:
            np.ndarray: Updated grid
        """
        height, width = initial_state.shape
        new_cells = self.rng.random(initial_state.shape) < self.p

        # Combine existing occupied cells with new ones
        state = np.logical_or(initial_state, new_cells).astype(float)

        # Update clusters - mark cells with value 5.0 for better visibility
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if state[i, j] == 1:
                    # Connect to neighboring clusters
                    neighbors = state[i - 1 : i + 2, j - 1 : j + 2]
                    if np.sum(neighbors) > 1:
                        state[i, j] = 5.0  # Higher value for better contrast

        return state

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the percolation pattern.

        Returns:
            np.ndarray: Final grid state
        """
        # Initialize grid with zeros
        try:
            height = int(self._x[0][0])
            width = int(self._x[1][0])
            state = np.zeros((height, width), dtype=float)
        except (IndexError, ValueError, TypeError):
            # Fallback if dimensions are not properly specified
            state = np.zeros((100, 100), dtype=float)

        # Run simulation
        for _ in range(self.max_iter):
            state = self.iterate_system(state)

        return state


class RandomWalkFractal(DynamicFractalFunction):
    r"""Implementation of a random walk fractal.

    A random walk fractal is generated by taking random steps in a bounded space,
    creating patterns that exhibit fractal properties.

    Notes:
        The random walk is a mathematical formalization of a path consisting of
        successive random steps. In two dimensions, it follows the equation:

        $$
        \vec{r}_{n+1} = \vec{r}_n + \delta \hat{e}(\theta_n)
        $$

        where $\vec{r}_n$ is the position at step $n$, $\delta$ is the step size,
        $\theta_n$ is a random angle uniformly distributed over $[0, 2\pi)$, and
        $\hat{e}(\theta_n)$ is the unit vector in the direction of $\theta_n$.

        Random walks have a fractal dimension of 2 in free space, but bounded
        walks exhibit a smaller dimension of approximately 1.5. They are related
        to Brownian motion and diffusion processes in physics.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.dynamic import RandomWalkFractal
        >>> # Generate random walk with more steps for better pattern
        >>> start = np.array([0.]), np.array([0.])
        >>> bounds = np.array([[-10, 10], [-10, 10]])
        >>> walk = RandomWalkFractal(*start, bounds=bounds, max_iter=20000)()
        >>> points = walk.result
        >>>
        >>> # Visualization with enhanced quality
        >>> fig = plt.figure(figsize=(10, 10), dpi=300)
        >>>
        >>> # Create a gradient colormap for better visualization
        >>> colors = [(0.0, 0.2, 0.5), (0.2, 0.5, 0.8), (0.9, 0.0, 0.2)]
        >>> cm = LinearSegmentedColormap.from_list('walk_colors', colors, N=256)
        >>>
        >>> # Use a segment-based coloring approach with step-based color and alpha
        >>> segment_size = 100  # Group points for better performance
        >>> for i in range(0, len(points)-segment_size, segment_size):
        ...     # Get subset of points for this segment
        ...     segment = points[i:i+segment_size+1]
        ...     # Normalize position for color mapping
        ...     position = i / len(points)
        ...     color = cm(position)
        ...     # Alpha increases with step number for depth effect
        ...     alpha = 0.3 + 0.5 * position
        ...     _ = plt.plot(segment[:, 0], segment[:, 1], '-',
        ...                  color=color, linewidth=0.5, alpha=alpha)
        >>>
        >>> # Add starting point marker
        >>> _ = plt.plot(points[0, 0], points[0, 1], 'o',
        ...              color='green', markersize=6, alpha=0.8)
        >>> _ = plt.plot(points[-1, 0], points[-1, 1], '*',
        ...              color='red', markersize=8, alpha=0.8)
        >>>
        >>> # Set equal aspect ratio and clean up the plot
        >>> _ = plt.axis('equal')
        >>> # Add subtle grid
        >>> _ = plt.grid(True, linestyle='--', alpha=0.2)
        >>> # Add border lines for the bounding box
        >>> _ = plt.plot([-10, -10, 10, 10, -10], [-10, 10, 10, -10, -10],
        ...              'k-', linewidth=1, alpha=0.5)
        >>> _ = plt.title("Random Walk Fractal", fontsize=14)
        >>> plt.tight_layout()
        >>> plt.savefig("RandomWalkFractal.png", dpi=300, transparent=True)

    Args:
        walk_data (Tuple[UniversalArray, UniversalArray]): Tuple containing:
            - start: Starting point [x, y]
            - bounds: Boundary limits [[xmin, xmax], [ymin, ymax]]
        max_iter (int, optional): Number of steps. Defaults to 10000.
        step_size (float, optional): Size of each step. Defaults to 1.0.
        dimension (int, optional): Dimensionality of the walk. Defaults to 2.
        fractal_dimension (float, optional): Fractal dimension. Defaults to 2.0.
    """

    def __init__(
        self,
        *walk_data: UniversalArray,
        bounds: UniversalArray | None = None,
        max_iter: int = 10000,
        step_size: float = 1.0,
        dimension: int = 2,
        fractal_dimension: float = 2.0,
    ) -> None:
        """Initialize the random walk fractal."""
        self.step_size = step_size
        self.dimension = dimension
        self.rng = np.random.default_rng()
        if bounds is not None:
            self.bounds = np.array(bounds)
        else:
            self.bounds = np.array([[-10, 10], [-10, 10]])
        # For bounded random walks in 2D
        fractal_dimension = 1.5 if self.dimension == __2d__ else 2.0

        if len(walk_data) != __2d__:
            raise OutOfDimensionError(
                function_name=self.__class__.__name__,
                dimension=__2d__,
            )
        super().__init__(
            *walk_data, max_iter=max_iter, fractal_dimension=fractal_dimension
        )

    def iterate_system(self, state: np.ndarray) -> np.ndarray:
        """Take a random step from current position.

        Args:
            state (np.ndarray): Current position

        Returns:
            np.ndarray: New position
        """
        if self.dimension == __2d__:
            # Generate random angle
            angle = self.rng.uniform(0, 2 * np.pi)
            # Take step in random direction
            new_state = state + self.step_size * np.array(
                [np.cos(angle), np.sin(angle)]
            )

            # Enforce boundaries
            bounds = np.asarray(self.bounds)
            for i in range(self.dimension):
                new_state[i] = np.clip(new_state[i], bounds[i, 0], bounds[i, 1])
        else:
            # Handle higher dimensions with normalized random steps
            step = self.rng.normal(0, 1, self.dimension)
            step = step / np.linalg.norm(step) * self.step_size
            new_state = state + step

            # Apply bounds if provided (assuming bounds is properly dimensioned)
            if self.bounds is not None and self.bounds.shape[0] == self.dimension:
                for i in range(self.dimension):
                    new_state[i] = np.clip(
                        new_state[i], self.bounds[i, 0], self.bounds[i, 1]
                    )

        return new_state

    @property
    def __eval__(self) -> np.ndarray:
        """Generate the random walk points.

        Returns:
            np.ndarray: Array of points visited by the walk
        """
        state = np.array(
            [self._x[0][0], self._x[1][0]]
        )  # Extract scalar values from arrays
        points = [state]

        for _ in range(self.max_iter):
            state = self.iterate_system(state)
            points.append(state.copy())

        return np.array(points)
