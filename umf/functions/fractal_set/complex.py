"""Complex plane fractals for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.meta.functions import ComplexFractalFunction
from umf.meta.functions import FractalFunction


if TYPE_CHECKING:
    from umf.meta.api import ResultsFractalAPI
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "FeigenbaumDiagram",
    "JuliaSet",
    "MandelbrotSet",
]


class MandelbrotSet(ComplexFractalFunction):
    r"""Implementation of the classic Mandelbrot set fractal.

    The Mandelbrot set is the set of complex numbers c for which the function
    $f_c(z) = z^2 + c$ does not diverge to infinity when iterated from $z = 0$.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.complex import MandelbrotSet
        >>> # Generate a Mandelbrot set with 100x100 resolution and 50 max iterations
        >>> x = np.linspace(-2.5, 1.5, 1000)
        >>> y = np.linspace(-1.5, 1.5, 1000)
        >>> X, Y = np.meshgrid(x, y)
        >>> C = X + 1j * Y
        >>> mandelbrot = MandelbrotSet(C, max_iter=150)()
        >>> iterations = mandelbrot.result

        >>> # Visualization Example
        >>> _ = plt.figure(figsize=(10, 8))
        >>> _ = plt.imshow(iterations, cmap='hot', extent=[-2.5, 1.5, -1.5, 1.5])
        >>> _ = plt.colorbar(label='Iterations')
        >>> _ = plt.title("Mandelbrot Set")
        >>> plt.savefig("MandelbrotSet.png", dpi=300, transparent=True)

    Notes:
        The Mandelbrot set is defined by the recurrence relation:

        $$
            z_{n+1} = z_n^2 + c
        $$

        where $z_0 = 0$ and $c$ is a complex parameter.

    Args:
        *x (UniversalArray): Complex plane coordinates.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        escape_radius (float, optional): Escape radius. Defaults to 2.0.
    """

    def __init__(
        self, *x: UniversalArray, max_iter: int = 100, escape_radius: float = 2.0
    ) -> None:
        """Initialize the Mandelbrot set."""
        # Initialize first, then call super() to register properly
        self.fractal_dimension = (
            2.0  # Approximate dimension for the Mandelbrot boundary
        )

        super().__init__(*x, max_iter=max_iter, escape_radius=escape_radius)

    @property
    def __eval__(self) -> np.ndarray:
        """Compute the Mandelbrot set.

        Returns:
            np.ndarray: Array of iteration counts.
        """
        c = self._x[0]
        shape = c.shape

        # Use the common implementation from ComplexFractalFunction
        return self.iterate_complex_function(z_start=np.zeros_like(c), c=c, shape=shape)

    def is_in_set(self, c_values: UniversalArray) -> np.ndarray:
        """Determine whether points are in the Mandelbrot set.

        Args:
            c_values (UniversalArray): Complex values to test

        Returns:
            np.ndarray: Boolean array where True indicates the point is in the set
        """
        iterations = MandelbrotSet(
            c_values, max_iter=self.max_iter, escape_radius=self.escape_radius
        )()
        return iterations.result == self.max_iter


class JuliaSet(ComplexFractalFunction):
    r"""Implementation of the Julia set fractal.

    The Julia set is closely related to the Mandelbrot set. While the Mandelbrot set
    starts with $z = 0$ and varies the parameter c, the Julia set fixes $c$ and varies
    the starting value of $z$ across the complex plane.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from umf.functions.fractal_set.complex import JuliaSet
        >>> # Generate a Julia set with 100x100 resolution, c = -0.7 + 0.27j
        >>> # 50 max iterations
        >>> x = np.linspace(-1.5, 1.5, 1000)
        >>> y = np.linspace(-1.5, 1.5, 1000)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = X + 1j * Y
        >>> c = -0.7 + 0.27j
        >>> julia = JuliaSet(Z, c, max_iter=150)()
        >>> iterations = julia.result

        >>> # Visualization Example
        >>> _ = plt.figure(figsize=(10, 8))
        >>> _ = plt.imshow(iterations, cmap='viridis', extent=[-1.5, 1.5, -1.5, 1.5])
        >>> _ = plt.colorbar(label='Iterations')
        >>> _ = plt.title(f"Julia Set (c = {c})")
        >>> plt.savefig("JuliaSet.png", dpi=300, transparent=True)

    Notes:
        The Julia set is defined by the recurrence relation:

        $$
            z_{n+1} = z_n^2 + c
        $$
        where $z_0$ is a complex number and c is a fixed complex parameter.

    Args:
        *x (UniversalArray): Complex values for [z, c], where z is the starting complex
            plane coordinates and c is the Julia set parameter.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        escape_radius (float, optional): Escape radius. Defaults to 2.0.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 100,
        escape_radius: float = 2.0,
    ) -> None:
        """Initialize the Julia set."""
        # Ensure c is converted to numpy array even if it's a scalar complex
        c_value = x[1] if len(x) > 1 else -0.7 + 0.27j
        self.c = np.asarray(c_value, dtype=complex)
        # Different c values give different fractal dimensions
        # 1.2 is a reasonable default for many Julia sets
        self.fractal_dimension = 1.2
        # We need to ensure x[1] is an ndarray for validation
        new_x = list(x)
        if len(new_x) > 1:
            new_x[1] = self.c
        super().__init__(*new_x, max_iter=max_iter, escape_radius=escape_radius)

    @property
    def __eval__(self) -> np.ndarray:
        """Compute the Julia set.

        Returns:
            np.ndarray: Array of iteration counts.
        """
        z = np.asarray(self._x[0])
        shape = z.shape

        # Ensure c is properly broadcast when it's a scalar
        c = (
            np.broadcast_to(self.c, shape)
            if np.isscalar(self.c) or self.c.size == 1
            else self.c
        )

        # Use the common implementation from ComplexFractalFunction
        return self.iterate_complex_function(z_start=z, c=c, shape=shape)

    def is_in_set(self, z_values: UniversalArray) -> np.ndarray:
        """Determine whether points are in the Julia set.

        Args:
            z_values (UniversalArray): Complex values to test

        Returns:
            np.ndarray: Boolean array where True indicates the point is in the set
        """
        iterations: ResultsFractalAPI = JuliaSet(
            z_values, self.c, max_iter=self.max_iter, escape_radius=self.escape_radius
        )()
        return np.array(iterations.result == self.max_iter)


class FeigenbaumDiagram(FractalFunction):
    r"""Implementation of the Feigenbaum diagram (bifurcation diagram).

    The Feigenbaum diagram illustrates the bifurcation of the logistic map
    $x_{n+1} = r * x_n * (1 - x_n)$ as the parameter $r$ increases.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.colors import LinearSegmentedColormap
        >>> from umf.functions.fractal_set.complex import FeigenbaumDiagram
        >>> # Generate a Feigenbaum diagram
        >>> r_values = np.linspace(2.8, 4.0, 1000)
        >>> feigenbaum = FeigenbaumDiagram(r_values)()
        >>> bifurcation_data = feigenbaum.result
        >>>
        >>> # Visualization Example with improved quality
        >>> fig = plt.figure(figsize=(12, 8), dpi=300)
        >>> # Create a custom colormap for a more appealing visualization
        >>> colors = [(0.0, 0.0, 0.4), (0.0, 0.0, 0.7), (0.0, 0.3, 1.0)]
        >>> cm = LinearSegmentedColormap.from_list('feigenbaum_colors', colors, N=256)
        >>>
        >>> # Plot points with a gradient based on r value for better visualization
        >>> for i, (r, x_values) in enumerate(bifurcation_data):
        ...     # Break long line into multiple lines
        ...     color_val = i / len(bifurcation_data)
        ...     _ = plt.plot(
        ...         [r] * len(x_values),
        ...         x_values,
        ...         ',',
        ...         color=cm(color_val),
        ...         alpha=0.1,
        ...         markersize=0.5
        ...     )
        >>>
        >>> # Add key bifurcation points as vertical lines
        >>> key_points = [3.0, 3.45, 3.57]  # Important bifurcation points
        >>> for point in key_points:
        ...     _ = plt.axvline(x=point, color='red', alpha=0.2, linestyle='--')
        >>>
        >>> _ = plt.xlabel('Parameter r', fontsize=14)
        >>> _ = plt.ylabel('Population values x', fontsize=14)
        >>> _ = plt.title('Feigenbaum Diagram (Logistic Map)', fontsize=16)
        >>> _ = plt.xlim(2.8, 4.0)  # Set explicit x limits
        >>> _ = plt.ylim(0, 1)      # Set explicit y limits
        >>> plt.grid(False)         # Remove grid for cleaner appearance
        >>> plt.tight_layout()
        >>> plt.savefig("FeigenbaumDiagram.png", dpi=300, transparent=True)

    Notes:
        The Feigenbaum diagram shows the values that the logistic map takes as
        the parameter $r$ increases, revealing the period-doubling route to chaos.

        The logistic map is defined as:

        $$
        x_{n+1} = r \cdot x_n \cdot (1 - x_n)
        $$

        where $r$ is the bifurcation parameter. As $r$ increases from 0 to 4,
        the behavior changes from a single stable fixed point to period doubling
        and eventually chaos.

        Key values of the bifurcation parameter:

        - For $0 < r < 1$: All iterations tend to 0
        - For $1 < r < 3$: Iterations tend to a single fixed point
          $x^* = 1 - \frac{1}{r}$
        - For $r > 3$: Period doubling begins
        - Near $r \approx 3.57$: Chaotic behavior emerges

        The bifurcations follow a geometric scaling governed by the Feigenbaum constant:

        $$
        \delta = \lim_{n \to \infty} \frac{r_n - r_{n-1}}{r_{n+1} - r_n} \approx 4.669
        $$

        where $r_n$ is the parameter value at the $n$-th bifurcation.

        > Reference: Feigenbaum, M. J. (1978). Quantitative universality for a class
        > of nonlinear transformations. Journal of Statistical Physics, 19(1), 25-52.

    Args:
        *x (UniversalArray): Range of parameter values.
        x0 (float, optional): Initial value for the iteration. Defaults to 0.5.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        n_discard (int, optional): Number of initial iterations to discard.
            Defaults to 100.
        fractal_dimension (float, optional): Fractal dimension of the set.
            Defaults to 0.58.
    """

    def __init__(
        self,
        *x: UniversalArray,
        x0: float = 0.5,
        max_iter: int = 1000,
        n_discard: int = 100,
        fractal_dimension: float = 0.58,
    ) -> None:
        """Initialize the Feigenbaum diagram."""
        self.x0 = x0
        self.n_discard = n_discard
        self.fractal_dimension = fractal_dimension
        super().__init__(*x, max_iter=max_iter)

    @property
    def __eval__(self) -> list[tuple[float, list[float]]]:
        """Compute the Feigenbaum diagram.

        Returns:
            list[tuple[float, list[float]]]: List of (r, x_values) pairs.
        """
        result = []

        for r in self._x[0]:
            x = self.x0
            x_values = []

            # Discard initial iterations
            for _ in range(self.n_discard):
                x = r * x * (1 - x)

            # Save subsequent iterations
            for _ in range(self.max_iter):  # Using max_iter consistently
                x = r * x * (1 - x)
                x_values.append(x)

            result.append((r, x_values))

        return result

    def find_bifurcation_points(self) -> list[float]:
        """Find approximate bifurcation points in the diagram.

        Returns:
            list[float]: List of r values where bifurcations occur
        """
        # This is a simplified implementation
        bifurcations = []

        # Parameters for identification
        window_size = 10
        # Get the r values from the first element of each tuple in the result
        # Fix: Instead of accessing self.r_values, we use self._x[0]
        r_values = self._x[0]

        if len(r_values) <= window_size * 2:
            return []

        # Analyze the bifurcation diagram
        diagram = self.__eval__
        for i in range(window_size, len(diagram) - window_size):
            # Simple heuristic: look for changes in the number of distinct values
            prev_values = set()
            for _, x_vals in diagram[i - window_size : i]:
                prev_values.update(x_vals)

            next_values = set()
            for _, x_vals in diagram[i : i + window_size]:
                next_values.update(x_vals)

            # If the number of values doubles (approximately), it might be a bifurcation
            if abs(len(next_values) - 2 * len(prev_values)) < 0.2 * len(prev_values):
                # Fix: Use r_values instead of self.r_values
                bifurcations.append(r_values[i])

        return bifurcations

    def is_in_set(self, point: UniversalArray) -> np.ndarray:
        """Determine whether a point is in the chaotic region of the bifurcation.

        Args:
            point (UniversalArray): Point to test, in the form [r, x]

        Returns:
            np.ndarray: Boolean array where True indicates chaotic behavior
        """
        # Define chaotic region approximately as r > 3.57
        r = point[0]
        # For numerical stability, return array type with proper shape
        chaotic_limiter = 3.57
        return np.array(r > chaotic_limiter, dtype=bool)
