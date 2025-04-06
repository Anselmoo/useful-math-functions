"""Complex plane fractals for the UMF package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from meta.api import ResultsFractalAPI
from umf.meta.functions import ComplexFractalFunction
from umf.meta.functions import FractalFunction


if TYPE_CHECKING:
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
        c (UniversalArray): Complex plane coordinates.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        escape_radius (float, optional): Escape radius. Defaults to 2.0.
    """

    def __init__(
        self, c: UniversalArray, max_iter: int = 100, escape_radius: float = 2.0
    ) -> None:
        """Initialize the Mandelbrot set."""
        self.c = c
        # Initialize first, then call super() to register properly
        self.fractal_dimension = (
            2.0  # Approximate dimension for the Mandelbrot boundary
        )
        super().__init__(c, max_iter=max_iter, escape_radius=escape_radius)

    @property
    def __eval__(self) -> np.ndarray:
        """Compute the Mandelbrot set.

        Returns:
            np.ndarray: Array of iteration counts.
        """
        c = np.asarray(self.c)
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
        z (UniversalArray): Starting complex plane coordinates.
        c (complex): Julia set parameter.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        escape_radius (float, optional): Escape radius. Defaults to 2.0.
    """

    def __init__(
        self,
        z: UniversalArray,
        c: complex,
        max_iter: int = 100,
        escape_radius: float = 2.0,
    ) -> None:
        """Initialize the Julia set."""
        self.z = z
        self.c = c
        # Different c values give different fractal dimensions
        # 1.2 is a reasonable default for many Julia sets
        self.fractal_dimension = 1.2
        super().__init__(z, max_iter=max_iter, escape_radius=escape_radius)

    @property
    def __eval__(self) -> np.ndarray:
        """Compute the Julia set.

        Returns:
            np.ndarray: Array of iteration counts.
        """
        z = np.asarray(self.z)
        shape = z.shape

        # Use the common implementation from ComplexFractalFunction
        return self.iterate_complex_function(z_start=z, c=self.c, shape=shape)

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
        >>> from umf.functions.fractal_set.complex import FeigenbaumDiagram
        >>> # Generate a Feigenbaum diagram
        >>> r_values = np.linspace(2.8, 4.0, 1000)  # Parameter range
        >>> feigenbaum = FeigenbaumDiagram(r_values)()
        >>> bifurcation_data = feigenbaum.result

        >>> # Visualization Example
        >>> _ = plt.figure(figsize=(12, 8))
        >>> for r, x_values in bifurcation_data:
        ...     _ = plt.plot([r] * len(x_values), x_values, ',r', alpha=0.2)
        >>> _ = plt.xlabel('r parameter')
        >>> _ = plt.ylabel('x values')
        >>> _ = plt.title('Feigenbaum Diagram (Logistic Map)')
        >>> plt.savefig("FeigenbaumDiagram.png", dpi=300, transparent=True)

    Notes:
        The Feigenbaum diagram shows the values that the logistic map takes as
        the parameter $r$ increases, revealing the period-doubling route to chaos.

    Args:
        r_values (UniversalArray): Range of parameter values.
        x0 (float, optional): Initial value for the iteration. Defaults to 0.5.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        n_discard (int, optional): Number of initial iterations to discard.
            Defaults to 100.
    """

    def __init__(
        self,
        r_values: UniversalArray,
        x0: float = 0.5,
        max_iter: int = 1000,  # Changed from n_iterations for consistency
        n_discard: int = 100,
    ) -> None:
        """Initialize the Feigenbaum diagram."""
        self.r_values = r_values
        self.x0 = x0
        self.n_discard = n_discard
        self.fractal_dimension = 0.538  # Feigenbaum constant-related dimension
        super().__init__(r_values, max_iter=max_iter)

    @property
    def __eval__(self) -> list[tuple[float, list[float]]]:
        """Compute the Feigenbaum diagram.

        Returns:
            list[tuple[float, list[float]]]: List of (r, x_values) pairs.
        """
        result = []

        for r in self.r_values:
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
        if len(self.r_values) <= window_size * 2:
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
                bifurcations.append(self.r_values[i])

        return bifurcations
