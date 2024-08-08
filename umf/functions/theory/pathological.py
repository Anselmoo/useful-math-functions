"""Pathological functions for the useful-math-functions library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.exceptions import NotInRangesError
from umf.constants.exceptions import NotLargerThanAnyError
from umf.meta.functions import PathologicalPure
from umf.meta.functions import PathologicalWithCoefficients


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "WeierstrassFunction",
    "RiemannFunction",
    "TakagiFunction",
    "MandelbrotsFractalFunction",
    "BesicovitchFunction",
]


class WeierstrassFunction(PathologicalWithCoefficients):
    r"""Weierstrass function.

    The Weierstrass function is a famous example of a real-valued function that is
    continuous everywhere but differentiable nowhere. It is defined by an infinite
    series that oscillates too wildly to settle down to a smooth curve.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation
        >>> import numpy as np
        >>> from umf.functions.theory.pathological import WeierstrassFunction
        >>> x = np.linspace(-3, 3, 100000)
        >>> wf = WeierstrassFunction(x, n=20, a=0.5, b=30)()
        >>> fig = plt.figure(figsize=(10, 6))
        >>> ax = fig.add_subplot(111)
        >>> (ax_return,) = ax.plot(x, wf.result)
        >>> def update(frame: int) -> tuple:
        ...     zoom_factor = frame / 25.0
        ...     ax.set_xlim(
        ...             -3 + zoom_factor / 1.3, 3 - zoom_factor/ 1.3
        ...         )
        ...     ax.set_ylim(
        ...             -2.5 + zoom_factor / 2.5, 2.5 - zoom_factor / 5
        ...         )
        ...     return (ax_return,)
        >>> ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)
        >>> ani.save('WeierstrassFunction.gif', writer='imagemagick', fps=10)

    Notes:
        The Weierstrass function is a prototypical example of a pathological
        function in mathematical analysis. Its definition challenges the
        intuition that a continuous function must be smooth.

        $$
        W(x) = \sum_{n=0}^{n_1} a^n \cos(b^n \pi x)
        $$

        with constraints $0 < a < 1$ and $ab > 1 + 3\pi/2$.

        > Reference: https://en.wikipedia.org/wiki/Weierstrass_function

    Args:
        *x (UniversalArray): Input values for which the Weierstrass function is
            evaluated.
        n (int, optional): The upper limit of the summation. Defaults to 10.
        a (float, optional): A parameter of the function that controls the amplitude of
            oscillations. Must be in the range (0, 1). Defaults to 0.9.
        b (float, optional): A parameter of the function that controls the frequency
            of oscillations. Must satisfy the condition a * b > (1 + 3 * np.pi / 2).
            Defaults to 7.

    Raises:
        NotInRangesError: If $a$ is not in the range $(0, 1)$.
        NotLargerThanAnyError: If $a * b$ is not larger than $(1 + 3 * np.pi / 2)$.
    """

    def __init__(
        self,
        *x: UniversalArray,
        n: int = 10,
        a: float = 0.9,
        b: float = 7,
    ) -> None:
        """Initialize the Weierstrass function."""
        if not 0 < a < 1:
            raise NotInRangesError(var_number="a", number=a, ranges=(0, 1))

        if a * b <= (1 + 3 * np.pi / 2):
            raise NotLargerThanAnyError(
                var_number="a * b",
                number=a * b,
                minimum=(1 + 3 * np.pi / 2),
            )
        super().__init__(*x, n_0=0, n_1=n, a=a, b=b)

    @property
    def __eval__(self) -> UniversalArray:
        """Calculate the Weierstrass function.

        Returns:
            UniversalArray: The Weierstrass function.
        """
        result = np.zeros_like(self._x, dtype=float)
        for n in range(self.n_0, self.n_1 + 1):
            result += self.a**n * np.cos(self.b**n * np.pi * self._x)
        return result


class RiemannFunction(PathologicalPure):
    r"""Riemann Function.

    The Riemann function is a mathematical function that is defined as the sum of
    a series of terms. Each term in the series is calculated using the
    Riemann zeta function and the sine function.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation
        >>> import numpy as np
        >>> from umf.functions.theory.pathological import RiemannFunction
        >>> x = np.linspace(-3, 3, 100000)
        >>> rf = RiemannFunction(x, n=20)()
        >>> fig = plt.figure(figsize=(10, 6))
        >>> ax = fig.add_subplot(111)
        >>> (ax_return,) = ax.plot(x, rf.result)
        >>> def update(frame: int) -> tuple:
        ...     zoom_factor = frame / 25.0
        ...     ax.set_xlim(
        ...         -3 + zoom_factor / 1.3, 3 - zoom_factor / 1.3
        ...         )
        ...     ax.set_ylim(
        ...             -2.5 + zoom_factor / 5, 2.5 - zoom_factor / 5
        ...         )
        ...     return (ax_return,)
        >>> ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)
        >>> ani.save('RiemannFunction.gif', writer='imagemagick', fps=10)

    Notes:
        The Riemann function is a mathematical function that is defined as the sum of
        a series of terms. Each term in the series is calculated using the
        Riemann zeta function and the sine function.

        $$
        R(x) = \sum_{n=1}^{n_1} \frac{1}{n^2} \sin(n^2 \pi x)
        $$

        > Reference: https://en.wikipedia.org/wiki/Riemann_function

    Args:
        *x (UniversalArray): The input values at which to evaluate the Riemann function.
        n (int, optional): The number of terms to include in the series.
            Defaults to 100.
    """

    def __init__(self, *x: UniversalArray, n: int = 100) -> None:
        """Initialize the Riemann function."""
        super().__init__(*x, n_0=1, n_1=n)

    @property
    def __eval__(self) -> UniversalArray:
        """Calculate the Riemann function.

        Returns:
            UniversalArray: The Riemann function.
        """
        result = np.zeros_like(self._x, dtype=float)
        for n in range(self.n_0, self.n_1 + 1):
            result += (1 / n**2) * np.sin(n**2 * np.pi * self._x)
        return result


class TakagiFunction(PathologicalPure):
    r"""Takagi Function.

    The Takagi function is a fractal-like continuous function defined on the real line.
    It is also known as the Takagi-Landsberg function or the Blancmange function.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation
        >>> import numpy as np
        >>> from umf.functions.theory.pathological import TakagiFunction
        >>> x = np.linspace(-1.5, 1.5, 100000)
        >>> tf = TakagiFunction(x, n=20)()
        >>> fig = plt.figure(figsize=(10, 6))
        >>> ax = fig.add_subplot(111)
        >>> (ax_return,) = ax.plot(x, tf.result)
        >>> def update(frame: int) -> tuple:
        ...     zoom_factor = frame / 25.0
        ...     ax.set_xlim(
        ...             -1.5 + zoom_factor / 2.5, 1.5 - zoom_factor / 2.5
        ...         )
        ...     ax.set_ylim(
        ...          0 + zoom_factor / 25, 0.6 - zoom_factor / 25
        ...         )
        ...     return (ax_return,)
        >>> ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)
        >>> ani.save('TakagiFunction.gif', writer='imagemagick', fps=10)

    Notes:
        The Takagi function is defined as the sum of a series of terms. Each term in the
        series is calculated using the absolute value of the fractional part of the
        input value raised to a power. It is also known as Blancmange curve.

        $$
        T(x) = \sum_{n=0}^{\infty} \frac{\phi(2^n x)}{2^n}
        $$

        where $(\phi(x))$ is the distance from $(x)$ to the nearest integer.

        > Reference: https://en.wikipedia.org/wiki/Takagi_function

    Args:
        *x (UniversalArray): Input values for which the Takagi function will be
            calculated.
        n (int, optional): Number of iterations to compute the Takagi function.
            Defaults to 100.
    """

    def __init__(self, *x: UniversalArray, n: int = 100) -> None:
        """Initialize the Takagi function."""
        super().__init__(*x, n_0=0, n_1=n)

    @property
    def __eval__(self) -> UniversalArray:
        """Calculate the Takagi function.

        Returns:
            UniversalArray: The Takagi function.
        """
        result = np.zeros_like(self._x, dtype=float)
        for n in range(1, self.n_1):
            result += self.phi(x=2**n * self._x) / 2**n
        return result

    @staticmethod
    def phi(x: UniversalArray) -> UniversalArray:
        """Calculate the distance from x to the nearest integer."""
        return np.abs(x - np.round(x))


class MandelbrotsFractalFunction(PathologicalPure):
    r"""Mandelbrot's Fractal Function.

    The Mandelbrot set is a famous example of a fractal, a mathematical object that
    exhibits self-similarity at different scales. It is named after the mathematician
    Benoit Mandelbrot, who first visualized and defined it in 1980. The set is created
    by iterating the function f_c(z) = z^2 + c over complex numbers c, where z starts
    at zero. A complex number c is part of the Mandelbrot set if, when applying this
    iteration, the absolute value of z does not diverge to infinity no matter how many
    times the iteration is applied. The beauty of the Mandelbrot set lies in its
    complex and boundary-defining structure, which reveals an infinitely detailed and
    varied pattern upon magnification.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation
        >>> import numpy as np
        >>> from umf.functions.theory.pathological import MandelbrotsFractalFunction
        >>> x = np.linspace(-2, 2, 1000)
        >>> rf = MandelbrotsFractalFunction(x, max_iter=50)()
        >>> fig = plt.figure(figsize=(10, 6))
        >>> ax = fig.add_subplot(111)
        >>> ax_return= ax.imshow(
        ...                rf.result,
        ...                cmap='seismic',
        ...                extent=(-2, 2, -2, 2),
        ...                interpolation='bilinear',
        ...               )
        >>> def update(frame: int) -> tuple:
        ...     zoom_factor = frame / 50.0
        ...     ax.set_xlim(-2 , 2 - zoom_factor)
        ...     ax.set_ylim(-2 + zoom_factor, 2 - zoom_factor)
        ...     return (ax_return,)
        >>> ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)
        >>> ani.save('MandelbrotsFractalFunction.gif', writer='imagemagick', fps=10)

    Notes:
        The Mandelbrot set is the set of complex numbers $\(c\)$ for which the function
        $\(f_c(z) = z^2 + c\)$ does not diverge when iterated from $\(z = 0\)$. The
        Mandelbrot set is a fractal, meaning that it exhibits self-similarity at
        different scales.

        $$
        M = \{c \in \mathbb{C} : \lim_{n \to \infty} |z_n| \leq 2\}
        $$

        where $(z_{n+1} = z_n^2 + c)$ and $(z_0 = 0)$.

        > Reference: https://en.wikipedia.org/wiki/Mandelbrot_set

    Args:
        *x (UniversalArray): The coordinates in the complex plane where the function
            will be evaluated.
        max_iter (int, optional): The maximum number of iterations to perform. Defaults
            to 100.
        escape_threshold (float, optional): The threshold for escaping the fractal
            region. Defaults to 2.0.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 100,
        escape_threshold: float = 2.0,
    ) -> None:
        """Initialize the Mandelbrot's Fractal function."""
        super().__init__(*x, n_0=0, n_1=max_iter)
        self.escape_threshold = escape_threshold

    @property
    def __eval__(self) -> UniversalArray:
        """Calculate the Mandelbrot's Fractal function.

        Returns:
            UniversalArray: The Mandelbrot's Fractal function.
        """
        height = len(self._x)
        width = 2 * len(self._x) // 3
        mandelbrot_set = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                c = complex(self._x[j], self._x[i])
                z = 0
                for k in range(self.n_1):
                    z: complex = z**2 + c
                    if abs(z) > self.escape_threshold:
                        mandelbrot_set[i, j] = k
                        break
                else:
                    mandelbrot_set[i, j] = self.n_1
        return mandelbrot_set


class BesicovitchFunction(PathologicalPure):
    r"""Besicovitch Function.

    The Besicovitch function is a fractal-like continuous function defined on the
    real line. It is also known as the Besicovitch-Eggleston function.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation
        >>> import numpy as np
        >>> from umf.functions.theory.pathological import BesicovitchFunction
        >>> x = np.linspace(0, 1, 10000)
        >>> wf = BesicovitchFunction(x, n=30)()
        >>> fig = plt.figure(figsize=(10, 6))
        >>> ax = fig.add_subplot(111)
        >>> (ax_return,) = ax.plot(x, wf.result)
        >>> def update(frame: int) -> tuple:
        ...     zoom_factor = frame / 100
        ...     ax.set_xlim(0 + zoom_factor/2 ,1 - zoom_factor/2)
        ...     ax.set_ylim(-1 + zoom_factor, 1 - zoom_factor)
        ...     return (ax_return,)
        >>> ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)
        >>> ani.save('BesicovitchFunction.gif', writer='imagemagick', fps=10)

    Notes:
        The Besicovitch function is a fractal-like continuous function defined on the
        real line. It is also known as the Besicovitch-Eggleston function.

        $$
        B(x) = \sum_{n=1}^{n_1} \frac{\sin(\pi 2^n x)}{\mu^n}
        $$

        with constraint $\mu \geq 1$.

        > Reference: https://en.wikipedia.org/wiki/Besicovitch

    Args:
        *x (UniversalArray): Input values for which the Weierstrass function is
            evaluated.
        n (int, optional): Number of iterations to compute the Besicovitch function.
            Defaults to 10.
        mu (float, optional): A parameter of the function that controls the frequency
            of oscillations. Must satisfy the condition b >= 1. Defaults to 2.

    Raises:
        NotLargerThanAnyError: If $\mu$ is not larger than 1.
    """

    def __init__(self, *x: UniversalArray, n: int = 10, mu: float = 2) -> None:
        """Initialize the Besicovitch function."""
        if mu < 1:
            raise NotLargerThanAnyError(var_number="mu", number=mu, minimum=1)
        super().__init__(*x, n_0=1, n_1=n)
        self.mu = mu

    @property
    def __eval__(self) -> UniversalArray:
        """Calculate the Besicovitch function."""
        result = np.zeros_like(self._x, dtype=float)
        for n in range(1, self.n_1 + 1):
            result += np.sin(2**n * np.pi * self._x) / (self.mu**n)
        return result
