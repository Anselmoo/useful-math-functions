"""Special optimization functions for the useful-math-functions library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.dimensions import __2d__
from umf.constants.exceptions import OutOfDimensionError
from umf.meta.api import MinimaAPI
from umf.meta.functions import OptFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__: list[str] = [
    "BealeFunction",
    "BraninFunction",
    "StyblinskiTangFunction",
    "GoldsteinPriceFunction",
    "GoldsteinPriceLogFunction",
]


class BealeFunction(OptFunction):
    r"""Beale function.

    The Beale function is a two-dimensional function with multimodal structure and
    sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.special import BealeFunction
        >>> x = np.linspace(-4.5, 4.5, 100)
        >>> y = np.linspace(-4.5, 4.5, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = BealeFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("BealeFunction.png", dpi=300, transparent=True)

    Notes:
        The Beale function is defined as:

        $$
            f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/beale.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.

    Raises:
        OutOfDimensionError: If the dimension of the input data is not 2.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="Beale",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Beale function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        x, y = self._x[0], self._x[1]
        return (
            (1.5 - x + x * y) ** 2
            + (2.25 - x + x * y**2) ** 2
            + (2.625 - x + x * y**3) ** 2
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Beale function.

        Returns:
            MinimaAPI: Minima of the Beale function.
        """
        return MinimaAPI(f_x=3, x=(0.5))


class BraninFunction(OptFunction):
    r"""Branin function.

    The Branin function is a two-dimensional function with multimodal structure and
    sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.special import BraninFunction
        >>> x = np.linspace(-5, 10, 100)
        >>> y = np.linspace(0, 15, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = BraninFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("BraninFunction.png", dpi=300, transparent=True)

    Notes:
        The Branin function is defined as:

        $$
            f(x, y) = a(y - bx^2 + cx - r)^2 + s(1 - t) \cos(y) + s
        $$

        where

        $$
            a = 1, b = 5.1/(4\pi^2), c = 5/\pi, r = 6, s = 10, t = 1/(8\pi)
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/branin.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.

    Raises:
        OutOfDimensionError: If the dimension of the input data is not 2.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="Branin",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Branin function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        x, y = self._x[0], self._x[1]
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * np.cos(y) + s

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Branin function.

        Returns:
            MinimaAPI: Minima of the Branin function.
        """
        return MinimaAPI(
            f_x=0.397887,
            x=tuple(
                np.array([-np.pi, 12.275]),
                np.array([np.pi, 2.275]),
                np.array([np.pi, 9.42478]),
                np.array([9.42478, 2.475]),
            ),
        )


class StyblinskiTangFunction(OptFunction):
    r"""Styblinski-Tang function.

    The Styblinski-Tang function is a D-dimensional function.


    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.special import StyblinskiTangFunction
        >>> x = np.linspace(-5, 5, 100)
        >>> y = np.linspace(-5, 5, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = StyblinskiTangFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("StyblinskiTangFunction.png", dpi=300, transparent=True)

    Notes:
        The Styblinski-Tang function is defined as:

        $$
            f(x) = \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16 x_i^2 + 5 x_i \right)
        $$

        with $D$ the dimension of the input. The hypercube of the function is defined
        as $x_i \in [-5, 5]$ for all $i$.

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/stybtang.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Styblinski-Tang function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        return np.array(
            0.5
            * sum(
                (self._x[i - 1] ** 4 - 16 * self._x[i - 1] ** 2 + 5 * self._x[i - 1])
                for i in range(1, self.dimension + 1)
            ),
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Styblinski-Tang function.

        Returns:
            MinimaAPI: Minima of the Styblinski-Tang function.
        """
        return MinimaAPI(
            f_x=-39.16616570377142,
            x=tuple(np.ones(self.dimension) * -2.903534),
        )


# GOLDSTEIN-PRICE FUNCTION
class GoldsteinPriceFunction(OptFunction):
    r"""Goldstein-Price function.

    The Goldstein-Price function is a two-dimensional function.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.special import GoldsteinPriceFunction
        >>> x = np.linspace(-2, 2, 100)
        >>> y = np.linspace(-2, 2, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = GoldsteinPriceFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("GoldsteinPriceFunction.png", dpi=300, transparent=True)

    Notes:
        The Goldstein-Price function is defined as:

        $$
            f(x) = \left( 1 + \left( x_1 + x_2 + 1 \right)^2
            \left( 19 - 14 x_1 + 3 x_1^2 - 14 x_2 + 6 x_1 x_2 + 3 x_2^2 \right) \right)
            \left( 30 + \left( 2 x - 3 x_2 \right)^2
            \left( 18 - 32 x_1 + 12 x^2 + 48 x_2 - 36 x_1 x_2 + 27 x_2^2 \right) \right)
        $$

        The hypercube of the function is defined as $x_1, x_2 \in [-2, 2]$.

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/goldpr.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="Goldstein-Price",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Goldstein-Price function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        return (
            1
            + (self._x[0] + self._x[1] + 1) ** 2
            * (
                19
                - 14 * self._x[0]
                + 3 * self._x[0] ** 2
                - 14 * self._x[1]
                + 6 * self._x[0] * self._x[1]
                + 3 * self._x[1] ** 2
            )
        ) * (
            30
            + (2 * self._x[0] - 3 * self._x[1]) ** 2
            * (
                18
                - 32 * self._x[0]
                + 12 * self._x[0] ** 2
                + 48 * self._x[1]
                - 36 * self._x[0] * self._x[1]
                + 27 * self._x[1] ** 2
            )
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Goldstein-Price function.

        Returns:
            MinimaAPI: Minima of the Goldstein-Price function.
        """
        return MinimaAPI(f_x=3, x=tuple(np.array([0, -1])))


class GoldsteinPriceLogFunction(OptFunction):
    r"""Goldstein-Price function in logarithmic form.

    The Goldstein-Price function in logarithmic form is a two-dimensional function.
    In this form, the function offers a better conditioning by using the logarithm.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.special import GoldsteinPriceLogFunction
        >>> x = np.linspace(-2, 2, 100)
        >>> y = np.linspace(-2, 2, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = GoldsteinPriceLogFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("GoldsteinPriceLogFunction.png", dpi=300, transparent=True)

    Notes:
        The Goldstein-Price function in logarithmic form is defined as:

        $$
            f(x, y) = \frac{1}{2.427} \left[
            \log \left( 1 + \left( \bar{x}_1 + \bar{x}_2 + 1 \right)^2
            \left( 19 - 14 x + 3 \bar{x}_1^2 - 14 \bar{x}_2 + 6 \bar{x}_1 \bar{x}_2
            + 3 \bar{x}_2^2 \right) \right) + \log \left( 30 + \left( 2 \bar{x}_1 - 3
            \bar{x}_2 \right)^2 \left( 18 - 32 \bar{x}_1 + 12 \bar{x}_1^2 + 48
            \bar{x}_2 - 36 \bar{x}_1 \bar{x}_2 + 27 y^2 \right) \right) - 8.683 \right]

            \text{with } \bar{x}_{1,2} = 4 x_{1,2} - 2
        $$

        The hypercube of the function is defined as $x_1, x_2 \in [-2, 2]$.

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/goldpr.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="Goldstein-Price",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Goldstein-Price function in logarithmic form at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        x_1 = 4 * self._x[0] - 2
        x_2 = 4 * self._x[1] - 2
        return (
            1
            / 2.427
            * (
                np.log(
                    1
                    + (x_1 + x_2 + 1) ** 2
                    * (
                        19
                        - 14 * x_1
                        + 3 * x_1**2
                        - 14 * x_2
                        + 6 * x_1 * x_2
                        + 3 * x_2**2
                    ),
                )
                + np.log(
                    30
                    + (2 * x_1 - 3 * x_2) ** 2
                    * (
                        18
                        - 32 * x_1
                        + 12 * x_1**2
                        + 48 * x_2
                        - 36 * x_1 * x_2
                        + 27 * x_2**2
                    ),
                )
                - 8.683
            )
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Goldstein-Price function in logarithmic form.

        Returns:
            MinimaAPI: Minima of the Goldstein-Price function in logarithmic form.
        """
        return MinimaAPI(f_x=3, x=tuple(np.array([0, -1])))
