"""Valley shaped functions for the useful-math-functions library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.dimensions import __2d__
from umf.constants.exceptions import OutOfDimensionError
from umf.constants.exceptions import TooSmallDimensionError
from umf.meta.api import MinimaAPI
from umf.meta.functions import OptFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__: list[str] = [
    "ThreeHumpCamelFunction",
    "SixHumpCamelFunction",
    "DixonPriceFunction",
    "RosenbrockFunction",
]


class ThreeHumpCamelFunction(OptFunction):
    r"""Three-hump camel function.

    The three-hump camel function is a two-dimensional function with three
    minima, where one of the minima is a global minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.valley_shaped import ThreeHumpCamelFunction
        >>> x = np.linspace(-5, 5, 100)
        >>> y = np.linspace(-5, 5, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = ThreeHumpCamelFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("ThreeHumpCamelFunction.png", dpi=300, transparent=True)

    Notes:
        The three-hump camel function is defined as:

        $$
        f(x) = 2x_1^2 - 1.05x_1^4 + \frac{x_1^6}{6} + x_1x_2 + x_2^2
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/camel3.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.

    Raises:
        OutOfDimensionError: If the dimension of the input data is not 2.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the three-hump camel function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="ThreeHumpCamel",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the three-hump camel function at x."""
        x_1 = self._x[0]
        x_2 = self._x[1]
        return 2 * x_1**2 - 1.05 * x_1**4 + (x_1**6) / 6 + x_1 * x_2 + x_2**2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the three-hump camel function."""
        return MinimaAPI(
            f_x=np.array([0.0, 0.0]),
            x=tuple(np.array([0.0])),
        )


class SixHumpCamelFunction(OptFunction):
    r"""Six-hump camel function.

    The six-hump camel function is a two-dimensional function with six
    minima, where two of them are global minima.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.valley_shaped import SixHumpCamelFunction
        >>> x = np.linspace(-3, 3, 100)
        >>> y = np.linspace(-2, 2, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = SixHumpCamelFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("SixHumpCamelFunction.png", dpi=300, transparent=True)

    Notes:
        The six-hump camel function is defined as:

        $$
        f(x) = (4 - 2.1x_1^2 + \frac{x_1^4}{3})x_1^2 + x_1x_2 + (-4 + 4x_2^2)x_2^2
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/camel6.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.

    Raises:
        OutOfDimensionError: If the dimension of the input data is not 2.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the six-hump camel function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="SixHumpCamel",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the six-hump camel function at x."""
        x_1 = self._x[0]
        x_2 = self._x[1]
        return (
            (4 - 2.1 * x_1**2 + (x_1**4) / 3) * x_1**2
            + x_1 * x_2
            + (-4 + 4 * x_2**2) * x_2**2
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the six-hump camel function."""
        return MinimaAPI(
            f_x=-1.031628453489877,
            x=tuple(np.array([0.0898, -0.7126], [-0.0898, 0.7126])),
        )


class DixonPriceFunction(OptFunction):
    r"""Dixon-Price function.

    The Dixon-Price function is a multi-dimensional function with a single
    global minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.valley_shaped import DixonPriceFunction
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = DixonPriceFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("DixonPriceFunction.png", dpi=300, transparent=True)

    Notes:
        The Dixon-Price function is defined as:

        $$
        f(x) = (x_1 - 1)^2 + \sum_{i=2}^n \left[ i(x_i - x_{i-1}^2)^2 \right]
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/dixonpr.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.

    Raises:
        TooSmallDimensionError: If the dimension of the input data is smaller than 2.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the Dixon-Price function."""
        if len(x) < __2d__:
            raise TooSmallDimensionError(
                function_name="DixonPrice",
                dimension=__2d__,
                len_x=len(x),
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Dixon-Price function at x."""
        x = self._x

        return (x[0] - 1) ** 2 + sum(
            (i + 1) * (x[i] - x[i - 1] ** 2) ** 2 for i in range(1, len(x))
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the zero function."""
        return MinimaAPI(
            f_x=0.0,
            x=tuple(np.array([2 ** (-i) for i in range(1, len(self._x) + 1)])),
        )


class RosenbrockFunction(OptFunction):
    r"""Rosenbrock function.

    The Rosenbrock function is a multi-dimensional function with a single
    global minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.valley_shaped import RosenbrockFunction
        >>> x = np.linspace(-2, 2, 100)
        >>> y = np.linspace(-1, 3, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = RosenbrockFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("RosenbrockFunction.png", dpi=300, transparent=True)

    Notes:
        The Rosenbrock function is defined as:

        $$
        f(x) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/rosen.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.

    Raises:
        TooSmallDimensionError: If the dimension of the input data is smaller than 2.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the Rosenbrock function."""
        if len(x) < __2d__:
            raise TooSmallDimensionError(
                function_name="Rosenbrock",
                dimension=__2d__,
                len_x=len(x),
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Rosenbrock function at x."""
        x = self._x

        return np.array(
            sum(
                100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)
            ),
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the zero function."""
        return MinimaAPI(
            f_x=0.0,
            x=tuple(np.array([1.0 for _ in range(len(self._x))])),
        )
