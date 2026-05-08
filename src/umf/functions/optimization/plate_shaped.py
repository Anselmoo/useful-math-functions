"""Plate shaped functions for the useful-math-functions library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.dimensions import __2d__
from umf.constants.exceptions import OutOfDimensionError
from umf.meta.api import MinimaAPI
from umf.meta.functions import OptFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray
    from umf.types.static_types import UniversalArrayTuple


__all__: list[str] = [
    "BoothFunction",
    "MatyasFunction",
    "McCormickFunction",
    "PowerSumFunction",
    "ZakharovFunction",
    "ZettlFunction",
]


class BoothFunction(OptFunction):
    r"""Booth Function.

    The Booth function is a two-dimensional function with a single global
    minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.plate_shaped import BoothFunction
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = BoothFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("BoothFunction.png", dpi=300, transparent=True)

    Notes:
        The Booth function is defined as:

        $$
            f(x) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/booth.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            msg = f"Expected 2 arguments, but got {len(x)}."
            raise ValueError(msg)
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the Booth function at x."""
        x_1 = self._x[0]
        x_2 = self._x[1]
        return (x_1 + 2 * x_2 - 7) ** 2 + (2 * x_1 + x_2 - 5) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Booth function."""
        return MinimaAPI(
            f_x=0.0,
            x=(1.0, 3.0),
        )


class MatyasFunction(OptFunction):
    r"""Matyas Function.

    The Matyas function is a two-dimensional function with a single global
    minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.plate_shaped import MatyasFunction
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = MatyasFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("MatyasFunction.png", dpi=300, transparent=True)

    Notes:
        The Matyas function is defined as:

        $$
            f(x) = 0.26(x_1^2 + x_2^2) - 0.48x_1 x_2
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/matya.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="Matyas",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the Matyas function at x."""
        x_1 = self._x[0]
        x_2 = self._x[1]
        return 0.26 * (x_1**2 + x_2**2) - 0.48 * x_1 * x_2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Matyas function."""
        return MinimaAPI(
            f_x=0.0,
            x=(0.0, 0.0),
        )


class McCormickFunction(OptFunction):
    r"""McCormick Function.

    The McCormick function is a two-dimensional function with a single global
    minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.plate_shaped import McCormickFunction
        >>> x = np.linspace(-1.5, 4, 100)
        >>> y = np.linspace(-3, 4, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = McCormickFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("McCormickFunction.png", dpi=300, transparent=True)

    Notes:
        The McCormick function is defined as:

        $$
            f(x) = \sin(x_1 + x_2) + (x_1 - x_2)^2 - 1.5x_1 + 2.5x_2 + 1
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/mccorm.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.
    """

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the McCormick function at x."""
        x_1 = self._x[0]
        x_2 = self._x[1]
        return np.sin(x_1 + x_2) + (x_1 - x_2) ** 2 - 1.5 * x_1 + 2.5 * x_2 + 1

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the McCormick function."""
        return MinimaAPI(
            f_x=-1.9133,
            x=(-0.54719, -1.54719),
        )


class PowerSumFunction(OptFunction):
    r"""Power Sum Function.

    The Power Sum function is a two-dimensional function with a single global
    minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.plate_shaped import PowerSumFunction
        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = PowerSumFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("PowerSumFunction.png", dpi=300, transparent=True)

    Notes:
        The Power Sum function is defined as:

        $$
            f(x) = x_1^2 + x_2^2 + x_1^4 + x_2^4
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/powersum.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="Power Sum",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArrayTuple:
        """Evaluate the Power Sum function at x."""
        x_1 = self._x[0]
        x_2 = self._x[1]
        return x_1**2 + x_2**2 + x_1**4 + x_2**4

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Power Sum function."""
        return MinimaAPI(
            f_x=0.0,
            x=(0.0, 0.0),
        )


class ZakharovFunction(OptFunction):
    r"""Zakharov Function.

    The Zakharov function is a two-dimensional function with a single global
    minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.plate_shaped import ZakharovFunction
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = ZakharovFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("ZakharovFunction.png", dpi=300, transparent=True)

    Notes:
        The Zakharov function is defined as:

        $$
            f(x) = \sum_{i=1}^2 x_i^2 + \left(\sum_{i=1}^2 0.5ix_i\right)^2
            + \left(\sum_{i=1}^2 0.5ix_i\right)^4
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/zakharov.html).

    Args:
        *x (UniversalArray): Input data, which has to be two-dimensional.
    """

    @property
    def __eval__(self) -> UniversalArrayTuple:
        """Evaluate the Zakharov function at x."""
        sum_1 = np.zeros_like(self._x[0])
        sum_2 = np.zeros_like(self._x[0])

        for i in range(self.dimension):
            sum_1 += self._x[i] ** 2
            sum_2 += 0.5 * (i + 1) * self._x[i]

        return sum_1 + sum_2**2 + sum_2**4

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Zakharov function."""
        return MinimaAPI(
            f_x=0.0,
            x=tuple(np.zeros_like(self.dimension)),
        )


class ZettlFunction(OptFunction):
    r"""Zettl function.

    The Zettl function is a D-dimensional function with multimodal structure and sharp
    peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.plate_shaped import ZettlFunction
        >>> x = np.linspace(-5, 10, 100)
        >>> y = np.linspace(-5, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = ZettlFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("ZettlFunction.png", dpi=300, transparent=True)

    Notes:
        The Zettl function is defined as:

        $$
            f(x) = (x_1^2 + x_2^2 - 2 x_1)^2 + \frac{1}{4} x_1
        $$

        > Reference: Original implementation can be found
        > [here](https://www.sfu.ca/~ssurjano/zettl.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            msg = f"Expected 2 arguments, but got {len(x)}."
            raise ValueError(msg)
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Zettl function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        return (
            self._x[0] ** 2 + self._x[1] ** 2 - 2 * self._x[0]
        ) ** 2 + 0.25 * self._x[0]

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Zettl function.

        Returns:
            MinimaAPI: Minima of the Zettl function.
        """
        return MinimaAPI(f_x=0, x=(0, 0))
