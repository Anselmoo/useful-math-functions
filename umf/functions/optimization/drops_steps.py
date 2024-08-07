"""Drop steps functions for the useful-math-functions library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from numpy import matlib

from umf.constants.dimensions import __2d__
from umf.constants.exceptions import OutOfDimensionError
from umf.meta.api import MinimaAPI
from umf.meta.functions import OptFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = ["DeJongN5Function", "EasomFunction", "MichalewiczFunction"]


class DeJongN5Function(OptFunction):
    r"""De Jong N.5 Function.

    The De Jong N.5 function is a two-dimensional function with a single global
    minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.drops_steps import DeJongN5Function
        >>> x = np.linspace(-65.536, 65.536, 1000)
        >>> y = np.linspace(-65.536, 65.536, 1000)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = DeJongN5Function(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("DeJongN5Function.png", dpi=300, transparent=True)

    Notes:
        The De Jong N.5 function is defined as:

        $$
            f(x, y) = \left(
                0.0002 + \sum_{i=1}^{25}
                \frac{1}{
                    i + \left(x_1 - a_{1i} \right)^6 + \left(x_2 - a_{2i} \right)^6
                }
            \right)^{-1}
        $$

        where

        $$
            a = \left(
                \begin{matrix}
                -32 & -16 & 0   & 16  & 32  & ... & -16 & 0 & 16 & 32 \\
                -32 & -32 & -32 & -32 & -32 & ... & 32 & 32 & 32 & 32
                \end{matrix}
                \right)
        $$

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/dejong5.html).


    Args:
        *x (UniversalArray): Input data, which can be one, two, three, or higher
             adddimensional.
        A (UniversalArray, optional): Elements of the matrix a, which has to become are
            2-dimensional with shape (2, 25). Defaults to None.
    """

    def __init__(
        self,
        *x: UniversalArray,
        A: UniversalArray | None = None,
    ) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="De Jong N.5",
                dimension=__2d__,
            )
        super().__init__(*x)

        if A is None:
            row_1 = matlib.repmat(np.arange(-32, 33, 16), 1, 5)
            row_2 = matlib.repmat(np.arange(-32, 33, 16), 5, 1).T.ravel()
            self.a_matrix = np.vstack((row_1, row_2))
        elif A.shape != (2, 25):
            msg = "The shape of a has to be (2, 25)."
            raise ValueError(msg)
        else:
            self.a_matrix = A

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the De Jong N.5 function.

        Returns:
            UniversalArray: The value of the De Jong N.5 function.
        """
        x_1 = self._x[0]
        x_2 = self._x[1]

        a_1 = self.a_matrix[0, :]
        a_2 = self.a_matrix[1, :]

        sum_ = np.zeros_like(x_1)

        for i in range(25):
            sum_ += 1 / ((i + 1) + (x_1 - a_1[i]) ** 6 + (x_2 - a_2[i]) ** 6)

        return (0.0002 + sum_) ** -1

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the De Jong N.5 function.

        Returns:
            MinimaAPI: The minima of the De Jong N.5 function.
        """
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class EasomFunction(OptFunction):
    r"""Easom Function.

    The Easom function is a two-dimensional function with a single global minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.drops_steps import EasomFunction
        >>> x = np.linspace(-100, 100, 1000)
        >>> y = np.linspace(-100, 100, 1000)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = EasomFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("EasomFunction.png", dpi=300, transparent=True)

    Notes:
        The Easom function is defined as:

        $$
            f(x, y) = -\cos(x)\cos(y)\exp\left(-\left(x-\pi\right)^2
            - \left(y-\pi\right)^2\right)
        $$

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/easom.html).

    Args:
        *x (UniversalArray): Input data, which can be one, two, three, or higher
             dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="Easom",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the Easom function.

        Returns:
            UniversalArray: The value of the Easom function.
        """
        x_1 = self._x[0]
        x_2 = self._x[1]

        return np.array(
            -np.cos(x_1)
            * np.cos(x_2)
            * np.exp(-((x_1 - np.pi) ** 2) - (x_2 - np.pi) ** 2),
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Easom function.

        Returns:
            MinimaAPI: The minima of the Easom function.
        """
        return MinimaAPI(
            f_x=-1.0,
            x=np.array([np.pi, np.pi]),
        )


class MichalewiczFunction(OptFunction):
    r"""Michalewicz Function.

    The Michalewicz function is a multi-dimensional function with a single
    global minimum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.drops_steps import MichalewiczFunction
        >>> x = np.linspace(0, np.pi, 1000)
        >>> y = np.linspace(0, np.pi, 1000)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = MichalewiczFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("MichalewiczFunction.png", dpi=300, transparent=True)

    Notes:
        The Michalewicz function is defined as:

        $$
            f(x, y) = -\sum_{i=1}^{2}\sin(x_i)\sin^2\left(\frac{i x_i^2}{\pi}\right)
        $$

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/michal.html).

    Args:
        *x (UniversalArray): Input data, which can be one, two, three, or higher
             dimensional.
        m (int, optional): The m parameter. Defaults to 10.
    """

    def __init__(self, *x: UniversalArray, m: int = 10) -> None:
        """Initialize the function."""
        super().__init__(*x)

        self.m = m

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the Michalewicz function.

        Returns:
            UniversalArray: The value of the Michalewicz function.
        """
        sum_ = np.zeros_like(self._x[0])

        for i, x_i in enumerate(self._x, start=1):
            sum_ += np.sin(x_i) * np.sin((i * x_i**2) / np.pi) ** (2 * self.m)

        return -sum_

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Michalewicz function.

         The minima of the Michalewicz function is not unique and depends on the
            m parameter and dimensionality of the function.

        Returns:
            MinimaAPI: The minima of the Michalewicz function.
        """
        return MinimaAPI(
            f_x=-1.8013,
            x=(2.20, 1.57),
        )
