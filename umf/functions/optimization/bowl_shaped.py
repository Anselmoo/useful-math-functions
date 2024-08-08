"""Bowl shaped functions for the useful-math-functions library."""

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
    "PermBetaDFunction",
    "TridFunction",
    "SumSquaresFunction",
    "SumOfDifferentPowersFunction",
    "ZirilliFunction",
    "SphereFunction",
    "RotatedHyperEllipseFunction",
    "PermFunction",
    "BohachevskyFunctionType1",
    "BohachevskyFunctionType2",
    "BohachevskyFunctionType3",
]


class PermBetaDFunction(OptFunction):
    r"""Perm Beta D function.

    The Perm Beta D function is a D-dimensional function with multimodal structure
    and sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import PermBetaDFunction
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = PermBetaDFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("PermBetaDFunction.png", dpi=300, transparent=True)

    Notes:
        The Perm Beta D function is defined as:

        $$
            f(x) =  \sum_{i=1}^D \left( \sum_{j=1}^D \frac{1}{j +
            \beta} \left( \frac{x_j}{j} \right)^i  -1 \right)^2
        $$

        with constant $\beta = 0.5$ and $D$ the dimension of the input. The hypercube
        of the function is defined as $x_i \in [-d, d]$ for all $i$.

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/sumpow.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Perm Beta D function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        beta = 0.5

        outer_sum = np.zeros(self._x[0].shape)

        for i in range(1, self.dimension + 1):
            inner_sum = sum(
                (j**i + beta) * ((self._x[j - 1] / j) ** i - 1)
                for j in range(1, self.dimension + 1)
            )
            outer_sum += inner_sum**2
        return outer_sum

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Perm Beta D function.

        Returns:
            MinimaAPI: Minima of the Perm Beta D function.
        """
        return MinimaAPI(f_x=0, x=tuple(np.arange(1, self.dimension + 1)))


class TridFunction(OptFunction):
    r"""Trid function.

    The Trid function is a D-dimensional function with multimodal structure and sharp
    peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import TridFunction
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = TridFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("TridFunction.png", dpi=300, transparent=True)

    Notes:
        The Trid function is defined as:

        $$
            f(x) = \sum_{i=1}^D \left( x_i - 1 \right)^2 - \sum_{i=2}^D x_i x_{i-1}
        $$

        with $D$ the dimension of the input. The hypercube of the function is defined
        as $x_i \in [-d, d]$ for all $i$.

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/trid.html)

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Trid function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        outer_sum = np.zeros(self._x[0].shape)

        for i in range(1, self.dimension + 1):
            inner_sum = (self._x[i - 1] - 1) ** 2 - self._x[i - 1] * self._x[i - 2]
            outer_sum += inner_sum
        return outer_sum

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Trid function.

        Returns:
            MinimaAPI: Minima of the Trid function.
        """
        return MinimaAPI(
            f_x=-self.dimension * (self.dimension + 4) * (self.dimension - 1) / 6,
            x=tuple(i * (self.dimension + 1 - i) for i in range(1, self.dimension + 1)),
        )


class SumSquaresFunction(OptFunction):
    r"""Sum squares function.

    The Sum squares function is a D-dimensional function with multimodal structure and
    sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import SumSquaresFunction
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = SumSquaresFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("SumSquaresFunction.png", dpi=300, transparent=True)

    Notes:
        The Sum squares function is defined as:

        $$
            f(x) = \sum_{i=1}^D i x_i^2
        $$

        with $D$ the dimension of the input. The hypercube of the function is defined as
        $x_i \in [-d, d]$ for all $i$.

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/sumsqu.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Sum squares function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        outer_sum = np.zeros(self._x[0].shape)

        for i in range(1, self.dimension + 1):
            inner_sum = i * self._x[i - 1] ** 2
            outer_sum += inner_sum
        return outer_sum

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Sum squares function.

        Returns:
            MinimaAPI: Minima of the Sum squares function.
        """
        return MinimaAPI(f_x=0, x=tuple(np.zeros(self.dimension)))


class SumOfDifferentPowersFunction(OptFunction):
    r"""Sum of different powers function.

    The Sum of different powers function is a D-dimensional function with multimodal
    structure and sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import (
        ... SumOfDifferentPowersFunction
        ... )
        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = SumOfDifferentPowersFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("SumOfDifferentPowersFunction.png", dpi=300, transparent=True)

    Notes:
        The Sum of different powers function is defined as:

        $$
            f(x) = \sum_{i=1}^D \left| x_i \right|^{i+1}
        $$

        with $D$ the dimension of the input. The hypercube of the function is defined as
        $x_i \in [-d, d]$ for all $i$.

        _Reference: Original implementation can be found
        [here](http://www.sfu.ca/~ssurjano/sumpow.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Sum of different powers function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        outer_sum = np.zeros(self._x[0].shape)

        for i in range(1, self.dimension + 1):
            inner_sum = abs(self._x[i - 1]) ** (i + 1)
            outer_sum += inner_sum
        return outer_sum

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Sum of different powers function.

        Returns:
            MinimaAPI: Minima of the Sum of different powers function.
        """
        return MinimaAPI(f_x=0, x=tuple(np.zeros(self.dimension)))


class ZirilliFunction(OptFunction):
    r"""Zirilli function.

    The Zirilli function is a 2D-dimensional function with multimodal structure and
    sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import ZirilliFunction
        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = ZirilliFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("ZirilliFunction.png", dpi=300, transparent=True)

    Notes:
        The Zirilli function is defined as:

        $$
            f(x) = 0.25 x_1^4 - 0.5 x_1^2 + 0.1 x_1 + 0.5 x_2^2
        $$


    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.

    Raises:
        OutOfDimensionError: If the dimension of the input data is not 2.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="Zirilli",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Zirilli function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        x_1 = self._x[0]
        x_2 = self._x[1]
        return 0.25 * x_1**4 - 0.5 * x_1**2 + 0.1 * x_1 + 0.5 * x_2**2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Zirilli function.

        Returns:
            MinimaAPI: Minima of the Zirilli function.
        """
        return MinimaAPI(
            f_x=-0.352386073800034,
            x=(-1.046680576580755, 0),
        )


class SphereFunction(OptFunction):
    r"""Sphere function.

    The Sphere function is a D-dimensional function with multimodal structure and
    sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import SphereFunction
        >>> x = np.linspace(-2.5, 2.5, 100)
        >>> y = np.linspace(-2.5, 2.5, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = SphereFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("SphereFunction.png", dpi=300, transparent=True)

    Notes:
        The Sphere function is defined as:

        $$
            f(x) = \sum_{i=1}^D x_i^2
        $$

        with $D$ the dimension of the input. The hypercube of the function is defined as
        $x_i \in [-d, d]$ for all $i$.

        Reference: Original implementation can be found
        [here](http://www.sfu.ca/~ssurjano/spheref.html).

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Sphere function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        return np.sum(np.power(self._x, 2), axis=0)

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Sphere function.

        Returns:
            MinimaAPI: Minima of the Sphere function.
        """
        return MinimaAPI(f_x=0, x=tuple(np.zeros(self.dimension)))


class RotatedHyperEllipseFunction(OptFunction):
    r"""Rotated hyper-ellipse function.

    The Rotated hyper-ellipse function is a D-dimensional function with multimodal
    structure and sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import (
        ... RotatedHyperEllipseFunction
        ... )
        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = RotatedHyperEllipseFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("RotatedHyperEllipseFunction.png", dpi=300, transparent=True)

    Notes:
        The Rotated hyper-ellipse function is defined as:

        $$
            f(x) = \sum_{i=1}^D \left( \sum_{j=1}^D a_{ij} x_j^2 \right)^2
        $$

        with $D$ the dimension of the input and $a_{ij}$ the rotation matrix. The
        hypercube of the function is defined as $x_i \in [-d, d]$ for all $i$.

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/rothyp.html)

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.

    Raises:
        OutOfDimensionError: If the dimension of the input data is not 2.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="RotatedHyperEllipse",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Rotated hyper-ellipse function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        outer_sum = np.zeros(self._x[0].shape)
        for i in range(self.dimension):
            inner_sum = np.zeros(self._x[0].shape)
            for j in range(i + 1):
                inner_sum += self._x[j] ** 2
            outer_sum = outer_sum + inner_sum
        return outer_sum

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Rotated hyper-ellipse function.

        Returns:
            MinimaAPI: Minima of the Rotated hyper-ellipse function.
        """
        return MinimaAPI(f_x=0, x=tuple(np.zeros(self.dimension)))


class PermFunction(OptFunction):
    r"""Perm function.

    The Perm function is a D-dimensional function with multimodal structure and
    sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import PermFunction
        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = PermFunction(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("PermFunction.png", dpi=300, transparent=True)

    Notes:
        The Perm function is defined as:

        $$
            f(x) = \sum_{i=1}^D \left( \sum_{j=1}^D j^{i+1} (j + \beta)
              \left( \frac{x_j}{j} \right)^i  -1 \right)^2
        $$

        with constant $\beta = 10$ and $D$ the dimension of the input. The hypercube
        of the function is defined as $x_i \in [-d, d]$ for all $i$.


    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
        beta (float, optional): The beta parameter. Defaults to 10.
    """

    def __init__(self, *x: UniversalArray, beta: float = 10) -> None:
        """Initialize the function."""
        super().__init__(*x)
        self.beta = beta

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Perm function at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        outer_sum = np.zeros(self._x[0].shape)

        for i in range(1, self.dimension + 1):
            inner_sum = sum(
                j ** (i + 1) * (j + self.beta) * ((self._x[j - 1] / j) ** i - 1)
                for j in range(1, self.dimension + 1)
            )
            outer_sum += inner_sum**2
        return outer_sum

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Perm function.

        Returns:
            MinimaAPI: Minima of the Perm function.
        """
        return MinimaAPI(f_x=0, x=tuple(np.arange(1, self.dimension + 1)))


class BohachevskyFunctionType1(OptFunction):
    r"""Bohachevsky function type 1.

    The Bohachevsky function type 1 is a 2D-dimensional function with multimodal
    structure and sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import BohachevskyFunctionType1
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = BohachevskyFunctionType1(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("BohachevskyFunctionType1.png", dpi=300, transparent=True)

    Notes:
        The Bohachevsky function type 1 is defined as:

        $$
            f(x) = x_1^2 + 2 x_2^2 - 0.3 \cos(3 \pi x_1) - 0.4 \cos(4 \pi x_2) + 0.7
        $$

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/boha.html)

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="BohachevskyType1",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Bohachevsky function type 1 at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        x_1 = self._x[0]
        x_2 = self._x[1]
        return (
            x_1**2
            + 2 * x_2**2
            - 0.3 * np.cos(3 * np.pi * x_1)
            - 0.4 * np.cos(4 * np.pi * x_2)
            + 0.7
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Bohachevsky function type 1.

        Returns:
            MinimaAPI: Minima of the Bohachevsky function type 1.
        """
        return MinimaAPI(f_x=0, x=(0, 0))


class BohachevskyFunctionType2(OptFunction):
    r"""Bohachevsky function tye 2.

    The Bohachevsky function type 2 is a 2D-dimensional function with multimodal
    structure and sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import BohachevskyFunctionType2
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = BohachevskyFunctionType2(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("BohachevskyFunctionType2.png", dpi=300, transparent=True)

    Notes:
        The Bohachevsky function type 2 is defined as:

        $$
            f(x) = x_1^2 + 2 x_2^2 - 0.3 \cos(3 \pi x_1) \cos(4 \pi x_2) + 0.3
        $$

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/boha.html)

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="BohachevskyType2",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Bohachevsky function type 2 at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        x_1 = self._x[0]
        x_2 = self._x[1]
        return (
            x_1**2
            + 2 * x_2**2
            - 0.3 * np.cos(3 * np.pi * x_1) * np.cos(4 * np.pi * x_2)
            + 0.3
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Bohachevsky function type 2.

        Returns:
            MinimaAPI: Minima of the Bohachevsky function type 2.
        """
        return MinimaAPI(f_x=0, x=(0, 0))


class BohachevskyFunctionType3(OptFunction):
    r"""Bohachevsky function type 3.

    The Bohachevsky function type 3 is a 2D-dimensional function with multimodal
    structure and sharp peaks.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.optimization.bowl_shaped import BohachevskyFunctionType3
        >>> x = np.linspace(-10, 10, 100)
        >>> y = np.linspace(-10, 10, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = BohachevskyFunctionType3(X, Y).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> _ = ax.plot_surface(X, Y, Z, cmap="viridis")
        >>> plt.savefig("BohachevskyFunctionType3.png", dpi=300, transparent=True)

    Notes:
        The Bohachevsky function type 3 is defined as:

        $$
            f(x) = x_1^2 + 2 x_2^2 - 0.3 \cos(3 \pi x_1 + 4 \pi x_2) + 0.3
        $$

        > Reference: Original implementation can be found
        > [here](http://www.sfu.ca/~ssurjano/boha.html)

    Args:
        *x (UniversalArray): Input data, which has to be two dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if len(x) != __2d__:
            raise OutOfDimensionError(
                function_name="BohachevskyType3",
                dimension=__2d__,
            )
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate Bohachevsky function type 3 at x.

        Returns:
            UniversalArray: Evaluated function value.
        """
        x_1 = self._x[0]
        x_2 = self._x[1]
        return (
            x_1**2 + 2 * x_2**2 - 0.3 * np.cos(3 * np.pi * x_1 + 4 * np.pi * x_2) + 0.3
        )

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the minima of the Bohachevsky function type 3.

        Returns:
            MinimaAPI: Minima of the Bohachevsky function type 3.
        """
        return MinimaAPI(f_x=0, x=(0, 0))
