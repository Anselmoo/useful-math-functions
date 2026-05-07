"""Additional optimization functions for the useful-math-functions library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.dimensions import __2d__
from umf.constants.dimensions import __3d__
from umf.constants.exceptions import OutOfDimensionError
from umf.constants.exceptions import TooSmallDimensionError
from umf.meta.api import MinimaAPI
from umf.meta.functions import OptFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__: list[str] = [
    "AbsoluteBowlFunction",
    "CosineMixtureFunction",
    "CrossQuadraticFunction",
    "CubicNormFunction",
    "CubicValleyFunction",
    "EllipticAbsoluteRootFunction",
    "EllipticParaboloidFunction",
    "ExponentialNormFunction",
    "ExponentialValleyFunction",
    "OffsetQuadraticFunction",
    "QuarticBowlFunction",
    "RippleBowlFunction",
    "RotatedQuadraticFunction",
    "SaddleSuppressedFunction",
    "ShiftedHyperSphereFunction",
    "ShiftedSphere2DFunction",
    "SinusoidalRosenbrockFunction",
    "SumAndProductFunction",
    "WeightedL1L2Function",
    "CosineProductSphereFunction",
]


def _validate_two_dimensional(*x: UniversalArray, function_name: str) -> None:
    """Validate the function is called with exactly two dimensions."""
    if len(x) != __2d__:
        raise OutOfDimensionError(function_name=function_name, dimension=__2d__)


def _validate_three_or_higher(*x: UniversalArray, function_name: str) -> None:
    """Validate the function is called with at least three dimensions."""
    if len(x) < __3d__:
        raise TooSmallDimensionError(
            function_name=function_name,
            dimension=__3d__,
            len_x=len(x),
        )


class EllipticParaboloidFunction(OptFunction):
    r"""Elliptic paraboloid function.

    Notes:
        $$
        f(x, y) = x^2 + 2y^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="EllipticParaboloid")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return x_1**2 + 2 * x_2**2

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class RotatedQuadraticFunction(OptFunction):
    r"""Rotated quadratic function.

    Notes:
        $$
        f(x, y) = x^2 + xy + y^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="RotatedQuadratic")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return x_1**2 + x_1 * x_2 + x_2**2

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class RippleBowlFunction(OptFunction):
    r"""Ripple bowl function.

    Notes:
        $$
        f(x, y) = x^2 + y^2 + 0.1\sin^2(3x) + 0.1\sin^2(3y)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="RippleBowl")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return x_1**2 + x_2**2 + 0.1 * np.sin(3 * x_1) ** 2 + 0.1 * np.sin(3 * x_2) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class ExponentialValleyFunction(OptFunction):
    r"""Exponential valley function.

    Notes:
        $$
        f(x, y) = (x + y - 1)^2 + e^{(x - y)^2} - 1
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="ExponentialValley")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return (x_1 + x_2 - 1) ** 2 + np.expm1((x_1 - x_2) ** 2)

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.5, 0.5))


class AbsoluteBowlFunction(OptFunction):
    r"""Absolute bowl function.

    Notes:
        $$
        f(x, y) = |x| + |y|
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="AbsoluteBowl")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return np.abs(x_1) + np.abs(x_2)

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class CubicValleyFunction(OptFunction):
    r"""Cubic valley function.

    Notes:
        $$
        f(x, y) = (x - 1)^2 + (y - x^3)^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="CubicValley")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return (x_1 - 1) ** 2 + (x_2 - x_1**3) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(1.0, 1.0))


class CosineMixtureFunction(OptFunction):
    r"""Cosine mixture function.

    Notes:
        $$
        f(x, y) = 0.1(x^2 + y^2) + (1 - \cos(x)) + (1 - \cos(y))
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="CosineMixture")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return 0.1 * (x_1**2 + x_2**2) + (1 - np.cos(x_1)) + (1 - np.cos(x_2))

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class SaddleSuppressedFunction(OptFunction):
    r"""Saddle-suppressed function.

    Notes:
        $$
        f(x, y) = (x^2 - y^2)^2 + 0.1(x^2 + y^2)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="SaddleSuppressed")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return (x_1**2 - x_2**2) ** 2 + 0.1 * (x_1**2 + x_2**2)

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class EllipticAbsoluteRootFunction(OptFunction):
    r"""Elliptic absolute-root function.

    Notes:
        $$
        f(x, y) = \sqrt{1 + x^2} + \sqrt{1 + y^2} - 2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="EllipticAbsoluteRoot")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return np.sqrt(1 + x_1**2) + np.sqrt(1 + x_2**2) - 2

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class ShiftedSphere2DFunction(OptFunction):
    r"""Shifted 2D sphere function.

    Notes:
        $$
        f(x, y) = (x - 2)^2 + (y + 1)^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="ShiftedSphere2D")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return (x_1 - 2) ** 2 + (x_2 + 1) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(2.0, -1.0))


class QuarticBowlFunction(OptFunction):
    r"""Quartic bowl function.

    Notes:
        $$
        f(x, y) = x^4 + y^4
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="QuarticBowl")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return x_1**4 + x_2**4

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class CrossQuadraticFunction(OptFunction):
    r"""Cross-quadratic function.

    Notes:
        $$
        f(x, y) = (x + y)^2 + (x - y)^4
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="CrossQuadratic")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return (x_1 + x_2) ** 2 + (x_1 - x_2) ** 4

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class SinusoidalRosenbrockFunction(OptFunction):
    r"""Sinusoidal Rosenbrock-like function.

    Notes:
        $$
        f(x, y) = (1 - x)^2 + 100(y - x^2)^2 + 0.1\sin^2(\pi x)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="SinusoidalRosenbrock")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return (1 - x_1) ** 2 + 100 * (x_2 - x_1**2) ** 2 + 0.1 * np.sin(np.pi * x_1) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(1.0, 1.0))


class WeightedL1L2Function(OptFunction):
    r"""Weighted mixed-norm function.

    Notes:
        $$
        f(x, y) = x^2 + y^2 + |x + y|
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="WeightedL1L2")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return x_1**2 + x_2**2 + np.abs(x_1 + x_2)

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(0.0, 0.0))


class OffsetQuadraticFunction(OptFunction):
    r"""Offset quadratic function.

    Notes:
        $$
        f(x, y) = (x + 1)^2 + 3(y - 2)^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_two_dimensional(*x, function_name="OffsetQuadratic")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        x_1, x_2 = self._x
        return (x_1 + 1) ** 2 + 3 * (x_2 - 2) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=(-1.0, 2.0))


class ShiftedHyperSphereFunction(OptFunction):
    r"""Shifted hypersphere function.

    Notes:
        $$
        f(\mathbf{x}) = \sum_{i=1}^{n} (x_i - 1)^2, \quad n \geq 3
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_three_or_higher(*x, function_name="ShiftedHyperSphere")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        return np.array(sum((axis - 1) ** 2 for axis in self._x))

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=tuple(1.0 for _ in range(self.dimension)))


class CosineProductSphereFunction(OptFunction):
    r"""Cosine-product sphere function.

    Notes:
        $$
        f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2 - \prod_{i=1}^{n}\cos\left(\frac{x_i}{\sqrt{i}}\right) + 1,
        \quad n \geq 3
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_three_or_higher(*x, function_name="CosineProductSphere")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        indices = np.arange(1, self.dimension + 1)
        cosine_term = np.ones_like(self._x[0])
        for index, axis in zip(indices, self._x, strict=True):
            cosine_term *= np.cos(axis / np.sqrt(index))
        return np.array(sum(axis**2 for axis in self._x) - cosine_term + 1)

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=tuple(0.0 for _ in range(self.dimension)))


class SumAndProductFunction(OptFunction):
    r"""Sum-and-product function.

    Notes:
        $$
        f(\mathbf{x}) = \sum_{i=1}^{n} |x_i| + \prod_{i=1}^{n} (|x_i| + 1) - 1,
        \quad n \geq 3
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_three_or_higher(*x, function_name="SumAndProduct")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        sum_term = np.array(sum(np.abs(axis) for axis in self._x))
        product_term = np.ones_like(self._x[0])
        for axis in self._x:
            product_term *= np.abs(axis) + 1
        return sum_term + product_term - 1

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=tuple(0.0 for _ in range(self.dimension)))


class ExponentialNormFunction(OptFunction):
    r"""Exponential norm function.

    Notes:
        $$
        f(\mathbf{x}) = \exp\left(\frac{1}{2} \sum_{i=1}^{n} x_i^2\right) - 1,
        \quad n \geq 3
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_three_or_higher(*x, function_name="ExponentialNorm")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        squared_norm = np.array(sum(axis**2 for axis in self._x))
        return np.exp(0.5 * squared_norm) - 1

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=tuple(0.0 for _ in range(self.dimension)))


class CubicNormFunction(OptFunction):
    r"""Cubic norm function.

    Notes:
        $$
        f(\mathbf{x}) = \sum_{i=1}^{n} |x_i|^3, \quad n \geq 3
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        _validate_three_or_higher(*x, function_name="CubicNorm")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        return np.array(sum(np.abs(axis) ** 3 for axis in self._x))

    @property
    def __minima__(self) -> MinimaAPI:
        return MinimaAPI(f_x=0.0, x=tuple(0.0 for _ in range(self.dimension)))
