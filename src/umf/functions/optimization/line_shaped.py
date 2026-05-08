"""One-dimensional line-shaped benchmark functions.

This module provides scalar one-dimensional optimization objectives of the form
$f(x)$, designed for testing gradient-based optimizers, autodiff pipelines,
and least-squares style solvers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.dimensions import __1d__
from umf.constants.exceptions import OutOfDimensionError
from umf.meta.api import MinimaAPI
from umf.meta.functions import OptFunction


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__ = [
    "AbsoluteLineFunction",
    "ArctanSquareLineFunction",
    "CauchyLossLineFunction",
    "CosineBowlLineFunction",
    "DampedOscillationLineFunction",
    "DoubleWellLineFunction",
    "ElasticNetLineFunction",
    "ExponentialSquareLineFunction",
    "GaussianValleyLineFunction",
    "IdentitySquareLineFunction",
    "LogCoshLineFunction",
    "QuarticLineFunction",
    "RationalBowlLineFunction",
    "SexticLineFunction",
    "ShiftedElasticLineFunction",
    "ShiftedIdentitySquareLineFunction",
    "SincSquareLineFunction",
    "SineSquaredLineFunction",
    "SoftplusSymmetricLineFunction",
    "TiltedParabolaRippleLineFunction",
]


def _validate_one_dimensional(*x: UniversalArray, function_name: str) -> None:
    """Validate that a function receives exactly one input array."""
    if len(x) != __1d__:
        raise OutOfDimensionError(function_name=function_name, dimension=__1d__)


class IdentitySquareLineFunction(OptFunction):
    r"""Identity-square line benchmark.

    Notes:
        $$
        f(x) = x^2
        $$

    Args:
        *x: Input data with one dimension.

    Raises:
        OutOfDimensionError: If input is not one-dimensional.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the identity-square line function."""
        _validate_one_dimensional(*x, function_name="IdentitySquareLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the identity-square line function."""
        x_1 = self._x[0]
        return x_1**2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class ShiftedIdentitySquareLineFunction(OptFunction):
    r"""Shifted identity-square line benchmark.

    Notes:
        $$
        f(x) = (x - 1)^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the shifted identity-square line function."""
        _validate_one_dimensional(*x, function_name="ShiftedIdentitySquareLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the shifted identity-square line function."""
        x_1 = self._x[0]
        return (x_1 - 1.0) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(1.0,))


class AbsoluteLineFunction(OptFunction):
    r"""Absolute-value line benchmark.

    Notes:
        $$
        f(x) = |x|
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the absolute line function."""
        _validate_one_dimensional(*x, function_name="AbsoluteLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the absolute line function."""
        x_1 = self._x[0]
        return np.abs(x_1)

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class QuarticLineFunction(OptFunction):
    r"""Quartic line benchmark.

    Notes:
        $$
        f(x) = x^4
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the quartic line function."""
        _validate_one_dimensional(*x, function_name="QuarticLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the quartic line function."""
        x_1 = self._x[0]
        return x_1**4

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class SexticLineFunction(OptFunction):
    r"""Sextic line benchmark.

    Notes:
        $$
        f(x) = x^6
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the sextic line function."""
        _validate_one_dimensional(*x, function_name="SexticLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the sextic line function."""
        x_1 = self._x[0]
        return x_1**6

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class DoubleWellLineFunction(OptFunction):
    r"""Double-well line benchmark.

    Notes:
        $$
        f(x) = (x^2 - 1)^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the double-well line function."""
        _validate_one_dimensional(*x, function_name="DoubleWellLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the double-well line function."""
        x_1 = self._x[0]
        return (x_1**2 - 1.0) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return representative global minima."""
        return MinimaAPI(f_x=0.0, x=(-1.0, 1.0))


class CosineBowlLineFunction(OptFunction):
    r"""Cosine bowl line benchmark.

    Notes:
        $$
        f(x) = 1 - \cos(x)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the cosine bowl line function."""
        _validate_one_dimensional(*x, function_name="CosineBowlLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the cosine bowl line function."""
        x_1 = self._x[0]
        return 1.0 - np.cos(x_1)

    @property
    def __minima__(self) -> MinimaAPI:
        """Return one global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class SineSquaredLineFunction(OptFunction):
    r"""Sine-squared line benchmark.

    Notes:
        $$
        f(x) = \sin^2(x)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the sine-squared line function."""
        _validate_one_dimensional(*x, function_name="SineSquaredLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the sine-squared line function."""
        x_1 = self._x[0]
        return np.sin(x_1) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return one global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class DampedOscillationLineFunction(OptFunction):
    r"""Damped oscillation line benchmark.

    Notes:
        $$
        f(x) = \sin^2(3x) + 0.01x^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the damped oscillation line function."""
        _validate_one_dimensional(*x, function_name="DampedOscillationLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the damped oscillation line function."""
        x_1 = self._x[0]
        return np.sin(3.0 * x_1) ** 2 + 0.01 * x_1**2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class ExponentialSquareLineFunction(OptFunction):
    r"""Exponential square line benchmark.

    Notes:
        $$
        f(x) = e^{x^2} - 1
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the exponential square line function."""
        _validate_one_dimensional(*x, function_name="ExponentialSquareLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the exponential square line function."""
        x_1 = self._x[0]
        return np.exp(x_1**2) - 1.0

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class LogCoshLineFunction(OptFunction):
    r"""Log-cosh line benchmark.

    Notes:
        $$
        f(x) = \log(\cosh(x))
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the log-cosh line function."""
        _validate_one_dimensional(*x, function_name="LogCoshLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the log-cosh line function."""
        x_1 = self._x[0]
        return np.log(np.cosh(x_1))

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class SoftplusSymmetricLineFunction(OptFunction):
    r"""Symmetric softplus line benchmark.

    Notes:
        $$
        f(x) = \log(1 + e^x) + \log(1 + e^{-x}) - 2\log(2)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the symmetric softplus line function."""
        _validate_one_dimensional(*x, function_name="SoftplusSymmetricLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the symmetric softplus line function."""
        x_1 = self._x[0]
        return np.logaddexp(0.0, x_1) + np.logaddexp(0.0, -x_1) - 2.0 * np.log(2.0)

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class RationalBowlLineFunction(OptFunction):
    r"""Rational bowl line benchmark.

    Notes:
        $$
        f(x) = \frac{x^2}{1 + x^2}
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the rational bowl line function."""
        _validate_one_dimensional(*x, function_name="RationalBowlLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the rational bowl line function."""
        x_1 = self._x[0]
        return x_1**2 / (1.0 + x_1**2)

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class ArctanSquareLineFunction(OptFunction):
    r"""Arctangent-square line benchmark.

    Notes:
        $$
        f(x) = \arctan^2(x)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the arctangent-square line function."""
        _validate_one_dimensional(*x, function_name="ArctanSquareLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the arctangent-square line function."""
        x_1 = self._x[0]
        return np.arctan(x_1) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class CauchyLossLineFunction(OptFunction):
    r"""Cauchy-loss line benchmark.

    Notes:
        $$
        f(x) = \log(1 + x^2)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the Cauchy-loss line function."""
        _validate_one_dimensional(*x, function_name="CauchyLossLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the Cauchy-loss line function."""
        x_1 = self._x[0]
        return np.log1p(x_1**2)

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class ElasticNetLineFunction(OptFunction):
    r"""Elastic-net style line benchmark.

    Notes:
        $$
        f(x) = |x| + 0.5x^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the elastic-net line function."""
        _validate_one_dimensional(*x, function_name="ElasticNetLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the elastic-net line function."""
        x_1 = self._x[0]
        return np.abs(x_1) + 0.5 * x_1**2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class ShiftedElasticLineFunction(OptFunction):
    r"""Shifted elastic line benchmark.

    Notes:
        $$
        f(x) = |x - 2| + 0.25(x - 2)^2
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the shifted elastic line function."""
        _validate_one_dimensional(*x, function_name="ShiftedElasticLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the shifted elastic line function."""
        x_1 = self._x[0]
        return np.abs(x_1 - 2.0) + 0.25 * (x_1 - 2.0) ** 2

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(2.0,))


class GaussianValleyLineFunction(OptFunction):
    r"""Gaussian valley line benchmark.

    Notes:
        $$
        f(x) = 1 - \exp\left(-(x - 1)^2\right)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the gaussian valley line function."""
        _validate_one_dimensional(*x, function_name="GaussianValleyLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the gaussian valley line function."""
        x_1 = self._x[0]
        return 1.0 - np.exp(-((x_1 - 1.0) ** 2))

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(1.0,))


class SincSquareLineFunction(OptFunction):
    r"""Sinc-square line benchmark.

    Notes:
        $$
        f(x) = 1 - \operatorname{sinc}(x/\pi)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the sinc-square line function."""
        _validate_one_dimensional(*x, function_name="SincSquareLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the sinc-square line function."""
        x_1 = self._x[0]
        return 1.0 - np.sinc(x_1 / np.pi)

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


class TiltedParabolaRippleLineFunction(OptFunction):
    r"""Tilted parabola-ripple line benchmark.

    Notes:
        $$
        f(x) = (x - 0.5)^2 + 0.05\left(1 - \cos\left(6(x - 0.5)\right)\right)
        $$
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the tilted parabola-ripple line function."""
        _validate_one_dimensional(*x, function_name="TiltedParabolaRippleLine")
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the tilted parabola-ripple line function."""
        x_1 = self._x[0]
        shifted = x_1 - 0.5
        return shifted**2 + 0.05 * (1.0 - np.cos(6.0 * shifted))

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.5,))
