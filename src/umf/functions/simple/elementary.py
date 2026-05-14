"""A collection of 100 simple one-dimensional mathematical functions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np

from umf.constants.dimensions import __1d__
from umf.constants.exceptions import OutOfDimensionError
from umf.meta.api import MinimaAPI
from umf.meta.functions import OptFunction


if TYPE_CHECKING:
    from collections.abc import Callable

    from umf.types.static_types import UniversalArray


def _validate_one_dimensional(*x: UniversalArray, function_name: str) -> None:
    """Validate that exactly one input array is provided."""
    if len(x) != __1d__:
        raise OutOfDimensionError(function_name=function_name, dimension=__1d__)


def _humanize_name(name: str) -> str:
    """Convert a CamelCase class name into a readable title."""
    base = name.removesuffix("SimpleFunction")
    words: list[str] = []
    current = base[0]
    for character in base[1:]:
        if character.isupper() and not current[-1].isupper():
            words.append(current)
            current = character
        else:
            current += character
    words.append(current)
    return " ".join(words) + " Simple Function"


class _SimpleFunctionBase(OptFunction):
    """Base class for generated simple functions."""

    _formula: ClassVar[str]
    evaluator: ClassVar[staticmethod]

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the simple function."""
        _validate_one_dimensional(*x, function_name=self.__class__.__name__)
        super().__init__(*x)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the simple function."""
        return type(self).evaluator(self._x[0])

    @property
    def __minima__(self) -> MinimaAPI:
        """Return the representative global minimum."""
        return MinimaAPI(f_x=0.0, x=(0.0,))


def _make_simple_function(
    name: str,
    formula: str,
    evaluator: Callable[[UniversalArray], UniversalArray],
) -> type[_SimpleFunctionBase]:
    """Create a simple function class from a name and evaluator."""
    title = _humanize_name(name)
    doc = f"""{title}.

    The {title.lower()} is a one-dimensional mathematical function that maps a
    one-dimensional input array to a non-negative output curve.

    Examples:
        >>> from umf.functions.simple.elementary import {name}
        >>> import numpy as np
        >>> x = np.array([0.0, 1.0])
        >>> result = {name}(x)()
        >>> result.result.shape
        (2,)

        >>> # Visualization Example
        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(-5, 5, 1000)
        >>> y = {name}(x).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y)
        >>> plt.savefig("{name}.png", dpi=300, transparent=True)

    Notes:
        The {title.lower()} is defined as:

        $$
        f(x) = {formula}
        $$

    Args:
        *x (UniversalArray): Input data, which has to be one-dimensional.
    """

    return type(
        name,
        (_SimpleFunctionBase,),
        {
            "__doc__": doc,
            "__module__": __name__,
            "_formula": formula,
            "evaluator": staticmethod(evaluator),
        },
    )


_LOG_TWO = float(np.log(2.0))

_FUNCTION_SPECS: list[tuple[str, str, Callable[[UniversalArray], UniversalArray]]] = [
    ("SquareSimpleFunction", "x^2", lambda x: x**2),
    ("CubicAbsoluteSimpleFunction", r"|x|^3", lambda x: np.abs(x) ** 3),
    ("QuarticSimpleFunction", "x^4", lambda x: x**4),
    ("QuinticAbsoluteSimpleFunction", r"|x|^5", lambda x: np.abs(x) ** 5),
    ("SexticSimpleFunction", "x^6", lambda x: x**6),
    ("SepticAbsoluteSimpleFunction", r"|x|^7", lambda x: np.abs(x) ** 7),
    ("OcticSimpleFunction", "x^8", lambda x: x**8),
    ("NonicAbsoluteSimpleFunction", r"|x|^9", lambda x: np.abs(x) ** 9),
    ("DecicSimpleFunction", "x^10", lambda x: x**10),
    ("TwelfthPowerSimpleFunction", "x^12", lambda x: x**12),
    ("RootAbsoluteSimpleFunction", r"\sqrt{|x|}", lambda x: np.sqrt(np.abs(x))),
    (
        "ThreeHalvesAbsoluteSimpleFunction",
        r"|x|^{3/2}",
        lambda x: np.abs(x) ** 1.5,
    ),
    (
        "FiveHalvesAbsoluteSimpleFunction",
        r"|x|^{5/2}",
        lambda x: np.abs(x) ** 2.5,
    ),
    (
        "SevenHalvesAbsoluteSimpleFunction",
        r"|x|^{7/2}",
        lambda x: np.abs(x) ** 3.5,
    ),
    (
        "PseudoHuberSimpleFunction",
        r"\sqrt{1 + x^2} - 1",
        lambda x: np.sqrt(1.0 + x**2) - 1.0,
    ),
    ("LogQuadraticSimpleFunction", r"\log(1 + x^2)", lambda x: np.log1p(x**2)),
    ("LogQuarticSimpleFunction", r"\log(1 + x^4)", lambda x: np.log1p(x**4)),
    ("LogSexticSimpleFunction", r"\log(1 + x^6)", lambda x: np.log1p(x**6)),
    (
        "RationalQuadraticSimpleFunction",
        r"\frac{x^2}{1 + x^2}",
        lambda x: x**2 / (1.0 + x**2),
    ),
    (
        "RationalQuarticSimpleFunction",
        r"\frac{x^4}{1 + x^4}",
        lambda x: x**4 / (1.0 + x**4),
    ),
    (
        "RationalSexticSimpleFunction",
        r"\frac{x^6}{1 + x^6}",
        lambda x: x**6 / (1.0 + x**6),
    ),
    (
        "RationalAbsoluteSimpleFunction",
        r"\frac{|x|}{1 + |x|}",
        lambda x: np.abs(x) / (1.0 + np.abs(x)),
    ),
    (
        "RationalCubicAbsoluteSimpleFunction",
        r"\frac{|x|^3}{1 + |x|^3}",
        lambda x: np.abs(x) ** 3 / (1.0 + np.abs(x) ** 3),
    ),
    (
        "RationalQuinticAbsoluteSimpleFunction",
        r"\frac{|x|^5}{1 + |x|^5}",
        lambda x: np.abs(x) ** 5 / (1.0 + np.abs(x) ** 5),
    ),
    (
        "RationalRootSimpleFunction",
        r"\frac{\sqrt{|x|}}{1 + \sqrt{|x|}}",
        lambda x: np.sqrt(np.abs(x)) / (1.0 + np.sqrt(np.abs(x))),
    ),
    (
        "InverseBellComplementSimpleFunction",
        r"1 - \frac{1}{1 + x^2}",
        lambda x: 1.0 - 1.0 / (1.0 + x**2),
    ),
    (
        "InverseQuarticBellComplementSimpleFunction",
        r"1 - \frac{1}{1 + x^4}",
        lambda x: 1.0 - 1.0 / (1.0 + x**4),
    ),
    (
        "InverseSexticBellComplementSimpleFunction",
        r"1 - \frac{1}{1 + x^6}",
        lambda x: 1.0 - 1.0 / (1.0 + x**6),
    ),
    (
        "InverseGaussianComplementSimpleFunction",
        r"1 - e^{-x^2}",
        lambda x: 1.0 - np.exp(-(x**2)),
    ),
    (
        "InverseSuperGaussianComplementSimpleFunction",
        r"1 - e^{-x^4}",
        lambda x: 1.0 - np.exp(-(x**4)),
    ),
    (
        "ExponentialSquareSimpleFunction",
        r"e^{x^2} - 1",
        lambda x: np.expm1(x**2),
    ),
    (
        "ExponentialQuarticSimpleFunction",
        r"e^{x^4} - 1",
        lambda x: np.expm1(x**4),
    ),
    (
        "ExponentialAbsoluteSimpleFunction",
        r"e^{|x|} - 1",
        lambda x: np.expm1(np.abs(x)),
    ),
    (
        "ExponentialCubicAbsoluteSimpleFunction",
        r"e^{|x|^3} - 1",
        lambda x: np.expm1(np.abs(x) ** 3),
    ),
    ("DampedSquareSimpleFunction", r"1 - e^{-x^2}", lambda x: 1.0 - np.exp(-(x**2))),
    ("DampedQuarticSimpleFunction", r"1 - e^{-x^4}", lambda x: 1.0 - np.exp(-(x**4))),
    (
        "DampedAbsoluteSimpleFunction",
        r"1 - e^{-|x|}",
        lambda x: 1.0 - np.exp(-np.abs(x)),
    ),
    (
        "DampedCubicAbsoluteSimpleFunction",
        r"1 - e^{-|x|^3}",
        lambda x: 1.0 - np.exp(-(np.abs(x) ** 3)),
    ),
    (
        "GaussianValleySimpleFunction",
        r"1 - e^{-x^2 / 2}",
        lambda x: 1.0 - np.exp(-(x**2) / 2.0),
    ),
    (
        "NarrowGaussianValleySimpleFunction",
        r"1 - e^{-2x^2}",
        lambda x: 1.0 - np.exp(-2.0 * x**2),
    ),
    (
        "WideGaussianValleySimpleFunction",
        r"1 - e^{-x^2 / 8}",
        lambda x: 1.0 - np.exp(-(x**2) / 8.0),
    ),
    (
        "SuperGaussianValleySimpleFunction",
        r"1 - e^{-x^4 / 2}",
        lambda x: 1.0 - np.exp(-(x**4) / 2.0),
    ),
    ("CosineBowlSimpleFunction", r"1 - \cos(x)", lambda x: 1.0 - np.cos(x)),
    (
        "DoubleCosineBowlSimpleFunction",
        r"1 - \cos(2x)",
        lambda x: 1.0 - np.cos(2.0 * x),
    ),
    (
        "TripleCosineBowlSimpleFunction",
        r"1 - \cos(3x)",
        lambda x: 1.0 - np.cos(3.0 * x),
    ),
    (
        "HalfCosineBowlSimpleFunction",
        r"1 - \cos(x / 2)",
        lambda x: 1.0 - np.cos(x / 2.0),
    ),
    ("SineSquaredSimpleFunction", r"\sin^2(x)", lambda x: np.sin(x) ** 2),
    ("DoubleSineSquaredSimpleFunction", r"\sin^2(2x)", lambda x: np.sin(2.0 * x) ** 2),
    ("TripleSineSquaredSimpleFunction", r"\sin^2(3x)", lambda x: np.sin(3.0 * x) ** 2),
    ("HalfSineSquaredSimpleFunction", r"\sin^2(x / 2)", lambda x: np.sin(x / 2.0) ** 2),
    (
        "CosineBowlSquaredSimpleFunction",
        r"(1 - \cos(x))^2",
        lambda x: (1.0 - np.cos(x)) ** 2,
    ),
    ("SineFourthSimpleFunction", r"\sin^4(x)", lambda x: np.sin(x) ** 4),
    ("SineSixthSimpleFunction", r"\sin^6(x)", lambda x: np.sin(x) ** 6),
    (
        "SincLossSimpleFunction",
        r"1 - \mathrm{sinc}(x / \pi)",
        lambda x: 1.0 - np.sinc(x / np.pi),
    ),
    (
        "DoubleSincLossSimpleFunction",
        r"1 - \mathrm{sinc}(2x / \pi)",
        lambda x: 1.0 - np.sinc(2.0 * x / np.pi),
    ),
    (
        "TripleSincLossSimpleFunction",
        r"1 - \mathrm{sinc}(3x / \pi)",
        lambda x: 1.0 - np.sinc(3.0 * x / np.pi),
    ),
    (
        "CosineDampedSquareSimpleFunction",
        r"x^2 \left(1 + \frac{1 - \cos(x)}{2}\right)",
        lambda x: x**2 * (1.0 + 0.5 * (1.0 - np.cos(x))),
    ),
    (
        "SineDampedSquareSimpleFunction",
        r"x^2 + \sin^2(x)",
        lambda x: x**2 + np.sin(x) ** 2,
    ),
    (
        "RippleQuarticSimpleFunction",
        r"x^4 + \frac{\sin^2(x)}{4}",
        lambda x: x**4 + 0.25 * np.sin(x) ** 2,
    ),
    (
        "RippleAbsoluteSimpleFunction",
        r"|x| + \frac{\sin^2(x)}{4}",
        lambda x: np.abs(x) + 0.25 * np.sin(x) ** 2,
    ),
    ("HyperbolicCosineSimpleFunction", r"\cosh(x) - 1", lambda x: np.cosh(x) - 1.0),
    (
        "SechBowlSimpleFunction",
        r"1 - \operatorname{sech}(x)",
        lambda x: 1.0 - 1.0 / np.cosh(x),
    ),
    ("TanhSquaredSimpleFunction", r"\tanh^2(x)", lambda x: np.tanh(x) ** 2),
    ("ArctanSquaredSimpleFunction", r"\arctan^2(x)", lambda x: np.arctan(x) ** 2),
    (
        "ArcsinhSquaredSimpleFunction",
        r"\operatorname{arsinh}^2(x)",
        lambda x: np.arcsinh(x) ** 2,
    ),
    (
        "SoftsignSquaredSimpleFunction",
        r"\left(\frac{x}{1 + |x|}\right)^2",
        lambda x: (x / (1.0 + np.abs(x))) ** 2,
    ),
    (
        "LogisticMirrorSimpleFunction",
        r"2\log(1 + e^x) - x - 2\log(2)",
        lambda x: 2.0 * np.logaddexp(0.0, x) - x - 2.0 * _LOG_TWO,
    ),
    ("LogCoshSimpleFunction", r"\log(\cosh(x))", lambda x: np.log(np.cosh(x))),
    (
        "CoshMinusSechSimpleFunction",
        r"\cosh(x) - \operatorname{sech}(x)",
        lambda x: np.cosh(x) - 1.0 / np.cosh(x),
    ),
    ("SinhSquaredSimpleFunction", r"\sinh^2(x)", lambda x: np.sinh(x) ** 2),
    ("TanhAbsoluteSimpleFunction", r"|\tanh(x)|", lambda x: np.abs(np.tanh(x))),
    (
        "ArcsinhAbsoluteSimpleFunction",
        r"|\operatorname{arsinh}(x)|",
        lambda x: np.abs(np.arcsinh(x)),
    ),
    ("Log1pAbsoluteSimpleFunction", r"\log(1 + |x|)", lambda x: np.log1p(np.abs(x))),
    (
        "Log1pRootAbsoluteSimpleFunction",
        r"\log(1 + \sqrt{|x|})",
        lambda x: np.log1p(np.sqrt(np.abs(x))),
    ),
    (
        "Log1pSinhSquaredSimpleFunction",
        r"\log(1 + \sinh^2(x))",
        lambda x: np.log1p(np.sinh(x) ** 2),
    ),
    ("QuadraticPlusAbsoluteSimpleFunction", r"x^2 + |x|", lambda x: x**2 + np.abs(x)),
    (
        "QuadraticPlusLogSimpleFunction",
        r"x^2 + \log(1 + x^2)",
        lambda x: x**2 + np.log1p(x**2),
    ),
    (
        "QuadraticPlusCosineSimpleFunction",
        r"x^2 + 1 - \cos(x)",
        lambda x: x**2 + 1.0 - np.cos(x),
    ),
    (
        "QuadraticPlusSineSquaredSimpleFunction",
        r"x^2 + \sin^2(x)",
        lambda x: x**2 + np.sin(x) ** 2,
    ),
    (
        "QuarticPlusCosineSimpleFunction",
        r"x^4 + 1 - \cos(x)",
        lambda x: x**4 + 1.0 - np.cos(x),
    ),
    (
        "QuarticPlusSineSquaredSimpleFunction",
        r"x^4 + \sin^2(x)",
        lambda x: x**4 + np.sin(x) ** 2,
    ),
    (
        "AbsolutePlusCosineSimpleFunction",
        r"|x| + 1 - \cos(x)",
        lambda x: np.abs(x) + 1.0 - np.cos(x),
    ),
    (
        "AbsolutePlusSineSquaredSimpleFunction",
        r"|x| + \sin^2(x)",
        lambda x: np.abs(x) + np.sin(x) ** 2,
    ),
    (
        "RootPlusQuadraticSimpleFunction",
        r"\sqrt{|x|} + x^2",
        lambda x: np.sqrt(np.abs(x)) + x**2,
    ),
    (
        "RootPlusSineSquaredSimpleFunction",
        r"\sqrt{|x|} + \sin^2(x)",
        lambda x: np.sqrt(np.abs(x)) + np.sin(x) ** 2,
    ),
    (
        "RationalPlusQuadraticSimpleFunction",
        r"\frac{x^2}{1 + x^2} + x^2",
        lambda x: x**2 / (1.0 + x**2) + x**2,
    ),
    (
        "RationalPlusAbsoluteSimpleFunction",
        r"\frac{|x|}{1 + |x|} + |x|",
        lambda x: np.abs(x) / (1.0 + np.abs(x)) + np.abs(x),
    ),
    (
        "RationalPlusCosineSimpleFunction",
        r"\frac{x^2}{1 + x^2} + 1 - \cos(x)",
        lambda x: x**2 / (1.0 + x**2) + 1.0 - np.cos(x),
    ),
    (
        "GaussianPlusQuadraticSimpleFunction",
        r"x^2 + 1 - e^{-x^2}",
        lambda x: x**2 + 1.0 - np.exp(-(x**2)),
    ),
    (
        "GaussianPlusAbsoluteSimpleFunction",
        r"|x| + 1 - e^{-x^2}",
        lambda x: np.abs(x) + 1.0 - np.exp(-(x**2)),
    ),
    (
        "HyperbolicPlusQuadraticSimpleFunction",
        r"\cosh(x) - 1 + x^2",
        lambda x: np.cosh(x) - 1.0 + x**2,
    ),
    (
        "HyperbolicPlusAbsoluteSimpleFunction",
        r"\cosh(x) - 1 + |x|",
        lambda x: np.cosh(x) - 1.0 + np.abs(x),
    ),
    (
        "ArctanPlusQuadraticSimpleFunction",
        r"\arctan^2(x) + x^2",
        lambda x: np.arctan(x) ** 2 + x**2,
    ),
    (
        "ArcsinhPlusQuadraticSimpleFunction",
        r"\operatorname{arsinh}^2(x) + x^2",
        lambda x: np.arcsinh(x) ** 2 + x**2,
    ),
    (
        "LogPlusSineSquaredSimpleFunction",
        r"\log(1 + x^2) + \sin^2(x)",
        lambda x: np.log1p(x**2) + np.sin(x) ** 2,
    ),
    (
        "LogPlusCosineSimpleFunction",
        r"\log(1 + x^2) + 1 - \cos(x)",
        lambda x: np.log1p(x**2) + 1.0 - np.cos(x),
    ),
    (
        "ExponentialPlusQuadraticSimpleFunction",
        r"e^{x^2} - 1 + x^2",
        lambda x: np.expm1(x**2) + x**2,
    ),
    (
        "ExponentialPlusAbsoluteSimpleFunction",
        r"e^{|x|} - 1 + |x|",
        lambda x: np.expm1(np.abs(x)) + np.abs(x),
    ),
    (
        "MixedEvenPolynomialSimpleFunction",
        r"x^2 + x^4",
        lambda x: x**2 + x**4,
    ),
    (
        "MixedAbsolutePolynomialSimpleFunction",
        r"|x| + x^2 + x^4",
        lambda x: np.abs(x) + x**2 + x**4,
    ),
]

__all__ = [name for name, _, _ in _FUNCTION_SPECS]

for _name, _formula, _evaluator in _FUNCTION_SPECS:
    globals()[_name] = _make_simple_function(_name, _formula, _evaluator)
