"""Tests for one-dimensional line-shaped optimization functions."""

from __future__ import annotations

import numpy as np
import pytest

from umf.constants.exceptions import OutOfDimensionError
from umf.functions.optimization.line_shaped import AbsoluteLineFunction
from umf.functions.optimization.line_shaped import ArctanSquareLineFunction
from umf.functions.optimization.line_shaped import CauchyLossLineFunction
from umf.functions.optimization.line_shaped import CosineBowlLineFunction
from umf.functions.optimization.line_shaped import DampedOscillationLineFunction
from umf.functions.optimization.line_shaped import DoubleWellLineFunction
from umf.functions.optimization.line_shaped import ElasticNetLineFunction
from umf.functions.optimization.line_shaped import ExponentialSquareLineFunction
from umf.functions.optimization.line_shaped import GaussianValleyLineFunction
from umf.functions.optimization.line_shaped import IdentitySquareLineFunction
from umf.functions.optimization.line_shaped import LogCoshLineFunction
from umf.functions.optimization.line_shaped import QuarticLineFunction
from umf.functions.optimization.line_shaped import RationalBowlLineFunction
from umf.functions.optimization.line_shaped import SexticLineFunction
from umf.functions.optimization.line_shaped import ShiftedElasticLineFunction
from umf.functions.optimization.line_shaped import ShiftedIdentitySquareLineFunction
from umf.functions.optimization.line_shaped import SincSquareLineFunction
from umf.functions.optimization.line_shaped import SineSquaredLineFunction
from umf.functions.optimization.line_shaped import SoftplusSymmetricLineFunction
from umf.functions.optimization.line_shaped import TiltedParabolaRippleLineFunction


@pytest.mark.parametrize(
    ("function_cls", "minimum_x"),
    [
        (IdentitySquareLineFunction, 0.0),
        (ShiftedIdentitySquareLineFunction, 1.0),
        (AbsoluteLineFunction, 0.0),
        (QuarticLineFunction, 0.0),
        (SexticLineFunction, 0.0),
        (DoubleWellLineFunction, 1.0),
        (CosineBowlLineFunction, 0.0),
        (SineSquaredLineFunction, 0.0),
        (DampedOscillationLineFunction, 0.0),
        (ExponentialSquareLineFunction, 0.0),
        (LogCoshLineFunction, 0.0),
        (SoftplusSymmetricLineFunction, 0.0),
        (RationalBowlLineFunction, 0.0),
        (ArctanSquareLineFunction, 0.0),
        (CauchyLossLineFunction, 0.0),
        (ElasticNetLineFunction, 0.0),
        (ShiftedElasticLineFunction, 2.0),
        (GaussianValleyLineFunction, 1.0),
        (SincSquareLineFunction, 0.0),
        (TiltedParabolaRippleLineFunction, 0.5),
    ],
)
def test_line_shaped_functions_minima(
    function_cls: type,
    minimum_x: float,
) -> None:
    """Test line-shaped functions evaluate to zero at representative minima."""
    function = function_cls(np.array([minimum_x]))
    result = float(np.sum(function.__eval__))
    assert np.isclose(result, 0.0, rtol=1e-6, atol=1e-8)


def test_line_shaped_rejects_wrong_dimension() -> None:
    """Test one-dimensional functions reject non-1D inputs."""
    with pytest.raises(OutOfDimensionError):
        IdentitySquareLineFunction(np.array([0.0]), np.array([0.0]))
