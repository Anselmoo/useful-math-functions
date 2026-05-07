"""Tests for additional optimization functions."""

from __future__ import annotations

import numpy as np
import pytest

from umf.constants.exceptions import OutOfDimensionError
from umf.constants.exceptions import TooSmallDimensionError
from umf.functions.optimization.additional import AbsoluteBowlFunction
from umf.functions.optimization.additional import CosineMixtureFunction
from umf.functions.optimization.additional import CosineProductSphereFunction
from umf.functions.optimization.additional import CrossQuadraticFunction
from umf.functions.optimization.additional import CubicNormFunction
from umf.functions.optimization.additional import CubicValleyFunction
from umf.functions.optimization.additional import EllipticAbsoluteRootFunction
from umf.functions.optimization.additional import EllipticParaboloidFunction
from umf.functions.optimization.additional import ExponentialNormFunction
from umf.functions.optimization.additional import ExponentialValleyFunction
from umf.functions.optimization.additional import OffsetQuadraticFunction
from umf.functions.optimization.additional import QuarticBowlFunction
from umf.functions.optimization.additional import RippleBowlFunction
from umf.functions.optimization.additional import RotatedQuadraticFunction
from umf.functions.optimization.additional import SaddleSuppressedFunction
from umf.functions.optimization.additional import ShiftedHyperSphereFunction
from umf.functions.optimization.additional import ShiftedSphere2DFunction
from umf.functions.optimization.additional import SinusoidalRosenbrockFunction
from umf.functions.optimization.additional import SumAndProductFunction
from umf.functions.optimization.additional import WeightedL1L2Function


@pytest.mark.parametrize(
    ("function_cls", "x_1", "x_2"),
    [
        (EllipticParaboloidFunction, 0.0, 0.0),
        (RotatedQuadraticFunction, 0.0, 0.0),
        (RippleBowlFunction, 0.0, 0.0),
        (ExponentialValleyFunction, 0.5, 0.5),
        (AbsoluteBowlFunction, 0.0, 0.0),
        (CubicValleyFunction, 1.0, 1.0),
        (CosineMixtureFunction, 0.0, 0.0),
        (SaddleSuppressedFunction, 0.0, 0.0),
        (EllipticAbsoluteRootFunction, 0.0, 0.0),
        (ShiftedSphere2DFunction, 2.0, -1.0),
        (QuarticBowlFunction, 0.0, 0.0),
        (CrossQuadraticFunction, 0.0, 0.0),
        (SinusoidalRosenbrockFunction, 1.0, 1.0),
        (WeightedL1L2Function, 0.0, 0.0),
        (OffsetQuadraticFunction, -1.0, 2.0),
    ],
)
def test_two_dimensional_functions_minima(
    function_cls: type,
    x_1: float,
    x_2: float,
) -> None:
    """Test all new two-dimensional functions at their minima."""
    function = function_cls(np.array([x_1]), np.array([x_2]))
    result = float(np.sum(function.__eval__))
    assert np.all(np.isclose(result, 0.0, rtol=1e-6, atol=1e-8))


@pytest.mark.parametrize(
    ("function_cls", "args"),
    [
        (ShiftedHyperSphereFunction, (1.0, 1.0, 1.0)),
        (CosineProductSphereFunction, (0.0, 0.0, 0.0)),
        (SumAndProductFunction, (0.0, 0.0, 0.0)),
        (ExponentialNormFunction, (0.0, 0.0, 0.0)),
        (CubicNormFunction, (0.0, 0.0, 0.0)),
    ],
)
def test_three_dimensional_functions_minima(
    function_cls: type,
    args: tuple[float, float, float],
) -> None:
    """Test all new three-dimensional-and-higher functions at their minima."""
    function = function_cls(*(np.array([value]) for value in args))
    result = float(np.sum(function.__eval__))
    assert np.all(np.isclose(result, 0.0, rtol=1e-6, atol=1e-8))


def test_two_dimensional_function_rejects_wrong_dimension() -> None:
    """Test 2D functions reject non-2D inputs."""
    with pytest.raises(OutOfDimensionError):
        EllipticParaboloidFunction(np.array([0.0]))


def test_three_dimensional_function_rejects_wrong_dimension() -> None:
    """Test 3D+ functions reject inputs with fewer than three dimensions."""
    with pytest.raises(TooSmallDimensionError):
        ShiftedHyperSphereFunction(np.array([0.0]), np.array([0.0]))
