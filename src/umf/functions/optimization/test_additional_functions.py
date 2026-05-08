"""Tests for additional optimization functions."""

from __future__ import annotations

import numpy as np
import pytest

from umf.constants.exceptions import OutOfDimensionError
from umf.constants.exceptions import TooSmallDimensionError
from umf.functions.optimization.additional import (
    AbsoluteBowlFunction as LegacyAbsoluteBowlFunction,
)
from umf.functions.optimization.bowl_shaped import AbsoluteBowlFunction
from umf.functions.optimization.bowl_shaped import CrossQuadraticFunction
from umf.functions.optimization.bowl_shaped import CubicNormFunction
from umf.functions.optimization.bowl_shaped import EllipticAbsoluteRootFunction
from umf.functions.optimization.bowl_shaped import EllipticParaboloidFunction
from umf.functions.optimization.bowl_shaped import ExponentialNormFunction
from umf.functions.optimization.bowl_shaped import OffsetQuadraticFunction
from umf.functions.optimization.bowl_shaped import QuarticBowlFunction
from umf.functions.optimization.bowl_shaped import RippleBowlFunction
from umf.functions.optimization.bowl_shaped import RotatedQuadraticFunction
from umf.functions.optimization.bowl_shaped import SaddleSuppressedFunction
from umf.functions.optimization.bowl_shaped import ShiftedHyperSphereFunction
from umf.functions.optimization.bowl_shaped import ShiftedSphere2DFunction
from umf.functions.optimization.bowl_shaped import SumAndProductFunction
from umf.functions.optimization.bowl_shaped import WeightedL1L2Function
from umf.functions.optimization.many_local_minima import CosineMixtureFunction
from umf.functions.optimization.many_local_minima import CosineProductSphereFunction
from umf.functions.optimization.valley_shaped import CubicValleyFunction
from umf.functions.optimization.valley_shaped import ExponentialValleyFunction
from umf.functions.optimization.valley_shaped import SinusoidalRosenbrockFunction


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
    assert result == pytest.approx(0.0, rel=1e-6, abs=1e-8)


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
    assert result == pytest.approx(0.0, rel=1e-6, abs=1e-8)


def test_two_dimensional_function_rejects_wrong_dimension() -> None:
    """Test 2D functions reject non-2D inputs."""
    with pytest.raises(OutOfDimensionError):
        EllipticParaboloidFunction(np.array([0.0]))


def test_three_dimensional_function_rejects_wrong_dimension() -> None:
    """Test 3D+ functions reject inputs with fewer than three dimensions."""
    with pytest.raises(TooSmallDimensionError):
        ShiftedHyperSphereFunction(np.array([0.0]), np.array([0.0]))


def test_additional_module_re_exports_semantic_class() -> None:
    """Test the legacy additional module re-exports semantic classes."""
    assert LegacyAbsoluteBowlFunction is AbsoluteBowlFunction
