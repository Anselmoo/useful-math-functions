"""Backward-compatible imports for newly categorized optimization functions."""

from __future__ import annotations

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


__all__: list[str] = [
    "AbsoluteBowlFunction",
    "CosineMixtureFunction",
    "CosineProductSphereFunction",
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
]
