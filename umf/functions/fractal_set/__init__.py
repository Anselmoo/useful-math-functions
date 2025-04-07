"""Fractal set module for the UMF package."""

from __future__ import annotations

from umf.functions.fractal_set.complex import FeigenbaumDiagram
from umf.functions.fractal_set.complex import JuliaSet
from umf.functions.fractal_set.complex import MandelbrotSet
from umf.functions.fractal_set.curve import CantorSet
from umf.functions.fractal_set.curve import DragonCurve
from umf.functions.fractal_set.curve import HilbertCurve
from umf.functions.fractal_set.curve import SpaceFillingCurve
from umf.functions.fractal_set.dynamic import CurlicueFractal
from umf.functions.fractal_set.dynamic import LorenzAttractor
from umf.functions.fractal_set.dynamic import PercolationModel
from umf.functions.fractal_set.dynamic import RandomWalkFractal
from umf.functions.fractal_set.geometric import KochCurve
from umf.functions.fractal_set.geometric import MengerSponge
from umf.functions.fractal_set.geometric import PythagorasTree
from umf.functions.fractal_set.geometric import SierpinskiCarpet
from umf.functions.fractal_set.geometric import SierpinskiTriangle
from umf.functions.fractal_set.geometric import UniformMassCenterTriangle


__all__ = [  # noqa: RUF022
    # Complex fractals
    "FeigenbaumDiagram",
    "JuliaSet",
    "MandelbrotSet",
    # Curve fractals
    "CantorSet",
    "DragonCurve",
    "HilbertCurve",
    "SpaceFillingCurve",
    # Dynamic fractals
    "CurlicueFractal",
    "LorenzAttractor",
    "PercolationModel",
    "RandomWalkFractal",
    # Geometric fractals
    "KochCurve",
    "MengerSponge",
    "PythagorasTree",
    "SierpinskiCarpet",
    "SierpinskiTriangle",
    "UniformMassCenterTriangle",
]
