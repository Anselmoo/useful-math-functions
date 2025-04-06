"""Fractal set module for the UMF package."""

from __future__ import annotations

from umf.functions.fractal_set.complex import FeigenbaumDiagram, JuliaSet, MandelbrotSet
from umf.functions.fractal_set.curve import (
    CantorSet,
    DragonCurve,
    HilbertCurve,
    SpaceFillingCurve,
)
from umf.functions.fractal_set.dynamic import (
    CurlicueFractal,
    LorenzAttractor,
    PercolationModel,
    RandomWalkFractal,
)
from umf.functions.fractal_set.geometric import (
    KochCurve,
    MengerSponge,
    PythagorasTree,
    SierpinskiCarpet,
    SierpinskiTriangle,
    UniformMassCenterTriangle,
)

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
