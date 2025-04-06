"""A comprehensive collection of fractal implementations for mathematical visualization.

This module provides classes for generating and analyzing various fractal patterns,
combining classical and complex mathematical concepts into visual and computational
representations.

Classes:
- MandelbrotSet: Implementation of the classic Mandelbrot set fractal
- JuliaSet: Parameterized Julia set fractal generator with surface rendering
- KochCurve: Koch snowflake curve fractal generator
- SierpinskiTriangle: Sierpinski gasket fractal generator
- SierpinskiCarpet: Sierpinski carpet fractal generator
- CantorSet: One-dimensional Cantor set visualization
- DragonCurve: Dragon curve fractal generator
- PythagorasTree: Tree-like fractal based on Pythagorean theorem with spherical projection
- HilbertCurve: Space-filling curve visualization with ball mapping
- MengerSponge: Three-dimensional fractal sponge visualization
- LorenzAttractor: Visualization of the chaotic Lorenz system dynamics
- CurlicueFractal: Fractal pattern based on cumulative angle visualization
- PercolationModel: Statistical physical model generating fractal patterns
- FeigenbaumDiagram: Visual representation of bifurcation in chaotic systems
- FractalDimension: Calculator for estimating fractal dimensions
- SpaceFillingCurve: Closed path space-filling curve implementation
- RandomWalkFractal: Random walk patterns in bounded spaces
- UniformMassCenterTriangle: Uniform mass center triangle fractal generator

Fractals are geometric shapes containing detailed structure at arbitrarily small scales,
typically having self-similarity and fractional dimensions. The implementation showcases
both deterministic and random fractals, combining traditional Euclidean geometry with
chaos theory and complex dynamics.

Mathematical Properties:
- Self-similarity at different scales
- Recursive patterns
- Complex number mapping
- Chaos theory applications
- Non-integer dimensions
- Topological transformations

References:
- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature
- Falconer, K. (2003). Fractal Geometry: Mathematical Foundations and Applications
- Barnsley, M. (1988). Fractals Everywhere
# trunk-ignore(ruff/E501)
- Peitgen, H.-O., Jürgens, H., & Saupe, D. (2004). Chaos and Fractals: New Frontiers of Science
"""  # noqa: E501

from __future__ import annotations
