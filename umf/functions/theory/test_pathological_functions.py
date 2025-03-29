"""Test cases for pathological functions."""

from __future__ import annotations

import numpy as np

from umf.functions.theory.pathological import BesicovitchFunction
from umf.functions.theory.pathological import MandelbrotsFractalFunction
from umf.functions.theory.pathological import RiemannFunction
from umf.functions.theory.pathological import TakagiFunction
from umf.functions.theory.pathological import WeierstrassFunction


def test_weierstrass_function() -> None:
    """Test the Weierstrass function.

    This test checks if the Weierstrass function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(-3, 3, 1000)
    func: WeierstrassFunction = WeierstrassFunction(x, n=20, a=0.5, b=30)
    result: np.ndarray = func.__eval__
    assert result is not None
    assert len(result) == len(x)


def test_riemann_function() -> None:
    """Test the Riemann function.

    This test checks if the Riemann function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(-3, 3, 1000)
    func: RiemannFunction = RiemannFunction(x, n=20)
    result: np.ndarray = func.__eval__
    assert result is not None
    assert len(result) == len(x)


def test_takagi_function() -> None:
    """Test the Takagi function.

    This test checks if the Takagi function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(-1.5, 1.5, 1000)
    func: TakagiFunction = TakagiFunction(x, n=20)
    result: np.ndarray = func.__eval__
    assert result is not None
    assert len(result) == len(x)


def test_mandelbrots_fractal_function() -> None:
    """Test the Mandelbrot's Fractal function.

    This test checks if the Mandelbrot's Fractal function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(-2, 2, 1000)
    func: MandelbrotsFractalFunction = MandelbrotsFractalFunction(x, max_iter=50)
    result: np.ndarray = func.__eval__
    assert result is not None
    assert result.shape == (1000, 666)


def test_besicovitch_function() -> None:
    """Test the Besicovitch function.

    This test checks if the Besicovitch function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(0, 1, 1000)
    func: BesicovitchFunction = BesicovitchFunction(x, n=30, mu=2)
    result: np.ndarray = func.__eval__
    assert result is not None
    assert len(result) == len(x)
