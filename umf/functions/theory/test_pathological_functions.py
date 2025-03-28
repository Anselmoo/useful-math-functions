from __future__ import annotations

import numpy as np

from umf.functions.theory.pathological import BesicovitchFunction
from umf.functions.theory.pathological import MandelbrotsFractalFunction
from umf.functions.theory.pathological import RiemannFunction
from umf.functions.theory.pathological import TakagiFunction
from umf.functions.theory.pathological import WeierstrassFunction


def test_weierstrass_function() -> None:
    x = np.linspace(-3, 3, 1000)
    func = WeierstrassFunction(x, n=20, a=0.5, b=30)
    result = func.__eval__
    assert result is not None
    assert len(result) == len(x)


def test_riemann_function() -> None:
    x = np.linspace(-3, 3, 1000)
    func = RiemannFunction(x, n=20)
    result = func.__eval__
    assert result is not None
    assert len(result) == len(x)


def test_takagi_function() -> None:
    x = np.linspace(-1.5, 1.5, 1000)
    func = TakagiFunction(x, n=20)
    result = func.__eval__
    assert result is not None
    assert len(result) == len(x)


def test_mandelbrots_fractal_function() -> None:
    x = np.linspace(-2, 2, 1000)
    func = MandelbrotsFractalFunction(x, max_iter=50)
    result = func.__eval__
    assert result is not None
    assert result.shape == (1000, 666)


def test_besicovitch_function() -> None:
    x = np.linspace(0, 1, 1000)
    func = BesicovitchFunction(x, n=30, mu=2)
    result = func.__eval__
    assert result is not None
    assert len(result) == len(x)
