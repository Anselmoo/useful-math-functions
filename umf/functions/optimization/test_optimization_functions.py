from __future__ import annotations

import numpy as np

from umf.functions.optimization.bowl_shaped import PermBetaDFunction
from umf.functions.optimization.bowl_shaped import SumOfDifferentPowersFunction
from umf.functions.optimization.bowl_shaped import SumSquaresFunction
from umf.functions.optimization.bowl_shaped import TridFunction
from umf.functions.optimization.bowl_shaped import ZirilliFunction


def test_perm_beta_d_function() -> None:
    x = np.array([1, 2, 3])
    func = PermBetaDFunction(x)
    result = func.__eval__
    expected = 0.0
    assert np.isclose(result, expected, rtol=1e-5, atol=1e-8)


def test_trid_function() -> None:
    x = np.array([1, 2, 3])
    func = TridFunction(x)
    result = func.__eval__
    expected = -2.0
    assert np.isclose(result, expected, rtol=1e-5, atol=1e-8)


def test_sum_squares_function() -> None:
    x = np.array([1, 2, 3])
    func = SumSquaresFunction(x)
    result = func.__eval__
    expected = 14.0
    assert np.isclose(result, expected, rtol=1e-5, atol=1e-8)


def test_sum_of_different_powers_function() -> None:
    x = np.array([1, 2, 3])
    func = SumOfDifferentPowersFunction(x)
    result = func.__eval__
    expected = 36.0
    assert np.isclose(result, expected, rtol=1e-5, atol=1e-8)


def test_zirilli_function() -> None:
    x = np.array([1, 2])
    func = ZirilliFunction(x)
    result = func.__eval__
    expected = 2.0
    assert np.isclose(result, expected, rtol=1e-5, atol=1e-8)
