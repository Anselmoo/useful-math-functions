"""Unit tests for optimization functions."""

from __future__ import annotations

import numpy as np

from umf.functions.optimization.bowl_shaped import PermBetaDFunction
from umf.functions.optimization.bowl_shaped import SumOfDifferentPowersFunction
from umf.functions.optimization.bowl_shaped import SumSquaresFunction
from umf.functions.optimization.bowl_shaped import TridFunction
from umf.functions.optimization.bowl_shaped import ZirilliFunction


def test_perm_beta_d_function() -> None:
    """Test the Perm Beta D function.

    This test checks if the Perm Beta D function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.array([1, 2, 3])
    func: PermBetaDFunction = PermBetaDFunction(x)
    result: float = float(np.sum(func.__eval__))
    expected: float = 11.25
    assert np.all(np.isclose(result, expected, rtol=1e-5, atol=1e-8))


def test_trid_function() -> None:
    """Test the Trid function.

    This test checks if the Trid function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.array([1, 2, 3])
    func: TridFunction = TridFunction(x)
    result: float = float(np.sum(func.__eval__))
    expected: float = -9.0
    assert np.all(np.isclose(result, expected, rtol=1e-5, atol=1e-8))


def test_sum_squares_function() -> None:
    """Test the Sum Squares function.

    This test checks if the Sum Squares function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.array([1, 2, 3])
    func: SumSquaresFunction = SumSquaresFunction(x)
    result: float = float(np.sum(func.__eval__))
    expected: float = 14.0
    assert np.all(np.isclose(result, expected, rtol=1e-5, atol=1e-8))


def test_sum_of_different_powers_function() -> None:
    """Test the Sum of Different Powers function.

    This test checks if the Sum of Different Powers function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.array([1, 2, 3])
    func: SumOfDifferentPowersFunction = SumOfDifferentPowersFunction(x)
    result: float = float(np.sum(func.__eval__))
    expected: float = 14.0
    assert np.all(np.isclose(result, expected, rtol=1e-5, atol=1e-8))


def test_zirilli_function() -> None:
    """Test the Zirilli function.

    This test checks if the Zirilli function returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.array([[1], [2]])
    func: ZirilliFunction = ZirilliFunction(*x)
    result: float = float(func.__eval__)
    expected: float = 1.85
    assert np.all(np.isclose(result, expected, rtol=1e-5, atol=1e-8))
