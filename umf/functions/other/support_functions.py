"""Support functions for the umf.functions.other module."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


def wofz(z: UniversalArray) -> UniversalArray:
    """Compute the Faddeeva function for a given complex argument."""
    return np.exp(-(z**2)) * numeric_erfc(-1j * z)


def erfc(z: UniversalArray) -> UniversalArray:
    """Compute the complementary error function for a given complex argument."""
    return 1 - erf(z)


def numeric_erfc(z: UniversalArray) -> UniversalArray:
    """Compute the complementary error function for a given complex argument."""
    return 1 - numeric_erf(z)


def erf(x: UniversalArray) -> UniversalArray:
    """Return the error function of x."""
    return np.vectorize(math.erf)(x)


def numeric_erf(z: UniversalArray) -> UniversalArray:
    """Compute the error function for a given complex argument."""
    z = np.asarray(z)
    sign: UniversalArray = np.sign(z)
    z = sign * z
    t = 1 / (1 + 0.5 * z)
    result = 1 - t * np.exp(
        -(z**2)
        - 1.26551223
        + 1.00002368 * t
        + 0.37409196 * t**2
        + 0.09678418 * t**3
        - 0.18628806 * t**4
        + 0.27886807 * t**5
        - 1.13520398 * t**6
        + 1.48851587 * t**7
        - 0.82215223 * t**8
        + 0.17087277 * t**9,
    )
    return sign * result


def gamma(x: UniversalArray, batch_size: int = 100) -> UniversalArray:
    """Return the gamma function of x using the Lanczos approximation.

    Args:
        x (UniversalArray): The value(s) at which the function is evaluated.
        batch_size (int): The batch size for the Lanczos approximation. Defaults to 100.

    Returns:
        UniversalArray: The gamma function of x using the Lanczos approximation.
    """

    # Define the coefficients of the Lanczos approximation
    def _gamma(x: UniversalArray) -> UniversalArray:
        """Return the gamma function of x using the Lanczos approximation."""
        g = 7
        p: list[float] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]

        if np.any(np.less(x, 0.5)):
            return np.pi / (np.sin(np.pi * x) * _gamma(1 - x))
        x = x - 1
        a = p[0] + np.sum([p[i] / (x + i) for i in range(1, len(p))], axis=0)
        t = x + g + 0.5
        return np.sqrt(2 * np.pi) * np.power(t, x + 0.5) * np.exp(-t) * a

    y: UniversalArray = np.zeros_like(x)

    for i in range(0, len(x), batch_size):
        y[i : i + batch_size] = _gamma(x[i : i + batch_size])
    return y


def combinations(n: int | np.ndarray, k: int | np.ndarray) -> int | np.ndarray:
    """Return the number of combinations of n things taken k at a time.

    Args:
        n (int | np.ndarray): The number of things.
        k (int | np.ndarray): The number of things taken at a time.

    Returns:
        int | np.ndarray: The number of combinations of n things taken k at a time.
    """
    n = np.array(n, dtype=int)
    k = np.array(k, dtype=int)
    if n.ndim == 0:
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    c: np.ndarray = np.zeros_like(n)
    for i in np.ndindex(n.shape):
        c[i] = math.factorial(n[i]) // (
            math.factorial(k[i]) * math.factorial(n[i] - k[i])
        )
    return c
