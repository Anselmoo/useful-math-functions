"""Test the support functions."""
from __future__ import annotations

import numpy as np

from scipy import special

from umf.functions.other.support_functions import combinations
from umf.functions.other.support_functions import erf
from umf.functions.other.support_functions import erfc
from umf.functions.other.support_functions import wofz


def test_combinations_accuracy() -> None:
    """Test the accuracy of the combinations function."""
    assert combinations(10, 5) == 252
    assert combinations(10, 5) == special.comb(10, 5)
    assert np.allclose(
        combinations(np.array([15, 10]), np.array([5, 5])),
        np.array(
            [
                special.comb(15, 5),
                special.comb(10, 5),
            ],
        ),
    )


def test_erf_accuracy() -> None:
    """Test the accuracy of the error function."""
    x = np.linspace(-5, 5, 100)
    y1 = erf(x)
    y2 = special.erf(x)
    assert np.allclose(y1, y2, rtol=1e-5, atol=1e-8)


def test_erfc_accuracy() -> None:
    """Test the accuracy of the complementary error function."""
    x = np.linspace(-5, 5, 100)
    y1 = erfc(x)
    y2 = 1 - special.erf(x)
    assert np.allclose(y1, y2, rtol=1e-5, atol=1e-8)


def test_wofz_accuracy() -> None:
    """Test the accuracy of the Faddeeva function."""
    x = np.linspace(-5, 5, 100)
    y1 = wofz(x)
    y2 = special.wofz(x)
    assert np.allclose(y1, y2, rtol=1e-3, atol=1e-3)
