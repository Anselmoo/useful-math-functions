"""Test the support functions."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.xfail(
    reason="Known precision issues with wofz function"
    " caused by migration from numpy v1 to v2",
)
def test_wofz_accuracy() -> None:
    """Test the accuracy of the Faddeeva function."""
    x = np.linspace(-5, 5, 100)
    y1 = wofz(x)  # Assuming wofz is defined/imported elsewhere
    y2 = special.wofz(x)
    close = np.isclose(y1, y2, rtol=1e-3, atol=1e-3, equal_nan=True)
    if not np.all(close):
        diff_indices = np.where(~close)[0]
        message = (
            f"Values differ at indices {diff_indices}."
            f" y1: {y1[~close]}, y2: {y2[~close]}"
        )
        pytest.fail(message)
