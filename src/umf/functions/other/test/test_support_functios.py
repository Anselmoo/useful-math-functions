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
    assert combinations(np.array([15, 10]), np.array([5, 5])) == pytest.approx(
        np.array([special.comb(15, 5), special.comb(10, 5)]), rel=1e-5, abs=1e-8
    )


def test_erf_accuracy() -> None:
    """Test the accuracy of the error function."""
    x = np.linspace(-5, 5, 100)
    y1 = erf(x)
    y2 = special.erf(x)
    assert y1 == pytest.approx(y2, rel=1e-5, abs=1e-8)


def test_erfc_accuracy() -> None:
    """Test the accuracy of the complementary error function."""
    x = np.linspace(-5, 5, 100)
    y1 = erfc(x)
    y2 = 1 - special.erf(x)
    assert y1 == pytest.approx(y2, rel=1e-5, abs=1e-8)


@pytest.mark.xfail(
    reason="Known precision issues with wofz function"
    " caused by migration from numpy v1 to v2",
)
def test_wofz_accuracy() -> None:
    """Test the accuracy of the Faddeeva function."""
    x = np.linspace(-5, 5, 100)
    y1 = wofz(x)  # Assuming wofz is defined/imported elsewhere
    y2 = special.wofz(x)
    assert y1 == pytest.approx(y2, rel=1e-3, abs=1e-3)
