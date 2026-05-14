"""Tests for the simple function collection."""

from __future__ import annotations

import numpy as np
import pytest

from umf.functions.simple import __all__ as simple_function_names
from umf.functions.simple import elementary as simple_module
from umf.functions.simple.elementary import LogisticMirrorSimpleFunction
from umf.functions.simple.elementary import LogQuadraticSimpleFunction
from umf.functions.simple.elementary import PseudoHuberSimpleFunction
from umf.functions.simple.elementary import SineSquaredSimpleFunction
from umf.functions.simple.elementary import SquareSimpleFunction
from umf.meta.functions import OptFunction


def test_simple_function_count() -> None:
    """Ensure the package exposes exactly one hundred simple functions."""
    assert len(simple_function_names) == 100


@pytest.mark.parametrize("name", simple_function_names)
def test_all_simple_functions_are_instantiable(name: str) -> None:
    """Ensure each generated simple function behaves like a 1D OptFunction."""
    function_class = getattr(simple_module, name)
    x = np.array([-1.0, 0.0, 1.0])
    result = function_class(x)

    assert issubclass(function_class, OptFunction)
    assert result.__eval__.shape == x.shape
    assert np.all(np.isfinite(result.__eval__))
    assert result.__minima__.f_x == 0.0
    assert result.__minima__.x == (0.0,)


def test_selected_simple_function_values() -> None:
    """Check a few reference formulas explicitly."""
    x = np.array([-1.0, 0.0, 1.0])

    assert SquareSimpleFunction(x).__eval__ == pytest.approx(x**2)
    assert LogQuadraticSimpleFunction(x).__eval__ == pytest.approx(np.log1p(x**2))
    assert SineSquaredSimpleFunction(x).__eval__ == pytest.approx(np.sin(x) ** 2)
    assert PseudoHuberSimpleFunction(x).__eval__ == pytest.approx(
        np.sqrt(1.0 + x**2) - 1.0
    )
    assert LogisticMirrorSimpleFunction(x).__eval__ == pytest.approx(
        2.0 * np.logaddexp(0.0, x) - x - 2.0 * np.log(2.0)
    )


@pytest.mark.parametrize("name", simple_function_names)
def test_all_simple_functions_have_png_doctest_examples(name: str) -> None:
    """Ensure each simple function docstring includes a PNG-generating example."""
    function_class = getattr(simple_module, name)
    doc = function_class.__doc__

    assert doc is not None
    assert "Examples:" in doc
    assert "Visualization Example" in doc
    assert "Notes:" in doc
    assert "Args:" in doc
    assert f'plt.savefig("{name}.png", dpi=300, transparent=True)' in doc
