"""Unit test configuration for UMF."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def close_figures() -> Generator[None, Any, None]:
    """Close all figures after each test."""
    yield
    plt.close("all")
