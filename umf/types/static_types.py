"""This module defines the types used in the useful-math-functions package.

!!! info "About"

    The following types are defined in this module:

    - `UniversalArray`: A numpy NDArray that can hold integers,
        floats, or complex numbers of any dimension.
    - `UniversalArrayTuple`: A tuple of numpy NDArrays that can hold integers,
        floats, or complex numbers of any dimension.
    - `UniversalFloatTuple`: A tuple of floats.
    - `MeshArray`:  A list of numpy NDArrays that can hold floats. This is used
        for plotting functions that take in two or more inputs.
    - `PlotlyScatterParameters`: A dictionary of parameters used for plotting
        functions that take in two or more inputs.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import NDArray


UniversalArray = NDArray[np.int_ | np.float64 | np.complex128]
UniversalArrayTuple = tuple[UniversalArray, ...]
UniversalFloatTuple = tuple[float, ...]
MeshArray = list[NDArray[np.float64]]
PlotlyScatterParameters = dict[
    str,
    UniversalArray | str | dict[str, UniversalArray | str | int],
]
