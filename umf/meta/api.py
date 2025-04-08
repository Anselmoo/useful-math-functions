"""API for function classes."""

from __future__ import annotations

import warnings

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from umf.constants.dimensions import __3d__
from umf.types.static_types import MeshArray  # noqa: TC001
from umf.types.static_types import UniversalArray  # noqa: TC001
from umf.types.static_types import UniversalArrayTuple  # noqa: TC001
from umf.types.static_types import UniversalFloatTuple  # noqa: TC001


class MinimaAPI(BaseModel):
    """Minima API for optimization functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    f_x: float | UniversalArray = Field(
        ...,
        description="Value of the function at the minimum or minima.",
    )
    x: UniversalArrayTuple | UniversalFloatTuple = Field(
        ...,
        description="Input data, where the minimum or minima is located.",
    )


class MaximaAPI(BaseModel):
    """Maxima API for optimization functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    f_x: float | UniversalArray = Field(
        ...,
        description="Value of the function at the maximum or maxima.",
    )
    x: UniversalArrayTuple | UniversalFloatTuple = Field(
        ...,
        description="Input data, where the maximum or maxima is located.",
    )


class ResultsFunctionAPI(BaseModel):
    """Results API for optimization functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: UniversalArrayTuple = Field(
        ...,
        description="Input data, which can be one, two, three, or higher dimensional.",
    )
    result: UniversalArray | MeshArray = Field(
        ...,
        description="Function value as numpy array or numpy mesh grid array.",
    )
    minima: MinimaAPI | None = Field(
        default=None,
        description="Tuple of minima as numpy arrays.",
    )
    maxima: MaximaAPI | None = Field(
        default=None,
        description="Tuple of maxima as numpy arrays.",
    )
    doc: str | None = Field(..., description="Function documentation string.")


class SummaryStatisticsAPI(BaseModel):
    """API for summary statistics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    mean: float | None = Field(
        default=...,
        description="Mean value of the data.",
    )
    variance: float | None = Field(
        default=...,
        description="Variance of the data.",
    )
    mode: float | UniversalFloatTuple | None = Field(
        default=...,
        description="Mode or modes of the data.",
    )
    doc: str | None = Field(
        default=...,
        description="Documentation string for the summary statistics.",
    )


class ResultsDistributionAPI(BaseModel):
    """Results API for distribution functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: UniversalArrayTuple = Field(
        ...,
        description="Input data, which can be one, two, three, or higher dimensional.",
    )
    result: UniversalArray | MeshArray = Field(
        ...,
        description="Function value as numpy array or numpy mesh grid array.",
    )
    summary: SummaryStatisticsAPI = Field(
        ...,
        description="Summary statistics of the data.",
    )
    doc: str | None = Field(..., description="Function documentation string.")


class ResultsPathologicalAPI(BaseModel):
    """Results API for pathological functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: UniversalArrayTuple = Field(
        ...,
        description="Input data, which can be one, two, three, or higher dimensional.",
    )
    result: UniversalArray | MeshArray = Field(
        ...,
        description="Function value as numpy array or numpy mesh grid array.",
    )

    doc: str | None = Field(..., description="Function documentation string.")


class ResultsChaoticOscillatorAPI(BaseModel):
    """Results API for chaotic oscillator functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    t: UniversalArrayTuple = Field(
        ...,
        description="Time array for the chaotic oscillator.",
    )
    initial_state: dict[str, UniversalArray] = Field(
        default=...,
        description="Initial conditions for the chaotic pendulum.",
    )
    result: UniversalArray = Field(
        default=...,
        description="Result of the chaotic oscillator.",
    )
    doc: str | None = Field(
        default=...,
        description="Function documentation string.",
    )


class ResultsHyperbolicAPI(BaseModel):
    """Results API for hyperbolic functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: UniversalArrayTuple = Field(
        ...,
        description="Input data, which can be one, two, three, or higher dimensional.",
    )
    result: UniversalArray | MeshArray = Field(
        ...,
        description="Function value as numpy array or numpy mesh grid array.",
    )
    doc: str | None = Field(..., description="Function documentation string.")


class ResultsFractalAPI(BaseModel):
    """Results API for fractal functions.

    This class provides a standardized structure for returning fractal data,
    including the input parameters, resulting fractal representation, and metadata.

    Attributes:
        x (UniversalArrayTuple): Input data for the fractal generation.
        result (UniversalArray | MeshArray | list | dict): Fractal data, which may
            be iteration counts, coordinates, or other representations.
        parameters (dict | None): Parameters used to generate the fractal.
        dimension (float | None): Fractal dimension, if calculated.
        doc (str | None): Function documentation string.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: UniversalArrayTuple = Field(
        ...,
        description="Input data for the fractal generation.",
    )
    result: UniversalArray | MeshArray | list | dict = Field(
        ...,
        description="Fractal data, which may be iteration counts, coordinates,"
        " or other representations.",
    )
    parameters: dict | None = Field(
        default=None,
        description="Parameters used to generate the fractal.",
    )
    dimension: float | None = Field(
        default=None,
        description="Fractal dimension, if calculated.",
    )
    doc: str | None = Field(..., description="Function documentation string.")

    @model_validator(mode="after")
    def validate_dimension(self) -> ResultsFractalAPI:
        """Validate the fractal dimension.

        Checks that the provided fractal dimension is within a reasonable range
        (between 0 and 3 for most fractals).

        Returns:
            ResultsFractalAPI: The validated model instance
        """
        if self.dimension is not None and not (0 <= self.dimension <= __3d__):
            warnings.warn(
                message=f"Unusual fractal dimension: {self.dimension}. "
                f"Most fractals have dimensions between 0 and 3.",
                category=UserWarning,
                stacklevel=1,
            )
        return self
