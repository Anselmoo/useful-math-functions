"""API for function classes."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from umf.types.static_types import MeshArray  # noqa: TCH001
from umf.types.static_types import UniversalArray  # noqa: TCH001
from umf.types.static_types import UniversalArrayTuple  # noqa: TCH001
from umf.types.static_types import UniversalFloatTuple  # noqa: TCH001


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
