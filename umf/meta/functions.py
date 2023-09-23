"""Reference class for functions to generate data for benchmarking."""
from __future__ import annotations

from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from umf.constants.dimensions import __1d__
from umf.constants.exceptions import MissingXError
from umf.constants.exceptions import NoCumulativeError
from umf.constants.exceptions import NotAPositiveNumberError
from umf.constants.exceptions import OutOfDimensionError
from umf.constants.exceptions import OutOfRangeError
from umf.meta.api import ResultsDistributionAPI
from umf.meta.api import ResultsFunctionAPI


if TYPE_CHECKING:
    from umf.meta.api import MinimaAPI
    from umf.meta.api import SummaryStatisticsAPI
    from umf.types.static_types import UniversalArray
    from umf.types.static_types import UniversalArrayTuple


class CoreElements(ABCMeta):
    """Metaclass for functions."""

    @property
    @abstractmethod
    def __eval__(cls) -> UniversalArray:
        """Evaluate the function."""

    @property
    @abstractmethod
    def __input__(cls) -> UniversalArrayTuple:
        """Return the input data."""


class OptFunction(ABC, metaclass=CoreElements):
    """Base class for functions for optimization.

    Args:
        x: Input data, which can be one, two, three, or higher dimensional.
        kwargs: Keyword arguments for the function.

    Raises:
        MissingXError: If no input data is specified.
    """

    def __init__(self, *x: UniversalArray) -> None:
        """Initialize the function."""
        if x is None:
            raise MissingXError

        self._x: tuple[UniversalArray, ...] = x
        self.dimension: int = len(x)

    @property
    def __input__(self) -> UniversalArrayTuple:
        """Return the input data."""
        return self._x

    @property
    @abstractmethod
    def __eval__(self) -> UniversalArray:
        """Evaluate the function."""

    @property
    @abstractmethod
    def __minima__(self) -> MinimaAPI:
        """Return the zero function."""

    def __call__(self) -> ResultsFunctionAPI:
        """Return the results of the function."""
        return ResultsFunctionAPI(
            x=self.__input__,
            result=self.__eval__,
            minima=self.__minima__,
            doc=self.__doc__,
        )


class ContinuousDistributionBase(ABC, metaclass=CoreElements):
    """Base class for distributions with a standard deviation and beta parameter.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float ): Mean of the distribution. Defaults to 0.
        beta (float): Shape parameter of the distribution. Defaults to 1.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if x is None:
            raise MissingXError

        if len(x) != __1d__:
            raise OutOfDimensionError(
                function_name=self.__class__.__name__,
                dimension=__1d__,
            )
        self._x = x[0]
        self.mu = mu
        self.cumulative = cumulative

    @property
    def __input__(self) -> UniversalArrayTuple:
        """Return the input data."""
        return (np.array(self._x),)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the function."""
        return (
            self.cumulative_distribution_function()
            if self.cumulative
            else self.probability_density_function()
        )

    @abstractmethod
    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""

    @property
    @abstractmethod
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        raise NoCumulativeError

    def __call__(self) -> ResultsDistributionAPI:
        """Return the results of the function."""
        return ResultsDistributionAPI(
            x=self.__input__,
            result=self.__eval__,
            summary=self.__summary__,
            doc=self.__doc__,
        )


class ContinuousPure(ContinuousDistributionBase):
    """Base class for continuous distributions with a standard deviation.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """


class ContinuousWBeta(ContinuousDistributionBase):
    """Base class for continuous distributions with beta parameter.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        beta (float): Shape parameter of the distribution. Defaults to 1.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        beta: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, mu=mu, cumulative=cumulative)
        if beta <= 0:
            raise NotAPositiveNumberError(number="beta")
        self.beta = beta


class ContinuousWSigma(ContinuousDistributionBase):
    """Base class for continuous distributions with a standard deviation.

    Args:
        x (UniversalArray): Input data, which can be one, two, three, or higher
            dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        sigma (float): Standard deviation of the distribution. Defaults to 1.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, mu=mu, cumulative=cumulative)
        self.sigma = sigma


class ContinuousMixed(ContinuousWSigma):
    r"""Base class for continuous distributions with mixed parameters.

    Args:
        x (UniversalArray): Input data, which can be one, two, three, or higher
            dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        sigma (float): Standard deviation of the distribution. Defaults to 1.
        zeta (float): Shape parameter of the distribution. Defaults to 0.0.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        zeta: float = 0.0,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, mu=mu, sigma=sigma, cumulative=cumulative)
        self.zeta = zeta


class ContinuousWLambda(ContinuousDistributionBase):
    """Base class for continuous distributions with beta parameter.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        beta (float): Shape parameter of the distribution. Defaults to 1.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """

    def __init__(
        self,
        *x: UniversalArray,
        lambda_: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, cumulative=cumulative)
        if lambda_ <= 0:
            raise NotAPositiveNumberError(number=lambda_)
        self.lambda_ = lambda_


class ContinousPseudo(ContinuousWSigma):
    """Base class for continuous distributions for pseudo Voigt like functions.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        sigma (float): Standard deviation of the distribution. Defaults to 1.
        eta (float): Shape parameter of the distribution. Defaults to 0.5.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
        OutOfRangeError: If the input data is not in the interval $[0, 1]$.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        eta: float = 0.5,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, mu=mu, sigma=sigma, cumulative=cumulative)
        if eta < 0 or eta > 1:
            raise OutOfRangeError(
                function_name=self.__class__.__name__,
                start_range=0,
                end_range=1,
            )
        self.eta = eta


class ContinousAsymmetricPseudo(ContinousPseudo):
    r"""Base class for continuous distributions for asym. pseudo Voigt like functions.

    Note:
        In terms of pseudo Voigt like functions, the $\gamma$ parameter is used to
        provide an asymetric line shape. The $\gamma$ parameter is defined as the ratio
        of the Lorentzian contribution to the Gaussian contribution. In case of
        **XPS** and **XAS**, these type of functions are popular to model the line shape
        of the photoemission and absorption spectra.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        sigma (float): Standard deviation of the distribution. Defaults to 1.
        eta (float): Shape parameter of the distribution. Defaults to 0.5.
        gamma (float): Asymmetry parameter of the distribution. Defaults to 0.0.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
        OutOfRangeError: If the input data is not in the interval $[0, 1]$.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        eta: float = 0.5,
        gamma: float = 0.0,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, mu=mu, sigma=sigma, eta=eta, cumulative=False)
        self.gamma = gamma


class SemiContinuous(ContinuousDistributionBase):
    """Base class for semi continuous distributions with a standard deviation.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """


class SemiContinuousWBeta(ContinuousDistributionBase):
    """Base class for continuous distributions with beta parameter.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        beta (float): Shape parameter of the distribution. Defaults to 1.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        beta: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if (min_x := np.min(x)) < 0:
            raise NotAPositiveNumberError(number=float(min_x))
        super().__init__(*x, mu=mu, cumulative=cumulative)
        if beta <= 0:
            raise NotAPositiveNumberError(number=beta)
        self.beta = beta


class SemiContinuousWSigma(ContinuousDistributionBase):
    """Base class for continuous distributions with a standard deviation.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        sigma (float): Standard deviation of the distribution. Defaults to 1.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if np.min(x) < 0:
            raise NotAPositiveNumberError(var_number="*x", number=float(np.min(x)))
        super().__init__(*x, mu=mu, cumulative=cumulative)
        self.sigma = sigma


class DiscreteDistributionBase(ABC, metaclass=CoreElements):
    """Base class for discrete distributions.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """

    def __init__(
        self,
        *x: UniversalArray,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if x is None:
            raise MissingXError

        if len(x) != __1d__:
            raise OutOfDimensionError(
                function_name=self.__class__.__name__,
                dimension=__1d__,
            )
        self._x = x[0]
        self.cumulative = cumulative

    @property
    def __input__(self) -> UniversalArrayTuple:
        """Return the input data."""
        return (np.array(self._x),)

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the function."""
        return (
            self.cumulative_distribution_function()
            if self.cumulative
            else self.probability_mass_function()
        )

    @abstractmethod
    def probability_mass_function(self) -> UniversalArray:
        """Return the probability mass function."""

    @property
    @abstractmethod
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        raise NoCumulativeError

    def __call__(self) -> ResultsDistributionAPI:
        """Return the results of the function."""
        return ResultsDistributionAPI(
            x=self.__input__,
            result=self.__eval__,
            summary=self.__summary__,
            doc=self.__doc__,
        )


class DiscretePure(DiscreteDistributionBase):
    """Base class for discrete distributions.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
    """


class DiscreteP(DiscreteDistributionBase):
    """Base class for discrete distributions with a probability parameter.

    Args:
        x (UniversalArray): Input data, which currently must be one dimensional.
        p (float): Probability parameter of the distribution. Defaults to 0.5.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
    """

    def __init__(
        self,
        *x: UniversalArray,
        p: float = 0.5,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, cumulative=cumulative)
        if p < 0 or p > 1:
            raise OutOfRangeError(
                function_name=self.__class__.__name__,
                start_range=0,
                end_range=1,
            )
        self.p = p
        self.q = 1 - p


class Continuous2PiInterval(ContinuousDistributionBase):
    r"""Base class for continuous distributions with fixed interval of $2\pi$.

    Args:
        *x (UniversalArray): Input data, which currently must be one dimensional.
        mu (float): Mean of the distribution. Defaults to 0.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        OutOfRangeError: If the input data is not in the interval $[-\pi, \pi]$.
        MissingXError: If no input data is specified.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        kappa: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if np.min(x) < -np.pi or np.max(x) > np.pi:
            raise OutOfRangeError(
                function_name=self.__class__.__name__,
                start_range=-np.pi,
                end_range=np.pi,
            )
        super().__init__(*x, mu=mu, cumulative=cumulative)
        self.kappa = kappa


class ContinuousBoundedInterval(ContinuousDistributionBase):
    """Base class for continuous distributions with a bounded interval.

    Args:
        *x (UniversalArray): Input data, which currently must be one dimensional.
        start (float): Start of the interval. Defaults to 0.
        end (float): End of the interval. Defaults to 1.
        cumulative (bool): Whether to return the cumulative distribution function.
            Defaults to False.

    Raises:
        OutOfRangeError: If the input data is not in the interval $[start, end]$.
        MissingXError: If no input data is specified.
    """

    def __init__(
        self,
        *x: UniversalArray,
        start: float = 0,
        end: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if np.min(x) < start or np.max(x) > end:
            raise OutOfRangeError(
                function_name=self.__class__.__name__,
                start_range=start,
                end_range=end,
            )
        super().__init__(*x, cumulative=cumulative)
