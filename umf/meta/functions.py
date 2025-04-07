"""Reference class for functions to generate data for benchmarking."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import odeint

from umf.constants.dimensions import __1d__
from umf.constants.exceptions import (
    ExcessiveExponentError,
    MissingXError,
    NoCumulativeError,
    NotAPositiveNumberError,
    NotTupleArrayError,
    OutOfDimensionError,
    OutOfRangeError,
    TimeFormatError,
)
from umf.meta.api import (
    ResultsChaoticOscillatorAPI,
    ResultsDistributionAPI,
    ResultsFractalAPI,
    ResultsFunctionAPI,
    ResultsHyperbolicAPI,
    ResultsPathologicalAPI,
)
from umf.types.static_types import UniversalArray

if TYPE_CHECKING:
    from umf.meta.api import MinimaAPI, SummaryStatisticsAPI
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
        if x[0] is None:
            raise MissingXError

        if isinstance(x, tuple):
            for i in x:
                if not isinstance(i, np.ndarray):
                    raise NotTupleArrayError

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


class HyperbolicFunction(ABC, metaclass=CoreElements):
    """Class for hyperbolic functions."""

    def __init__(self, *args: UniversalArray) -> None:
        """Initialize the hyperbolic function."""
        if args[0] is None:
            raise MissingXError

        self._x: tuple[UniversalArray, ...] = args
        self.dimension: int = len(args)

    @property
    def __input__(self) -> UniversalArrayTuple:
        """Return the input data for the hyperbolic function."""
        return self._x

    @property
    @abstractmethod
    def __eval__(self) -> UniversalArray:
        """Evaluate the hyperbolic function."""

    def __call__(self) -> ResultsHyperbolicAPI:
        """Return the results of the hyperbolic function."""
        return ResultsHyperbolicAPI(
            x=self.__input__,
            result=np.asarray(self.__eval__),
            doc=self.__doc__,
        )


class PathologicalBase(ABC, metaclass=CoreElements):
    """Base class for pathological functions.

    This class serves as a template for creating objects that represent pathological
    functions. These functions are defined over a specific interval and are
    characterized by their shape parameters. The class provides the structure
    for handling input data, defining the interval of interest, and specifying
    the shape parameters of the function.

    Args:
        x (UniversalArray): Input data, which currently must be one-dimensional.
        n_0 (int): Start of the interval. Defaults to 0.
        n_1 (int): End of the interval. Defaults to 100.
        max_safe_exponent (int | float): Maximum safe exponent. Defaults to 200.

    Raises:
        MissingXError: If no input data (x) is specified. This error is raised to ensure
            that the function has the necessary data to operate on.
        OutOfDimensionError: If the input data (x) has more than one dimension.
            This error is raised because the function is designed to work with
            one-dimensional data only.
    """

    def __init__(
        self,
        *x: UniversalArray,
        n_0: int = 0,
        n_1: int = 100,
        max_safe_exponent: float = 200,
    ) -> None:
        """Initialize the function."""
        if x[0] is None:
            raise MissingXError

        if len(x) != __1d__:
            raise OutOfDimensionError(
                function_name=self.__class__.__name__,
                dimension=__1d__,
            )
        if np.abs(n_1) > max_safe_exponent:
            raise ExcessiveExponentError(
                max_exponent=max_safe_exponent,
                current_exponent=n_1,
            )

        self._x = x[0]
        self.n_0 = n_0
        self.n_1 = n_1

    @property
    def __input__(self) -> UniversalArrayTuple:
        """Return the input data."""
        return (np.array(self._x),)

    @property
    @abstractmethod
    def __eval__(self) -> UniversalArray:
        """Evaluate the function."""

    def __call__(self) -> ResultsPathologicalAPI:
        """Return the results of the function."""
        return ResultsPathologicalAPI(
            x=self.__input__,
            result=self.__eval__,
            doc=self.__doc__,
        )


class PathologicalPure(PathologicalBase):
    """Base class for pathological functions with a standard deviation.

    Args:
        x (UniversalArray): Input data, which currently must be one-dimensional.
        n_0 (int): Start of the interval. Defaults to 0.
        n_1 (int): End of the interval. Defaults to 100.
    """

    def __init__(
        self,
        *x: UniversalArray,
        n_0: int = 0,
        n_1: int = 100,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, n_0=n_0, n_1=n_1)


class PathologicalWithCoefficients(PathologicalBase):
    """Base class for pathological functions with coefficients.

    Args:
        x (UniversalArray): Input data, which currently must be one-dimensional.
        n_1 (int): End of the interval. Defaults to 100.
        a (float): First shape parameter of the function.
        b (float): Second shape parameter of the function.
    """

    def __init__(
        self,
        *x: UniversalArray,
        n_0: int = 0,
        n_1: int = 100,
        a: float,
        b: float,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, n_0=n_0, n_1=n_1)
        self.a = a
        self.b = b


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
        if x[0] is None:
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
        NotAPositiveNumberError: If the beta parameter is not a positive number.
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
            raise NotAPositiveNumberError(var_number="beta", number=beta)
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
            raise NotAPositiveNumberError(var_number="lambda_", number=lambda_)
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
            raise NotAPositiveNumberError(var_number="*x", number=float(min_x))
        super().__init__(*x, mu=mu, cumulative=cumulative)
        if beta <= 0:
            raise NotAPositiveNumberError(var_number="beta", number=beta)
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


class OscillatorsFuncBase(ABC, metaclass=CoreElements):
    """Base class for chaotic oscillators.

    Args:
        *time_points (UniversalArray): The array of time points at which the
            oscillator's state is evaluated.
        time_format (str): The format of the time data. Defaults to "seconds".
        velocity (bool): Whether to return the velocity of the oscillator.
            Defaults to False.

    Raises:
        MissingXError: If no input data is specified.
        OutOfDimensionError: If the input data has more than one dimension.
        TimeFormatError: If the time format is not valid.
    """

    def __init__(
        self,
        *t: UniversalArray,
        time_format: str = "seconds",
        velocity: bool = False,
    ) -> None:
        """Initialize the function."""
        if t[0] is None:
            raise MissingXError

        if len(t) != __1d__:
            raise OutOfDimensionError(
                function_name=self.__class__.__name__,
                dimension=__1d__,
            )

        self.time_format = time_format
        self.t = self.convert2sec(t[0])
        self.velocity = velocity

    def convert2sec(self, t: UniversalArray) -> UniversalArray:
        """Convert the time data to seconds.

        Args:
            t (UniversalArray): The initial time data as input.

        Returns:
            UniversalArray: The time data converted to seconds.
        """
        conversion_factors = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}

        if self.time_format in conversion_factors:
            return t * conversion_factors[self.time_format]
        raise TimeFormatError(
            time_format=self.time_format,
            valid_formats=list(conversion_factors.keys()),
        )

    @property
    def __input__(self) -> UniversalArrayTuple:
        """Return the input data."""
        return (self.t,)

    def setup_initial_state(self) -> UniversalArray:
        """Setup the initial state of the oscillator."""
        raise NotImplementedError

    @property
    @abstractmethod
    def __initial_configuration__(self) -> dict[str, UniversalArray]:
        """Initialize the state of the oscillator."""

    @property
    @abstractmethod
    def initial_state(self) -> list[float]:
        """Return the initial state of the oscillator."""

    @abstractmethod
    def equation_of_motion(self, initial_state: list[float], t: float) -> tuple:
        """Return the equation of motion of the oscillator."""

    def solve(self, **kwargs: dict[str, Any]) -> UniversalArrayTuple:
        """Solve the equation of motion of the oscillator."""
        return odeint(
            func=self.equation_of_motion,
            y0=self.initial_state,
            t=self.t,
            **kwargs,
        )

    @property
    @abstractmethod
    def to_position(self) -> UniversalArray:
        """Return the position of the oscillator."""

    @property
    @abstractmethod
    def to_velocity(self) -> UniversalArray:
        """Return the velocity of the oscillator."""

    @property
    def __eval__(self) -> UniversalArray:
        """Evaluate the function."""
        return self.to_velocity if self.velocity else self.to_position

    def __call__(self) -> ResultsChaoticOscillatorAPI:
        """Return the results of the function."""
        return ResultsChaoticOscillatorAPI(
            t=self.__input__,
            result=self.__eval__,
            initial_state=self.__initial_configuration__,
            doc=self.__doc__,
        )


class OscillatorsFunc2D(OscillatorsFuncBase):
    """Base class for two-dimensional chaotic oscillators."""

    @property
    def to_position(self) -> UniversalArrayTuple:
        """Return the position of the oscillator."""
        y = self.solve()
        return y[:, 0], y[:, 2]

    @property
    def to_velocity(self) -> UniversalArrayTuple:
        """Return the velocity of the oscillator."""
        y = self.solve()
        return y[:, 1], y[:, 3]


class OscillatorsFunc3D(OscillatorsFuncBase):
    """Base class for three-dimensional chaotic oscillators."""

    @property
    def to_position(self) -> UniversalArrayTuple:
        """Return the position of the oscillator."""
        y = self.solve()
        return y[:, 0], y[:, 1], y[:, 2]

    @property
    def to_velocity(self) -> UniversalArrayTuple:
        """Return the velocity of the oscillator."""
        y = self.solve()
        return y[:, 3], y[:, 4], y[:, 5]


class FractalFunction(ABC, metaclass=CoreElements):
    """Base class for fractal functions.

    This class provides a template for implementing fractal generation algorithms.
    It defines the basic interface for all fractal functions including inputs,
    iteration control, and evaluation methods.

    Args:
        *x (UniversalArray): Input parameters for the fractal (coordinates,
            complex values, etc.)
        max_iter (int): Maximum number of iterations for the fractal generation.
        scale_factor (float, optional): Scaling factor for certain fractals.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 100,
        scale_factor: float | None = None,
    ) -> None:
        """Initialize the fractal function."""
        if x[0] is None:
            raise MissingXError

        # Store input as a tuple of UniversalArray objects
        self._x: tuple[UniversalArray, ...] = x
        self.max_iter = max_iter
        self.scale_factor = scale_factor
        self.fractal_dimension = 0.0  # Default dimension, to be overridden

    @property
    def __input__(self) -> UniversalArrayTuple:
        """Return the input data.

        Returns:
            UniversalArrayTuple: Tuple of input arrays
        """
        return self._x

    @property
    @abstractmethod
    def __eval__(self) -> UniversalArray | list | dict:
        """Generate the fractal.

        This method should implement the actual fractal generation algorithm.

        Returns:
            UniversalArray | list | dict: Fractal representation, which could be:
                - Array of iteration counts (e.g., for Mandelbrot set)
                - List of coordinates (e.g., for Koch curve)
                - Dictionary of parameters and values (e.g., for L-systems)
        """

    def calculate_dimension(self) -> float | None:
        """Calculate the fractal dimension.

        For many fractals, this is a fixed value that can be calculated analytically.
        For others, it may require box-counting or other numerical methods.

        Returns:
            float | None: Fractal dimension if available, otherwise None
        """
        return getattr(self, "fractal_dimension", None)

    def __call__(self) -> ResultsFractalAPI:
        """Return the results of the fractal function.

        Returns:
            ResultsFractalAPI: Object containing the fractal data and metadata
        """
        parameters = {
            "max_iter": self.max_iter,
        }
        if self.scale_factor is not None:
            parameters["scale_factor"] = self.scale_factor

        return ResultsFractalAPI(
            x=self.__input__,
            result=self.__eval__,
            parameters=parameters,
            dimension=self.calculate_dimension(),
            doc=self.__doc__,
        )


class ComplexFractalFunction(FractalFunction):
    """Base class for complex fractal functions.

    This class extends the FractalFunction to handle complex numbers in the input data.
    It provides common functionality for fractals like the Mandelbrot set and Julia set.

    Args:
        *x (UniversalArray): Input data, which can vary depending on the specific
            fractal.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        escape_radius (float, optional): Escape radius. Defaults to 2.0.

    Raises:
        MissingXError: If no input data is specified.
    """

    def __init__(
        self, *x: UniversalArray, max_iter: int = 100, escape_radius: float = 2.0
    ) -> None:
        """Initialize the complex fractal function."""
        super().__init__(*x, max_iter=max_iter)

        self.escape_radius = escape_radius

    def iterate_complex_function(
        self, z_start: np.ndarray, c: np.ndarray | complex, shape: tuple
    ) -> np.ndarray:
        """Common iteration method for complex fractals.

        Implements the $z = z^2 + c$ iteration common to both Mandelbrot and Julia sets.

        Args:
            z_start (np.ndarray): Starting z values
            c (np.ndarray | complex): Parameter c values
            shape (tuple): Original shape of the array

        Returns:
            np.ndarray: Iteration counts
        """
        z = z_start.flatten()
        if isinstance(c, np.ndarray):
            c = c.flatten()

        iterations = np.zeros(z.shape, dtype=int)
        escape_radius_squared = self.escape_radius**2

        # Points still iterating
        mask = np.full(shape=z.shape, fill_value=True, dtype=bool)

        for i in range(self.max_iter):
            # Update z for points still iterating
            z[mask] = (
                z[mask] ** 2 + c[mask]
                if isinstance(c, np.ndarray)
                else z[mask] ** 2 + c
            )
            # Find points that escape
            escaped = np.abs(z) ** 2 > escape_radius_squared

            # Update iterations for newly escaped points
            iterations[mask & escaped] = i + 1

            # Update mask to exclude escaped points
            mask[escaped] = False

            # If all points have escaped, break
            if not np.any(mask):
                break

        return iterations.reshape(shape)


class CurveFractalFunction(FractalFunction):
    """Base class for curve-based fractals.

    This class extends FractalFunction to handle fractals based on iterative curve
    generation, like the Cantor set, Dragon curve, and space-filling curves.

    Args:
        *x (UniversalArray): Input points defining the curve space
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        scale_factor (float, optional): Scale factor between iterations.
            Defaults to 0.5.
        fractal_dimension (float, optional): Fractal dimension of the curve.
            Defaults to 2.0.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 100,
        scale_factor: float = 0.5,
        fractal_dimension: float = 2.0,
    ) -> None:
        """Initialize the curve fractal function."""
        super().__init__(*x, max_iter=max_iter)
        self.scale_factor = scale_factor
        self.fractal_dimension = fractal_dimension

    def generate_points(self) -> np.ndarray:
        """Generate points for the curve at the current iteration.

        Returns:
            np.ndarray: Array of points defining the curve
        """
        raise NotImplementedError


class DynamicFractalFunction(FractalFunction):
    """Base class for fractals generated by dynamic systems.

    This class handles fractals that emerge from iterating dynamic systems,
    like attractors and percolation models.

    Args:
        *x (UniversalArray): Input data, which can vary depending on the specific
            fractal.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        transient_steps (int, optional): Initial steps to discard. Defaults to 100.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 1000,
        transient_steps: int = 100,
    ) -> None:
        """Initialize the dynamic fractal function."""
        super().__init__(*x, max_iter=max_iter)
        self.transient_steps = transient_steps

    def iterate_system(self, initial_state: np.ndarray) -> np.ndarray:
        """Iterate the dynamic system.

        Args:
            initial_state (np.ndarray): Starting state of the system

        Returns:
            np.ndarray: Array of system states
        """
        raise NotImplementedError


class GeometricFractalFunction(FractalFunction):
    """Base class for fractals based on geometric transformations.

    This class handles self-similar fractals generated through geometric
    transformations, like the Sierpinski triangle and Koch snowflake.

    Args:
        *x (UniversalArray): Initial geometric shape vertices
        max_iter (int, optional): Maximum number of iterations. Defaults to 6.
        scale_factor (float, optional): Scale factor for transformations. Defaults
            to 0.5.
        fractal_dimension (float, optional): Fractal dimension of the shape.
            Defaults to 2.0.
    """

    def __init__(
        self,
        *x: UniversalArray,
        max_iter: int = 6,
        scale_factor: float = 0.5,
        fractal_dimension: float = 2.0,
    ) -> None:
        """Initialize the geometric fractal function."""
        super().__init__(*x, max_iter=max_iter)
        self.scale_factor = scale_factor
        self.fractal_dimension = fractal_dimension

    def transform_points(self, points: np.ndarray) -> list[np.ndarray]:
        """Apply geometric transformations to points.

        Args:
            points (np.ndarray): Array of points to transform

        Returns:
            list[np.ndarray]: List of transformed point arrays
        """
        raise NotImplementedError
