"""Continuous distributions with support for semi-infinite intervals for the umf."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.special import gamma
from scipy.special import gammainc

from umf.constants.exceptions import NotAPositiveNumberError
from umf.constants.exceptions import NotLargerThanZeroError
from umf.functions.other.support_functions import erf
from umf.meta.api import SummaryStatisticsAPI
from umf.meta.functions import ContinuousPure
from umf.meta.functions import SemiContinuous
from umf.meta.functions import SemiContinuousWSigma


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "RayleighDistribution",
    "WeibullDistribution",
    "LogNormalDistribution",
    "ChiSquaredDistribution",
    "DagumDistribution",
]


class RayleighDistribution(SemiContinuousWSigma):
    r"""Rayleigh distribution.

    The Rayleigh distribution is a continuous probability distribution that is commonly
    used in statistics to model the magnitude of a vector whose components are
    independent and identically distributed Gaussian random variables with zero mean.
    It is also used to describe the distribution of the magnitude of the sum of
    independent, identically distributed Gaussian random variables with zero mean
    and equal variance.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... RayleighDistribution
        ... )
        >>> x = np.linspace(0, 15, 1000)
        >>> y_sigma_1 = RayleighDistribution(x, sigma=1).__eval__
        >>> y_sigma_2 = RayleighDistribution(x, sigma=2).__eval__
        >>> y_sigma_3 = RayleighDistribution(x, sigma=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_sigma_1, label="sigma=1")
        >>> _ = ax.plot(x, y_sigma_2, label="sigma=2")
        >>> _ = ax.plot(x, y_sigma_3, label="sigma=3")
        >>> _ = ax.legend()
        >>> plt.savefig("RayleighDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... RayleighDistribution
        ... )
        >>> x = np.linspace(0, 15, 1000)
        >>> y_sigma_1 = RayleighDistribution(
        ... x,
        ... sigma=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_sigma_2 = RayleighDistribution(
        ... x,
        ... sigma=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_sigma_3 = RayleighDistribution(
        ... x,
        ... sigma=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_sigma_1, label="sigma=1")
        >>> _ = ax.plot(x, y_sigma_2, label="sigma=2")
        >>> _ = ax.plot(x, y_sigma_3, label="sigma=3")
        >>> _ = ax.legend()
        >>> plt.savefig("RayleighDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The Rayleigh distribution is generally defined for the PDF as:

        $$
        f(x | \sigma) = \frac{x}{\sigma^2} \exp\left(-\frac{x^2}{2\sigma^2}\right)
        $$

        and for the CDF as:

        $$
        F(x | \sigma) = 1 - \exp\left(-\frac{x^2}{2\sigma^2}\right)
        $$

        where $\sigma$ is the scale parameter.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        sigma (float): Standard deviation. Defaults to 1.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        return self._x / self.sigma**2 * np.exp(-(self._x**2) / (2 * self.sigma**2))

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        return 1 - np.exp(-(self._x**2) / (2 * self.sigma**2))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=np.sqrt(np.pi / 2) * self.sigma,
            variance=(4 - np.pi) / 2 * self.sigma**2,
            mode=self.sigma,
            doc=self.__doc__,
        )


class WeibullDistribution(SemiContinuous):
    r"""Weibull distribution.

    The Weibull distribution is a continuous probability distribution that is
    commonly used in statistics to model variables that are the product of many small,
    independent factors. It is a transformation of the normal distribution, where the
    logarithm of the variable is normally distributed. The Weibull distribution has
    applications in various fields, such as finance, biology, and engineering.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... WeibullDistribution
        ... )
        >>> x = np.linspace(0, 15, 1000)
        >>> y_lambda_1 = WeibullDistribution(x, lambda_=1, k=0.5).__eval__
        >>> y_lambda_2 = WeibullDistribution(x, lambda_=2, k=1.0).__eval__
        >>> y_lambda_3 = WeibullDistribution(x, lambda_=3, k=1.5).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_lambda_1, label="lambda=1 and k=0.5")
        >>> _ = ax.plot(x, y_lambda_2, label="lambda=2 and k=1.0")
        >>> _ = ax.plot(x, y_lambda_3, label="lambda=3 and k=1.5")
        >>> _ = ax.legend()
        >>> plt.savefig("WeibullDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... WeibullDistribution
        ... )
        >>> x = np.linspace(0, 15, 1000)
        >>> y_lambda_1 = WeibullDistribution(
        ... x,
        ... lambda_=1,
        ... k=0.5,
        ... cumulative=True,
        ... ).__eval__
        >>> y_lambda_2 = WeibullDistribution(
        ... x,
        ... lambda_=2,
        ... k=1.0,
        ... cumulative=True,
        ... ).__eval__
        >>> y_lambda_3 = WeibullDistribution(
        ... x,
        ... lambda_=3,
        ... k=1.5,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_lambda_1, label="lambda=1 and k=0.5")
        >>> _ = ax.plot(x, y_lambda_2, label="lambda=2 and k=1.0")
        >>> _ = ax.plot(x, y_lambda_3, label="lambda=3 and k=1.5")
        >>> _ = ax.legend()
        >>> plt.savefig("WeibullDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The Weibull distribution is generally defined for the PDF as:

        $$
        f(x | \lambda, k) = \frac{k}{\lambda} \left(\frac{x}{\lambda}\right)^{k - 1}
            \exp\left(-\left(\frac{x}{\lambda}\right)^k\right)
        $$

        and for the CDF as:

        $$
        F(x | \lambda, k) = 1 - \exp\left(-\left(\frac{x}{\lambda}\right)^k\right)
        $$

        where $\lambda$ is the scale parameter and $k$ is the shape parameter.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        lambda (float): Scale parameter. Defaults to 1.
        k (float): Shape parameter. Defaults to 1.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def __init__(
        self,
        *x: UniversalArray,
        lambda_: float = 1.0,
        k: float = 1.0,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if lambda_ < 0:
            msg = "lambda_"
            raise NotAPositiveNumberError(msg, lambda_)
        if k <= 0:
            msg = "k"
            raise NotLargerThanZeroError(msg, k)

        super().__init__(*x, cumulative=cumulative)
        self.lambda_ = lambda_
        self.k = k

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        return (
            self.k
            / self.lambda_
            * (self._x / self.lambda_) ** (self.k - 1)
            * np.exp(-((self._x / self.lambda_) ** self.k))
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        return 1 - np.exp(-((self._x / self.lambda_) ** self.k))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.lambda_ * gamma(1 + 1 / self.k),
            variance=self.lambda_**2
            * (gamma(1 + 2 / self.k) - gamma(1 + 1 / self.k) ** 2),
            mode=self.lambda_ * (self.k - 1) ** (1 / self.k) if self.k > 1 else 0,
            doc=self.__doc__,
        )


class LogNormalDistribution(SemiContinuousWSigma):
    r"""Log-normal distribution.

    The log-normal distribution is a semi continuous probability distribution that is
    commonly used in statistics to model variables that are the product of many small,
    independent factors. It is a transformation of the normal distribution, where the
    logarithm of the variable is normally distributed. The log-normal distribution has
    applications in various fields, such as finance, biology, and engineering.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... LogNormalDistribution
        ... )
        >>> x = np.linspace(0, 5, 1000)
        >>> y_sigma_1 = LogNormalDistribution(x, sigma=1).__eval__
        >>> y_sigma_2 = LogNormalDistribution(x, sigma=2).__eval__
        >>> y_sigma_3 = LogNormalDistribution(x, sigma=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_sigma_1, label="sigma=1")
        >>> _ = ax.plot(x, y_sigma_2, label="sigma=2")
        >>> _ = ax.plot(x, y_sigma_3, label="sigma=3")
        >>> _ = ax.legend()
        >>> plt.savefig("LogNormalDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... LogNormalDistribution
        ... )
        >>> x = np.linspace(0, 5, 1000)
        >>> y_sigma_1 = LogNormalDistribution(
        ... x,
        ... sigma=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_sigma_2 = LogNormalDistribution(
        ... x,
        ... sigma=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_sigma_3 = LogNormalDistribution(
        ... x,
        ... sigma=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_sigma_1, label="sigma=1")
        >>> _ = ax.plot(x, y_sigma_2, label="sigma=2")
        >>> _ = ax.plot(x, y_sigma_3, label="sigma=3")
        >>> _ = ax.legend()
        >>> plt.savefig("LogNormalDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The log-normal distribution is generally defined for the PDF as:

        $$
        f(x | \mu, \sigma) = \frac{1}{x \sigma \sqrt{2\pi}}
          \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)
        $$

        and for the CDF as:

        $$
        F(x | \mu, \sigma) = \frac{1}{2} \left[1 + \mathrm{erf}
        \left(\frac{\ln x - \mu}{\sigma \sqrt{2}}\right)\right]
        $$

        where $\mu$ is the mean and $\sigma$ is the standard deviation.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        sigma (float): Standard deviation. Defaults to 1.
        mu (float): Mean. Defaults to 0.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        return (
            1
            / (self._x * self.sigma * np.sqrt(2 * np.pi))
            * np.exp(-((np.log(self._x) - self.mu) ** 2) / (2 * self.sigma**2))
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        return 0.5 * (1 + erf((np.log(self._x) - self.mu) / (self.sigma * np.sqrt(2))))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=np.exp(self.mu + self.sigma**2 / 2),
            variance=(np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2),
            mode=np.exp(self.mu - self.sigma**2),
            doc=self.__doc__,
        )


class ChiSquaredDistribution(SemiContinuous):
    r"""Chi-square distribution.

    The chi-square distribution is a semi continuous probability distribution that is
    commonly used in statistics to model variables that are the sum of the squares of
    independent standard normal random variables. It is a transformation of the normal
    distribution, where the logarithm of the variable is normally distributed. The
    chi-square distribution has applications in various fields, such as finance,
    biology, and engineering.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... ChiSquaredDistribution
        ... )
        >>> x = np.linspace(0, 15, 1000)
        >>> y_k_1 = ChiSquaredDistribution(x, k=1).__eval__
        >>> y_k_2 = ChiSquaredDistribution(x, k=2).__eval__
        >>> y_k_3 = ChiSquaredDistribution(x, k=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_k_1, label="k=1")
        >>> _ = ax.plot(x, y_k_2, label="k=2")
        >>> _ = ax.plot(x, y_k_3, label="k=3")
        >>> _ = ax.legend()
        >>> plt.savefig("ChiSquaredDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... ChiSquaredDistribution
        ... )
        >>> x = np.linspace(0, 15, 1000)
        >>> y_k_1 = ChiSquaredDistribution(
        ... x,
        ... k=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_k_2 = ChiSquaredDistribution(
        ... x,
        ... k=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_k_3 = ChiSquaredDistribution(
        ... x,
        ... k=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_k_1, label="k=1")
        >>> _ = ax.plot(x, y_k_2, label="k=2")
        >>> _ = ax.plot(x, y_k_3, label="k=3")
        >>> _ = ax.legend()
        >>> plt.savefig("ChiSquaredDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The chi-square distribution is generally defined for the PDF as:

        $$
        f(x | k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2 - 1} \exp(-x/2)
        $$

        and for the CDF as:

        $$
        F(x | k) = \frac{1}{\Gamma(k/2)} \gamma(k/2, x/2)
        $$

        where $k$ is the degrees of freedom.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        k (float): Degrees of freedom. Defaults to 1.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def __init__(
        self,
        *x: UniversalArray,
        k: float = 1.0,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if k <= 0:
            raise NotLargerThanZeroError(k)

        super().__init__(*x, cumulative=cumulative)
        self.k = k

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        return (
            1
            / (2 ** (self.k / 2) * gamma(self.k / 2))
            * self._x ** (self.k / 2 - 1)
            * np.exp(-self._x / 2)
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        return 1 / gamma(self.k / 2) * gammainc(self.k / 2, self._x / 2)

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.k,
            variance=2 * self.k,
            mode=self.k - 2 if self.k > 2 else 0,  # noqa: PLR2004
            doc=self.__doc__,
        )


class DagumDistribution(ContinuousPure):
    r"""Dagum distribution.

    The Dagum distribution is a continuous probability distribution that is defined on
    the semi-infinite interval 0, ∞). It is a three-parameter distribution that is
    characterized by its shape, scale, and shape parameters. The Dagum distribution is
    used in various fields, including economics, finance, and engineering, to model
    data that is non-negative and skewed to the right. It has a probability density
    function (PDF) and a cumulative distribution function (CDF) that can be used to
    calculate various statistical measures, such as mean, variance, and mode.

    Notes:
        The Dagum distribution is generally defined for the PDF as:

        $$
        f(x | p, a, b) = \frac{p a^{p}}{x^{p + 1} \left[1 + \left(\frac{a}{b}
        x\right)^{p}\right]^{(p+1)/p}}
        $$

        and for the CDF as:

        $$
        F(x | p, a, b) = 1 - \left[1 + \left(\frac{a}{b} x\right)^{p}\right]^{-p}
        $$

        where $p$ is the shape parameter, $a$ is the scale parameter, and $b$ is the
        shape parameter.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_semi_infinite_interval import (
        ... DagumDistribution
        ... )
        >>> x = np.linspace(0, 10, 1000)
        >>> y_p_1_a_1_b_1 = DagumDistribution(x, p=1, a=1, b=1).__eval__
        >>> y_p_2_a_1_b_1 = DagumDistribution(x, p=2, a=1, b=1).__eval__
        >>> y_p_3_a_1_b_1 = DagumDistribution(x, p=3, a=1, b=1).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_p_1_a_1_b_1, label="p=1, a=1, b=1")
        >>> _ = ax.plot(x, y_p_2_a_1_b_1, label="p=2, a=1, b=1")
        >>> _ = ax.plot(x, y_p_3_a_1_b_1, label="p=3, a=1, b=1")
        >>> _ = ax.legend()
        >>> plt.savefig("DagumDistribution.png", dpi=300, transparent=True)

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        p (float): Shape parameter. Must be greater than 0. Defaults to 1.
        a (float): Scale parameter. Must be greater than 0. Defaults to 1.
        b (float): Shape parameter. Must be greater than 0. Defaults to 1.

    Raises:
        NotLargerThanZeroError: If p, a, or b is not larger than 0.
    """

    def __init__(
        self,
        *x: UniversalArray,
        p: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
    ) -> None:
        """Initialize the function."""
        if p <= 0:
            msg = "p"
            raise NotLargerThanZeroError(msg, p)
        if a <= 0:
            msg = "a"
            raise NotLargerThanZeroError(msg, a)
        if b <= 0:
            msg = "b"
            raise NotLargerThanZeroError(msg, b)
        super().__init__(*x)
        self.p = p
        self.a = a
        self.b = b

    def probability_density_function(self) -> np.ndarray:
        """Return the probability density function."""
        return (
            self.p
            * self.a**self.p
            / (
                self._x ** (self.p + 1)
                * (1 + (self.a / self.b) ** self.p * self._x**self.p)
                ** ((self.p + 1) / self.p)
            )
        )

    def cumulative_distribution_function(self) -> np.ndarray:
        """Return the cumulative distribution function."""
        return 1 - (1 + (self.a / self.b) ** self.p * self._x**self.p) ** (-self.p)

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.a
            * gamma((self.p - 1) / self.p)
            * gamma((self.p + 1) / self.p)
            / gamma(self.p / self.p),
            variance=self.a**2
            * (
                gamma((self.p - 2) / self.p)
                * gamma((self.p + 1) / self.p)
                / gamma(self.p / self.p)
                - (
                    self.a
                    * gamma((self.p - 1) / self.p)
                    * gamma((self.p + 1) / self.p)
                    / gamma(self.p / self.p)
                )
                ** 2
            ),
            mode=self.a
            * ((self.p / self.b) ** (1 / self.p))
            * ((self.p - 1) / self.p) ** (1 / self.p),
            doc=self.__doc__,
        )
