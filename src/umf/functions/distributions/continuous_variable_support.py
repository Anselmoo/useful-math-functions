"""Continuous variable support distributions for the umf module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.special import gamma

from umf.constants.exceptions import NotAPositiveNumberError
from umf.meta.api import SummaryStatisticsAPI
from umf.meta.functions import ContinuousMixed


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "GeneralizedExtremeValueDistribution",
    "GeneralizedParetoDistribution",
]


class GeneralizedExtremeValueDistribution(ContinuousMixed):
    r"""Generalized extreme value distribution.

    The generalized extreme value distribution is a family of continuous probability
    distributions developed within extreme value theory to combine the Gumbel, Fréchet
    and Weibull families also known as type I, II and III extreme value distributions.
    The generalized extreme value distribution is also sometimes referred to as the
    Fisher-Tippett distribution or the extreme value type I distribution.


    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_variable_support import (
        ... GeneralizedExtremeValueDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_00 = GeneralizedExtremeValueDistribution(
        ... x,
        ... mu=0,
        ... zeta=0,
        ... sigma=1,
        ... ).__eval__
        >>> y_01 = GeneralizedExtremeValueDistribution(
        ... x,
        ... mu=0,
        ... zeta=0.1,
        ... sigma=1,
        ... ).__eval__
        >>> y_05 = GeneralizedExtremeValueDistribution(
        ... x,
        ... mu=0,
        ... zeta=0.5,
        ... sigma=1,
        ... ).__eval__
        >>> y_10 = GeneralizedExtremeValueDistribution(
        ... x,
        ... mu=0,
        ... zeta=1,
        ... sigma=1,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _  = ax.plot(x, y_00, label="zeta=0")
        >>> _  = ax.plot(x, y_01, label="zeta=0.1")
        >>> _  = ax.plot(x, y_05, label="zeta=0.5")
        >>> _  = ax.plot(x, y_10, label="zeta=1")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("f(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("GeneralizedExtremeValueDistribution.png",
        ... dpi=300,
        ... transparent=True,
        ... )

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_variable_support import (
        ... GeneralizedExtremeValueDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_00 = GeneralizedExtremeValueDistribution(
        ... x,
        ... mu=0,
        ... zeta=0,
        ... sigma=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_01 = GeneralizedExtremeValueDistribution(
        ... x,
        ... mu=0,
        ... zeta=0.1,
        ... sigma=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_05 = GeneralizedExtremeValueDistribution(
        ... x,
        ... mu=0,
        ... zeta=0.5,
        ... sigma=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_10 = GeneralizedExtremeValueDistribution(
        ... x,
        ... mu=0,
        ... zeta=1,
        ... sigma=1,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _  = ax.plot(x, y_00, label="zeta=0")
        >>> _  = ax.plot(x, y_01, label="zeta=0.1")
        >>> _  = ax.plot(x, y_05, label="zeta=0.5")
        >>> _  = ax.plot(x, y_10, label="zeta=1")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("F(x)")
        >>> _  = ax.legend()
        >>> plt.savefig(
        ... "GeneralizedExtremeValueDistribution-cml.png",
        ... dpi=300,
        ... transparent=True,
        ... )


    Notes:
        The generalized extreme value distribution is defined as follows for probability
        density function:

        $$
        f(x;\mu,\sigma,\zeta) = \begin{cases}
        \frac{1}{\sigma} \left[ 1 + \zeta \left( \frac{x - \mu}{\sigma} \right)
        \right]^{-1/\zeta - 1} \exp \left[ - \left( 1 + \zeta
        \left( \frac{x - \mu}{\sigma} \right)
        \right)^{-1/\zeta} \right] & \text{if } \zeta \neq 0 \\
        \frac{1}{\sigma} \exp \left[ - \left( \frac{x - \mu}{\sigma}
        \right) \right] \exp \left[ - \exp \left( -
        \left( \frac{x - \mu}{\sigma} \right) \right) \right] & \text{if } \zeta = 0
        \end{cases}
        $$

        where $\mu \in \mathbb{R}$, $\sigma > 0$ and $\zeta \in \mathbb{R}$.

        The generalized extreme value distribution is defined as follows for cumulative
        distribution function:

        $$
        F(x;\mu,\sigma,\zeta) = \exp \left[ - \left( 1 + \zeta \left( \frac{x -
        \mu}{\sigma} \right) \right)^{-1/\zeta} \right]
        $$

        where $\mu \in \mathbb{R}$, $\sigma > 0$ and $\zeta \in \mathbb{R}$.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        mu (float): Location parameter. Defaults to 0.
        zeta (float): Shape parameter. Defaults to 0.
        sigma (float): Scale parameter. Defaults to 1.
        cumulative (bool): If True, the cumulative distribution function is returned.
            Defaults to False.
    """

    @property
    def t_factor(self) -> UniversalArray:
        """Factor of the t-distribution."""
        if self.zeta == 0:
            return np.exp(-self._x - self.mu) / self.sigma
        return np.exp(-((1 + self.zeta * self._x) ** (-1 / self.zeta))) / self.sigma

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the gen. extreme value distribution."""
        return self.t_factor**self.zeta * np.exp(-self.t_factor)

    def cumulative_distribution_function(self) -> UniversalArray:
        """Cumulative distribution function of the gen. extreme value distribution."""
        return np.exp(-self.t_factor)

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the gen. extreme value distribution."""
        if self.zeta != 0 and self.zeta < 1:
            mean = self.mu + self.sigma * (gamma(1 - self.zeta) - 1) / self.zeta
        elif self.zeta == 0:
            mean = self.mu + self.sigma * np.euler_gamma
        else:
            mean = np.inf

        if self.zeta != 0 and self.zeta < 0.5:  # noqa: PLR2004
            variance = (
                self.sigma**2
                * (gamma(1 - 2 * self.zeta) - gamma(1 - self.zeta) ** 2)
                / self.zeta**2
            )
        elif self.zeta == 0:
            variance = self.sigma**2 * np.pi**2 / 6
        else:
            variance = np.inf

        if self.zeta != 0:
            mode = self.mu + self.sigma * (1 - self.zeta ** (-1)) ** (-1 / self.zeta)
        else:
            mode = self.mu
        return SummaryStatisticsAPI(
            mean=mean,
            variance=variance,
            mode=mode,
            doc=self.__doc__,
        )


class GeneralizedParetoDistribution(ContinuousMixed):
    r"""Generalized Pareto distribution.

    The generalized Pareto distribution is a family of continuous probability
    distributions that includes the exponential, Weibull, and uniform distributions.
    The generalized Pareto distribution is often used to model the tails of another
    distribution.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_variable_support import (
        ... GeneralizedParetoDistribution
        ... )
        >>> x = np.linspace(0, 5, 1000)
        >>> y_00 = GeneralizedParetoDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... zeta=0,
        ... ).__eval__
        >>> y_01 = GeneralizedParetoDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... zeta=0.1,
        ... ).__eval__
        >>> y_05 = GeneralizedParetoDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... zeta=5,
        ... ).__eval__
        >>> y_10 = GeneralizedParetoDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... zeta=20,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _  = ax.plot(x, y_00, label=r"$\zeta$=0")
        >>> _  = ax.plot(x, y_01, label=r"$\zeta$=0.1")
        >>> _  = ax.plot(x, y_05, label=r"$\zeta$=5")
        >>> _  = ax.plot(x, y_10, label=r"$\zeta$=20")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("f(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("GeneralizedParetoDistribution.png",
        ... dpi=300,
        ... transparent=True,
        ... )

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_variable_support import (
        ... GeneralizedParetoDistribution
        ... )
        >>> x = np.linspace(0, 5, 1000)
        >>> y_00 = GeneralizedParetoDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... zeta=0,
        ... cumulative=True,
        ... ).__eval__
        >>> y_01 = GeneralizedParetoDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... zeta=0.1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_05 = GeneralizedParetoDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... zeta=5,
        ... cumulative=True,
        ... ).__eval__
        >>> y_10 = GeneralizedParetoDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... zeta=20,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _  = ax.plot(x, y_00, label=r"$\zeta$=0")
        >>> _  = ax.plot(x, y_01, label=r"$\zeta$=0.1")
        >>> _  = ax.plot(x, y_05, label=r"$\zeta$=5")
        >>> _  = ax.plot(x, y_10, label=r"$\zeta$=20")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("F(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("GeneralizedParetoDistribution-cml.png",
        ... dpi=300,
        ... transparent=True,
        ... )

    Notes:
        The generalized Pareto distribution is defined as follows for probability
        density function:

        $$
        f(x;\mu,\sigma,\zeta) = \begin{cases}
        \frac{1}{\sigma} \left( 1 + \zeta \frac{x - \mu}{\sigma} \right)^{-1/\zeta -
        1} & \text{if } \zeta \neq 0 \\
        \frac{1}{\sigma} \exp \left( - \frac{x - \mu}{\sigma} \right) & \text{if }
        \zeta = 0
        \end{cases}
        $$

        where $\mu \in \mathbb{R}$, $\sigma > 0$ and $\zeta \in \mathbb{R}$.

        The generalized Pareto distribution is defined as follows for cumulative
        distribution function:

        $$
        F(x;\mu,\sigma,\zeta) = \begin{cases}
        1 - \left( 1 + \zeta \frac{x - \mu}{\sigma} \right)^{-1/\zeta} & \text{if }
        \zeta \neq 0 \\
        1 - \exp \left( - \frac{x - \mu}{\sigma} \right) & \text{if } \zeta = 0
        \end{cases}
        $$

        where $\mu \in \mathbb{R}$, $\sigma > 0$ and $\zeta \in \mathbb{R}$.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        mu (float): Location parameter. Defaults to 0.
        sigma (float): Scale parameter. Defaults to 1.
        zeta (float): Shape parameter. Defaults to 0.
        cumulative (bool): If True, the cumulative distribution function is returned.
            Defaults to False.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        zeta: float = 0,
        cumulative: bool = False,
    ) -> None:
        """Initialize the generalized Pareto distribution."""
        if (min_x := np.min(x)) < 0:
            msg = "*x"
            raise NotAPositiveNumberError(msg, number=float(min_x))
        super().__init__(*x, mu=mu, sigma=sigma, zeta=zeta, cumulative=cumulative)

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the generalized Pareto distribution."""
        if self.zeta == 0:
            return np.exp(-(self._x - self.mu) / self.sigma)
        return (1 + self.zeta * (self._x - self.mu) / self.sigma) ** (
            -1 / (self.zeta + 1)
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Cumulative distribution function of the generalized Pareto distribution."""
        if self.zeta == 0:
            return 1 - np.exp(-(self._x - self.mu) / self.sigma)
        return 1 - (1 + self.zeta * (self._x - self.mu) / self.sigma) ** (
            -1 / self.zeta
        )

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the generalized Pareto distribution."""
        SummaryStatisticsAPI(
            mean=self.mu + self.sigma / (1 - self.zeta) if self.zeta < 1 else np.inf,
            variance=(
                self.sigma**2 / (1 - self.zeta) ** 2 / (1 - 2 * self.zeta)
                if self.zeta < 0.5  # noqa: PLR2004
                else np.inf
            ),
            mode=self.mu,
            doc=self.__doc__,
        )
