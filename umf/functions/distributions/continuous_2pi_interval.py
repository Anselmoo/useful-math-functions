r"""Continuous distributions on the interval $[0, 2\pi]$."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy import special

from umf.meta.api import SummaryStatisticsAPI
from umf.meta.functions import Continuous2PiInterval


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__: list[str] = ["VonMisesDistribution", "WrappedAsymLaplaceDistribution"]


class VonMisesDistribution(Continuous2PiInterval):
    r"""von Mises distribution.

    The von Mises distribution is a continuous probability distribution on the
    circle. It is a close approximation to the wrapped normal distribution,
    which is the circular analogue of the normal distribution.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_2pi_interval import (
        ... VonMisesDistribution
        ... )
        >>> x = np.linspace(-np.pi, np.pi, 1000)
        >>> y_05 = VonMisesDistribution(x, mu=0, kappa=0.5).__eval__
        >>> y_07 = VonMisesDistribution(x, mu=0, kappa=0.7).__eval__
        >>> y_09 = VonMisesDistribution(x, mu=0, kappa=0.9).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _  = ax.plot(x, y_05, label=r"$\kappa=0.5$")
        >>> _  = ax.plot(x, y_07, label=r"$\kappa=0.7$")
        >>> _  = ax.plot(x, y_09, label=r"$\kappa=0.9$")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("f(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("VonMisesDistribution.png", dpi=300, transparent=True)


    Notes:
        The von Mises distribution is defined as follows for probability density:

        $$
        f(x;\mu,\kappa) = \frac{e^{\kappa \cos(x-\mu)}}{2\pi I_0(\kappa)}
        $$

        where $x \in [-\pi, \pi]$ and $\mu \in [-\pi, \pi]$. The parameter
        $\kappa \in [0, \infty)$ controls the concentration of the distribution
        around $\mu$. The function $I_0$ is the modified Bessel function of the
        first kind and order zero.


    Args:
        *x (UniversalArray): The points at which to evaluate the distribution.
        mu (float): The mean of the distribution. Defaults to 0.
        kappa (float): The concentration of the distribution around mu.
            Defaults to 1.
    """

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the von Mises distribution."""
        return np.exp(self.kappa * np.cos(self._x - self.mu)) / (
            2 * np.pi * special.i0(self.kappa)
        )

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the von Mises distribution."""

        def _mode() -> float | tuple[float, float]:
            """Mode of the von Mises distribution."""
            return self.mu

        return SummaryStatisticsAPI(
            mean=self.mu,
            variance=1 - special.i1(self.kappa) / special.i0(self.kappa),
            mode=_mode(),
            doc=self.__doc__,
        )


class WrappedAsymLaplaceDistribution(Continuous2PiInterval):
    r"""Wrapped asymmetric Laplace distribution.

    The wrapped (asymmetric) Laplace distribution is a continuous probability
    distribution on the circle. It is a close approximation to the wrapped normal
    distribution, which is the circular analogue of the normal distribution.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_2pi_interval import (
        ... WrappedAsymLaplaceDistribution
        ... )
        >>> x = np.linspace(-np.pi, np.pi, 1000)
        >>> y_05 = WrappedAsymLaplaceDistribution(
        ... x,
        ... mu=0,
        ... lambda_=0.5,
        ... kappa=0.5,
        ... ).__eval__
        >>> y_07 = WrappedAsymLaplaceDistribution(
        ... x,
        ... mu=0,
        ... lambda_=0.7,
        ... kappa=0.7,
        ... ).__eval__
        >>> y_09 = WrappedAsymLaplaceDistribution(
        ... x,
        ... mu=0,
        ... lambda_=0.9,
        ... kappa=0.9,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _  = ax.plot(x, y_05, label=r"$\lambda=0.5$")
        >>> _  = ax.plot(x, y_07, label=r"$\lambda=0.7$")
        >>> _  = ax.plot(x, y_09, label=r"$\lambda=0.9$")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("f(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("WrappedAsymLaplaceDistribution.png", dpi=300, transparent=True)

    Notes:
        The wrapped Laplace distribution is defined as follows for probability
        density:

        $$
        \begin{aligned}f_{WAL}(\theta ;m,\lambda ,\kappa )&=
        \sum _{k=-\infty }^{\infty }f_{AL}(\theta +2\pi k,m,\lambda ,\kappa )\\[10pt]
        &={\dfrac {\kappa \lambda }{\kappa ^{2}+1}}{\begin{cases}
        {\dfrac {e^{-(\theta -m)\lambda \kappa }}
        {1-e^{-2\pi \lambda \kappa }}}-{\dfrac {e^{(\theta -m)\lambda /\kappa }}
        {1-e^{2\pi \lambda /\kappa }}}&{\text{if }}
        \theta \geq m\\[12pt]{\dfrac {e^{-(\theta -m)\lambda \kappa }}
        {e^{2\pi \lambda \kappa }-1}}-{\dfrac {e^{(\theta -m)\lambda /\kappa }}
        {e^{-2\pi \lambda /\kappa }-1}}&{\text{if }}\theta <m\end{cases}}\end{aligned}
        $$

        where $x \in [-\pi, \pi]$ and $\mu \in [-\pi, \pi]$. The parameter
        $\kappa \in [0, \infty)$ controls the concentration of the distribution
        around $\mu$. The function $I_0$ is the modified Bessel function of the
        first kind and order zero.[^1]

        [^1]: Wrapped asymmetric Laplace distribution. (2022, January 24).
            _In Wikipedia._
            https://en.wikipedia.org/wiki/Wrapped_asymmetric_Laplace_distribution

    Args:
        x: The points at which to evaluate the distribution.
        mu: The mean of the distribution.
        lambda_: The scale parameter of the distribution.
        kappa: The concentration of the distribution around mu.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        lambda_: float = 1,
        kappa: float = 1,
    ) -> None:
        r"""Initialize a wrapped Laplace distribution.

        Args:
            *x (UniversalArray): The points at which to evaluate the distribution.
            mu (float): The mean of the distribution. Defaults to 0.
            lambda_ (float): The scale parameter of the distribution. Defaults to 1.
            kappa (float): The concentration of the distribution around mu.
                Defaults to 1.
        """
        super().__init__(*x, mu=mu, kappa=kappa)
        self.lambda_ = lambda_

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the wrapped Laplace distribution."""
        part_1 = (
            self.kappa
            * self.lambda_
            / (self.kappa**2 + 1)
            * (
                np.exp(-(self._x - self.mu) * self.lambda_ * self.kappa)
                / (1 - np.exp(-2 * np.pi * self.lambda_ * self.kappa))
                - np.exp((self._x - self.mu) * self.lambda_ / self.kappa)
                / (1 - np.exp(2 * np.pi * self.lambda_ / self.kappa))
            )
        )
        part_2 = (
            self.kappa
            * self.lambda_
            / (self.kappa**2 + 1)
            * (
                np.exp(-(self._x - self.mu) * self.lambda_ * self.kappa)
                / (np.exp(2 * np.pi * self.lambda_ * self.kappa) - 1)
                - np.exp((self._x - self.mu) * self.lambda_ / self.kappa)
                / (np.exp(-2 * np.pi * self.lambda_ / self.kappa) - 1)
            )
        )

        # Combine the two parts
        return np.where(self._x >= self.mu, part_1, part_2)

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the wrapped Laplace distribution."""
        return SummaryStatisticsAPI(
            mean=self.mu,
            variance=1
            - self.lambda_**2
            / np.sqrt(
                (1 / self.kappa**2 + self.lambda_**2)
                * (self.kappa**2 + self.lambda_**2),
            ),
            mode=None,
            doc=self.__doc__,
        )
