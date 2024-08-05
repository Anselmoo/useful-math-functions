"""Discrete distributions with infinite support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umf.constants.exceptions import NotAPositiveNumberError
from umf.constants.exceptions import NotLargerThanZeroError
from umf.functions.other.support_functions import erf
from umf.meta.api import SummaryStatisticsAPI
from umf.meta.functions import DiscretePure


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__: list[str] = [
    "BoltzmannDistribution",
    "MaxwellBoltzmannDistribution",
    "GausKuzminDistribution",
]


class BoltzmannDistribution(DiscretePure):
    r"""Boltzmann distribution.

    The Boltzmann distribution is a discrete probability distribution with discrete
    infinite support. It is used to describe the distribution of energy among particles
    in a system at a given temperature in statistical mechanics and thermodynamics.

    Examples:
        >>> # PMF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.discrete_infinite_support import (
        ... BoltzmannDistribution
        ... )
        >>> x = np.linspace(0.5, 20, 1000)
        >>> y_12 = BoltzmannDistribution(x, energy_i=1, energy_j=2).__eval__
        >>> y_13 = BoltzmannDistribution(x, energy_i=1, energy_j=3).__eval__
        >>> y_31 = BoltzmannDistribution(x, energy_i=3, energy_j=1).__eval__
        >>> y_21 = BoltzmannDistribution(x, energy_i=2, energy_j=1).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot(111)
        >>> _  = ax.plot(x, y_12, label=r"$\frac{p_{1}}{p_{2}}$")
        >>> _  = ax.plot(x, y_13, label=r"$\frac{p_{1}}{p_{3}}$")
        >>> _  = ax.plot(x, y_31, label=r"$\frac{p_{3}}{p_{1}}$")
        >>> _  = ax.plot(x, y_21, label=r"$\frac{p_{2}}{p_{1}}$")
        >>> _  = ax.set_xlabel("Temperature (K)")
        >>> _  = ax.set_ylabel(r"$\frac{p_i}{p_j}$")
        >>> _  = ax.legend()
        >>> plt.savefig("BoltzmannDistribution.png", dpi=300, transparent=True)

    Notes:
        The Boltzmann distribution is defined for the probability mass function as:

        $$
        F(x; a) = {\frac {p_{i}}{p_{j}}}=\exp \left({\frac
        {\varepsilon _{j}-\varepsilon _{i}}{kT}}\right)
        $$

        where $p_i$ is the probability of a system being in state $i$, $p_j$ is the
        probability of a system being in state $j$, $\varepsilon_i$ is the energy of
        state $i$, $\varepsilon_j$ is the energy of state $j$, $k$ is the Boltzmann
        constant, and $T$ is the temperature.

    Info:
        For simplicity, the exponentianal term of the Boltzmann factor $k$ is
        simpflified from  $1.380649 \times 10^{-23}$ to 1.

    Args:
        x (UniversalArray): The value(s) at which the function is evaluated.
        energy_i (float): The energy of state $i$.
        energy_j (float): The energy of state $j$.
        temperature (float): The temperature of the system.
    """

    def __init__(
        self,
        *x: UniversalArray,
        energy_i: float,
        energy_j: float,
        k: float = 1,
    ) -> None:
        """Initialize the Boltzmann distribution."""
        if energy_i == energy_j:
            msg = "'energy_i' and 'energy_j' cannot be equal."
            raise ValueError(msg)
        if energy_i < 0:
            raise NotLargerThanZeroError(
                var_number="energy_i",
                number=energy_i,
            )
        if energy_j < 0:
            raise NotLargerThanZeroError(
                var_number="energy_j",
                number=energy_j,
            )
        if (min_temp := float(np.min(x))) <= 0:
            raise NotLargerThanZeroError(
                var_number="temperature",
                number=min_temp,
            )
        super().__init__(*x)
        self.energy_i = energy_i
        self.energy_j = energy_j
        self.temperature = self._x
        self.k = k

    def probability_mass_function(self) -> UniversalArray:
        """Probability mass function of the Boltzmann distribution."""
        return np.exp(-(self.energy_j - self.energy_i) / (self.k * self.temperature))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Boltzmann distribution."""
        return SummaryStatisticsAPI(
            mean=None,
            variance=None,
            mode=None,
            doc=self.__doc__,
        )


class MaxwellBoltzmannDistribution(DiscretePure):
    r"""Maxwell-Boltzmann distribution.

    The Maxwell-Boltzmann distribution is a discrete probability distribution with
    discrete infinite support. It is used to describe the distribution of the speeds of
    particles in a gas at a given temperature in statistical mechanics and
    thermodynamics.

    Examples:
        >>> # PMF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.discrete_infinite_support import (
        ... MaxwellBoltzmannDistribution
        ... )
        >>> x = np.linspace(0.5, 20, 1000)
        >>> y_1 = MaxwellBoltzmannDistribution(x, a=1).__eval__
        >>> y_2 = MaxwellBoltzmannDistribution(x, a=2).__eval__
        >>> y_3 = MaxwellBoltzmannDistribution(x, a=3).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot(111)
        >>> _  = ax.plot(x, y_1, label=r"$a=1$")
        >>> _  = ax.plot(x, y_2, label=r"$a=2$")
        >>> _  = ax.plot(x, y_3, label=r"$a=3$")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel(r"$p$")
        >>> _  = ax.legend()
        >>> plt.savefig("MaxwellBoltzmannDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.discrete_infinite_support import (
        ... MaxwellBoltzmannDistribution
        ... )
        >>> x = np.linspace(0.5, 20, 1000)
        >>> y_1 = MaxwellBoltzmannDistribution(
        ... x,
        ... a=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_2 = MaxwellBoltzmannDistribution(
        ... x,
        ... a=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_3 = MaxwellBoltzmannDistribution(
        ... x,
        ... a=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot(111)
        >>> _  = ax.plot(x, y_1, label=r"$a=1$")
        >>> _  = ax.plot(x, y_2, label=r"$a=2$")
        >>> _  = ax.plot(x, y_3, label=r"$a=3$")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel(r"$F(x)$")
        >>> _  = ax.legend()
        >>> plt.savefig(
        ... "MaxwellBoltzmannDistribution-cml.png",
        ... dpi=300,
        ... transparent=True,
        ... )

    Notes:
        The Maxwell-Boltzmann distribution is defined for the PMF as follows:

        $$
        F(x; a) = \sqrt {\frac {2}{\pi }}\,{\frac {x^{2}}{a^{3}}}
        \,\exp \left({\frac {-x^{2}}{2a^{2}}}\right)
        $$

        where $x$ is the speed of a particle, $a$ is the most probable speed of $a$,
        $\pi$ is the constant pi, and $a$ is a parametrization.

        The Maxwell-Boltzmann distribution is defined for the CDF as follows:

        $$
        F(x; a) = \operatorname {erf} \left({\frac {x}{{\sqrt {2}}a}}\right)
        -{\sqrt {\frac {2}{\pi }}}\,{\frac {x}{a}}\,\exp
        \left({\frac {-x^{2}}{2a^{2}}}\right)
        $$

        For more informtation about the Maxwell-Boltzmann distribution, see also
        https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution


    Args:
        *x (UniversalArray): The value(s) at which the function is evaluated.
        a(float): A parametrization for the Co-Factors of the Maxwell-Boltzmann
            distribution.
    """

    def __init__(self, *x: UniversalArray, a: float, cumulative: bool = False) -> None:
        """Initialize the Maxwell-Boltzmann distribution."""
        if a <= 0:
            msg = "a"
            raise NotAPositiveNumberError(msg, a)
        super().__init__(*x, cumulative=cumulative)
        self.a = a

    def probability_mass_function(self) -> UniversalArray:
        """Probability mass function of the Maxwell-Boltzmann distribution."""
        return (
            np.sqrt(2 / np.pi)
            * (self._x**2 / self.a**3)
            * np.exp(-(self._x**2) / (2 * self.a**2))
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Cumulative distribution function of the Maxwell-Boltzmann distribution."""
        return erf(self._x / (np.sqrt(2) * self.a)) - np.sqrt(
            2 / np.pi,
        ) * self._x / self.a * np.exp(-(self._x**2) / (2 * self.a**2))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Maxwell-Boltzmann distribution."""
        return SummaryStatisticsAPI(
            mean=2 * np.sqrt(2 / np.pi) * self.a,
            variance=(self.a**2 * (3 * np.pi - 8)) / (np.pi),
            mode=np.sqrt(2) * self.a,
            doc=self.__doc__,
        )


class GausKuzminDistribution(DiscretePure):
    r"""Gaus-Kuzmin distribution.

    The Gaus-Kuzmin distribution is a discrete probability distribution with discrete
    infinite support. It is used to describe the distribution of the number of steps
    taken by a random walker on a line before reaching a given distance from the origin.

    Examples:
        >>> # PMF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.discrete_infinite_support import (
        ... GausKuzminDistribution
        ... )
        >>> x = np.arange(1, 100, dtype=int)
        >>> y = GausKuzminDistribution(x).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot(111)
        >>> _  = ax.plot(x, y, label=r"$p$")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel(r"$p$")
        >>> _  = ax.legend()
        >>> plt.savefig("GausKuzminDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.discrete_infinite_support import (
        ... GausKuzminDistribution
        ... )
        >>> x = np.arange(1, 100, dtype=int)
        >>> y = GausKuzminDistribution(x, cumulative=True).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot(111)
        >>> _  = ax.plot(x, y, label=r"$F(x)$")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel(r"$F(x)$")
        >>> _  = ax.legend()
        >>> plt.savefig("GausKuzminDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The Gaus-Kuzmin distribution is defined for the PMF as
        follows:

        $$
        F(x) = -\log _{2}\left[1-{\frac  {1}{(x+1)^{2}}}\right]
        $$

        where $k$ is the number of steps taken by a random walker on a line before
        reaching a given distance from the origin.

        The Gaus-Kuzmin distribution is defined for the CDF as follows:

        $$
        F(x) = 1-\log _{2}\left({\frac  {x+2}{x+1}}\right)
        $$

        For more information about the Gaus-Kuzmin distribution, see also
        <https://en.wikipedia.org/wiki/Gauss-Kuzmin_distribution>

    Args:
        *x (UniversalArray): The value(s) at which the function is evaluated.
    """

    def probability_mass_function(self) -> UniversalArray:
        """Probability mass function of the Gaus-Kuzmin distribution."""
        return -np.log2(1 - 1 / (self._x + 1) ** 2)

    def cumulative_distribution_function(self) -> UniversalArray:
        """Cumulative distribution function of the Gaus-Kuzmin distribution."""
        return 1 - np.log2((self._x + 2) / (self._x + 1))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Gaus-Kuzmin distribution."""
        return SummaryStatisticsAPI(
            mean=np.inf,
            variance=np.inf,
            mode=1,
            doc=self.__doc__,
        )
