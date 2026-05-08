"""Discrete distributions with finite support for the umf package."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

from umf.functions.other.support_functions import combinations
from umf.meta.api import SummaryStatisticsAPI
from umf.meta.functions import DiscreteP


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = ["BernoulliDistribution", "BinomialDistribution"]


class BernoulliDistribution(DiscreteP):
    r"""Bernoulli distribution.

    The Bernoulli distribution is a discrete distribution with two possible
    outcomes, 0 and 1. It is the simplest discrete distribution. It is a
    special case of the binomial distribution where a single trial is
    conducted (n=1).

    Examples:
        >>> # PMF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.discrete_finite_support import (
        ... BernoulliDistribution
        ... )
        >>> x = np.linspace(0, 1, 1000)
        >>> y_05 = BernoulliDistribution(x, p=0.5).__eval__
        >>> y_07 = BernoulliDistribution(x, p=0.7).__eval__
        >>> y_09 = BernoulliDistribution(x, p=0.9).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _  = ax.plot(x, y_05, label="p=0.5")
        >>> _  = ax.plot(x, y_07, label="p=0.7")
        >>> _  = ax.plot(x, y_09, label="p=0.9")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("f(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("BernoulliDistribution.png", dpi=300, transparent=True)

    Notes:
        The Bernoulli distribution is defined as follows:

        $$
        f(x;p) = p^x (1-p)^{1-x}
        $$

        where $x \in \{0, 1\}$ and $p \in [0, 1]$.

    Args:
        *x (UniversalArray): The value(s) at which the function is evaluated.
        p (float): The probability of success.
    """

    def probability_mass_function(self) -> UniversalArray:
        """Probability mass function of the Bernoulli distribution."""
        return self.p**self._x * (self.q) ** (1 - self._x)

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Bernoulli distribution."""

        def _mode() -> float | tuple[float, float]:
            """Mode of the Bernoulli distribution."""
            threshold = 0.5
            if self.p > threshold:
                return 1
            return 0 if self.p < threshold else (0, 1)

        return SummaryStatisticsAPI(
            mean=self.p,
            variance=self.p * self.q,
            mode=_mode(),
            doc=self.__doc__,
        )


class BinomialDistribution(DiscreteP):
    r"""Binomial distribution.

    The binomial distribution is a discrete distribution with two possible
    outcomes, 0 and 1. It is a generalization of the Bernoulli distribution
    where $n$ trials are conducted.

    Examples:
        >>> # PMF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.discrete_finite_support import (
        ... BinomialDistribution
        ... )
        >>> x = np.arange(0, 50, dtype=int)
        >>> y_05 = BinomialDistribution(x, p=0.5).__eval__
        >>> y_07 = BinomialDistribution(x,p=0.7).__eval__
        >>> y_09 = BinomialDistribution(x,p=0.9).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _  = ax.plot(x, y_05, label="p=0.5")
        >>> _  = ax.plot(x, y_07, label="p=0.7")
        >>> _  = ax.plot(x, y_09, label="p=0.9")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("f(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("BinomialDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.discrete_finite_support import (
        ... BinomialDistribution
        ... )
        >>> x = np.arange(0, 50, dtype=int )
        >>> y_05 = BinomialDistribution(x,  p=0.5, cumulative=True).__eval__
        >>> y_07 = BinomialDistribution(x,  p=0.7, cumulative=True).__eval__
        >>> y_09 = BinomialDistribution(x,  p=0.9, cumulative=True).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _  = ax.plot(x, y_05, label="p=0.5")
        >>> _  = ax.plot(x, y_07, label="p=0.7")
        >>> _  = ax.plot(x, y_09, label="p=0.9")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("F(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("BinomialDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The binomial distribution is defined as follows for probability mass function:

        $$
        f(x;n,p) = \binom{n}{k} p^k (1-p)^{n-k}
        $$

        where $k \in \{0, 1, ..., n\}$, $n \in \mathbb{N}$, and $p \in [0, 1]$ and
        $\binom{n}{k}$ is the binomial coefficient. $1 - p$ is also denoted as $q$.

        The binomial distribution is defined as follows for cumulative distribution
        function:

        $$
        F(x;n,p) = \sum_{k=0}^x \binom{n}{k} p^k (1-p)^{n-k}
        $$

        where $k \in \{0, 1, ..., n\}$, $n \in \mathbb{N}$, and $p \in [0, 1]$ and
        $\binom{n}{k}$ is the binomial coefficient. $1 - p$ is also denoted as $q$.
        This expression is also known as the regularized incomplete beta function.

        $$
        F(x;n,p) = I_{1-p}(n-k, k+1)
        $$



    Args:
        *x (UniversalArray): The value(s) at which the function is evaluated.
        p (float): The probability of success.
        cumulative: If True, the cumulative distribution function is returned.
            Defaults to False.
    """

    def __init__(self, *x: UniversalArray, p: float, cumulative: bool = False) -> None:
        """Initialize the Binomial distribution."""
        super().__init__(*x, p=p, cumulative=cumulative)
        self.n = np.full_like(self._x, self._x[-1])
        self.k = self._x

    def probability_mass_function(self) -> UniversalArray:
        """Probability mass function of the Binomial distribution."""
        return (
            combinations(self.n, self.k) * self.p**self.k * self.q ** (self.n - self.k)
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Cumulative distribution function of the Binomial distribution."""
        return np.array(
            [
                np.sum(
                    [
                        combinations(self.n[i], k)
                        * self.p**k
                        * self.q ** (self.n[i] - k)
                        for k in range(self.k[i] + 1)
                    ],
                )
                for i in range(len(self._x))
            ],
        )

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Binomial distribution."""
        return SummaryStatisticsAPI(
            mean=self.n.max() * self.p,
            variance=self.n.max() * self.p * self.q,
            mode=math.ceil((self.n.max() + 1) * self.p) - 1,
            doc=self.__doc__,
        )
