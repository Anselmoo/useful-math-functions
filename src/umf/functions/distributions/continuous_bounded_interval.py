"""Continuous bounded interval distributions for the umf package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.special import gamma

from umf.constants.exceptions import NotAPositiveNumberError
from umf.meta.api import SummaryStatisticsAPI
from umf.meta.functions import ContinuousBoundedInterval


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray


__all__: list[str] = ["KumaraswamyDistribution"]


class KumaraswamyDistribution(ContinuousBoundedInterval):
    r"""Kumaraswamy distribution.

    The Kumaraswamy distribution is a continuous probability distribution with
    support on the interval [0, 1]. It is a generalization of the beta
    distribution.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_bounded_interval import (
        ... KumaraswamyDistribution
        ... )
        >>> x = np.linspace(0, 1, 1000)
        >>> kumaraswamy_a1_b1 = KumaraswamyDistribution(x, a=1, b=1).__eval__
        >>> kumaraswamy_a2_b2 = KumaraswamyDistribution(x, a=2, b=2).__eval__
        >>> kumaraswamy_a1_b2 = KumaraswamyDistribution(x, a=1, b=2).__eval__
        >>> kumaraswamy_a2_b1 = KumaraswamyDistribution(x, a=2, b=1).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, kumaraswamy_a1_b1, label=r"$a=1, b=1$")
        >>> _ = ax.plot(x, kumaraswamy_a2_b2, label=r"$a=2, b=2$")
        >>> _ = ax.plot(x, kumaraswamy_a1_b2, label=r"$a=1, b=2$")
        >>> _ = ax.plot(x, kumaraswamy_a2_b1, label=r"$a=2, b=1$")
        >>> _ = ax.legend()
        >>> plt.savefig("KumaraswamyDistribution.png", dpi=300, transparent=True)
        >>> plt.close()

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_bounded_interval import (
        ... KumaraswamyDistribution
        ... )
        >>> x = np.linspace(0, 1, 1000)
        >>> kumaraswamy_a1_b1 = KumaraswamyDistribution(
        ... x,
        ... a=1,
        ... b=1,
        ... cumulative=True,
        ... ).__eval__
        >>> kumaraswamy_a2_b2 = KumaraswamyDistribution(
        ... x,
        ... a=2,
        ... b=2,
        ... cumulative=True,
        ... ).__eval__
        >>> kumaraswamy_a1_b2 = KumaraswamyDistribution(
        ... x,
        ... a=1,
        ... b=2,
        ... cumulative=True,
        ... ).__eval__
        >>> kumaraswamy_a2_b1 = KumaraswamyDistribution(
        ... x,
        ... a=2,
        ... b=1,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, kumaraswamy_a1_b1, label=r"$a=1, b=1$")
        >>> _ = ax.plot(x, kumaraswamy_a2_b2, label=r"$a=2, b=2$")
        >>> _ = ax.plot(x, kumaraswamy_a1_b2, label=r"$a=1, b=2$")
        >>> _ = ax.plot(x, kumaraswamy_a2_b1, label=r"$a=2, b=1$")
        >>> _ = ax.legend()
        >>> plt.savefig(
        ... "KumaraswamyDistribution-cml.png",
        ... dpi=300,
        ... transparent=True
        ... )

    Notes:
        The Kumaraswamy distribution is generally defined for the PDF as:

        $$
        f(x; a, b) = abx^{a-1}(1-x^a)^{b-1}
        $$

        where $a, b > 0$ and $0 \leq x \leq 1$. The CDF is given by:

        $$
        F(x; a, b) = 1 - (1 - x^a)^b
        $$

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        a (float): The first shape parameter, which must be positive. Default is 1.
        b (float): The second shape parameter, which must be positive. Default is 1.
        cumulative (bool): If True, the cumulative distribution function is returned.
    """

    def __init__(
        self,
        *x: UniversalArray,
        a: float = 1,
        b: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the Kumaraswamy distribution."""
        if a <= 0:
            msg = "a"
            raise NotAPositiveNumberError(msg, a)
        if b <= 0:
            msg = "b"
            raise NotAPositiveNumberError(msg, b)
        super().__init__(*x, start=0, end=1, cumulative=cumulative)
        self.a = a
        self.b = b

    def probability_density_function(self) -> UniversalArray:
        """Calculate the probability density function of the Kumaraswamy distribution.

        Returns:
            UniversalArray: The value of the probability density function of the
                Kumaraswamy distribution.
        """
        return (
            self.a
            * self.b
            * self._x ** (self.a - 1)
            * (1 - self._x**self.a) ** (self.b - 1)
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Calculate the cumulative distribution function of the Kumaraswamy distribution.

        Returns:
            UniversalArray: The value of the cumulative distribution function of the
                Kumaraswamy distribution.
        """  # noqa: E501
        return 1 - (1 - self._x**self.a) ** self.b

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Calculate the summary statistics of the Kumaraswamy distribution.

        Returns:
            SummaryStatisticsAPI: The summary statistics of the Kumaraswamy
                distribution.
        """
        mode = (
            (self.a - 1) / (self.a + self.b - 2) if self.a > 1 and self.b > 1 else None
        )
        mean = (self.b * gamma(1 + 1 / self.a) * gamma(self.a - 1)) / (
            self.a * gamma(self.a + self.b)
        )
        return SummaryStatisticsAPI(
            mean=mean,
            mode=mode,
            variance=None,
            doc=self.__doc__,
        )
