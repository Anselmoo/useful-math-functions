"""Continous distributions with whole line support for the umf package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.special import gamma
from scipy.special import gammainc

from umf.constants.exceptions import NotAPositiveNumberError
from umf.constants.exceptions import NotLargerThanZeroError
from umf.functions.other.support_functions import erf
from umf.functions.other.support_functions import wofz
from umf.meta.api import SummaryStatisticsAPI
from umf.meta.functions import ContinuousDistributionBase
from umf.meta.functions import ContinuousWBeta
from umf.meta.functions import ContinuousWLambda
from umf.meta.functions import ContinuousWSigma


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "GeneralizedNormalDistribution",
    "GaussianDistribution",
    "SkewGaussianDistribution",
    "LaplaceDistribution",
    "LogisticDistribution",
    "VoigtDistribution",
    "CauchyDistribution",
    "GumbelDistribution",
    "ExponentialDistribution",
]


class GeneralizedNormalDistribution(ContinuousWBeta):
    r"""Generalized normal distribution.

    The generalized normal distribution is a probability distribution that extends the
    normal distribution to incorporate an additional shape parameter, allowing for
    greater flexibility in modeling a wider range of data distributions.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... GeneralizedNormalDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_beta_1 = GeneralizedNormalDistribution(x, beta=1).__eval__
        >>> y_beta_2 = GeneralizedNormalDistribution(x, beta=2).__eval__
        >>> y_beta_3 = GeneralizedNormalDistribution(x, beta=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_beta_1, label="beta=1")
        >>> _ = ax.plot(x, y_beta_2, label="beta=2")
        >>> _ = ax.plot(x, y_beta_3, label="beta=3")
        >>> _ = ax.legend()
        >>> plt.savefig("GeneralizedNormalDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... GeneralizedNormalDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_beta_1 = GeneralizedNormalDistribution(
        ... x,
        ... beta=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_beta_2 = GeneralizedNormalDistribution(
        ... x,
        ... beta=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_beta_3 = GeneralizedNormalDistribution(
        ... x,
        ... beta=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_beta_1, label="beta=1")
        >>> _ = ax.plot(x, y_beta_2, label="beta=2")
        >>> _ = ax.plot(x, y_beta_3, label="beta=3")
        >>> _ = ax.legend()
        >>> plt.savefig(
        ... "GeneralizedNormalDistribution-cml.png",
        ... dpi=300,
        ... transparent=True,
        ... )

    Notes:
        The generalized normal distribution is generally defined for the PDF as:

        $$
        f(x | \beta, \mu, \alpha) = \frac{\alpha}{2\beta \Gamma(1/\alpha)}
          \exp\left(-\left|\frac{x - \mu}{\beta}\right|^\alpha\right)
        $$

        and for the CDF as:

        $$
        F(x | \beta, \mu, \alpha) = \frac{1}{2} + \frac{\mathrm{sign}(x - \mu)}{2}
        \left(1 - \frac{\Gamma\left(\frac{1}{\alpha},
        \left|\frac{x - \mu}{\beta}\right|^\alpha\right)}{\Gamma
        \left(\frac{1}{\alpha}\right)}\right)
        $$

        where $\alpha$ is the shape parameter, $\beta$ is the scale parameter, and
        $\mu$ is the location parameter plus $\Gamma$ as the gamma function.
        The PDF is defined for $x \in \mathbb{R}$ and $\alpha, \beta > 0$.
        The CDF is defined for $x \in \mathbb{R}$ and $\alpha > 0$ and rquires the
        unnormalized lower incomplete gamma function.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        mu (float): Location parameter. Defaults to 0.
        alpha (float): Shape parameter. Defaults to 1.
        beta (float): Scale parameter. Defaults to 1.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        alpha: float = 1,
        beta: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if beta < 0:
            msg = "beta"
            raise NotAPositiveNumberError(msg, beta)
        if alpha <= 0:
            msg = "alpha"
            raise NotLargerThanZeroError(msg, alpha)

        super().__init__(*x, mu=mu, beta=beta, cumulative=cumulative)
        self.alpha = alpha

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        return (
            self.beta
            / (2 * self.alpha * gamma(1.0 / self.beta))
            * np.exp(-(np.abs((self._x - self.mu) / self.alpha) ** self.beta))
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        return 0.5 + np.sign(self._x - self.mu) * (
            1 / (2 * gamma(1 / self.beta))
        ) * gammainc(
            1 / self.beta,
            np.abs((self._x - self.mu) / self.alpha) ** self.beta,
        )

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.mu,
            variance=self.alpha**2 * gamma(3 / self.beta) / gamma(1 / self.beta),
            mode=self.mu,
            doc=self.__doc__,
        )


class GaussianDistribution(ContinuousWSigma):
    r"""Gaussian distribution.

    The Gaussian distribution is a continuous probability distribution that is widely
    used in statistics to describe the normal distributions.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... GaussianDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_sigma_1 = GaussianDistribution(x, sigma=1).__eval__
        >>> y_sigma_2 = GaussianDistribution(x, sigma=2).__eval__
        >>> y_sigma_3 = GaussianDistribution(x, sigma=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_sigma_1, label="sigma=1")
        >>> _ = ax.plot(x, y_sigma_2, label="sigma=2")
        >>> _ = ax.plot(x, y_sigma_3, label="sigma=3")
        >>> _ = ax.legend()
        >>> plt.savefig("GaussianDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... GaussianDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_sigma_1 = GaussianDistribution(
        ... x,
        ... sigma=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_sigma_2 = GaussianDistribution(
        ... x,
        ... sigma=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_sigma_3 = GaussianDistribution(
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
        >>> plt.savefig("GaussianDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The Gaussian distribution is generally defined for the PDF as:

        $$
        f(x | \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}}
          \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
        $$

        and for the CDF as:

        $$
        F(x | \mu, \sigma) = \frac{1}{2} \left[1 + \mathrm{erf}
        \left(\frac{x - \mu}{\sigma \sqrt{2}}\right)\right]
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
            / (self.sigma * np.sqrt(2 * np.pi))
            * np.exp(-((self._x - self.mu) ** 2) / (2 * self.sigma**2))
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        return 0.5 * (1 + erf((self._x - self.mu) / (self.sigma * np.sqrt(2))))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.mu,
            variance=self.sigma**2,
            mode=self.mu,
            doc=self.__doc__,
        )


class SkewGaussianDistribution(ContinuousWSigma):
    r"""Skew Gaussian distribution.

    The Skew Gaussian distribution is a continuous probability distribution that is
    widely used in statistics to describe the normal distributions with skewness.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... SkewGaussianDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_sigma_1_alpha_0 = SkewGaussianDistribution(
        ... x,
        ... sigma=1,
        ... alpha=0,
        ... ).__eval__
        >>> y_sigma_1_alpha_1 = SkewGaussianDistribution(
        ... x,
        ... sigma=1,
        ... alpha=1,
        ... ).__eval__
        >>> y_sigma_1_alpha_minus_1 = SkewGaussianDistribution(
        ... x,
        ... sigma=1,
        ... alpha=-1,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_sigma_1_alpha_0, label="sigma=1, alpha=0")
        >>> _ = ax.plot(x, y_sigma_1_alpha_1, label="sigma=1, alpha=1")
        >>> _ = ax.plot(x, y_sigma_1_alpha_minus_1, label="sigma=1, alpha=-1")
        >>> _ = ax.legend()
        >>> plt.savefig("SkewGaussianDistribution.png", dpi=300, transparent=True)

    Notes:
        The Skew Gaussian distribution is generally defined for the PDF as:

        $$
        f(x | \mu, \sigma, \alpha) = \frac{1}{\sigma \sqrt{2\pi}}
        \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
        \left[1 + \mathrm{erf}\left(\frac{\alpha(x - \mu)}{\sigma\sqrt{2}}\right)\right]
        $$

        where $\mu$ is the mean, $\sigma$ is the standard deviation, and $\alpha$ is the
        skewness.

        The CDF is not available in closed form.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        sigma (float): Standard deviation. Defaults to 1.
        mu (float): Mean. Defaults to 0.
        alpha (float): Skewness. Defaults to 0.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        alpha: float = 0,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*x, mu=mu, sigma=sigma, cumulative=cumulative)
        self.alpha = alpha

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        return (
            1
            / (self.sigma * np.sqrt(2 * np.pi))
            * np.exp(-((self._x - self.mu) ** 2) / (2 * self.sigma**2))
            * (1 + erf(self.alpha * (self._x - self.mu) / (self.sigma * np.sqrt(2))))
        )

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.mu,
            variance=self.sigma**2,
            mode=self.mu,
            doc=self.__doc__,
        )


class LaplaceDistribution(ContinuousWBeta):
    r"""Laplace distribution.

    The Laplace distribution is a continuous probability distribution that is widely
    used in statistics to describe the normal distributions.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... LaplaceDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_beta_1 = LaplaceDistribution(x, beta=1).__eval__
        >>> y_beta_2 = LaplaceDistribution(x, beta=2).__eval__
        >>> y_beta_3 = LaplaceDistribution(x, beta=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_beta_1, label="beta=1")
        >>> _ = ax.plot(x, y_beta_2, label="beta=2")
        >>> _ = ax.plot(x, y_beta_3, label="beta=3")
        >>> _ = ax.legend()
        >>> plt.savefig("LaplaceDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... LaplaceDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_beta_1 = LaplaceDistribution(
        ... x,
        ... beta=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_beta_2 = LaplaceDistribution(
        ... x,
        ... beta=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_beta_3 = LaplaceDistribution(
        ... x,
        ... beta=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_beta_1, label="beta=1")
        >>> _ = ax.plot(x, y_beta_2, label="beta=2")
        >>> _ = ax.plot(x, y_beta_3, label="beta=3")
        >>> _ = ax.legend()
        >>> plt.savefig("LaplaceDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The Laplace distribution is generally defined for the PDF as:

        $$
        f(x | \mu, \beta) = \frac{1}{2\beta} \exp\left(-\frac{|x - \mu|}{\beta}\right)
        $$

        and for the CDF as:

        $$
        F(x | \mu, \beta) = \frac{1}{2} + \frac{1}{2}\mathrm{sign}(x - \mu)
        \left(1 - \exp\left(-\frac{|x - \mu|}{\beta}\right)\right)
        $$

        where $\mu$ is the mean and $\beta$ is the scale parameter.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        beta (float): Scale parameter. Defaults to 1.
        mu (float): Mean. Defaults to 0.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        return 1 / (2 * self.beta) * np.exp(-np.abs((self._x - self.mu) / self.beta))

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        return np.array(
            0.5
            + 0.5
            * np.sign(self._x - self.mu)
            * (1 - np.exp(-np.abs((self._x - self.mu) / self.beta))),
        )

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.mu,
            variance=2 * self.beta**2,
            mode=self.mu,
            doc=self.__doc__,
        )


class LogisticDistribution(ContinuousWBeta):
    r"""Logistic distribution.

    The logistic distribution is a continuous probability distribution that is widely
    used in statistics to describe the normal distributions.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... LogisticDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_beta_1 = LogisticDistribution(x, beta=1).__eval__
        >>> y_beta_2 = LogisticDistribution(x, beta=2).__eval__
        >>> y_beta_3 = LogisticDistribution(x, beta=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_beta_1, label="beta=1")
        >>> _ = ax.plot(x, y_beta_2, label="beta=2")
        >>> _ = ax.plot(x, y_beta_3, label="beta=3")
        >>> _ = ax.legend()
        >>> plt.savefig("LogisticDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... LogisticDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_beta_1 = LogisticDistribution(
        ... x,
        ... beta=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_beta_2 = LogisticDistribution(
        ... x,
        ... beta=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_beta_3 = LogisticDistribution(
        ... x,
        ... beta=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_beta_1, label="beta=1")
        >>> _ = ax.plot(x, y_beta_2, label="beta=2")
        >>> _ = ax.plot(x, y_beta_3, label="beta=3")
        >>> _ = ax.legend()
        >>> plt.savefig("LogisticDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The Logistic distribution is generally defined for the PDF as:

        $$
        f(x | \mu, \beta) = \frac{1}{\beta} \exp\left(-\frac{x -
        \mu}{\beta}\right)\left(1 + \exp\left(-\frac{x - \mu}{\beta}\right)\right)^{-2}
        $$

        and for the CDF as:

        $$
        F(x | \mu, \beta) = \frac{1}{1 + \exp\left(-\frac{x - \mu}{\beta}\right)}
        $$

        where $\mu$ is the mean and $\beta$ is the scale parameter.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        beta (float): Scale parameter. Defaults to 1.
        mu (float): Mean. Defaults to 0.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        return np.array(
            1
            / self.beta
            * np.exp(-(self._x - self.mu) / self.beta)
            / (1 + np.exp(-(self._x - self.mu) / self.beta)) ** 2,
        )

    def cumulative_distribution_function(self) -> UniversalArray:
        """Return the cumulative distribution function."""
        return 1 / (1 + np.exp(-(self._x - self.mu) / self.beta))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.mu,
            variance=(np.pi**2 * self.beta**2) / 3,
            mode=self.mu,
            doc=self.__doc__,
        )


class VoigtDistribution(ContinuousWSigma):
    r"""Voigt distribution.

    The Voigt distribution is a continuous probability distribution that is widely used
    in physics and spectroscopy to describe the line shape of spectral lines. It is a
    convolution of a Gaussian distribution and a Lorentzian distribution, and is useful
    for modeling the effects of both natural and instrumental broadening on spectral
    lines.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... VoigtDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_sigma_1 = VoigtDistribution(x, sigma=1).__eval__
        >>> y_sigma_2 = VoigtDistribution(x, sigma=2).__eval__
        >>> y_sigma_3 = VoigtDistribution(x, sigma=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_sigma_1, label="sigma=1")
        >>> _ = ax.plot(x, y_sigma_2, label="sigma=2")
        >>> _ = ax.plot(x, y_sigma_3, label="sigma=3")
        >>> _ = ax.legend()
        >>> plt.savefig("VoigtDistribution.png", dpi=300, transparent=True)

    Notes:
        The Voigt distribution is generally defined for the PDF as:

        $$
        f(x | \mu, \sigma, \gamma) = \frac{1}{\sigma \sqrt{2\pi}}
          \int_{-\infty}^\infty \exp\left(-\frac{(x - y)^2}{2\sigma^2}\right)
          \frac{\gamma}{\pi\left((x - y)^2 + \gamma^2\right)} dy
        $$

        which can be further simplified to:

        $$
        V(x;\sigma ,\gamma )={\frac {\operatorname {Re} [w(z)]}{\sigma {\sqrt {2\pi }}}}
        $$

        with $\operatorname {Re} [w(z)$ as the real part of the Faddeeva function and
        $z$ as:

        $$
        z={\frac {x+i\gamma }{\sigma {\sqrt {2}}}}
        $$

        and for the CDF as:

        $$
        F(x | \mu, \sigma, \gamma) = \frac{1}{\sigma \sqrt{2\pi}}
          \int_{-\infty}^x \exp\left(-\frac{(x - y)^2}{2\sigma^2}\right)
          \frac{\gamma}{\pi\left((x - y)^2 + \gamma^2\right)} dy
        $$

        where $\mu$ is the mean, $\sigma$ is the standard deviation and $\gamma$ is the
        Lorentzian width.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        sigma (float): Standard deviation. Defaults to 1.
        gamma (float): Lorentzian width. Defaults to 1.
        mu (float): Mean. Defaults to 0.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def __init__(
        self,
        *x: UniversalArray,
        sigma: float = 1,
        gamma: float = 1,
        mu: float = 0,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if sigma < 0:
            msg = "sigma"
            raise NotAPositiveNumberError(msg, sigma)
        if gamma < 0:
            msg = "gamma"
            raise NotAPositiveNumberError(msg, gamma)

        super().__init__(*x, mu=mu, sigma=sigma, cumulative=cumulative)
        self.gamma = gamma

    def probability_density_function(self) -> UniversalArray:
        """Return the probability density function."""
        z = (self._x + 1j * self.gamma) / (self.sigma * np.sqrt(2))
        return np.real(wofz(z)) / (self.sigma * np.sqrt(2 * np.pi))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=None,
            variance=None,
            mode=0,
            doc=self.__doc__,
        )


class CauchyDistribution(ContinuousDistributionBase):
    r"""Cauchy distribution.

    The Cauchy distribution is a continuous probability distribution that has no mean
    or variance. It is also known as the Lorentz distribution, after Hendrik Lorentz.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... CauchyDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_gamma_1 = CauchyDistribution(x, gamma=1).__eval__
        >>> y_gamma_2 = CauchyDistribution(x, gamma=2).__eval__
        >>> y_gamma_3 = CauchyDistribution(x, gamma=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_gamma_1, label="gamma=1")
        >>> _ = ax.plot(x, y_gamma_2, label="gamma=2")
        >>> _ = ax.plot(x, y_gamma_3, label="gamma=3")
        >>> _ = ax.legend()
        >>> plt.savefig("CauchyDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... CauchyDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_gamma_1 = CauchyDistribution(
        ... x,
        ... gamma=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_gamma_2 = CauchyDistribution(
        ... x,
        ... gamma=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_gamma_3 = CauchyDistribution(
        ... x,
        ... gamma=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_gamma_1, label="gamma=1")
        >>> _ = ax.plot(x, y_gamma_2, label="gamma=2")
        >>> _ = ax.plot(x, y_gamma_3, label="gamma=3")
        >>> _ = ax.legend()
        >>> plt.savefig("CauchyDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The Cauchy distribution is defined as:

        $$
        f(x | x_0, \gamma) = \frac{1}{\pi \gamma \left[1 +
        \left(\frac{x - x_0}{\gamma}\right)^2\right]}
        $$

        where $x_0$ is the location parameter and $\gamma$ is the scale parameter.

        The cumulative distribution function (CDF) of the Cauchy distribution is:

        $$
        F(x | x_0, \gamma) = \frac{1}{\pi}
        \arctan\left(\frac{x - x_0}{\gamma}\right) + \frac{1}{2}
        $$

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        mu (float): Location parameter. Defaults to 0.
        gamma (float): Scale parameter. Defaults to 1.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        gamma: float = 1,
        cumulative: bool = False,
    ) -> None:
        """Initialize the function."""
        if gamma <= 0:
            msg = "gamma must be positive"
            raise ValueError(msg)
        self.gamma = gamma
        super().__init__(*x, mu=mu, cumulative=cumulative)

    def probability_density_function(self) -> np.ndarray:
        """Return the probability density function."""
        return 1 / (np.pi * self.gamma * (1 + ((self._x - self.mu) / self.gamma) ** 2))

    def cumulative_distribution_function(self) -> np.ndarray:
        """Return the cumulative distribution function."""
        return 1 / np.pi * np.arctan((self._x - self.mu) / self.gamma) + 1 / 2

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=np.nan,
            variance=np.nan,
            mode=self.mu,
            doc=self.__doc__,
        )


class GumbelDistribution(ContinuousWBeta):
    r"""Gumbel distribution.

    The Gumbel distribution is a continuous probability distribution that is used to
    model the distribution of the maximum (or the minimum) of a number of samples of
    various distributions. It is a two-parameter family of curves, with the location
    parameter $\mu$ controlling the location of the distribution and the scale
    parameter $\beta$ controlling the spread of the distribution.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... GumbelDistribution
        ... )
        >>> x = np.linspace(-10, 20, 1000)
        >>> y_mu_0_beta_1 = GumbelDistribution(x, mu=0, beta=1).__eval__
        >>> y_mu_5_beta_2 = GumbelDistribution(x, mu=5, beta=2).__eval__
        >>> y_mu_10_beta_3 = GumbelDistribution(x, mu=10, beta=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_mu_0_beta_1, label="mu=0, beta=1")
        >>> _ = ax.plot(x, y_mu_5_beta_2, label="mu=5, beta=2")
        >>> _ = ax.plot(x, y_mu_10_beta_3, label="mu=10, beta=3")
        >>> _ = ax.legend()
        >>> plt.savefig("GumbelDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... GumbelDistribution
        ... )
        >>> x = np.linspace(-10, 20, 1000)
        >>> y_mu_0_beta_1 = GumbelDistribution(
        ... x,
        ... mu=0,
        ... beta=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_mu_5_beta_2 = GumbelDistribution(
        ... x,
        ... mu=5,
        ... beta=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_mu_10_beta_3 = GumbelDistribution(
        ... x,
        ... mu=10,
        ... beta=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_mu_0_beta_1, label="mu=0, beta=1")
        >>> _ = ax.plot(x, y_mu_5_beta_2, label="mu=5, beta=2")
        >>> _ = ax.plot(x, y_mu_10_beta_3, label="mu=10, beta=3")
        >>> _ = ax.legend()
        >>> plt.savefig("GumbelDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The Gumbel distribution is defined as:

        $$
        f(x | \mu, \beta) = \frac{1}{\beta}
        e^{-\frac{x - \mu + e^{-(x - \mu)/\beta}}{\beta}}
        $$

        where $\mu$ is the location parameter and $\beta$ is the scale parameter.

        The cumulative distribution function (CDF) of the Gumbel distribution is:

        $$
        F(x | \mu, \beta) = e^{-e^{-(x - \mu)/\beta}}
        $$

    Args:
        *x (UniversalArray): Input data, which can be only one dimensional.
        mu (float): Location parameter. Defaults to 0.
        beta (float): Scale parameter. Defaults to 1.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def probability_density_function(self) -> np.ndarray:
        """Return the probability density function."""
        return (
            1
            / self.beta
            * np.exp(
                -(self._x - self.mu + np.exp(-(self._x - self.mu) / self.beta))
                / self.beta,
            )
        )

    def cumulative_distribution_function(self) -> np.ndarray:
        """Return the cumulative distribution function."""
        return np.exp(-np.exp(-(self._x - self.mu) / self.beta))

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=self.mu + self.beta * np.euler_gamma,
            variance=(np.pi**2 / 6) * self.beta**2,
            mode=self.mu + self.beta * np.log(np.log(3)),
            doc=self.__doc__,
        )


class ExponentialDistribution(ContinuousWLambda):
    r"""Exponential distribution.

    The exponential distribution is a continuous probability distribution that
    describes the time between events in a Poisson process, where events occur
    continuously and independently at a constant average rate. It is a one-parameter
    family of curves, with the rate parameter $\lambda$ controlling the shape of the
    distribution. The exponential distribution is widely used in reliability theory,
    queueing theory, and other fields.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... ExponentialDistribution
        ... )
        >>> x = np.linspace(0, 5, 1000)
        >>> y_lambda_1 = ExponentialDistribution(x, lambda_=1).__eval__
        >>> y_lambda_2 = ExponentialDistribution(x, lambda_=2).__eval__
        >>> y_lambda_3 = ExponentialDistribution(x, lambda_=3).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_lambda_1, label="lambda=1")
        >>> _ = ax.plot(x, y_lambda_2, label="lambda=2")
        >>> _ = ax.plot(x, y_lambda_3, label="lambda=3")
        >>> _ = ax.legend()
        >>> plt.savefig("ExponentialDistribution.png", dpi=300, transparent=True)

        >>> # CDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.continuous_whole_line_support import (
        ... ExponentialDistribution
        ... )
        >>> x = np.linspace(0, 5, 1000)
        >>> y_lambda_1 = ExponentialDistribution(
        ... x,
        ... lambda_=1,
        ... cumulative=True,
        ... ).__eval__
        >>> y_lambda_2 = ExponentialDistribution(
        ... x,
        ... lambda_=2,
        ... cumulative=True,
        ... ).__eval__
        >>> y_lambda_3 = ExponentialDistribution(
        ... x,
        ... lambda_=3,
        ... cumulative=True,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> _ = ax.plot(x, y_lambda_1, label="lambda=1")
        >>> _ = ax.plot(x, y_lambda_2, label="lambda=2")
        >>> _ = ax.plot(x, y_lambda_3, label="lambda=3")
        >>> _ = ax.legend()
        >>> plt.savefig("ExponentialDistribution-cml.png", dpi=300, transparent=True)

    Notes:
        The exponential distribution is generally defined for the PDF as:

        $$
        f(x | \lambda) = \lambda e^{-\lambda x}
        $$

        and for the CDF as:

        $$
        F(x | \lambda) = 1 - e^{-\lambda x}
        $$

        where $\lambda$ is the rate parameter.

    Args:
        *x (UniversalArray): Input data, which can be only one  dimensional.
        lambda_ (float): Rate parameter. Defaults to 1.
        cumulative (bool): If True, the CDF is returned. Defaults to False.
    """

    def probability_density_function(self) -> np.ndarray:
        """Return the probability density function."""
        return self.lambda_ * np.exp(-self.lambda_ * self._x)

    def cumulative_distribution_function(self) -> np.ndarray:
        """Return the cumulative distribution function."""
        return 1 - np.exp(-self.lambda_ * self._x)

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Return the summary statistics."""
        return SummaryStatisticsAPI(
            mean=1 / self.lambda_,
            variance=1 / self.lambda_**2,
            mode=0,
            doc=self.__doc__,
        )
