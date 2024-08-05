"""Mixed discrete-continuous distributions for the for the umf package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.special import gammaln

from umf.constants.exceptions import NotLargerThanZeroError
from umf.functions.distributions.continuous_whole_line_support import (
    CauchyDistribution as LorentzianDistribution,
)
from umf.functions.distributions.continuous_whole_line_support import (
    GaussianDistribution,
)
from umf.functions.other.support_functions import erf
from umf.meta.api import SummaryStatisticsAPI
from umf.meta.functions import ContinousAsymmetricPseudo
from umf.meta.functions import ContinousPseudo
from umf.meta.functions import ContinuousWSigma


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray

__all__: list[str] = [
    "CrystalBallDistribution",
    "PearsonVIIDistribution",
    "PseudoVoigtDistribution",
    "AsymmetricRamanLineshape",
    "ModifiedDoniachSunjicDistribution",
]


class CrystalBallDistribution(ContinuousWSigma):
    r"""Crystal Ball distribution.

    The Crystal Ball distribution, which is sometimes also called the Crystal
    Ball function, is a continuous but asymmetric probability distribution on the real
    line.[^1] This type of function is often used in physics to model the invariant
    mass of a particle or system of particles, especially when there is a known
    background contribution.[^2]

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.mixed_discrete_continuous import (
        ... CrystalBallDistribution
        ... )
        >>> x = np.linspace(-25, 5, 1000)
        >>> y_111 = CrystalBallDistribution(x, mu=0, sigma=1, n=1, alpha=1).__eval__
        >>> y_211 = CrystalBallDistribution(x, mu=0, sigma=2, n=1, alpha=1).__eval__
        >>> y_121 = CrystalBallDistribution(x, mu=0, sigma=1, n=2, alpha=1).__eval__
        >>> y_221 = CrystalBallDistribution(x, mu=0, sigma=2, n=2, alpha=1).__eval__
        >>> y_112 = CrystalBallDistribution(x, mu=0, sigma=1, n=1, alpha=2).__eval__
        >>> y_222 = CrystalBallDistribution(x, mu=0, sigma=2, n=2, alpha=2).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _  = ax.plot(x, y_111, label=r"$\sigma=1, n=1, \alpha=1$")
        >>> _  = ax.plot(x, y_211, label=r"$\sigma=2, n=1, \alpha=1$")
        >>> _  = ax.plot(x, y_121, label=r"$\sigma=1, n=2, \alpha=1$")
        >>> _  = ax.plot(x, y_221, label=r"$\sigma=2, n=2, \alpha=1$")
        >>> _  = ax.plot(x, y_112, label=r"$\sigma=1, n=1, \alpha=2$")
        >>> _  = ax.plot(x, y_222, label=r"$\sigma=2, n=2, \alpha=2$")
        >>> _  = ax.set_xlabel("x")
        >>> _  = ax.set_ylabel("f(x)")
        >>> _  = ax.legend()
        >>> plt.savefig("CrystalBallDistribution.png", dpi=300, transparent=True)


    Notes:
        The Crystal Ball distribution is defined as follows for probability density
        [^2]:

        $$
        f(x;\alpha ,n,{\bar {x}},\sigma )=N\cdot {\begin{cases}\exp
        \left(-{\frac {(x-{\bar {x}})^{2}}{2\sigma ^{2}}}\right),&{\mbox{for }}{
        \frac {x-{\bar {x}}}{\sigma }}>-\alpha \\A\cdot
        \left(B-{\frac {x-{\bar {x}}}{\sigma }}
        \right)^{-n},&{\mbox{for }}{\frac {x-{\bar {x}}}{\sigma }}\leqslant -\alpha
        \end{cases}}
        $$

        with:

        $$
        A=\left({\frac  {n}{\left|\alpha \right|}}\right)^{n}\cdot \exp
        \left(-{\frac  {\left|\alpha \right|^{2}}{2}}\right)
        $$

        $$
        B={\frac  {n}{\left|\alpha \right|}}-\left|\alpha \right|
        $$

        $$
        N={\frac  {1}{\sigma (C+D)}}
        $$

        $$
        C={\frac  {n}{\left|\alpha \right|}}\cdot {\frac  {1}{n-1}}
        \cdot \exp \left(-{\frac  {\left|\alpha \right|^{2}}{2}}\right)
        $$

        $$
        D={\sqrt  {{\frac  {\pi }{2}}}}\left(1+\operatorname {erf}
        \left({\frac  {\left|\alpha \right|}{{\sqrt  2}}}\right)\right)
        $$

        See also: https://www.jlab.org/primex/weekly_meetings/slides_2009_07_17/dmitry/
        crystalball.html

        !!! warning "About the Normalization"
            The normalization constant $N$ might be not correct implemented because
            for the zero-division case $n=1$ the normalization constant $N$ is set to
            one to achieve a optical match with the reference figures of [^2].

        [^1]: Tomasz Skwarnicki, _A study of the radiative CASCADE transitions between_
            _ the Upsilon-Prime and Upsilon resonances_, **PHD-Thesis**, DESY-F31-86-02,
            Apr. 1986
        [^2]: Crystal Ball function. (2020, November 27). _In Wikipedia._
            https://en.wikipedia.org/wiki/Crystal_Ball_function


    Args:
        *x (UniversalArray): The points at which to evaluate the distribution.
        mu (float): The mean of the Gaussian region.
        sigma (float): The standard deviation of the Gaussian region.
        n (float): The power of the power-law region.
        alpha (float):: The location of the transition between the Gaussian and
            power-law regions.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        n: float = 1,
        alpha: float = 1,
    ) -> None:
        """Initialize the Crystal Ball distribution."""
        super().__init__(*x, mu=mu, sigma=sigma)

        if n <= 0:
            raise NotLargerThanZeroError(number=n)

        if alpha < 0:
            raise NotLargerThanZeroError(number=alpha)

        self.n = n
        self.alpha = alpha

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the Crystal Ball distribution."""
        _a = (self.n / abs(self.alpha)) ** self.n * np.exp(
            -(self.alpha**2) / 2,
        )
        _b = self.n / abs(self.alpha) - abs(self.alpha)
        if self.n == 1:
            _n = self.n
        else:
            _c = self.n / abs(self.alpha) / (self.n - 1) * np.exp(-(self.alpha**2) / 2)
            _d = np.sqrt(np.pi / 2) * (1 + erf(x=abs(self.alpha) / np.sqrt(2)))
            _n = 1 / (self.sigma * (_c + _d))
        return np.where(
            (self._x - self.mu) / self.sigma > -self.alpha,
            _n * np.exp(-((self._x - self.mu) ** 2) / (2 * self.sigma**2)),
            _n * _a * (_b - (self._x - self.mu) / self.sigma) ** (-self.n),
        )

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Crystal Ball distribution."""
        return SummaryStatisticsAPI(
            mean=None,
            variance=None,
            mode=None,
            doc=self.__doc__,
        )


class PearsonVIIDistribution(ContinuousWSigma):
    r"""Pearson VII distribution.

    The Pearson VII distribution is a continuous probability distribution on the real
    line. It is a generalization of the Student's t-distribution and the Cauchy
    distribution. This function becomes popular for X-ray diffraction data analysis.[^1]

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.mixed_discrete_continuous import (
        ... PearsonVIIDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_1 = PearsonVIIDistribution(x, mu=0, sigma=1, p=1).__eval__
        >>> y_2 = PearsonVIIDistribution(x, mu=0, sigma=1, p=2).__eval__
        >>> y_3 = PearsonVIIDistribution(x, mu=0, sigma=1, p=3).__eval__
        >>> y_4 = PearsonVIIDistribution(x, mu=0, sigma=1, p=4).__eval__
        >>> y_5 = PearsonVIIDistribution(x, mu=0, sigma=1, p=5).__eval__
        >>> y_6 = PearsonVIIDistribution(x, mu=0, sigma=1, p=6).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _ = ax.plot(x, y_1, label=r"$p=1$")
        >>> _ = ax.plot(x, y_2, label=r"$p=2$")
        >>> _ = ax.plot(x, y_3, label=r"$p=3$")
        >>> _ = ax.plot(x, y_4, label=r"$p=4$")
        >>> _ = ax.plot(x, y_5, label=r"$p=5$")
        >>> _ = ax.plot(x, y_6, label=r"$p=6$")
        >>> _ = ax.set_xlabel("x")
        >>> _ = ax.set_ylabel("f(x)")
        >>> _ = ax.legend()
        >>> plt.savefig("PearsonVIIDistribution.png", dpi=300, transparent=True)

    Notes:
        The Pearson VII distribution is defined as follows for probability density
        [^1]:

        $$
        f(x;\mu ,\sigma ,p)=\frac {\Gamma \left({\frac {p}{2}}\right)}
        {\sigma {\sqrt {\pi }}\,\Gamma \left({\frac {p-1}{2}}\right)}\left[1+
        \left({\frac {x-\mu }{\sigma }}\right)^{2}\right]^{-{\frac {p}{2}}}
        $$

        [^1] Gupta, S. K. (1998). Peak Decomposition using Pearson Type VII Function.
            Journal of Applied Crystalography, 31(3), 474-476.
            https://doi.org/10.1107/S0021889897011047

        See also: https://en.wikipedia.org/wiki/Pearson_distribution and
        https://www.originlab.com/doc/Origin-Help/PearsonVII-FitFunc

    Args:
        *x (UniversalArray): The points at which to evaluate the distribution.
        mu (float): The mean of the distribution.
        sigma (float): The standard deviation of the distribution.
        p (float): The shape parameter of the distribution.
    """

    def __init__(
        self,
        *x: UniversalArray,
        mu: float = 0,
        sigma: float = 1,
        p: float = 1,
    ) -> None:
        """Initialize the Pearson VII distribution."""
        super().__init__(*x, mu=mu, sigma=sigma)
        self.p = p

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the Pearson VII distribution."""
        return (
            gammaln(self.p / 2)
            / (self.sigma * np.sqrt(np.pi) * gammaln((self.p - 1) / 2))
            * (1 + ((self._x - self.mu) / self.sigma) ** 2) ** (-self.p / 2)
        )

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Pearson VII distribution."""
        return SummaryStatisticsAPI(
            mean=None,
            variance=None,
            mode=None,
            doc=self.__doc__,
        )


class PseudoVoigtDistribution(ContinousPseudo):
    r"""Pseudo-Voigt distribution.

    The Pseudo-Voigt distribution is a continuous probability distribution on the real
    line. It is a convolution of a Gaussian distribution and a Lorentzian distribution.
    This function becomes popular for X-ray diffraction data analysis.[^1]

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.mixed_discrete_continuous import (
        ... PseudoVoigtDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_1 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.1).__eval__
        >>> y_2 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.2).__eval__
        >>> y_3 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.3).__eval__
        >>> y_4 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.4).__eval__
        >>> y_5 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.5).__eval__
        >>> y_6 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.6).__eval__
        >>> y_7 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.7).__eval__
        >>> y_8 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.8).__eval__
        >>> y_9 = PseudoVoigtDistribution(x, mu=0, sigma=1, eta=0.9).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _ = ax.plot(x, y_1, label=r"$f=0.1$")
        >>> _ = ax.plot(x, y_2, label=r"$f=0.2$")
        >>> _ = ax.plot(x, y_3, label=r"$f=0.3$")
        >>> _ = ax.plot(x, y_4, label=r"$f=0.4$")
        >>> _ = ax.plot(x, y_5, label=r"$f=0.5$")
        >>> _ = ax.plot(x, y_6, label=r"$f=0.6$")
        >>> _ = ax.plot(x, y_7, label=r"$f=0.7$")
        >>> _ = ax.plot(x, y_8, label=r"$f=0.8$")
        >>> _ = ax.plot(x, y_9, label=r"$f=0.9$")
        >>> _ = ax.set_xlabel("x")
        >>> _ = ax.set_ylabel("f(x)")
        >>> _ = ax.legend()
        >>> plt.savefig("PseudoVoigtDistribution.png", dpi=300, transparent=True)

    Notes:
        The Pseudo-Voigt distribution is defined as follows for probability density

        $$
        f(x;\mu ,\sigma ,\eta)=(1 - \eta) \cdot \frac {1}{\sigma
        {\sqrt {2\pi }}}\exp \left(-{\frac {(x-\mu )^{2}}{2\sigma^{2}}}
        \right)+\eta \cdot {\frac {1}{\pi }}\left(
        {\frac {\sigma }{(\frac{\sigma}{2})^{2}+(x-\mu )^{2}}}\right)
        $$

        with the mixing parameter $\eta$ in the range $0 \leq \eta \leq 1$.
        See also: https://en.wikipedia.org/wiki/Voigt_profile

    Args:
        *x: The points at which to evaluate the distribution.
        mu: The mean of the distribution.
        sigma: The standard deviation of the distribution.
        eta: The mixing parameter of the distribution.
    """

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the Pseudo-Voigt distribution."""
        return (1 - self.eta) * GaussianDistribution(
            self._x,
            mu=self.mu,
            sigma=self.sigma,
        ).probability_density_function() + self.eta * LorentzianDistribution(
            self._x,
            mu=self.mu,
            gamma=self.sigma,
        ).probability_density_function()

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Pseudo-Voigt distribution."""
        return SummaryStatisticsAPI(
            mean=None,
            variance=None,
            mode=None,
            doc=self.__doc__,
        )


class AsymmetricRamanLineshape(ContinousAsymmetricPseudo):
    r"""Asymmetric Raman lineshape distribution.

    The Asymmetric Raman lineshape distribution is a continuous probability, which is
    a modified version of the Pseudo-Voigt distribution. It is a convolution of a
    Gaussian distribution and a Lorentzian distribution plus a damped sigmoidal
    term.[^1] This function becomes popular for Raman spectroscopy data analysis.

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.mixed_discrete_continuous import (
        ... AsymmetricRamanLineshape
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_1 = AsymmetricRamanLineshape(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.2,
        ... gamma=0.1
        ... ).__eval__
        >>> y_2 = AsymmetricRamanLineshape(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.2,
        ... gamma=0.2
        ... ).__eval__
        >>> y_3 = AsymmetricRamanLineshape(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.3,
        ... gamma=0.3,
        ... ).__eval__
        >>> y_4 = AsymmetricRamanLineshape(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.3,
        ... gamma=0.4,
        ... ).__eval__
        >>> y_5 = AsymmetricRamanLineshape(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.5,
        ... gamma=0.5,
        ... ).__eval__
        >>> y_6 = AsymmetricRamanLineshape(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.5,
        ... gamma=0.6,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _ = ax.plot(x, y_1, label=r"$\eta=0.2$, $\gamma=0.1$")
        >>> _ = ax.plot(x, y_2, label=r"$\eta=0.2$, $\gamma=0.2$")
        >>> _ = ax.plot(x, y_3, label=r"$\eta=0.3$, $\gamma=0.3$")
        >>> _ = ax.plot(x, y_4, label=r"$\eta=0.3$, $\gamma=0.4$")
        >>> _ = ax.plot(x, y_5, label=r"$\eta=0.5$, $\gamma=0.5$")
        >>> _ = ax.plot(x, y_6, label=r"$\eta=0.5$, $\gamma=0.6$")
        >>> _ = ax.set_xlabel("x")
        >>> _ = ax.set_ylabel("f(x)")
        >>> _ = ax.legend()
        >>> plt.savefig("AsymmetricRamanLineshape.png", dpi=300, transparent=True)

    Notes:
        The Asymmetric Raman lineshape distribution is defined as follows for
        probability density [^1]:

        $$
        f(x;\mu ,\sigma ,\eta, \gamma)=(1 - \eta) \cdot \textrm{Gauss}(x \cdot
        p(x; \gamma)) + \eta \cdot \textrm{Lorentzian}(x \cdot p(x; \gamma))
        $$

        with the mixing parameter $\eta$ in the range $0 \leq \eta \leq 1$ and the
        damped sigmoidal term $p(x)$:

        $$
        p(x; \gamma) = 1 - \gamma \cdot \frac{x-\mu}{\sigma} \cdot
        \exp \left(-\frac{(x-\mu)^2}{2 \sigma^2}\right)
        $$

        [^1] Korepanov, V, I.and Sedlovets, D. M. (2018),
            An asymmetric fitting function for condensed-phase Raman spectroscopy,
            Analyst RSC, 2674-2679 (143)
            http://dx.doi.org/10.1039/C8AN00710A

    Args:
        *x: The points at which to evaluate the distribution.
        mu: The mean of the distribution.
        sigma: The standard deviation of the distribution.
        eta: The mixing parameter of the distribution.
        gamma: The damping parameter of the distribution.
    """

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the Asym. Raman lineshape distribution."""
        p = 1 - self.gamma * (self._x - self.mu) / self.sigma * np.exp(
            -((self._x - self.mu) ** 2) / (2 * self.sigma**2),
        )
        return (1 - self.eta) * GaussianDistribution(
            self._x * p,
            mu=self.mu,
            sigma=self.sigma,
        ).probability_density_function() + self.eta * LorentzianDistribution(
            self._x * p,
            mu=self.mu,
            gamma=self.sigma,
        ).probability_density_function()

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Asym. Raman lineshape distribution."""
        return SummaryStatisticsAPI(
            mean=None,
            variance=None,
            mode=None,
            doc=self.__doc__,
        )


class ModifiedDoniachSunjicDistribution(ContinousAsymmetricPseudo):
    r"""Doniach-Sunjic distribution.

    The Doniach-Sunjic distribution is a continuous probability, which is
    a modified version of the Pseudo-Voigt distribution. It is a convolution of a
    Gaussian distribution and a Lorentzian distribution plus a damped sigmoidal
    term.[^1] This function becomes popular for XPS/AES data analysis. See also:
    http://www.casaxps.com

    Examples:
        >>> # PDF Example
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from umf.functions.distributions.mixed_discrete_continuous import (
        ... ModifiedDoniachSunjicDistribution
        ... )
        >>> x = np.linspace(-5, 5, 1000)
        >>> y_1 = ModifiedDoniachSunjicDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.2,
        ... gamma=0.1
        ... ).__eval__
        >>> y_2 = ModifiedDoniachSunjicDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.2,
        ... gamma=0.2
        ... ).__eval__
        >>> y_3 = ModifiedDoniachSunjicDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.3,
        ... gamma=0.3,
        ... ).__eval__
        >>> y_4 = ModifiedDoniachSunjicDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.3,
        ... gamma=0.4,
        ... ).__eval__
        >>> y_5 = ModifiedDoniachSunjicDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.5,
        ... gamma=0.5,
        ... ).__eval__
        >>> y_6 = ModifiedDoniachSunjicDistribution(
        ... x,
        ... mu=0,
        ... sigma=1,
        ... eta=0.5,
        ... gamma=0.6,
        ... ).__eval__
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> _ = ax.plot(x, y_1, label=r"$\eta=0.2$, $\gamma=0.1$")
        >>> _ = ax.plot(x, y_2, label=r"$\eta=0.2$, $\gamma=0.2$")
        >>> _ = ax.plot(x, y_3, label=r"$\eta=0.3$, $\gamma=0.3$")
        >>> _ = ax.plot(x, y_4, label=r"$\eta=0.3$, $\gamma=0.4$")
        >>> _ = ax.plot(x, y_5, label=r"$\eta=0.5$, $\gamma=0.5$")
        >>> _ = ax.plot(x, y_6, label=r"$\eta=0.5$, $\gamma=0.6$")
        >>> _ = ax.set_xlabel("x")
        >>> _ = ax.set_ylabel("f(x)")
        >>> _ = ax.legend()
        >>> plt.savefig(
        ... "ModifiedDoniachSunjicDistribution.png",
        ... dpi=300,
        ... transparent=True,
        ... )

    Warning:
        The Doniach-Sunjic distribution as defined for **casaxps** is not the same as
        used in the current implementation. In the current implementation, the damping
        factor is included into the lorentzian model.

    Notes:
        The Doniach-Sunjic distribution is defined as follows for probability density:

        $$
        f(x;\mu ,\sigma ,\eta, \gamma)= GL(x) + (1 - GL(x) \cdot p(x; \gamma))
        $$

        with GL(x) as the Gaussian-Lorentzian mixture function and the damped sigmoidal
        is defined as:

        $$
        p(x; \gamma) = \begin{cases} \exp \left(- \gamma \cdot
        \frac{(x-\mu)}{2 \sigma^2}\right) & \text{if } x < \mu \\
        0 & \text{if } x \geq \mu \end{cases}
        $$

        and the mixing parameter $\eta$ in the range $0 \leq \eta \leq 1$.
        See also: http://www.casaxps.com/help_manual/line_shapes.htm

    Args:
        *x: The points at which to evaluate the distribution.
        mu: The mean of the distribution.
        sigma: The standard deviation of the distribution.
        eta: The mixing parameter of the distribution.
        gamma: The damping parameter of the distribution.
    """

    def probability_density_function(self) -> UniversalArray:
        """Probability density function of the Doniach-Sunjic distribution."""
        p = np.where(
            self._x < self.mu,
            np.exp(-self.gamma * (self._x - self.mu) / (2 * self.sigma**2)),
            0,
        )
        return (1 - self.eta) * GaussianDistribution(
            self._x,
            mu=self.mu,
            sigma=self.sigma,
        ).probability_density_function() + self.eta * LorentzianDistribution(
            self._x * p,
            mu=self.mu,
            gamma=self.sigma,
        ).probability_density_function()

    @property
    def __summary__(self) -> SummaryStatisticsAPI:
        """Summary statistics of the Doniach-Sunjic distribution."""
        return SummaryStatisticsAPI(
            mean=None,
            variance=None,
            mode=None,
            doc=self.__doc__,
        )
