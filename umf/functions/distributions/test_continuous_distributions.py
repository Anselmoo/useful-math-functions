from __future__ import annotations

import numpy as np

from scipy import special

from umf.functions.distributions.continuous_2pi_interval import VonMisesDistribution
from umf.functions.distributions.continuous_2pi_interval import (
    WrappedAsymLaplaceDistribution,
)
from umf.functions.distributions.continuous_bounded_interval import (
    KumaraswamyDistribution,
)


def test_von_mises_distribution_pdf() -> None:
    x = np.linspace(-np.pi, np.pi, 1000)
    mu = 0
    kappa = 1
    distribution = VonMisesDistribution(x, mu=mu, kappa=kappa)
    pdf = distribution.probability_density_function()
    expected_pdf = np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * special.i0(kappa))
    assert np.allclose(pdf, expected_pdf, rtol=1e-5, atol=1e-8)


def test_wrapped_asym_laplace_distribution_pdf() -> None:
    x = np.linspace(-np.pi, np.pi, 1000)
    mu = 0
    lambda_ = 1
    kappa = 1
    distribution = WrappedAsymLaplaceDistribution(x, mu=mu, lambda_=lambda_, kappa=kappa)
    pdf = distribution.probability_density_function()
    part_1 = (
        kappa
        * lambda_
        / (kappa**2 + 1)
        * (
            np.exp(-(x - mu) * lambda_ * kappa) / (1 - np.exp(-2 * np.pi * lambda_ * kappa))
            - np.exp((x - mu) * lambda_ / kappa) / (1 - np.exp(2 * np.pi * lambda_ / kappa))
        )
    )
    part_2 = (
        kappa
        * lambda_
        / (kappa**2 + 1)
        * (
            np.exp(-(x - mu) * lambda_ * kappa) / (np.exp(2 * np.pi * lambda_ * kappa) - 1)
            - np.exp((x - mu) * lambda_ / kappa) / (np.exp(-2 * np.pi * lambda_ / kappa) - 1)
        )
    )
    expected_pdf = np.where(x >= mu, part_1, part_2)
    assert np.allclose(pdf, expected_pdf, rtol=1e-5, atol=1e-8)


def test_kumaraswamy_distribution_pdf() -> None:
    x = np.linspace(0, 1, 1000)
    a = 2
    b = 2
    distribution = KumaraswamyDistribution(x, a=a, b=b)
    pdf = distribution.probability_density_function()
    expected_pdf = a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)
    assert np.allclose(pdf, expected_pdf, rtol=1e-5, atol=1e-8)


def test_kumaraswamy_distribution_cdf() -> None:
    x = np.linspace(0, 1, 1000)
    a = 2
    b = 2
    distribution = KumaraswamyDistribution(x, a=a, b=b, cumulative=True)
    cdf = distribution.cumulative_distribution_function()
    expected_cdf = 1 - (1 - x**a) ** b
    assert np.allclose(cdf, expected_cdf, rtol=1e-5, atol=1e-8)
