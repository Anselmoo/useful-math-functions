"""Test cases for continuous distributions."""

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
    """Test the probability density function of the Von Mises distribution.

    This test checks if the probability density function of the Von Mises distribution
    returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(-np.pi, np.pi, 1000)
    mu: float = 0
    kappa: float = 1
    distribution: VonMisesDistribution = VonMisesDistribution(x, mu=mu, kappa=kappa)
    pdf: np.ndarray = distribution.probability_density_function()
    expected_pdf: np.ndarray = np.exp(kappa * np.cos(x - mu)) / (
        2 * np.pi * special.i0(kappa)
    )
    assert np.allclose(pdf, expected_pdf, rtol=1e-5, atol=1e-8)


def test_wrapped_asym_laplace_distribution_pdf() -> None:
    """Test the PDF of the Wrapped Asymmetric Laplace distribution.

    This test checks if the probability density function of the Wrapped Asymmetric
    Laplace distribution returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(-np.pi, np.pi, 1000)
    mu: float = 0
    lambda_: float = 1
    kappa: float = 1
    distribution: WrappedAsymLaplaceDistribution = WrappedAsymLaplaceDistribution(
        x, mu=mu, lambda_=lambda_, kappa=kappa
    )
    pdf: np.ndarray = distribution.probability_density_function()
    part_1: np.ndarray = (
        kappa
        * lambda_
        / (kappa**2 + 1)
        * (
            np.exp(-(x - mu) * lambda_ * kappa)
            / (1 - np.exp(-2 * np.pi * lambda_ * kappa))
            - np.exp((x - mu) * lambda_ / kappa)
            / (1 - np.exp(2 * np.pi * lambda_ / kappa))
        )
    )
    part_2: np.ndarray = (
        kappa
        * lambda_
        / (kappa**2 + 1)
        * (
            np.exp(-(x - mu) * lambda_ * kappa)
            / (np.exp(2 * np.pi * lambda_ * kappa) - 1)
            - np.exp((x - mu) * lambda_ / kappa)
            / (np.exp(-2 * np.pi * lambda_ / kappa) - 1)
        )
    )
    expected_pdf: np.ndarray = np.where(x >= mu, part_1, part_2)
    assert np.allclose(pdf, expected_pdf, rtol=1e-5, atol=1e-8)


def test_kumaraswamy_distribution_pdf() -> None:
    """Test the probability density function of the Kumaraswamy distribution.

    This test checks if the probability density function of the Kumaraswamy distribution
    returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(0, 1, 1000)
    a: float = 2
    b: float = 2
    distribution: KumaraswamyDistribution = KumaraswamyDistribution(x, a=a, b=b)
    pdf: np.ndarray = distribution.probability_density_function()
    expected_pdf: np.ndarray = a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)
    assert np.allclose(pdf, expected_pdf, rtol=1e-5, atol=1e-8)


def test_kumaraswamy_distribution_cdf() -> None:
    """Test the cumulative distribution function of the Kumaraswamy distribution.

    This test checks if the cumulative distribution function of the Kumaraswamy
    distribution returns the correct result.

    Returns:
        None
    """
    x: np.ndarray = np.linspace(0, 1, 1000)
    a: float = 2
    b: float = 2
    distribution: KumaraswamyDistribution = KumaraswamyDistribution(
        x, a=a, b=b, cumulative=True
    )
    cdf: np.ndarray = distribution.cumulative_distribution_function()
    expected_cdf: np.ndarray = 1 - (1 - x**a) ** b
    assert np.allclose(cdf, expected_cdf, rtol=1e-5, atol=1e-8)
