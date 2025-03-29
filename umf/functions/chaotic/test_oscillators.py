"""Test cases for chaotic oscillators."""

from __future__ import annotations

import numpy as np

from umf.functions.chaotic.oscillators import DoublePendulum
from umf.functions.chaotic.oscillators import DoubleSpringMassSystem
from umf.functions.chaotic.oscillators import LorenzAttractor
from umf.functions.chaotic.oscillators import MagneticPendulum
from umf.functions.chaotic.oscillators import RoesslerAttractor


def test_double_pendulum_initial_state() -> None:
    """Test the initial state of the Double Pendulum.

    This test checks if the initial state of the Double Pendulum is set correctly.

    Returns:
        None
    """
    pendulum = DoublePendulum(np.linspace(0, 10, 1000))
    assert pendulum.initial_state == [np.pi / 2, 0.0, np.pi / 2, 0.0]


def test_double_pendulum_equation_of_motion() -> None:
    """Test the equation of motion of the Double Pendulum.

    This test checks if the equation of motion of the Double Pendulum returns the
    correct result.

    Returns:
        None
    """
    pendulum = DoublePendulum(np.linspace(0, 10, 1000))
    initial_state = pendulum.initial_state
    t = pendulum.t
    result = pendulum.equation_of_motion(initial_state, t)
    assert len(result) == 4


def test_magnetic_pendulum_initial_state() -> None:
    """Test the initial state of the Magnetic Pendulum.

    This test checks if the initial state of the Magnetic Pendulum is set correctly.

    Returns:
        None
    """
    pendulum = MagneticPendulum(np.linspace(0, 2.5, 500))
    assert pendulum.initial_state == [np.pi / 4, np.pi / 2, 0.5, 0.5]


def test_magnetic_pendulum_equation_of_motion() -> None:
    """Test the equation of motion of the Magnetic Pendulum.

    This test checks if the equation of motion of the Magnetic Pendulum returns the
    correct result.

    Returns:
        None
    """
    pendulum = MagneticPendulum(np.linspace(0, 2.5, 500))
    initial_state = pendulum.initial_state
    t = pendulum.t
    result = pendulum.equation_of_motion(initial_state, t)
    assert len(result) == 4


def test_double_spring_mass_system_initial_state() -> None:
    """Test the initial state of the Double Spring Mass System.

    This test checks if the initial state of the Double Spring Mass System is set
    correctly.

    Returns:
        None
    """
    system = DoubleSpringMassSystem(np.linspace(0, 100, 500))
    assert system.initial_state == [0.0, 0.0, -1.0, 0.0]


def test_double_spring_mass_system_equation_of_motion() -> None:
    """Test the equation of motion of the Double Spring Mass System.

    This test checks if the equation of motion of the Double Spring Mass System returns
    the correct result.

    Returns:
        None
    """
    system = DoubleSpringMassSystem(np.linspace(0, 100, 500))
    initial_state = system.initial_state
    t = system.t
    result = system.equation_of_motion(initial_state, t)
    assert len(result) == 4


def test_lorenz_attractor_initial_state() -> None:
    """Test the initial state of the Lorenz Attractor.

    This test checks if the initial state of the Lorenz Attractor is set correctly.

    Returns:
        None
    """
    attractor = LorenzAttractor(np.linspace(0, 20, 1000))
    assert attractor.initial_state == [1.0, 1.0, 1.0]


def test_lorenz_attractor_equation_of_motion() -> None:
    """Test the equation of motion of the Lorenz Attractor.

    This test checks if the equation of motion of the Lorenz Attractor returns the
    correct result.

    Returns:
        None
    """
    attractor = LorenzAttractor(np.linspace(0, 20, 1000))
    initial_state = attractor.initial_state
    t = attractor.t
    result = attractor.equation_of_motion(initial_state, t)
    assert len(result) == 3


def test_roessler_attractor_initial_state() -> None:
    """Test the initial state of the Roessler Attractor.

    This test checks if the initial state of the Roessler Attractor is set correctly.

    Returns:
        None
    """
    attractor = RoesslerAttractor(np.linspace(0, 100, 1000))
    assert attractor.initial_state == [0.1, 0.0, 0.0]


def test_roessler_attractor_equation_of_motion() -> None:
    """Test the equation of motion of the Roessler Attractor.

    This test checks if the equation of motion of the Roessler Attractor returns the
    correct result.

    Returns:
        None
    """
    attractor = RoesslerAttractor(np.linspace(0, 100, 1000))
    initial_state = attractor.initial_state
    t = attractor.t
    result = attractor.equation_of_motion(initial_state, t)
    assert len(result) == 3
