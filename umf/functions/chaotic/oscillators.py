"""Chaotic Oscillators functions."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from numpy import pi
from scipy.constants import g
from scipy.constants import mu_0

from umf.meta.functions import OscillatorsFunc2D
from umf.meta.functions import OscillatorsFunc3D
from umf.meta.functions import OscillatorsFuncBase


if TYPE_CHECKING:
    from umf.types.static_types import UniversalArray
    from umf.types.static_types import UniversalArrayTuple

__all__: list[str] = [
    "DoublePendulum",
    "MagneticPendulum",
    "DoubleSpringMassSystem",
    "LorenzAttractor",
    "RoesslerAttractor",
    "DuffingOscillator",
    "ChuaSCircuit",
]


class DoublePendulum(OscillatorsFuncBase):
    r"""Double Pendulum differential equation.

    The double pendulum is a simple physical system that exhibits chaotic behavior.
    The double pendulum consists of two pendulums attached to each other, where the
    motion of the second pendulum is influenced by the motion of the first pendulum.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation, PillowWriter
        >>> from umf.functions.chaotic.oscillators import DoublePendulum
        >>> pendulum = DoublePendulum(np.linspace(0, 10, 1000))
        >>> x1, y1, x2, y2 = pendulum.__eval__
        >>> t = pendulum.t
        >>> fig, ax = plt.subplots()
        >>> _ = ax.set_xlabel("X")
        >>> _ = ax.set_ylabel("Y")
        >>> _ = ax.set_xlim(min(x2) - 0.5, max(x2) + 0.5)
        >>> _ = ax.set_ylim(min(y2) - 0.5, max(y2) + 0.5)
        >>> (line,) = ax.plot([], [], "o-", lw=2)
        >>> def init():
        ...     line.set_data([], [])
        ...     return (line,)
        >>> def update(frame):
        ...     line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        ...     _ = ax.set_title(f"t = {t[frame]:.2f} seconds")
        ...     return (line,)
        >>> ani = FuncAnimation(
        ...     fig=fig, func=update,
        ...     init_func=init,
        ...     frames=len(t),
        ...     interval=10,
        ...     blit=True
        ... )
        >>> ani.save('DoublePendulum.gif', writer=PillowWriter(fps=10))

    Notes:
        The double pendulum differential equation is defined as:

        $$
        \begin{align*}
        \dot{z_1} &= \frac{m_2 g \sin(\theta_2) \cos(\theta_1 - \theta_2)
        - m_2 \sin(\theta_1 - \theta_2) (l_1 z_1^2 \cos(\theta_1 - \theta_2)
        + l_2 z_2^2) - (m_1 + m_2) g \sin(\theta_1)}{l_1 (m_1
        + m_2 \sin^2(\theta_1 - \theta_2))} \\
        \dot{z_2} &= \frac{(m_1 + m_2) (l_1 z_1^2 \sin(\theta_1 - \theta_2)
        - g \sin(\theta_2) + g \sin(\theta_1) \cos(\theta_1 - \theta_2))
        + m_2 l_2 z_2^2 \sin(\theta_1 - \theta_2) \cos(\theta_1
        - \theta_2)}{l_2 (m_1 + m_2 \sin^2(\theta_1 - \theta_2))}
        \end{align*}
        $$

        with $\dot{\theta_1} = z_1 \\$ and $\dot{\theta_2} = z_2$.

    Args:
        *time_points (UniversalArray): The array of time points at which the
            oscillator's state is evaluated.
        time_format (str, optional): The time format. Defaults to "seconds".
        theta1 (float, optional): The initial angle of the first pendulum. Defaults to
            pi / 2.
        theta2 (float, optional): The initial angle of the second pendulum. Defaults to
            pi / 2.
        z1 (float, optional): The initial angular velocity of the first pendulum.
            Defaults to 0.0.
        z2 (float, optional): The initial angular velocity of the second pendulum.
            Defaults to 0.0.
        m1 (float, optional): The mass of the first pendulum. Defaults to 1.0.
        m2 (float, optional): The mass of the second pendulum. Defaults to 1.0.
        l1 (float, optional): The length of the first pendulum. Defaults to 1.0.
        l2 (float, optional): The length of the second pendulum. Defaults to 1.0.
        g (float, optional): The acceleration due to gravity. Defaults to 9.81.
        velocity (bool, optional): Whether to return the velocity of the double
            pendulum. Defaults to False.
    """

    def __init__(  # noqa: PLR0913
        self,
        *t: UniversalArray,
        time_format: str = "seconds",
        theta1: float = pi / 2,
        theta2: float = pi / 2,
        z1: float = 0.0,
        z2: float = 0.0,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        g: float = g,
        velocity: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(
            *t,
            time_format=time_format,
            velocity=velocity,
        )
        self.theta1 = theta1
        self.theta2 = theta2
        self.z1 = z1
        self.z2 = z2
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    @property
    def __initial_configuration__(self) -> dict[str, float]:
        """Return the initial configuration of the double pendulum.

        Returns:
            dict[str, float]: The initial configuration of the double pendulum.
        """
        return OrderedDict(
            sorted(
                {
                    "theta1": self.theta1,
                    "theta2": self.theta2,
                    "z1": self.z1,
                    "z2": self.z2,
                    "m1": self.m1,
                    "m2": self.m2,
                    "l1": self.l1,
                    "l2": self.l2,
                    "g": self.g,
                }.items(),
            ),
        )

    @property
    def initial_state(self) -> list[float]:
        """Return the initial state of the double pendulum.

        Returns:
            list[float]: The initial state of the double pendulum.
        """
        return [self.theta1, self.z1, self.theta2, self.z2]

    def equation_of_motion(
        self,
        initial_state: list[float],
        t: UniversalArray,  # noqa: ARG002
    ) -> tuple[float, float, float, float]:
        """Return the equation of motion of the double pendulum.

        Args:
            initial_state (list[float]): The initial state of the double pendulum.
            t (UniversalArray): The time array.

        Returns:
            tuple[float, float, float, float]: The equation of motion of the double
                pendulum.
        """
        theta1, z1, theta2, z2 = initial_state
        c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

        theta1_dot = z1
        z1_dot = (
            (
                self.m2 * self.g * np.sin(theta2) * c
                - self.m2 * s * (self.l1 * z1**2 * c + self.l2 * z2**2)
                - (self.m1 + self.m2) * self.g * np.sin(theta1)
            )
            / self.l1
            / (self.m1 + self.m2 * s**2)
        )
        theta2_dot = z2
        z2_dot = (
            (
                (self.m1 + self.m2)
                * (
                    self.l1 * z1**2 * s
                    - self.g * np.sin(theta2)
                    + self.g * np.sin(theta1) * c
                )
                + self.m2 * self.l2 * z2**2 * s * c
            )
            / self.l2
            / (self.m1 + self.m2 * s**2)
        )
        return theta1_dot, z1_dot, theta2_dot, z2_dot

    @property
    def to_position(self) -> UniversalArrayTuple:
        """Return the position of the double pendulum.

        Returns:
            UniversalArrayTuple: The position of the double pendulum.
        """
        y = self.solve()
        x1 = self.l1 * np.sin(y[:, 0])
        y1 = -self.l1 * np.cos(y[:, 0])
        x2 = x1 + self.l2 * np.sin(y[:, 2])
        y2 = y1 - self.l2 * np.cos(y[:, 2])

        return x1, y1, x2, y2

    @property
    def to_velocity(self) -> UniversalArrayTuple:
        """Return the velocity of the double pendulum.

        Returns:
            UniversalArrayTuple: The velocity of the double pendulum.
        """
        y = self.solve()
        vx1 = self.l1 * np.sin(y[:, 1])
        vy1 = -self.l1 * np.cos(y[:, 1])
        vx2 = vx1 + self.l2 * np.sin(y[:, 3])
        vy2 = vy1 - self.l2 * np.cos(y[:, 3])

        return vx1, vy1, vx2, vy2


class MagneticPendulum(OscillatorsFuncBase):
    r"""Magnetic Pendulum differential equation.

    The magnetic pendulum is an intriguing physical system that exhibits chaotic
    behavior when subjected to magnetic fields. It consists of a pendulum bob influenced
    by the magnetic fields of several surrounding magnets. The motion of the pendulum
    bob is affected by these magnetic fields, and the chaotic behavior is further
    enhanced by the presence of magnets with both north and south poles.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation, PillowWriter
        >>> from umf.functions.chaotic.oscillators import MagneticPendulum
        >>> pendulum = MagneticPendulum(np.linspace(0, 2.5, 500))
        >>> x, y, z = pendulum.to_position
        >>> t = pendulum.t
        >>> fig, ax = plt.subplots()
        >>> magnet_x, magnet_y, _ = zip(*pendulum.magnets, strict=True)
        >>> _ = ax.scatter(magnet_x, magnet_y, marker="x")
        >>> scat = ax.scatter([], [], cmap="viridis", s=10)
        >>> (line,) = ax.plot([], [], linestyle="dashed", color="grey", alpha=0.6)
        >>> _ = ax.set_xlabel("X")
        >>> _ = ax.set_ylabel("Y")
        >>> _ = ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
        >>> _ = ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        >>> def init():
        ...     scat.set_offsets([])
        ...     line.set_data([], [])
        ...     return scat, line
        >>> def update(frame):
        ...     start = max(0, frame - 50)
        ...     scat.set_offsets(np.c_[x[start:frame], y[start:frame]])
        ...     line.set_data(x[:frame], y[:frame])
        ...     _ = ax.set_title(f"t = {t[frame]:.2f} seconds")
        ...     return scat, line
        >>> ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
        >>> ani.save('MagneticPendulum.gif', writer=PillowWriter(fps=10))

    Notes:
        The magnetic pendulum differential equation is defined as:

        $$
        \begin{align*}
        \frac{d\omega_\theta}{dt} &= -\frac{g}{l} \sin(\theta) + \frac{1}{m \cdot l}
        \left( f_{mx} \cos(\theta) \cos(\phi) + f_{my} \cos(\theta) \sin(\phi)
        + f_{mz} \sin(\theta) \right), \\
        \frac{d\omega_\phi}{dt} &= \frac{1}{m \cdot l \sin(\theta)} \left( f_{mx}
        \sin(\phi) - f_{my} \cos(\phi) \right).
        \end{align*}
        $$

        with $\frac{d\theta}{dt} = \omega_\theta$ and
        $\frac{d\phi}{dt} = \omega_\phi$, while components of the magnetic force
        $((f_{mx}, f_{my}, f_{mz}))$ are calculated as follows:

        $$
        \begin{align*}
        f_{mx} &= \sum_{i} \frac{\mu_0 \cdot k_i \cdot (x - x_i)}{d_i^3}, \\
        f_{my} &= \sum_{i} \frac{\mu_0 \cdot k_i \cdot (y - y_i)}{d_i^3}, \\
        f_{mz} &= \sum_{i} \frac{\mu_0 \cdot k_i \cdot z}{d_i^3},
        \end{align*}
        $$

        with $d_i = \sqrt{(x - x_i)^2 + (y - y_i)^2 + z^2}$.

    Args:
        *time_points (UniversalArray): The array of time points at which the
            oscillator's state is evaluated.
        time_format (str, optional): The time format. Defaults to "seconds".
        l (float, optional): The length of the pendulum. Defaults to 2.
        m (float, optional): The mass of the pendulum. Defaults to 0.5.
        x0 (float, optional): The initi`al x-coordinate of the pendulum bob. Defaults
            to 0.5.
        y0 (float, optional): The initial y-coordinate of the pendulum bob. Defaults
            to 0.5.
        theta (float, optional): The initial angle theta of the pendulum. Defaults to
            pi/4.
        phi (float, optional): The initial angle phi of the pendulum. Defaults to pi/2.
        magnetic_constant (float, optional): The magnetic constant. Defaults to 1.0e10.
        magnets (list[tuple[float, float, int]], optional): A list of magnets, each
            defined by a tuple of (x, y, pole). Defaults to None.
        velocity (bool, optional): Whether to return the velocity of the magnetic
            pendulum. Defaults to False.
    """

    def __init__(  # noqa: PLR0913
        self,
        *t: UniversalArray,
        time_format: str = "seconds",
        l: float = 2,  # noqa: E741
        m: float = 0.5,
        x0: float = 0.5,
        y0: float = 0.5,
        theta: float = pi / 4,
        phi: float = pi / 2,
        magnetic_constant: float = 1.0e10,
        magnets: list[tuple[float, float, int]] | None = None,
        velocity: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*t, time_format=time_format, velocity=velocity)
        self.l = l
        self.m = m
        self.x0 = x0
        self.y0 = y0
        self.theta = theta
        self.phi = phi
        self.magnetic_constant = magnetic_constant
        self.magnets: list[tuple[float, float, int]] = (
            magnets
            if magnets is not None
            else [
                (1.0, 1.0, +1),
                (-1.0, 1.0, +1),
                (-1.0, -1.0, +1),
                (1.0, -1.0, +1),
                (0.5, 0.5, -1),
                (-0.5, 0.5, -1),
                (-0.5, -0.5, -1),
                (0.5, -0.5, -1),
            ]
        )

    @property
    def __initial_configuration__(
        self,
    ) -> dict[str, float | list[tuple[float, float, int]]]:
        """Return the initial configuration of the pendulum."""
        return {
            "l": self.l,
            "m": self.m,
            "x0": self.x0,
            "y0": self.y0,
            "theta": self.theta,
            "phi": self.phi,
            "magnetic_constant": self.magnetic_constant,
            "magnets": self.magnets,
        }

    @property
    def initial_state(self) -> list[float]:
        """Return the initial state of the pendulum."""
        return [self.theta, self.phi, self.x0, self.y0]

    def equation_of_motion(
        self,
        initial_state: list[float],
        t: UniversalArray,  # noqa: ARG002
    ) -> tuple[float, float, float, float]:
        """Return the equation of motion of the magnetic pendulum.

        Args:
            initial_state (list[float]): The initial state of the magnetic pendulum.
            t (UniversalArray): The time array.

        Returns:
            tuple[float, float, float, float]: The equation of motion of the magnetic
                pendulum.
        """
        theta, phi, omega_theta, omega_phi = initial_state
        x = self.l * np.sin(theta) * np.cos(phi)
        y = self.l * np.sin(theta) * np.sin(phi)
        z = -self.l * np.cos(theta)
        f_mx, f_my, f_mz = self.magnetic_force(x, y, z)

        dtheta_dt = omega_theta
        dphi_dt = omega_phi
        domega_theta_dt = -(g / self.l) * np.sin(theta) + (
            f_mx * np.cos(theta) * np.cos(phi)
            + f_my * np.cos(theta) * np.sin(phi)
            + f_mz * np.sin(theta)
        ) / (self.m * self.l)
        domega_phi_dt = (f_mx * np.sin(phi) - f_my * np.cos(phi)) / (
            self.m * self.l * np.sin(theta)
        )

        return dtheta_dt, dphi_dt, domega_theta_dt, domega_phi_dt

    def magnetic_force(
        self,
        x: UniversalArray,
        y: UniversalArray,
        z: UniversalArray,
    ) -> UniversalArrayTuple:
        """Compute the magnetic force on the pendulum bob.

        Args:
            x (UniversalArray): The x-coordinate of the pendulum bob.
            y (UniversalArray): The y-coordinate of the pendulum bob.
            z (UniversalArray): The z-coordinate of the pendulum bob.

        Returns:
            UniversalArrayTuple: The magnetic force on the pendulum bob.
        """
        f_mx, f_my, f_mz = 0.0, 0.0, 0.0
        for x_m, y_m, pole in self.magnets:
            d = np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2 + z**2)
            if d != 0:
                force_magnitude = self.magnetic_constant * mu_0 * pole / d**2
                f_mx += force_magnitude * (x - x_m) / d
                f_my += force_magnitude * (y - y_m) / d
                f_mz += force_magnitude * z / d
        return f_mx, f_my, f_mz

    @property
    def to_position(self) -> UniversalArrayTuple:
        """Compute the 3D position of the pendulum bob."""
        y = self.solve()
        x1 = self.l * np.sin(y[:, 0]) * np.cos(y[:, 1])
        y1 = self.l * np.sin(y[:, 0]) * np.sin(y[:, 1])
        z1 = -self.l * np.cos(y[:, 0])
        return x1, y1, z1

    @property
    def to_velocity(self) -> UniversalArrayTuple:
        """Compute the 3D velocity of the pendulum bob."""
        y = self.solve()
        vx = self.l * y[:, 2] * np.cos(y[:, 0]) * np.cos(y[:, 1])
        vy = self.l * y[:, 2] * np.sin(y[:, 0]) * np.sin(y[:, 1])
        vz = self.l * y[:, 2] * np.sin(y[:, 0])
        return vx, vy, vz


class DoubleSpringMassSystem(OscillatorsFunc2D):
    r"""Double Spring Mass System differential equation.

    The double spring mass system is a simple physical system that consists of two
    masses connected by springs. The motion of the masses is influenced by the spring
    constants and the values of the masses. The double spring mass system exhibits
    oscillatory behavior, and the motion of the masses is described by a set of coupled
    differential equations.

    !!! warning "About the Double Spring Mass System"

        The current implementation of the double spring mass system is partially
        incorrect, because it allows that $m_2$ can skip over $m_1$.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation, PillowWriter
        >>> from umf.functions.chaotic.oscillators import DoubleSpringMassSystem
        >>> pendulum = DoubleSpringMassSystem(np.linspace(0, 100, 500))
        >>> x1, x2 = pendulum.to_position
        >>> t = pendulum.t
        >>> fig, ax = plt.subplots()
        >>> _ = ax.set_ylim( min(x2) - 0.5, 0)
        >>> _ = ax.set_xticks([])
        >>> _ = ax.set_xticklabels([])
        >>> _ = ax.set_ylabel("Z")
        >>> (mass1,) = ax.plot([], [], "ro", lw=2)
        >>> (mass2,) = ax.plot([], [], "bo", lw=2)
        >>> (spring1,) = ax.plot([], [], "k-", lw=2)
        >>> (spring2,) = ax.plot([], [], "k-", lw=2)
        >>> def init():
        ...     mass1.set_data([], [])
        ...     mass2.set_data([], [])
        ...     spring1.set_data([], [])
        ...     spring2.set_data([], [])
        ...     return mass1, mass2, spring1, spring2
        >>> def update(frame):
        ...     mass1.set_data([0], [x1[frame]])
        ...     mass2.set_data([0], [x2[frame]])
        ...     spring1.set_data([0, 0], [0, x1[frame]])
        ...     spring2.set_data([0, 0], [x1[frame], x2[frame]])
        ...     _ = ax.set_title(f"t = {t[frame]:.2f} seconds")
        ...     return mass1, mass2, spring1, spring2
        >>> ani = FuncAnimation(
        ...     fig=fig, func=update,
        ...     init_func=init,
        ...     frames=len(t),
        ...     interval=10,
        ...     blit=True
        ... )
        >>> ani.save('DoubleSpringMassSystem.gif', writer='imagemagick', fps=10)

    Notes:
        The double spring mass system differential equation is defined as:

        $$
        \begin{align*}
        \ddot{x_1} &= -\frac{k_1}{m_1} x_1 - \frac{k_2}{m_1} (x_1 - x_2) - g, \\
        \ddot{x_2} &= -\frac{k_2}{m_2} (x_2 - x_1) - g.
        \end{align*}
        $$

    Args:
        *time_points (UniversalArray): The array of time points at which the
            oscillator's state is evaluated.
        time_format (str, optional): The time format. Defaults to "seconds".
        m1 (float, optional): The mass of the first spring. Defaults to 1.0.
        m2 (float, optional): The mass of the second spring. Defaults to 1.0.
        k1 (float, optional): The spring constant of the first spring. Defaults to 1.0.
        k2 (float, optional): The spring constant of the second spring. Defaults to 1.0.
        z1 (float, optional): The initial velocity of the first spring. Defaults to 0.0.
        z2 (float, optional): The initial velocity of the second spring.
            Defaults to 1.0.
        velocity (bool, optional): Whether to return the velocity of the double
            spring mass system. Defaults to False.
    """

    def __init__(  # noqa: PLR0913
        self,
        *t: UniversalArray,
        time_format: str = "seconds",
        m1: float = 1.0,
        m2: float = 1.0,
        k1: float = 1.0,
        k2: float = 1.0,
        z1: float = 0.0,
        z2: float = -1.0,
        velocity: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*t, time_format=time_format, velocity=velocity)
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.z1 = z1
        self.z2 = z2
        self.g = g

    @property
    def __initial_configuration__(self) -> dict[str, float]:
        """Return the initial configuration of the double spring mass system."""
        return {
            "m1": self.m1,
            "m2": self.m2,
            "k1": self.k1,
            "k2": self.k2,
            "z1": self.z1,
            "z2": self.z2,
            "g": self.g,
        }

    @property
    def initial_state(self) -> list[float]:
        """Return the initial state of the double spring mass system."""
        return [self.z1, 0.0, self.z2, 0.0]

    def equation_of_motion(
        self,
        initial_state: list[float],
        t: UniversalArray,  # noqa: ARG002
    ) -> tuple[float, float, float, float]:
        """Return the equation of motion of the double spring mass system."""
        x1, z1, x2, z2 = initial_state
        x1_dot = z1
        z1_dot = -self.k1 / self.m1 * x1 - self.k2 / self.m1 * (x1 - x2) - self.g
        x2_dot = z2
        z2_dot = -self.k2 / self.m2 * (x2 - x1) - self.g
        return x1_dot, z1_dot, x2_dot, z2_dot


class LorenzAttractor(OscillatorsFunc3D):
    r"""Lorenz Attractor differential equation.

    The Lorenz attractor is a set of differential equations that exhibit chaotic
    behavior. The Lorenz attractor consists of three coupled differential
    equations that describe the motion of a system in a simplified model of atmospheric
    convection.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation, PillowWriter
        >>> from umf.functions.chaotic.oscillators import LorenzAttractor
        >>> pendulum = LorenzAttractor(np.linspace(0, 20, 1000))
        >>> x, y, z = pendulum.to_position
        >>> t = pendulum.t
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> (line,) = ax.plot([], [], [], lw=0.5)
        >>> (point,) = ax.plot([], [], [], "o", markersize=2, color="red")
        >>> _ = ax.set_xlabel("X")
        >>> _ = ax.set_ylabel("Y")
        >>> _ = ax.set_zlabel("Z")
        >>> _ = ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
        >>> _ = ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        >>> _ = ax.set_zlim(min(z) - 0.5, max(z) + 0.5)
        >>> def init() -> tuple:
        ...     line.set_data([], [])
        ...     line.set_3d_properties([])
        ...     point.set_data([], [])
        ...     point.set_3d_properties([])
        ...     _ = ax.set_title("")
        ...     return line, point
        >>> def update(frame) -> tuple:
        ...     line.set_data(x[:frame], y[:frame])
        ...     line.set_3d_properties(z[:frame])
        ...     point.set_data([x[frame]], [y[frame]])
        ...     point.set_3d_properties([z[frame]])
        ...     _ = ax.set_title(f"t = {t[frame]:.2f} seconds")
        ...     return line, point
        >>> ani = FuncAnimation(
        ...     fig=fig, func=update,
        ...     init_func=init,
        ...     frames=len(t),
        ...     interval=10,
        ...     blit=True
        ... )
        >>> ani.save('LorenzAttractor.gif', writer=PillowWriter(fps=10))

    Notes:
        The Lorenz attractor differential equation is defined as:

        $$
        \begin{align*}
        \dot{x} &= \sigma (y - x), \\
        \dot{y} &= x (\rho - z) - y, \\
        \dot{z} &= x y - \beta z.
        \end{align*}
        $$

        with the parameters of the system are as follows: $ \sigma$ is the Prandtl
        number, which describes the ratio of momentum diffusivity to thermal
        diffusivity, while  \rho $ is the Rayleigh number, which describes
        the difference in temperature between the top and bottom of the fluid layer.
        Finally, the $ \beta $ is a geometric factor related to the physical dimensions
        of the system.


    Args:
        *time_points (UniversalArray): The array of time points at which the
            oscillator's state is evaluated.
        time_format (str, optional): The time format. Defaults to "seconds".
        rho (float, optional): The rho parameter of the Lorenz attractor. Defaults to
            28.0.
        sigma (float, optional): The sigma parameter of the Lorenz attractor. Defaults
            to 10.0.
        beta (float, optional): The beta parameter of the Lorenz attractor. Defaults to
            8/3.
        velocity (bool, optional): Whether to return the velocity of the Lorenz
            attractor. Defaults to False.
    """

    def __init__(
        self,
        *t: UniversalArray,
        time_format: str = "seconds",
        rho: float = 28.0,
        sigma: float = 10.0,
        beta: float = 8 / 3,
        velocity: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*t, time_format=time_format, velocity=velocity)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    @property
    def __initial_configuration__(self) -> dict[str, float]:
        """Return the initial configuration of the Lorenz attractor."""
        return {"rho": self.rho, "sigma": self.sigma, "beta": self.beta}

    @property
    def initial_state(self) -> list[float]:
        """Return the initial state of the Lorenz attractor."""
        return [1.0, 1.0, 1.0]

    def equation_of_motion(
        self,
        initial_state: list[float],
        t: UniversalArray,  # noqa: ARG002
    ) -> tuple[float, float, float]:
        """Return the equation of motion of the Lorenz attractor.

        Args:
            initial_state (list[float]): The initial state of the Lorenz attractor.
            t (UniversalArray): The time array.

        Returns:
            tuple[float, float, float]: The equation of motion of the Lorenz attractor.
        """
        x, y, z = initial_state
        x_dot = self.sigma * (y - x)
        y_dot = x * (self.rho - z) - y
        z_dot = x * y - self.beta * z
        return x_dot, y_dot, z_dot


class RoesslerAttractor(OscillatorsFunc3D):
    r"""Roessler Attractor differential equation.

    The Roessler attractor is a set of differential equations that exhibit chaotic
    behavior. The Roessler attractor consists of three coupled differential equations
    that describe the motion of a system in a simplified model of atmospheric
    convection.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation, PillowWriter
        >>> from umf.functions.chaotic.oscillators import RoesslerAttractor
        >>> pendulum = RoesslerAttractor(np.linspace(0, 100, 1000))
        >>> x, y, z = pendulum.to_position
        >>> t = pendulum.t
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> (line,) = ax.plot([], [], [], lw=0.5)
        >>> (point,) = ax.plot([], [], [], "o", markersize=2, color="red")
        >>> _ = ax.set_xlabel("X")
        >>> _ = ax.set_ylabel("Y")
        >>> _ = ax.set_zlabel("Z")
        >>> _ = ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
        >>> _ = ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        >>> _ = ax.set_zlim(min(z) - 0.5, max(z) + 0.5)
        >>> def init() -> tuple:
        ...     line.set_data([], [])
        ...     line.set_3d_properties([])
        ...     point.set_data([], [])
        ...     point.set_3d_properties([])
        ...     _ = ax.set_title("")
        ...     return line, point
        >>> def update(frame) -> tuple:
        ...     line.set_data(x[:frame], y[:frame])
        ...     line.set_3d_properties(z[:frame])
        ...     point.set_data([x[frame]], [y[frame]])
        ...     point.set_3d_properties([z[frame]])
        ...     _ = ax.set_title(f"t = {t[frame]:.2f} seconds")
        ...     return line, point
        >>> ani = FuncAnimation(
        ...     fig=fig, func=update,
        ...     init_func=init,
        ...     frames=len(t),
        ...     interval=10,
        ...     blit=True
        ... )
        >>> ani.save('RoesslerAttractor.gif', writer=PillowWriter(fps=10))

    Notes:
        The Roessler attractor differential equation is defined as:

        $$
        \begin{align*}
        \dot{x} &= -y - z, \\
        \dot{y} &= x + a y, \\
        \dot{z} &= b + z (x - c).
        \end{align*}
        $$
    Args:
        *time_points (UniversalArray): The array of time points at which the
            oscillator's state is evaluated.
        time_format (str, optional): The time format. Defaults to "seconds".
        a (float, optional): The a parameter of the Roessler attractor. Defaults to 0.2.
        b (float, optional): The b parameter of the Roessler attractor. Defaults to 0.2.
        c (float, optional): The c parameter of the Roessler attractor. Defaults to 5.7.
        velocity (bool, optional): Whether to return the velocity of the Roessler
            attractor. Defaults to False.
    """

    def __init__(
        self,
        *t: UniversalArray,
        time_format: str = "seconds",
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        velocity: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*t, time_format=time_format, velocity=velocity)
        self.a = a
        self.b = b
        self.c = c

    @property
    def __initial_configuration__(self) -> dict[str, float]:
        """Return the initial configuration of the Roessler attractor."""
        return {"a": self.a, "b": self.b, "c": self.c}

    @property
    def initial_state(self) -> list[float]:
        """Return the initial state of the Roessler attractor."""
        return [0.1, 0.0, 0.0]

    def equation_of_motion(
        self,
        initial_state: list[float],
        t: UniversalArray,  # noqa: ARG002
    ) -> tuple[float, float, float]:
        """Return the equation of motion of the Roessler attractor.

        Args:
            initial_state (list[float]): The initial state of the Roessler attractor.
            t (UniversalArray): The time array.

        Returns:
            tuple[float, float, float]: The equation of motion of the Roessler
                attractor.
        """
        x, y, z = initial_state
        x_dot = -y - z
        y_dot = x + self.a * y
        z_dot = self.b + z * (x - self.c)
        return x_dot, y_dot, z_dot


class DuffingOscillator(OscillatorsFunc2D):
    r"""Duffing Oscillator differential equation.

    The Duffing oscillator is a simple physical system that exhibits chaotic behavior.
    The Duffing oscillator consists of a mass attached to
    a spring and a damper. The motion of the mass is influenced by the spring constant,
    the damping coefficient, and the nonlinearity of the system. The Duffing oscillator
    exhibits chaotic behavior when the nonlinearity of the system is increased.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation, PillowWriter
        >>> from umf.functions.chaotic.oscillators import DuffingOscillator
        >>> pendulum = DuffingOscillator(np.linspace(0, 100, 1000))
        >>> x, y = pendulum.to_position
        >>> t = pendulum.t
        >>> fig, ax = plt.subplots()
        >>> (line,) = ax.plot([], [], lw=1, alpha=0.6)
        >>> (dots,) = ax.plot([], [], "ro", markersize=2)
        >>> _ = ax.set_xlabel("X")
        >>> _ = ax.set_ylabel("Y")
        >>> _ = ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
        >>> _ = ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        >>> def init() -> tuple:
        ...     line.set_data([], [])
        ...     dots.set_data([], [])
        ...     _ = ax.set_title("")
        ...     return line, dots
        >>> def update(frame: int) -> Tuple[Line2D, Line2D]:
        ...     line.set_data(x[:frame], y[:frame])
        ...     dots.set_data(x[:frame], y[:frame])
        ...     ax.set_title(f"t = {t[frame]:.2f} seconds")
        ...     return line, dots
        >>> ani = FuncAnimation(
        ...     fig=fig, func=update,
        ...     init_func=init,
        ...     frames=len(t),
        ...     interval=10,
        ...     blit=True
        ... )
        >>> ani.save('DuffingOscillator.gif', writer=PillowWriter(fps=10))

    Notes:
        The Duffing oscillator differential equation is defined as:

        $$
        \begin{align*}
        \ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 &= \gamma \cos(\omega t), \\
        \dot{x} &= y.
        \end{align*}
        $$
    Args:
        *time_points (UniversalArray): The array of time points at which the
            oscillator's state is evaluated.
        time_format (str, optional): The time format. Defaults to "seconds".
        alpha (float, optional): The alpha parameter of the Duffing oscillator.
            Defaults to -1.0.
        beta (float, optional): The beta parameter of the Duffing oscillator.
            Defaults to 1.0.
        delta (float, optional): The delta parameter of the Duffing oscillator.
            Defaults to 0.2.
        gamma (float, optional): The gamma parameter of the Duffing oscillator.
            Defaults to 0.3.
        omega (float, optional): The omega parameter of the Duffing oscillator.
            Defaults to 1.2.
        velocity (bool, optional): Whether to return the velocity of the Duffing
            oscillator. Defaults to False.
    """

    def __init__(  # noqa: PLR0913
        self,
        *t: UniversalArray,
        time_format: str = "seconds",
        alpha: float = -1.0,
        beta: float = 1.0,
        delta: float = 0.2,
        gamma: float = 0.3,
        omega: float = 1.2,
        velocity: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*t, time_format=time_format, velocity=velocity)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.omega = omega

    @property
    def __initial_configuration__(self) -> dict[str, float]:
        """Return the initial configuration of the Duffing oscillator."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "delta": self.delta,
            "gamma": self.gamma,
            "omega": self.omega,
        }

    @property
    def initial_state(self) -> list[float]:
        """Return the initial state of the Duffing oscillator."""
        return [0.0, 1.0]

    def equation_of_motion(
        self,
        initial_state: list[float],
        t: UniversalArray,
    ) -> tuple[float, float]:
        """Return the equation of motion of the Duffing oscillator.

        Args:
            initial_state (list[float]): The initial state of the Duffing oscillator.
            t (UniversalArray): The time array.

        Returns:
            tuple[float, float]: The equation of motion of the Duffing oscillator.
        """
        x, y = initial_state
        x_dot = y
        y_dot = (
            self.gamma * np.cos(self.omega * t)
            - self.alpha * x
            - self.beta * x**3
            - self.delta * y
        )
        return x_dot, y_dot

    @property
    def to_position(self) -> UniversalArrayTuple:
        """Return the position of the Duffing oscillator."""
        y = self.solve()
        return y[:, 0], y[:, 1]

    @property
    def to_velocity(self) -> UniversalArrayTuple:
        """Return the velocity of the Duffing oscillator."""
        y = self.solve()
        return y[:, 2], y[:, 3]


class ChuaSCircuit(OscillatorsFunc3D):
    r"""Chua's Circuit differential equation.

    Chua's circuit is a simple physical system that exhibits chaotic behavior. It
    consists of a set of nonlinear differential equations that describe the evolution
    of the system's state over time. The behavior of the system is influenced by its
    nonlinear components and the values of its circuit elements.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import FuncAnimation, PillowWriter
        >>> from umf.functions.chaotic.oscillators import ChuaSCircuit
        >>> circuit = ChuaSCircuit(np.linspace(0, 100, 1000))
        >>> x, y, z = circuit.to_position
        >>> t = circuit.t
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="3d")
        >>> (line,) = ax.plot([], [], [], lw=0.5)
        >>> (point,) = ax.plot([], [], [], marker="o", markersize=2, color="red")
        >>> _ = ax.set_xlabel("X")
        >>> _ = ax.set_ylabel("Y")
        >>> _ = ax.set_zlabel("Z")
        >>> _ = ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
        >>> _ = ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        >>> _ = ax.set_zlim(min(z) - 0.5, max(z) + 0.5)
        >>> def init() -> tuple:
        ...     line.set_data([], [])
        ...     line.set_3d_properties([])
        ...     point.set_data([], [])
        ...     point.set_3d_properties([])
        ...     ax.set_title("")
        ...     return line, point
        >>> def update(frame: int) -> tuple:
        ...     line.set_data(x[:frame], y[:frame])
        ...     line.set_3d_properties(z[:frame])
        ...     point.set_data([x[frame]], [y[frame]])
        ...     point.set_3d_properties([z[frame]])
        ...     ax.set_title(f"Current Time: {t[frame]:.2f} seconds")
        ...     return line, point
        >>> ani = FuncAnimation(
        ...     fig=fig, func=update,
        ...     init_func=init,
        ...     frames=len(t),
        ...     interval=10,
        ...     blit=True
        ... )
        >>> ani.save('ChuaSCircuit.gif', writer=PillowWriter(fps=10))

    Notes:
        Chua's Circuit differential equation is defined as:

        $$
        \begin{align*}
        \dot{x} &= \alpha \left( y - x - m_0 x + 0.5 \left( \lvert x + 1 \rvert -
        \lvert x - 1 \rvert \right) \right), \\
        \dot{y} &= x - y + z, \\
        \dot{z} &= -\beta y.
        \end{align*}
        $$

        with $\alpha$ as a parameter related to the system's linear components, $\beta$
        as a parameter related to the system's damping factor, and $m_0$ as a parameter
        that determines the slope of the piecewise-linear function within the interval
        $([-1, 1])$.

    Args:
        *time_points (UniversalArray): The array of time points at which the
            oscillator's state is evaluated.
        time_format (str, optional): The time format. Defaults to "seconds".
        alpha (float, optional): The alpha parameter of Chua's circuit. Defaults to
            15.6.
        beta (float, optional): The beta parameter of Chua's circuit. Defaults to 28.0.
        m0 (float, optional): The m0 parameter of Chua's circuit. Defaults to -1.143.
        m1 (float, optional): The m1 parameter of Chua's circuit. Defaults to -0.714.
        R (float, optional): The resistance of Chua's circuit. Defaults to 220.0.
        C1 (float, optional): The capacitance of Chua's circuit. Defaults to 1.0e-6.
        C2 (float, optional): The capacitance of Chua's circuit. Defaults to 1.0e-6.
        L (float, optional): The inductance of Chua's circuit. Defaults to 1.0e-3.
        velocity (bool, optional): Whether to return the velocity of Chua's circuit.
            Defaults to False.
    """

    def __init__(  # noqa: PLR0913
        self,
        *t: UniversalArray,
        alpha: float = 15.6,
        beta: float = 28.0,
        m0: float = -1.143,
        m1: float = -0.714,
        R: float = 220.0,
        C1: float = 1.0e-6,
        C2: float = 1.0e-6,
        L: float = 1.0e-3,
        time_format: str = "seconds",
        velocity: bool = False,
    ) -> None:
        """Initialize the function."""
        super().__init__(*t, time_format=time_format, velocity=velocity)

        self.alpha = alpha
        self.beta = beta
        self.m0 = m0
        self.m1 = m1
        self.R = R
        self.C1 = C1
        self.C2 = C2
        self.L = L

    @property
    def __initial_configuration__(self) -> dict[str, float]:
        """Return the initial configuration of Chua's circuit."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "m0": self.m0,
            "m1": self.m1,
            "R": self.R,
            "C1": self.C1,
            "C2": self.C2,
            "L": self.L,
        }

    @property
    def initial_state(self) -> list[float]:
        """Return the initial state of Chua's circuit."""
        return [0.1, 0.0, 0.0]  # Small perturbation from zero

    def equation_of_motion(
        self,
        initial_state: list[float],
        t: UniversalArray,  # noqa: ARG002
    ) -> tuple[float, float, float]:
        """Return the equation of motion of Chua's circuit.

        Args:
            initial_state (list[float]): The initial state of Chua's circuit.
            t (UniversalArray): The time array.

        Returns:
            tuple[float, float, float]: The equation of motion of Chua's circuit.
        """
        x, y, z = initial_state

        # Nonlinear function of Chua's circuit
        h = self.m1 * x + 0.5 * (self.m0 - self.m1) * (abs(x + 1) - abs(x - 1))

        # Chua's Circuit differential equations
        x_dot = self.alpha * (y - x - h)
        y_dot = x - y + z
        z_dot = -self.beta * y
        return x_dot, y_dot, z_dot
