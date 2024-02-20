from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

import scipy


class Potential(ABC):
    """
    Describes a potential function including its first and second derivatives.

    This is an abstract class; always extend and override, never call directly.

    For example implementations see LogPotential, HernquistPotential, NFWPotential, and PowerlawPotential.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, r: float):
        """
        The expression for the potential function with r being the variable

        Parameters
        ----------
        r : float
            The variable (radius) for central potential.
        """
        pass

    @abstractmethod
    def ddr(self, r: float):
        """
        The first derivative for the potential function with r being the variable

        Parameters
        ----------
        r : float
            The variable (radius) for central potential.
        """
        pass

    @abstractmethod
    def ddr2(self, r: float):
        """
        The second derivative for the potential function with r being the variable

        Parameters
        ----------
        r : float
            The variable (radius) for central potential.
        """
        pass

    @abstractmethod
    def name(self):
        """
        Describes the potential for use in uniquely identifying the Precomputer object once computation has finished
        """
        pass


class LogPotential(Potential):
    def __init__(self, vcirc, nur=None):
        self.vcirc = vcirc

    def __call__(self, r):
        return -self.vcirc**2 * np.log(r)

    def ddr(self, r, IZ=0):
        return -self.vcirc**2 / r

    def ddr2(self, r, IZ=0):
        return self.vcirc**2 / (r * r)

    def name(self):
        """A unique identifier"""
        return "logpotential" + str(self.vcirc).replace(".", "p")


class NFWPotential(Potential):
    def __init__(self):
        return 0

    def __call__(self, r):
        return 0

    def ddr(self, r):
        return 0

    def ddr2(self, r):
        return 0

    def name(self):
        return 0


class HernquistPotential(Potential):
    def __init__(self, scale, mass, gravity=0.00449987):
        """
        Create a HernquistPotential object.

        Parameters
        ----------
        scale : float
            The scale radius of the potential in parsecs.
        mass : float
            The mass of the material producing the potential, in solar masses.
        gravity : float
            The gravitational constant.
        """

        self.mass = mass
        self.scale = scale
        self.gravity = gravity

    @classmethod
    def vcirc(cls, scale, vcirc, gravity=0.00449987) -> HernquistPotential:
        return HernquistPotential(scale, vcirc * vcirc * 4.0 * scale / gravity, gravity)

    @classmethod
    def mass(cls, scale, mass, gravity=0.00449987) -> HernquistPotential:
        return HernquistPotential(scale, mass, gravity)

    def __call__(self, r):
        return self.gravity * self.mass / (r + self.scale)

    def ddr(self, r):
        return -self.gravity * self.mass / (r + self.scale) ** 2

    def ddr2(self, r):
        return 2.0 * self.gravity * self.mass / (r + self.scale) ** 3

    def name(self):
        """A unique name for the object"""
        return (
            "HernquistPotential_scale_"
            + str(self.scale).replace(".", "p")
            + "_mass_"
            + str(self.mass).replace(".", "p")
        )


class PowerlawPotential(Potential):
    def __init__(self):
        return 0

    def __call__(self, r):
        return 0

    def ddr(self, r):
        return 0

    def ddr2(self, r):
        return 0

    def name(self):
        return 0


class PotentialWrapper:
    def __init__(self, potential: Potential, nur=None, dlnnur=None):
        """DOCSTRING"""
        self.potential = potential
        self.dlnnur = dlnnur
        self.nur = nur
        self.deltapsi_of_logr_fac = (
            None if self.nur == None else self.initialize_deltapsi()
        )

    def __call__(self, r, Iz0=0):
        return (
            self.potential(r)
            if Iz0 == 0
            else self.potential(r) + Iz0 * self.deltapsi_of_logr_fac(np.log10(r))
        )

    def initialize_deltapsi(self):
        def to_integrate(r, _):
            return (
                1.0
                / (2.0*r * r * self.nur(r))
                * (r * self.potential.ddr2(r) -  self.potential.ddr(r))
            )

        t_eval = np.logspace(-5, np.log10(300) * 0.99999, 1000)
        logr_eval = np.linspace(-5, np.log10(300) * 0.99999, 1000)
        res = scipy.integrate.solve_ivp(
            to_integrate,
            [1.0e-5, 300],
            [0],
            method="DOP853",
            rtol=1.0e-13,
            atol=1.0e-13,
            t_eval=t_eval,
        )

        return scipy.interpolate.CubicSpline(logr_eval, res.y.flatten())

    def omega(self, r, Iz0=0):
        return self.vc(r, Iz0=Iz0) / r

    def kappa(self, r, Iz0=0):
        return self.Omega(r, Iz0=Iz0) * self.gamma(r)

    def ddr(self, r, Iz0=0):
        return (
            self.potential.ddr(r)
            if Iz0 == 0
            else self.potential.ddr(r)
            + Iz0
            / (r * r * self.nu(r))
            * (r * self.potential.ddr2(r) - 0.5 * self.potential.ddr(r))
        )

    def ddr2(self, r, Iz0=0):
        return (
            self.potential.ddr2(r)
            if Iz0 == 0
            else self.potential.ddr2(r)
            + Iz0
            / (r * r * self.nu(r))
            * (
                (-2.0 / r - self.dlnnudr(r))
                * (r * self.potential.ddr2(r) - 0.5 * self.potential.ddr(r))
                + (r * self.potential.ddr3(r) + 0.5 * self.potential.ddr2(r))
            )
        )

    def vc(self, r, Iz0=0):
        """The minus sign is due to the force being inward towards the centre and thus having a negative sign in the potential"""
        return np.sqrt(-r * self.ddr(r, Iz0))

    def gamma(self, r, Iz0=0):
        beta = (r / self.vc(r, Iz0=Iz0)) * (
            self.ddr(r, Iz0=Iz0) + r * self.ddr2(r, Iz0=Iz0)
        )
        return np.sqrt(2 * (beta + 1))

    def nu(self, r, Iz0=0):
        # TODO Fix badness of recursive function
        return np.sqrt(
            self.nur(r) ** 2
            - 0.5*Iz0
            / (r * r * r * self.nur(r))
            * (r * self.potential.ddr2(r) -  self.potential.ddr(r))
        )

    def name(self) -> str:
        return "PotentialWrapper_" + self.potential.name()
