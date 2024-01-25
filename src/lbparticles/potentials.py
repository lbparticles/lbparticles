from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


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
    def __call__(self, r):
        pass

    @abstractmethod
    def ddr(self, r):
        pass

    @abstractmethod
    def ddr2(self, r):
        pass

    @abstractmethod
    def name(self):
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

    def __call__(self):
        return 0

    def ddr(self):
        return 0

    def ddr2(self):
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

    def __call__(self):
        return 0

    def ddr(self):
        return 0

    def ddr2(self):
        return 0

    def name(self):
        return 0
