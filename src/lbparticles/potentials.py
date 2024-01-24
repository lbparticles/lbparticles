import numpy as np
from abc import ABC, abstractmethod


class Potential(ABC):
    """This is an abstract class, always extend and override, never call directly"""

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
    def __init__(self, scale, gravity=0.00449987, mass=None, vcirc=None):
        """Create a hernquistpotential object.
        Parameters:
            scale - the scale radius of the potential in parsecs
            gravity - gravitational constant
            mass - the mass of the material producing the potential, in solar masses
            vcirc - the circular velocity at r=scale, in pc/Myr (close to km/s)
        Exactly one of mass or vcirc must be specified (not both)
        """

        if (mass is None and vcirc is None) or (not mass is None and not vcirc is None):
            raise Exception("Need to specify exactly one of mass, or vcirc.")
        if mass is None:
            self.mass = vcirc * vcirc * 4.0 * scale / gravity
        else:
            self.mass = mass
        self.scale = scale
        self.gravity = gravity

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
