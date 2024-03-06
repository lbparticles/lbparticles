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


class galpyPotential(Potential):
    '''
    Wrapper for potentials to be imported from galpy.
    '''
    def __init__(self, pot, units, name=''):
        self.pot = pot
        self.isiterable = hasattr(self.pot,'__iter__')
        self.name_input = name
        self.units = units
    def _rz(self, r, z):
        return self.units.Quantity(r,'pc'),self.units.Quantity(z,'pc')
    def __call__(self, r, **kwargs):
        ret = 0.0
        ru,zu = self._rz(r,0.0)
        if self.isiterable:
            for i in range(len(self.pot)):
                ret += self.pot[i](ru,zu,**kwargs)
        else:
            ret += self.pot(ru,zu,**kwargs)
        if hasattr(ret,'to'):
            return -ret.to('pc**2/Myr**2').value
        else:
            return self.__call__(r,ro=8.,vo=220.)
    def ddr(self,r,**kwargs):
        ret = 0.0
        ru,zu = self._rz(r,0.0)
        if self.isiterable:
            for i in range(len(self.pot)):
                ret += self.pot[i].rforce(ru,zu,**kwargs)
        else:
            ret += self.pot.rforce(ru,zu,**kwargs)
        if hasattr(ret,'to'):
            return ret.to('pc/Myr**2').value
        else:
            return self.ddr(r,ro=8.,vo=220.)
    def ddr2(self,r, **kwargs):
        ret = 0.0
        ru,zu = self._rz(r,0.0)
        if self.isiterable:
            for i in range(len(self.pot)):
                ret += self.pot[i].R2deriv(ru,zu,**kwargs)
        else:
            ret += self.pot.R2deriv(ru,zu,**kwargs)
        if hasattr(ret,'to'):
            return -ret.to('Myr**-2').value
        else:
            return self.ddr2(r,ro=8.,vo=220.)
    def name(self):
        return self.name_input

class galpyFreq:
    def __init__(self,pot, units):
        self.pot = pot
        self.isiterable = hasattr(self.pot,'__iter__')
        self.units = units
    def _rz(self, r, z):
        return self.units.Quantity(r,'pc'),self.units.Quantity(z,'pc')
    def _check_spherical(self):
        pass
    def __call__(self,r,**kwargs):
        ret = 0.0
        ru,zu = self._rz(r,0.0)
        if self.isiterable:
            for i in range(len(self.pot)):
                ret += self.pot[i].z2deriv(ru,zu,**kwargs)
        else:
            ret += self.pot.z2deriv(ru,zu,**kwargs)
        if hasattr(ret,'to'):
            return np.sqrt(ret.to('1/Myr**2').value)
        else:
            return self.__call__(r,ro=8.,vo=220.0)

class numericalFreqDeriv:
    def __init__(self, nu):
        self.nu = nu
    def __call__(self,r):
        dr = 1.0e-3
        return (np.log(self.nu(r+dr))-np.log(self.nu(r-dr)))/(2*dr)




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
        return ("Hernquist_scale_"
            + "{:.2e}".format(self.scale)
            +"_mass_{:.2e}".format(self.mass))


class PotentialWrapper:
    def __init__(self, potential: Potential, nur=None, dlnnudr=None):
        """DOCSTRING"""
        self.potential = potential
        self.dlnnudr = dlnnudr
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
            return [(
                1.0
                / (2.0*r * r * self.nur(r))
                * (r * self.potential.ddr2(r) -  self.potential.ddr(r))
            )]

        t_eval = np.logspace(1.0, np.log10(30000) * 0.99999, 1000)
        logr_eval = np.linspace(1.0, np.log10(30000) * 0.99999, 1000)
        res = scipy.integrate.solve_ivp(
            to_integrate,
            [10.0, 30000],
            [0],
            method="BDF",
            rtol=1.0e-12,
            atol=1.0e-12,
            t_eval=t_eval,
        )

        return scipy.interpolate.CubicSpline(logr_eval, res.y.flatten())

    def Omega(self, r, Iz0=0):
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
        return self.potential.name()
