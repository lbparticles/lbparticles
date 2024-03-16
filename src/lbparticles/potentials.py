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
    def __init__(self, pot, units, galpy, ro=8., vo=220., name=''):
        self.pot = pot
        self.isiterable = hasattr(self.pot,'__iter__')
        self.name_input = name
        self.units = units
        self.ro=ro
        self.vo=vo
        self.galpyDotPotential = galpy # pointer to galpy
    def _rz(self, r, z):
        return self.units.Quantity(r,'pc'),self.units.Quantity(z,'pc')
    def __call__(self, r, z=0.):
        ret = 0.0
        ru,zu = self._rz(r,z)
        return -self.galpyDotPotential.evaluatePotentials(self.pot,ru,zu,ro=self.ro,vo=self.vo).to('pc**2/Myr**2').value
    def ddr(self,r,**kwargs):
        ret = 0.0
        ru,zu = self._rz(r,0.0)
        return self.galpyDotPotential.evaluateRforces(self.pot, ru, zu, ro=self.ro,vo=self.vo).to('pc/Myr**2').value
    def ddr2(self,r, **kwargs):
        ret = 0.0
        ru,zu = self._rz(r,0.0)
        return -self.galpyDotPotential.evaluateR2derivs(self.pot, ru,zu,ro=self.ro,vo=self.vo).to('Myr**-2').value
    def name(self):
        return self.name_input

class galpyFreq:
    def __init__(self,pot, units, galpy, ro=8., vo=220., forSureNotSpherical=False):
        self.pot = pot
        self.isiterable = hasattr(self.pot,'__iter__')
        self.units = units
        self.galpyDotPotential = galpy
        self.ro=ro
        self.vo=vo
        
        self.likelySpherical = self._check_spherical()
        if self.likelySpherical:
            print("WARNING: it looks like you're inferring a vertical frequency from the shape of a spherical potential!")
            print("If you want to just look at orbits in a spherical potential, you are better off not specifying a value of nu,")
            print(" and simply using the TILT option when initializing particles in this potential.")
            print("If you want to include the effects of a disk, you have to specify nu(r) another way, either as its own")
            print(" function, or using this class but with a galpy potential that is not spherically symmetric.")
            if not forSureNotSpherical:
                print("Raising an error because of this issue. To forge ahead because you're certain the potential isn't actually spherical")
                print("you can suppress this error message by passing forSureNotSpherical=True to this constructor.")
                raise ValueError

    def _rz(self, r, z):
        return self.units.Quantity(r,'pc'),self.units.Quantity(z,'pc')
    def _check_spherical(self, scale=1.):
        ''' Check whether the potential is spherical so we can warn the user not to 
        use nu(r) and just use tilt'''
        # check a couple of spots.
        gpp = galpyPotential(self.pot,self.units,self.galpyDotPotential,ro=self.ro,vo=self.vo)
        rz = gpp(4000.*scale,z=3000.*scale)
        rsph = gpp(5000.*scale)

        rz2 = gpp(12000.*scale,z=5000.*scale)
        rsph2 = gpp(13000.*scale)
        return np.isclose(rz,rsph) and np.isclose(rz2, rsph2)
    def __call__(self,r, **kwargs):
        ret = 0.0
        ru,zu = self._rz(r,0.)
        return np.sqrt(self.galpyDotPotential.evaluatez2derivs(self.pot,ru,zu,ro=self.ro,vo=self.vo).to('1/Myr**2').value)

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
        ''' gamma = kappa/Omega = sqrt(2(beta+1)), where beta=dlnvc/dlnr '''
        # vc = sqrt( - r ddr(r) )
        #  => dv/dr = (1/2) ( - r ddr(r) )^(-1/2) (-r ddr2(r) - ddr(r)) 
        #  => dv/dr = - (1/2vc)  (r ddr2(r) + ddr(r)) 
        #  => beta  = - (r/2vc^2)  (r ddr2(r) + ddr(r)) 
        #beta = (r / self.vc(r, Iz0=Iz0)) * (
        #    self.ddr(r, Iz0=Iz0) + r * self.ddr2(r, Iz0=Iz0)
        #)
        vc = self.vc(r,Iz0=Iz0)
        beta = -(r/(2*vc*vc)) * (self.ddr(r,Iz0=Iz0) + r*self.ddr2(r,Iz0=Iz0)) 
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
