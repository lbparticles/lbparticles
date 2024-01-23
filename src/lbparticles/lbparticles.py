import numpy as np
import pickle
import copy
import scipy.integrate
import scipy.fft
import scipy.spatial
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


# TODO Push inside precomputer so that precomputer holds ground truth
GRAVITY = 0.00449987  # pc^3 / (solar mass Myr^2)


@dataclass
class CartVec():
    x: float = 0
    y: float = 0
    z: float = 0
    vx: float = 0
    vy: float = 0
    vz: float = 0


@dataclass
class CylindVec():
    r: float = 0
    theta: float = 0
    z: float = 0
    vr: float = 0
    vtheta: float = 0
    vz: float = 0


class VertOptionEnum(Enum):
    INTEGRATE = 1
    FOURIER = 2
    TILT = 3
    FIRST = 4
    ZERO = 5


class Potential(ABC):
    """Never use, override all functions"""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def ddr(self):
        pass

    @abstractmethod
    def ddr2(self):
        pass

    @abstractmethod
    def name(self):
        pass


class Precomputer():
    def __init__():
        return 0


class Particle():
    def __init__():
        return 0


class PerturbationWrapper():
    def __init__():
        return 0


class PotentialWrapper():
    def __init__(self, potential: Potential, nur=None, dlnnur=None):
        self.potential = potential
        self.dlnnur = dlnnur
        self.nur = nur
        self.deltapsi_of_logr_fac = self.initialize_deltapsi()

    def initialize_deltapsi(self):
        def to_integrate(r, _):
            return 1.0/(r*r*self.nur) * (r * self.potential.ddr2(r) - 0.5 * self.potential.ddr(r))
        t_eval = np.logspace(-5, np.log10(300)*0.99999, 1000)
        logr_eval = np.linspace(-5, np.log10(300)*0.99999, 1000)
        res = scipy.integrate.solve_ivp(to_integrate, [
                                        1.0e-5, 300], [0], method='DOP853', rtol=1.0e-13, atol=1.0e-13, t_eval=t_eval)

        return scipy.interpolate.CubicSpline(logr_eval,  res.y.flatten())


def coszeros():
    return 0


def precompute_inverses_up_to():
    return 0
