import numpy as np
import pickle
import copy
import scipy.integrate
import scipy.fft
import scipy.spatial
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from enum import Enum

GRAVITY = 0.00449987  # pc^3 / (solar mass Myr^2)

@dataclass
class cartVec():
    x:float
    y:float
    z:float
    vx:float
    vy:float
    vz:float

@dataclass
class cylindVec():
    r:float
    theta:float
    z:float
    vr:float
    vtheta:float
    vz:float

class vertOptionEnum(Enum):
    INTEGRATE = 1
    FOURIER = 2
    TILT = 3
    FIRST = 4
    ZERO = 5


class precomputer():
    def __init__():
        return 0

class particle():
    def __init__():
        return 0

class perturbationWrapper():
    def __init__():
        return 0

class potentialWrapper():
    def __init__(self, potential, nur=None, dlnnur=None):
        self.potential = potential
        self.dlnnur = dlnnur
        self.nur = nur
        self.deltapsi_of_logr_fac = self.initialize_deltapsi()

    def initialize_deltapsi(self):
        def to_integrate( r, dummy ):
            return 1.0/(r*r*self.nu(r)) * (r* self.potential.ddr2(r) - 0.5 * self.potential.ddr(r))
        t_eval = np.logspace(-5, np.log10(300)*0.99999, 1000 )
        logr_eval = np.linspace(-5, np.log10(300)*0.99999, 1000)
        res = scipy.integrate.solve_ivp(to_integrate, [1.0e-5, 300], [0], method='DOP853', rtol=1.0e-13, atol=1.0e-13, t_eval=t_eval)

        return scipy.interpolate.CubicSpline( logr_eval,  res.y.flatten() )
    
    
def coszeros():
    return 0

def precompute_inverses_up_to():
    return 0