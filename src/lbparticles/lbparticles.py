import numpy as np
import pickle
import copy
import scipy.integrate
import scipy.fft
import scipy.spatial
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from enum import Enum

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
    def __init__():
        return 0
    
def coszeros():
    return 0

def precompute_inverses_up_to():
    return 0