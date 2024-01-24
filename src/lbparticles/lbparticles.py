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

from src.lbparticles.potentials import LogPotential


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
    """This is an abstract class, always extend and override, never call directly"""

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


class PotentialWrapper():
    def __init__(self, potential: Potential, nur=None, dlnnur=None):
        self.potential = potential
        self.dlnnur = dlnnur
        self.nur = nur
        self.deltapsi_of_logr_fac = self.initialize_deltapsi()

    def __call__(self, r, Iz0=0):
        return self.potential(r) + Iz0 * self.deltapsi_of_logr_fac(np.log10(r))

    def initialize_deltapsi(self):
        def to_integrate(r, _):
            return 1.0 / (r * r * self.nur) * (r * self.potential.ddr2(r) - 0.5 * self.potential.ddr(r))

        t_eval = np.logspace(-5, np.log10(300) * 0.99999, 1000)
        logr_eval = np.linspace(-5, np.log10(300) * 0.99999, 1000)
        res = scipy.integrate.solve_ivp(to_integrate, [
            1.0e-5, 300], [0], method='DOP853', rtol=1.0e-13, atol=1.0e-13, t_eval=t_eval)

        return scipy.interpolate.CubicSpline(logr_eval, res.y.flatten())

    def omega(self, r, Iz0=0):
        return self.vc(r, Iz0=Iz0) / r

    def kappa(self, r, Iz0=0):
        return self.Omega(r, Iz0=Iz0) * self.gamma(r)

    def ddr(self, r, Iz0=0):
        return self.potential.ddr(r) + Iz0 / (r * r * self.nu(r)) * (
                r * self.potential.ddr2(r) - 0.5 * self.potential.ddr(r))

    def ddr2(self, r, Iz0=0):
        return self.potential.ddr2(r) + Iz0 / (r * r * self.nu(r)) * (
                (-2.0 / r - self.dlnnudr(r)) * (r * self.potential.ddr2(r) - 0.5 * self.potential.ddr(r)) + (
                r * self.potential.ddr3(r) + 0.5 * self.potential.ddr2(r)))

    def vc(self, r, Iz0=0):
        return np.sqrt(r * self.ddr(r, Iz0))

    def gamma(self, r, Iz0=0):
        beta = (r / self.vc(r, Iz0=Iz0)) * \
               (self.ddr(r, Iz0=Iz0) + r * self.ddr2(r, Iz0=Iz0))
        return np.sqrt(2 * (beta + 1))

    def nu(self, r, Iz0=0):
        return np.sqrt(self.nur(r) ** 2 - Iz0 / (r * r * r * self.nu(r)) * (
                r * self.potential.ddr2(r) - 0.5 * self.potential.ddr(r)))

    def name(self) -> str:
        return "PotentialWrapper_" + self.potential.name()


class Precomputer():
    def __init__(
            self,
            filename=None,
            time_order=10,
            shape_order=100,
            psir=PotentialWrapper(LogPotential(220.0)),
            e_target=0.08,
            nchis=1000,
            nks=100,
            alpha=2.2,
            vwidth=20,
            R=8100.0,
            eps=1.0e-8,
            gravity=0.00449987
    ):
        self.time_order = time_order
        self.shape_order = shape_order
        self.psir = psir
        self.e_target = e_target
        self.nchis = nchis
        self.N = nks
        self.alpha = alpha
        self.vwidth = vwidth
        self.R = R
        self.eps = eps
        self.gravity = gravity
        self.vc = self.psir.vc(R)
        self.ks = np.zeros(self.N)
        self.es = np.zeros(self.N)
        self.identifier = f"{time_order:0>2}_{nchis:0>4}_alpha{str(alpha).replace('.', 'p')}"

        v_target = self.init_first_pass()
        self.target_data, self.target_data_nuphase, self.chi_eval = self.init_second_pass(v_target)
        self.interpolators, self.interpolators_nuphase = self.generate_interpolators()

        if filename is not None:
            self.load(filename)
            self.add_new_data(1000)

    def init_first_pass(self):
        vs = np.linspace(self.vc / 10, self.vc * 2, self.N)
        for i in range(self.N):
            x_cart = [self.R, 0, 0]
            v_cart = [1.0, vs[i], 0]
            particle = Particle(x_cart, v_cart, self.psir, 1.0, None, quickreturn=True)
            self.es[i] = particle.e
            self.ks[i] = particle.k

        i = np.nanargmin(np.abs(self.es - self.e_target))
        v_target = vs[i]
        return v_target

    def init_second_pass(self, v_target):
        """
        Complete a second pass on a more useful range of velocities <--> e's <--> k's.
        """
        vs = np.linspace(v_target - self.vwidth, v_target + self.vwidth, self.N - 22)
        vs = np.concatenate([vs, np.zeros(22) + 1000])
        v_close_ind = np.argmin(np.abs(vs - self.vc))
        v_close = np.abs(vs[v_close_ind] - self.vc)
        # Add in more points close to zero eccentricity
        vs[-11:] = np.linspace(self.vc - 0.9 * v_close, self.vc + 0.9 * v_close, 11)
        v_close_ind = np.argmin(np.abs(vs))
        v_close = np.abs(vs[v_close_ind])
        vs[-22:-11] = np.linspace(v_close / 10.0, v_close * 0.9, 11)
        for i in range(self.N):
            x_cart = [self.R, 0, 0]
            v_cart = [0.01, vs[i], 0]
            particle = Particle(x_cart, v_cart, self.psir, 1.0, None, quickreturn=True)
            self.es[i] = particle.e
            self.ks[i] = particle.k

        target_data = np.zeros((self.nchis, self.N,
                                # TBD how to interpolate this. rn I'm thinking of doing a 1D interpolation
                                # at each chi, then leave the particle to decide how it will interpolate that.
                                self.time_order + 2))

        target_data_nuphase = np.zeros((self.nchis, self.N,
                                        # tbd how to interpolate this. rn I'm thinking of doing a 1D interpolation
                                        # at each chi, then leave the particle to decide how it will interpolate
                                        # that.
                                        self.time_order + 2))

        chi_eval = np.linspace(0, 2.0 * np.pi, self.nchis)
        for j in range(self.N):
            nuk = 2.0 / self.ks[j] - 1.0
            for i in range(self.time_order + 2):
                def to_integrate(chi, val):
                    return (1.0 - self.es[j] * np.cos(chi)) ** nuk * np.cos(i * chi)

                res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi + 0.001], [0], vectorized=True,
                                                rtol=1.0e-13, atol=1.0e-14, t_eval=chi_eval)
                assert np.all(np.isclose(res.t, chi_eval))
                target_data[:, j, i] = res.y.flatten()

                def to_integrate(chi, val):
                    return (1.0 - self.es[j] * np.cos(chi)) ** (nuk - self.alpha / (2.0 * self.ks[j])) * np.cos(i * chi)

                res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi + 0.001], [0], vectorized=True,
                                                rtol=1.0e-13, atol=1.0e-14, t_eval=chi_eval)
                target_data_nuphase[:, j, i] = res.y.flatten()

        return target_data, target_data_nuphase, chi_eval

    def generate_interpolators(self):
        sort_e = np.argsort(self.es)
        sorted_e = self.es[sort_e]
        interpolators = np.zeros(
            (self.time_order + 2, self.nchis), dtype=object)
        interpolators_nuphase = np.zeros(
            (self.time_order + 2, self.nchis), dtype=object)
        for i in range(self.time_order + 2):
            for k in range(self.nchis):
                interpolators[i, k] = scipy.interpolate.CubicSpline(sorted_e, self.target_data[
                    k, sort_e, i])  # not obvious if we should pick es, ks, or do something fancier

                interpolators_nuphase[i, k] = scipy.interpolate.CubicSpline(sorted_e, self.target_data_nuphase[
                    k, sort_e, i])  # not obvious if we should pick es, ks, or do something fancier
        return interpolators, interpolators_nuphase

    def gravity(self) -> float:
        return self.gravity

    def add_new_data(self, nnew):
        N_start = len(self.es)

        # Do the hardest ones first
        new_es = sorted(np.random.beta(0.9, 2.0, size=nnew))
        new_ks = [self.get_k_given_e(new_es[i]) for i in range(nnew)]

        self.ks = np.concatenate((self.ks, new_ks))
        self.es = np.concatenate((self.es, new_es))
        news_shape = np.array(self.target_data.shape)
        news_shape[1] = nnew
        assert news_shape[0] == self.nchis
        assert news_shape[2] == self.time_order + 2

        self.target_data = np.concatenate((self.target_data, np.zeros(news_shape)), axis=1)
        self.target_data_nuphase = np.concatenate((self.target_data_nuphase, np.zeros(news_shape)), axis=1)

        for j in range(N_start, self.N + nnew):
            nuk = 2.0 / self.ks[j] - 1.0
            for i in range(self.time_order + 2):
                def to_integrate(chi, val):
                    return (1.0 - self.es[j] * np.cos(chi)) ** nuk * np.cos(i * chi)

                res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi + 0.001], [0], vectorized=True,
                                                rtol=1.0e-13, atol=1.0e-14, t_eval=self.chi_eval)
                assert np.all(np.isclose(res.t, self.chi_eval))
                self.target_data[:, j, i] = res.y.flatten()

                def to_integrate(chi, val):
                    return (1.0 - self.es[j] * np.cos(chi)) ** (nuk - self.alpha / (2.0 * self.ks[j])) * np.cos(i * chi)

                res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi + 0.001], [0], vectorized=True,
                                                rtol=1.0e-13, atol=1.0e-14, t_eval=self.chi_eval)
                assert np.all(np.isclose(res.t, self.chi_eval))
                self.target_data_nuphase[:, j, i] = res.y.flatten()

        self.N = self.N + nnew
        self.generate_interpolators()

    def get_k_given_e(self, ein):
        x_cart0 = [8100.0, 0, 0]
        v_cart0 = [0.0003, 220, 0]

        def to_zero(vin):
            vCart = v_cart0[:]
            vCart[1] = vin
            xCart = x_cart0[:]
            particle = Particle(xCart, vCart, self.psir, 1.0, None, quickreturn=True)
            return particle.e - ein

        a = 1.0
        b = 220.0
        if to_zero(a) * to_zero(b) > 0:
            # TODO Log instead of print?
            print("Initial guess for bounds failed - trying fine sampling")
            trial_x = np.linspace(b - 10, b + 1, 10000)
            trial_y = np.array([to_zero(trial_x[i]) for i in range(len(trial_x))])
            switches = trial_y[1:] * trial_y[:-1] < 0
            if np.any(switches):
                inds = np.ones(len(trial_x) - 1)[switches]
                a = trial_x[inds[0]]
                b = trial_x[inds[0] + 1]

        res = scipy.optimize.brentq(to_zero, a, b, xtol=1.0e-14)
        v_cart = v_cart0[:]
        v_cart[1] = res
        part = Particle(x_cart0, v_cart, self.psir, 1.0, None, quickreturn=True)
        return part.k

    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def save(self):
        with open(f"{self.identifier}__lbpre.pickle", 'wb') as file:
            pickle.dump(self, file)


class Particle():
    def __init__(self):
        return 0


class PerturbationWrapper():
    def __init__(self):
        return 0


def coszeros():
    return 0


def precompute_inverses_up_to():
    return 0
