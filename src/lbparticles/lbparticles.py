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


class Particle():
    def __init__(self, xCartIn, vCartIn, psir, nunought, lbdata, rnought=8100.0, ordershape=1, ordertime=1, tcorr=True,
                 emcorr=1.0, Vcorr=1.0, wcorrs=None, wtwcorrs=None, debug=False, quickreturn=False, profile=False,
                 tilt=False, alpha=2.2, adhoc=None, nchis=300, Nevalz=1000, atolz=1.0e-7, rtolz=1.0e-7,
                 zopt='integrate',
                 Necc=10):
        self.adhoc = adhoc
        self.nunought = nunought
        self.alpha = alpha
        if not lbdata is None:
            if hasattr(lbdata, 'alpha'):
                self.alpha = lbdata.alpha
        self.rnought = rnought
        self.psi = psir
        self.ordershape = ordershape
        self.ordertime = ordertime
        self.Necc = Necc
        self.nchis = nchis
        self.xCart0 = copy.deepcopy(xCartIn)
        self.vCart0 = copy.deepcopy(vCartIn)
        self.hvec = np.cross(xCartIn, vCartIn)
        self.hhat = self.hvec / np.sqrt(np.sum(self.hvec * self.hvec))

        v = np.cross(np.array([0, 0, 1.0]), self.hhat)
        cose = np.dot(self.hhat, np.array([0, 0, 1.0]))

        vcross = np.zeros((3, 3))
        vcross[0, 1] = -v[2]
        vcross[1, 0] = v[2]
        vcross[0, 2] = v[1]
        vcross[2, 0] = -v[1]
        vcross[1, 2] = -v[0]
        vcross[2, 1] = v[0]
        self.zopt = zopt
        tilt = zopt == 'tilt'

        if tilt:
            rot = np.eye(3) + vcross + vcross @ vcross * 1.0 / (1.0 + cose)
        else:
            rot = np.eye(3)

        self.rot = Rotation.from_matrix(rot)

        xCart = self.rot.apply(xCartIn, inverse=True)
        vCart = self.rot.apply(vCartIn, inverse=True)

        x, y, z = xCart
        vx, vy, vz = vCart

        if tilt:
            assert np.isclose(z, 0)
            assert np.isclose(vz, 0)

        R = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)
        u = (x * vx + y * vy) / R
        v = ((x * vy - vx * y) / R)
        w = vz
        self.Ez = 0.5 * (w * w + (nunought * (R / self.rnought)
                                  ** (-alpha / 2.0)) ** 2 * z * z)
        self.IzIC = self.Ez / (nunought * (R / self.rnought) ** -(alpha / 2.0))
        self.psiIC = np.arctan2(
            z * (nunought * (R / self.rnought) ** -(alpha / 2.0)), w)
        self.h = R * v
        self.epsilon = 0.5 * (vCart[0] ** 2 + vCart[1] ** 2) - self.psi(
            R)

        def fpp(r, epsi, hi):
            return 2.0 * epsi + 2.0 * self.psi(r) - hi * hi / (r * r), 2.0 * (
                self.psi.ddr(r) + hi * hi / (r * r * r)), 2.0 * (self.psi.ddr2(r) - hi * hi / (r * r * r * r))

        rcirc = self.h / self.psi.vc(R)
        eff, effprime, effpp = fpp(rcirc, self.epsilon, self.h)
        curvature = -0.5 * effpp
        peri_zero = np.min([rcirc / 2.0, R])
        apo_zero = np.max([rcirc * 2.0, R])

        res_peri = scipy.optimize.root_scalar(fpp, args=(self.epsilon, self.h), fprime=True, fprime2=True, x0=peri_zero,
                                              method='halley', rtol=1.0e-8, xtol=1.0e-10)
        res_apo = scipy.optimize.root_scalar(fpp, args=(self.epsilon, self.h), fprime=True, fprime2=True, x0=apo_zero,
                                             method='halley', rtol=1.0e-8, xtol=1.0e-10)
        self.peri = res_peri.root
        self.apo = res_apo.root
        self.X = self.apo / self.peri
        dr = 0.00001
        self.cRa = self.apo * self.apo * self.apo / (self.h * self.h) * self.psi.ddr(
            self.apo)
        self.cRp = self.peri * self.peri * self.peri / (self.h * self.h) * self.psi.ddr(
            self.peri)

        self.k = np.log((-self.cRa - 1) / (self.cRp + 1)) / np.log(self.X)

        self.m0sq = 2 * self.k * (1.0 + self.cRp) / \
            (1.0 - self.X ** -self.k) / (emcorr ** 2)
        self.m0 = np.sqrt(self.m0sq)

        self.perU = self.peri ** -self.k
        self.apoU = self.apo ** -self.k

        self.e = (self.perU - self.apoU) / (self.perU + self.apoU)
        if quickreturn:
            return
        self.ubar = 0.5 * (self.perU + self.apoU)
        self.Ubar = 0.5 * (1.0 / self.perU + 1.0 / self.apoU)

        self.ell = self.ubar ** (-1.0 / self.k)

        chi_eval = lbdata.get_chi_arr(nchis)

        nuk = 2.0 / self.k - 1.0
        tfac = self.ell * self.ell / \
            (self.h * self.m0 * (1.0 - self.e * self.e) ** (nuk + 0.5)) / tcorr
        nufac = nunought * tfac * \
            self.Ubar ** (-self.alpha / (2.0 * self.k)) / \
            self.rnought ** (-self.alpha / 2.0)
        if self.ordertime >= 0:
            timezeroes = coszeros(self.ordertime)
            wt_arr = np.zeros((self.ordertime, self.ordertime))
            wtzeroes = np.zeros(self.ordertime)
            for i in range(self.ordertime):
                coeffs = np.zeros(self.ordertime)
                coeffs[0] = 0.5
                for j in np.arange(1, len(coeffs)):
                    coeffs[j] = np.cos(j * timezeroes[i])
                wt_arr[i, :] = coeffs[:] * \
                    (self.e * self.e * np.sin(timezeroes[i]) ** 2)

                ui = self.ubar * \
                    (1.0 + self.e * np.cos(self.eta_given_chi(timezeroes[i])))
                ui2 = 1.0 / (0.5 * (1.0 / self.perU + 1.0 / self.apoU)
                             * (1.0 - self.e * np.cos(timezeroes[i])))
                assert np.isclose(ui, ui2)
                wtzeroes[i] = (np.sqrt(self.essq(ui) / self.ess(ui)) - 1.0)

            wt_inv_arr = np.linalg.inv(wt_arr)
            self.wts = np.dot(wt_inv_arr, wtzeroes)
            if wtwcorrs is None:
                pass
            else:
                self.wts = self.wts + wtwcorrs

            self.wts_padded = list(self.wts) + list([0, 0, 0, 0])

            t_terms, nu_terms = lbdata.get_t_terms(self.k, self.e, maxorder=self.ordertime + 2,
                                                   includeNu=(zopt == 'first' or zopt == 'zero'), nchis=nchis,
                                                   Necc=self.Necc)

            tee = (1.0 + 0.25 * self.e * self.e *
                   (self.wts_padded[0] - self.wts_padded[2])) * t_terms[0]
            nuu = (1.0 + 0.25 * self.e * self.e *
                   (self.wts_padded[0] - self.wts_padded[2])) * nu_terms[0]
            if self.ordertime > 0:
                for i in np.arange(1, self.ordertime + 2):
                    prefac = -self.wts_padded[i - 2] + 2 * \
                        self.wts_padded[i] - self.wts_padded[i + 2]
                    if i == 1:
                        prefac = self.wts_padded[i] - self.wts_padded[
                            i + 2]
                    prefac = prefac * 0.25 * self.e * self.e
                    tee = tee + prefac * t_terms[i]
                    nuu = nuu + prefac * nu_terms[i]

            tee = tee.flatten()
            nuu = nuu.flatten()

            if zopt == 'first':
                dchi = chi_eval[1] - chi_eval[0]
                integrands = np.sin(
                    chi_eval) / (1.0 - self.e * np.cos(chi_eval)) * np.cos(2.0 * nuu * nufac)
                to_integrate = scipy.interpolate.CubicSpline(
                    chi_eval, integrands)
                lefts = integrands[:-1]
                rights = integrands[1:]
                self.cosine_integral = np.zeros(len(chi_eval))
                self.cosine_integral[1:] = np.cumsum(
                    [to_integrate.integrate(chi_eval[k], chi_eval[k + 1]) for k in range(len(chi_eval) - 1)])

                integrands = np.sin(
                    chi_eval) / (1.0 - self.e * np.cos(chi_eval)) * np.sin(2.0 * nuu * nufac)
                to_integrate = scipy.interpolate.CubicSpline(
                    chi_eval, integrands)
                self.sine_integral = np.zeros(len(chi_eval))
                self.sine_integral[1:] = np.cumsum(
                    [to_integrate.integrate(chi_eval[k], chi_eval[k + 1]) for k in range(len(chi_eval) - 1)])
            try:
                self.t_of_chi = scipy.interpolate.CubicSpline(
                    chi_eval, tee * tfac)
                self.chi_of_t = scipy.interpolate.CubicSpline(
                    tee * tfac, chi_eval)
                self.nut_of_chi = scipy.interpolate.CubicSpline(
                    chi_eval, nuu * nufac)
                self.nut_of_t = scipy.interpolate.CubicSpline(
                    tee * tfac, nuu * nufac)
                if zopt == 'first':
                    self.sine_integral_of_chi = scipy.interpolate.CubicSpline(
                        chi_eval, self.sine_integral)
                    self.cosine_integral_of_chi = scipy.interpolate.CubicSpline(
                        chi_eval, self.cosine_integral)
            except:
                print("chi doesn't seem to be monotonic!!")
            self.Tr = tee[-1] * tfac
            self.phase_per_Tr = nuu[-1] * nufac
        else:
            def to_integrate(chi, dummy):
                ui = self.ubar * \
                    (1.0 + self.e * np.cos(self.eta_given_chi(chi)))
                ret = (1.0 - self.e*np.cos(chi))**nuk * \
                    np.sqrt(self.essq(ui) / self.ess(ui))
                if np.isnan(ret) or not np.isfinite(ret):
                    return 0.0
                return ret

            chis = lbdata.get_chi_arr(nchis)
            res = scipy.integrate.solve_ivp(to_integrate, [1.0e-6, 2.0 * np.pi + 0.001], [0.0], vectorized=True,
                                            rtol=10 ** ordertime, atol=1.0e-14, t_eval=chis[1:], method='DOP853')
            ys = np.zeros(nchis)
            if res.success:
                ys[1:] = res.y.flatten() * tfac
                self.chi_of_t = scipy.interpolate.CubicSpline(ys, chi_eval)
                self.t_of_chi = scipy.interpolate.CubicSpline(chi_eval, ys)
                self.Tr = self.t_of_chi(2.0*np.pi)
        Wzeroes = np.zeros(self.ordershape)
        W_inv_arr, shapezeroes = lbdata.invert(self.ordershape)
        for i in range(self.ordershape):
            ui = self.ubar * (1.0 + self.e * np.cos(shapezeroes[i]))
            Wzeroes[i] = (np.sqrt(self.essq(ui) / self.ess(ui)) - 1.0) * self.ubar * self.ubar / (
                (self.perU - ui) * (ui - self.apoU))

        self.Ws = np.dot(W_inv_arr, Wzeroes)

        if wcorrs is None:
            pass
        else:
            assert len(wcorrs) == len(self.Ws)
            self.Ws = self.Ws + wcorrs

        self.Wpadded = np.array(list(self.Ws) + [0, 0, 0, 0])

        ustar = 2.0 / (self.peri ** self.k + self.apo ** self.k)
        self.half_esq_w0 = np.sqrt(self.essq(ustar) / self.ess(ustar)) - 1.0

        nulg = 2.0 / self.k - 1.0
        zlg = 1.0 / np.sqrt(1 - self.e * self.e)
        dz = zlg * 1.0e-5
        etaIC = np.arccos((R ** -self.k / self.ubar - 1.0) / self.e)
        if u > 0:
            self.etaIC = etaIC
        else:
            self.etaIC = np.pi + (np.pi - etaIC)

        self.phiIC = self.phi(self.etaIC)

        self.thetaIC = theta
        chiIC = np.arccos(
            (1.0 - R ** self.k / (0.5 * (1.0 / self.apoU + 1.0 / self.perU))) / self.e)
        if u > 0:
            self.chiIC = chiIC
        else:
            self.chiIC = np.pi + (np.pi - chiIC)
        self.tperiIC = self.t_of_chi(self.chiIC)

        if zopt == 'first' or zopt == 'zero':
            self.nu_t_0 = np.arctan2(-w, z * self.nu(0)) - \
                self.zphase_given_tperi(self.tperiIC)

        if zopt == 'fourier':
            self.initialize_z_fourier(40, profile=profile)

        if zopt == 'integrate':
            self.initialize_z_numerical(rtol=rtolz, atol=atolz, Neval=Nevalz)

    def getpart(self, t):
        return 0.0, self

    def xvinclined(self, t):
        r, phiabs, rdot, vphi = self.rphi(t)
        z, vz = self.zvz(t)
        vx = rdot * np.cos(phiabs) - vphi * np.sin(phiabs)
        vy = rdot * np.sin(phiabs) + vphi * np.cos(phiabs)

        return r * np.cos(phiabs), r * np.sin(phiabs), z, vx, vy, vz

    def xvabs(self, t):
        res = self.xvinclined(t)
        if hasattr(self, 'rot'):
            rotated = self.rot.apply([res[:3], res[3:]], inverse=False)
        else:
            rotated = (res[:3], res[3:])
        return rotated[0], rotated[1]

    def xabs(self, t):

        return self.xvabs(t)[0]

    def getR(self, t):
        r, _, _, _ = self.rphi(t)
        return r

    def vabs(self, t):
        return self.xvabs(t)[1]

    def uvwinclined(self, t):
        r, phiabs, rdot, vphi = self.rphi(t)
        z, vz = self.zvz(t)
        return rdot, vphi, vz

    def xrelPart(self, t, part, tpart):
        x, y, z = self.xabs(t)
        rsq = x * x + y * y
        r = np.sqrt(rsq)

        xref, yref, zref = part.xabs(tpart)
        rrefsq = xref * xref + yref * yref

        distsq = (xref - x) * (xref - x) + (yref - y) * (yref - y)

        thetarel = np.arccos((rsq + rrefsq - distsq) /
                             (2.0 * np.sqrt(rsq * rrefsq)))

        th = np.arctan2(y, x)
        thref = np.arctan2(yref, xref)

        dtheta = th - thref
        ang = np.nan
        if np.isclose(np.abs(dtheta), thetarel, atol=1.0e-6):
            ang = dtheta
        else:
            if dtheta > np.pi:
                ang = dtheta - 2.0 * np.pi
            elif dtheta < -np.pi:
                ang = dtheta + 2.0 * np.pi

        resX = r * np.cos(ang) - np.sqrt(rrefsq)
        resY = r * np.sin(ang)

        return resX, resY, z - zref

    def nu(self, t):
        return self.nunought * (self.rvectorized(t) / self.rnought) ** (-self.alpha / 2.0)

    def Norb(self, t):
        past_peri = t % self.Tr
        NOrb = (t - past_peri) / self.Tr
        return int(NOrb)

    def effcos(self, chi):
        return -self.alpha * self.e / (2.0 * self.k) * (np.cos(2.0 * self.nu_t_0) * (
            self.cosine_integral_of_chi(chi) - self.cosine_integral_of_chi(self.chiIC)) - np.sin(
            2.0 * self.nu_t_0) * (self.sine_integral_of_chi(chi) - self.sine_integral_of_chi(self.chiIC)))

    def effsin(self, chi):
        return -self.alpha * self.e / (2.0 * self.k) * (np.sin(2.0 * self.nu_t_0) * (
            self.cosine_integral_of_chi(chi) - self.cosine_integral_of_chi(self.chiIC)) + np.cos(
            2.0 * self.nu_t_0) * (self.sine_integral_of_chi(chi) - self.sine_integral_of_chi(self.chiIC)))

    def initialize_z_fourier(self, zorder=20, profile=False):
        matr = np.zeros((zorder, zorder))
        coszeroes = coszeros(zorder)
        for i in range(zorder):
            row = np.zeros(zorder)
            row[0] = 0.5
            js = np.arange(1, zorder)
            row[1:] = np.cos(js * coszeroes[i])
            matr[i, :] = row[:]

        tPeri = coszeroes * self.Tr / (2.0 * np.pi)
        chi = self.chi_given_tperi(tPeri)
        rs = self.r_given_chi(chi)

        nusqs = self.nunought**2 * (rs / self.rnought) ** (-self.alpha)

        nusqs = nusqs * (self.Tr/np.pi)**2
        thetans = np.linalg.inv(matr) @ nusqs
        thetans = thetans/2.0

        thetans_padded = np.zeros(8 * zorder + 1)
        thetans_padded[:zorder] = thetans[:]

        def get_bmp(bmpsize=2*zorder+1):
            Bmp = np.zeros((bmpsize, bmpsize))
            diag = zip(np.arange(bmpsize), np.arange(bmpsize))
            ms, ps = np.meshgrid(np.arange(bmpsize), np.arange(
                bmpsize), indexing='ij')
            mneg = ms - (bmpsize-1)/2
            pneg = ps - (bmpsize-1)/2

            ms = ms.flatten()
            ps = ps.flatten()
            mneg = mneg.flatten()
            pneg = pneg.flatten()
            diffs = np.abs(mneg - pneg).astype(int)

            vals = thetans_padded[diffs] / (thetans[0] - 4 * mneg * mneg)
            Bmp[ms, ps] = vals

            Bmp[np.arange(bmpsize), np.arange(bmpsize)] = 1.0
            return Bmp

        Bmp = get_bmp(4*zorder+1)
        det = np.linalg.det(Bmp)

        rhs = np.array(
            [-det * np.sin(np.pi/2.0 * np.sqrt(thetans[0]))**2]).astype(complex)
        mu = np.arcsinh(np.sqrt(rhs)) * 2.0/np.pi
        # mu1 = np.arcsinh(np.sqrt(np.array([-det*np.sin(np.pi/2.0 * np.sqrt(thetans[0]))**2]).astype(complex))) * 2.0/np.pi # likely to be complex

        # then we solve the linear equation for b_n (likely need to construct the corresponding matrix first). Keep in mind for both this matrix and the B_mp matrix, the indices used in the literature are symmetric about zero!

        bmatr = np.zeros((4*zorder+1, 4*zorder+1)).astype(complex)

        rowind = np.arange(-2 * zorder, 2 * zorder + 1)
        for i, nn in enumerate(np.arange(-2 * zorder, 2 * zorder + 1)):
            row = thetans_padded[np.abs(rowind - nn)].astype(complex)
            bnfac = (mu + 2 * nn * 1j) ** 2
            row[i] += bnfac
            row = row / bnfac

            bmatr[i, :] = row[:]
        U, ess, Vt = np.linalg.svd(bmatr)
        assert np.argmin(np.abs(ess)) == len(ess) - 1
        assert np.min(np.abs(ess)) < 1.0e-3
        bvec = Vt[-1, :]

        tau = self.tperiIC * np.pi / self.Tr
        icmatr = np.zeros((2, 2)).astype(complex)
        icmatr[0, 0] = np.sum(bvec * np.exp((mu + 2 * rowind * 1j) * tau))
        icmatr[0, 1] = np.sum(bvec * np.exp(-(mu + 2 * rowind * 1j) * tau))
        icmatr[1, 0] = np.sum(bvec * (mu + 2 * rowind * 1j) *
                              np.exp((mu + 2 * rowind * 1j) * tau))
        icmatr[1, 1] = np.sum(bvec * -(mu + 2 * rowind * 1j) *
                              np.exp(-(mu + 2 * rowind * 1j) * tau))

        ics = np.array([self.xCart0[2], self.vCart0[2]]).reshape((2, 1))
        ics[1] = ics[1] * self.Tr / np.pi
        self.zDs = np.linalg.inv(icmatr) @ ics
        self.bvec = bvec
        self.zrowind = rowind
        self.zmu = mu

        return True

    def zvz_fourier(self, t):
        tau = (t + self.tperiIC) * np.pi / self.Tr
        z = self.zDs[0] * np.sum(self.bvec * np.exp((self.zmu + 2 * self.zrowind * 1j) * tau)) + \
            self.zDs[1] * np.sum(self.bvec *
                                 np.exp(-(self.zmu + 2 * self.zrowind * 1j) * tau))
        vz = self.zDs[0] * np.sum(
            self.bvec * (self.zmu + 2 * self.zrowind * 1j) * np.exp((self.zmu + 2 * self.zrowind * 1j) * tau)) + \
            self.zDs[1] * np.sum(self.bvec * -(self.zmu + 2 * self.zrowind * 1j)
                                 * np.exp(-(self.zmu + 2 * self.zrowind * 1j) * tau))
        return z.real.flatten()[0], vz.real.flatten()[0] * np.pi / self.Tr

    def initialize_z_numerical(self, atol=1.0e-8, rtol=1.0e-8, Neval=1000):
        def to_integrate(tt, y):
            zz = y[0]
            vz = y[1]

            nu = self.nu(tt)

            res = np.zeros(y.shape)
            res[0] = vz
            res[1] = -zz * nu * nu
            return res

        ic0 = [1.0, 0.0]
        ic1 = [0.0, 1.0]
        ts = np.linspace(0, self.Tr, Neval)
        res0 = scipy.integrate.solve_ivp(to_integrate, [np.min(ts), np.max(
            ts)], ic0, t_eval=ts, atol=atol, rtol=rtol, method='DOP853', vectorized=True)
        res1 = scipy.integrate.solve_ivp(to_integrate, [np.min(ts), np.max(
            ts)], ic1, t_eval=ts, atol=atol, rtol=rtol, method='DOP853', vectorized=True)

        z0s = res0.y[0, :]
        vz0s = res0.y[1, :]
        z1s = res1.y[0, :]
        vz1s = res1.y[1, :]

        self.z0_interp = scipy.interpolate.CubicSpline(ts, z0s)
        self.z1_interp = scipy.interpolate.CubicSpline(ts, z1s)
        self.vz0_interp = scipy.interpolate.CubicSpline(ts, vz0s)
        self.vz1_interp = scipy.interpolate.CubicSpline(ts, vz1s)

        self.monodromy = np.zeros((2, 2))
        self.monodromy[0, 0] = self.z0_interp(self.Tr)
        self.monodromy[1, 0] = self.vz0_interp(self.Tr)
        self.monodromy[0, 1] = self.z1_interp(self.Tr)
        self.monodromy[1, 1] = self.vz1_interp(self.Tr)

        self.zICvec = np.array(
            [self.xCart0[2], self.vCart0[2]]).reshape((2, 1))

    def zvz_floquet(self, t):
        texcess = t % self.Tr
        norb = round((t - texcess) / self.Tr)
        to_mult = np.zeros((2, 2))
        to_mult[0, 0] = self.z0_interp(texcess)
        to_mult[1, 0] = self.vz0_interp(texcess)
        to_mult[0, 1] = self.z1_interp(texcess)
        to_mult[1, 1] = self.vz1_interp(texcess)
        ics = self.zICvec
        ret = to_mult @ np.linalg.matrix_power(self.monodromy, norb) @ ics
        return ret[0][0], ret[1][0]

    def zvz(self, t):
        if self.zopt == 'tilt':
            return 0.0, 0.0

        tPeri = t + self.tperiIC
        nu_now = self.nu(t)

        if hasattr(self, 'IzIC'):
            pass
        else:
            r, _, _, _ = self.rphi(0)
            self.IzIC = self.Ez / \
                (self.nunought * (r / self.rnought) ** -(self.alpha / 2.0))

        IZ = self.IzIC
        if self.zopt == 'zero':
            nu_t = self.zphase_given_tperi(tPeri) + self.nu_t_0
            w = -np.sqrt(2 * IZ * nu_now) * np.sin(nu_t)
            z = np.sqrt(2 * IZ / nu_now) * np.cos(nu_t)
            return z, w

        elif self.zopt == 'first':

            nu_t = self.zphase_given_tperi(tPeri) + self.nu_t_0
            phiconst = self.nu_t_0
            chi_excess = self.chi_excess_given_tperi(tPeri)

            Norb = self.Norb(tPeri)

            if Norb == 0:
                cosine_integral = self.effcos(chi_excess)
                sine_integral = self.effsin(chi_excess)
            else:
                cosine_integral = self.effcos(2 * np.pi) \
                    - self.alpha * self.e / (2.0 * self.k) * \
                    (np.cos(2 * (self.nu_t_0 + Norb * self.phase_per_Tr)) * self.cosine_integral_of_chi(
                        chi_excess)
                     - np.sin(2 * (self.nu_t_0 + Norb * self.phase_per_Tr)) * self.sine_integral_of_chi(
                        chi_excess))
                sine_integral = self.effsin(2 * np.pi) \
                    - self.alpha * self.e / (2.0 * self.k) * \
                    (np.sin(2 * (self.nu_t_0 + Norb * self.phase_per_Tr)) * self.cosine_integral_of_chi(
                        chi_excess)
                     + np.cos(2 * (self.nu_t_0 + Norb * self.phase_per_Tr)) * self.sine_integral_of_chi(
                        chi_excess))

                if Norb > 1:
                    arrCos = [np.cos(2.0 * (self.nu_t_0 + (i + 1) * self.phase_per_Tr))
                              for i in range(Norb - 1)]
                    arrSin = [np.sin(2.0 * (self.nu_t_0 + (i + 1) * self.phase_per_Tr))
                              for i in range(Norb - 1)]
                    to_add_cosine = -self.alpha * self.e / (2.0 * self.k) * (self.cosine_integral_of_chi(
                        2.0 * np.pi) * np.sum(arrCos) - self.sine_integral_of_chi(2.0 * np.pi) * np.sum(arrSin))
                    to_add_sine = -self.alpha * self.e / (2.0 * self.k) * (self.sine_integral_of_chi(
                        2.0 * np.pi) * np.sum(arrCos) + self.cosine_integral_of_chi(2.0 * np.pi) * np.sum(arrSin))

                    cosine_integral = cosine_integral + to_add_cosine
                    sine_integral = sine_integral + to_add_sine

            nu_t = nu_t - 0.5 * sine_integral

            IZ = self.IzIC * np.exp(cosine_integral)

            w = -np.sqrt(2 * IZ * nu_now) * np.sin(nu_t)
            z = np.sqrt(2 * IZ / nu_now) * np.cos(nu_t)
            return z, w

        elif self.zopt == 'integrate':
            return self.zvz_floquet(t)
        elif self.zopt == 'fourier':
            return self.zvz_fourier(t)
        else:
            raise Exception(
                "Need to specify an implemented zopt. Options: fourier, integrate, first, zeroeth, tilt")

    def zphase_given_tperi(self, t):
        past_peri = t % self.Tr
        return (t - past_peri) / self.Tr * self.phase_per_Tr + self.nut_of_t(past_peri)

    def chi_excess_given_tperi(self, t):
        past_peri = t % self.Tr
        return self.chi_of_t(past_peri)

    def chi_given_tperi(self, t):

        past_peri = t % self.Tr
        return (t - past_peri) / self.Tr * 2.0 * np.pi + self.chi_of_t(past_peri)

    def phi(self, eta):

        ret = eta + 0.25 * self.e * self.e * \
            (self.Wpadded[0] - self.Wpadded[2]) * eta
        for i in np.arange(1, self.ordershape + 2):
            prefac = -self.Wpadded[i - 2] + 2 * \
                self.Wpadded[i] - self.Wpadded[i + 2]
            if i == 1:
                prefac = self.Wpadded[i] - self.Wpadded[i + 2]

            ret = ret + 0.25 * prefac * self.e * \
                self.e * (1.0 / i) * np.sin(i * eta)

        def to_integrate(etaIn):
            ui = self.ubar * (1.0 + self.e * np.cos(etaIn))
            W = (np.sqrt(self.essq(ui) / self.ess(ui)) - 1.0) * self.ubar * self.ubar / (
                (self.perU - ui) * (ui - self.apoU))
            return 1.0 + self.e * self.e * np.sin(etaIn) * np.sin(etaIn) * W

        return ret / self.m0

    def essq(self, u):
        return self.m0sq * (self.perU - u) * (u - self.apoU)

    def ess(self, u):
        r = u ** (-1.0 / self.k)
        return (2.0 * self.epsilon + 2.0 * self.psi(r) - self.h * self.h / (r * r)) * r * r / (self.h * self.h) * (
            u * u * self.k * self.k)

    def t(self, chi):
        return self.Tr / (2.0 * np.pi) * (chi - (self.V2 / self.V1 * self.e * np.sin(chi)))

    def r_given_chi(self, chi):
        return (self.Ubar * (1.0 - self.e * np.cos(chi))) ** (1.0 / self.k)

    def u_given_chi(self, chi):
        return 1.0 / (self.Ubar * (1.0 - self.e * np.cos(chi)))

    def eta_given_chi(self, chi):
        sinchi = np.asarray(np.sin(chi))
        coschi = np.asarray(np.cos(chi))
        sqrte = np.sqrt(1 - self.e * self.e)

        eta_from_arccos = np.arccos(
            ((1.0 - self.e * self.e) / (1.0 - self.e * coschi) - 1.0) / self.e)
        eta_ret = None
        ret = np.where(sinchi > 0, eta_from_arccos,
                       2 * np.pi - eta_from_arccos)

        nrot = (chi - (chi % (2.0 * np.pi))) / (2.0 * np.pi)

        return ret + 2.0 * np.pi * nrot

    def emphi(self, eta):
        ''' m*phi = eta - (1/8)*e^2 W0twiddle * sin(2 eta)'''

        phi = self.phi(eta)

        u = self.ubar * (1.0 + self.e * np.cos(eta))
        r = u ** (-1.0 / self.k)

        return r, self.m0 * phi

    def rvectorized(self, t):
        tPeri = t + self.tperiIC
        chi = self.chi_given_tperi(tPeri)
        return self.r_given_chi(chi)
        eta = self.eta_given_chi(chi)
        u = self.ubar * (1.0 + self.e * np.cos(eta))
        r = u ** (-1.0 / self.k)
        return r

    def rphi(self, t):
        tPeri = t + self.tperiIC
        chi = self.chi_given_tperi(tPeri)
        eta = self.eta_given_chi(chi)

        r, mphi = self.emphi(eta)
        phiabs = mphi / self.m0 + (self.thetaIC - self.phiIC)

        rdot = np.sqrt(2 * self.epsilon - self.h * self.h / (r * r) + 2.0 * self.psi(r)) * np.sign(
            np.sin(chi))
        vphi = self.h / r

        return r, phiabs, rdot, vphi


class PerturbationWrapper():
    def __init__(self):
        return 0


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
        self.identifier = f"{time_order:0>2}_{
            nchis:0>4}_alpha{str(alpha).replace('.', 'p')}"

        v_target = self.init_first_pass()
        self.target_data, self.target_data_nuphase, self.chi_eval = self.init_second_pass(
            v_target)
        self.interpolators, self.interpolators_nuphase = self.generate_interpolators()

        if filename is not None:
            self.load(filename)
            self.add_new_data(1000)

    def init_first_pass(self):
        vs = np.linspace(self.vc / 10, self.vc * 2, self.N)
        for i in range(self.N):
            x_cart = [self.R, 0, 0]
            v_cart = [1.0, vs[i], 0]
            particle = Particle(x_cart, v_cart, self.psir,
                                1.0, None, quickreturn=True)
            self.es[i] = particle.e
            self.ks[i] = particle.k

        i = np.nanargmin(np.abs(self.es - self.e_target))
        v_target = vs[i]
        return v_target

    def init_second_pass(self, v_target):
        """
        Complete a second pass on a more useful range of velocities <--> e's <--> k's.
        """
        vs = np.linspace(v_target - self.vwidth, v_target +
                         self.vwidth, self.N - 22)
        vs = np.concatenate([vs, np.zeros(22) + 1000])
        v_close_ind = np.argmin(np.abs(vs - self.vc))
        v_close = np.abs(vs[v_close_ind] - self.vc)
        # Add in more points close to zero eccentricity
        vs[-11:] = np.linspace(self.vc - 0.9 * v_close,
                               self.vc + 0.9 * v_close, 11)
        v_close_ind = np.argmin(np.abs(vs))
        v_close = np.abs(vs[v_close_ind])
        vs[-22:-11] = np.linspace(v_close / 10.0, v_close * 0.9, 11)
        for i in range(self.N):
            x_cart = [self.R, 0, 0]
            v_cart = [0.01, vs[i], 0]
            particle = Particle(x_cart, v_cart, self.psir,
                                1.0, None, quickreturn=True)
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

        self.target_data = np.concatenate(
            (self.target_data, np.zeros(news_shape)), axis=1)
        self.target_data_nuphase = np.concatenate(
            (self.target_data_nuphase, np.zeros(news_shape)), axis=1)

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
            particle = Particle(xCart, vCart, self.psir,
                                1.0, None, quickreturn=True)
            return particle.e - ein

        a = 1.0
        b = 220.0
        if to_zero(a) * to_zero(b) > 0:
            # TODO Log instead of print?
            print("Initial guess for bounds failed - trying fine sampling")
            trial_x = np.linspace(b - 10, b + 1, 10000)
            trial_y = np.array([to_zero(trial_x[i])
                               for i in range(len(trial_x))])
            switches = trial_y[1:] * trial_y[:-1] < 0
            if np.any(switches):
                inds = np.ones(len(trial_x) - 1)[switches]
                a = trial_x[inds[0]]
                b = trial_x[inds[0] + 1]

        res = scipy.optimize.brentq(to_zero, a, b, xtol=1.0e-14)
        v_cart = v_cart0[:]
        v_cart[1] = res
        part = Particle(x_cart0, v_cart, self.psir,
                        1.0, None, quickreturn=True)
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
