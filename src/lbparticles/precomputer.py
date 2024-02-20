from __future__ import annotations

import multiprocessing
import pickle
import numpy as np
import scipy.fft
import scipy.integrate
import scipy.spatial

from lbparticles.potentials import PotentialWrapper
from lbparticles.particle import Particle
from lbparticles.util import cos_zeros


class Precomputer:
    def __init__(
        self,
        psir: PotentialWrapper,
        time_order=10,
        shape_order=100,
        e_target=0.08,
        nchis=1000,
        Nclusters=10,
        Necc=10,
        Ninterp=1000,
        Nnuk=5,
        alpha=2.2,
        logodds_initialized=False,
        vwidth=50,
        R=8100.0,
        eps=1.0e-8,
        gravity=0.00449987,
        use_multiprocessing=True
    ):
        """DOCSTRING"""
        self.time_order = time_order
        self.shape_order = shape_order
        self.psir = psir
        self.e_target = e_target
        self.nchis = nchis
        self.Nclusters = Nclusters
        self.alpha = alpha
        self.vwidth = vwidth
        self.R = R
        self.eps = eps
        self.gravity = gravity
        self.vc = self.psir.vc(R)
        self.Ninterp = Ninterp
        self.ks = np.zeros(self.Ninterp)
        self.es = np.zeros(self.Ninterp)
        self.Nnuk = Nnuk
        self.Necc = Necc
        self.logodds_initialized = logodds_initialized
        self.Warrs = None
        self.shape_zeros = None
        self.ek_logodds = None
        self.chi_eval = None
        self.use_multiprocessing = use_multiprocessing
        self.identifier = (
            f"big_{time_order:0>2}_{nchis:0>4}_alpha{str(alpha).replace('.', 'p')}"
        )

        v_target = self._init_first_pass()
        (self.target_data, self.target_data_nuphase) = self._init_second_pass(v_target)

    def _init_first_pass(self):
        vs = np.linspace(self.vc / 10, self.vc * 2, self.Ninterp)
        for i in range(self.Ninterp):
            x_cart = [self.R, 0, 0]
            v_cart = [1.0, vs[i], 0]
            particle = Particle(x_cart, v_cart, self.psir, 1.0, None, quickreturn=True)
            self.es[i] = particle.e
            self.ks[i] = particle.k
        i = np.nanargmin(np.abs(self.es - self.e_target))
        v_target = vs[i]
        return v_target

    def _init_second_pass(self, v_target):
        """
        Complete a second pass on a more useful range of velocities <--> e's <--> k's.
        """
        vs = np.linspace(
            v_target - self.vwidth, v_target + self.vwidth, self.Ninterp - 22
        )
        vs = np.concatenate([vs, np.zeros(22) + 1000])
        v_close_ind = np.argmin(np.abs(vs - self.vc))
        v_close = np.abs(vs[v_close_ind] - self.vc)
        # Add in more points close to zero eccentricity
        vs[-11:] = np.linspace(self.vc - 0.9 * v_close, self.vc + 0.9 * v_close, 11)
        v_close_ind = np.argmin(np.abs(vs))
        v_close = np.abs(vs[v_close_ind])
        vs[-22:-11] = np.linspace(v_close / 10.0, v_close * 0.9, 11)
        for i in range(self.Ninterp):
            x_cart = [self.R, 0, 0]
            v_cart = [0.01, vs[i], 0]
            particle = Particle(x_cart, v_cart, self.psir, 1.0, None, quickreturn=True)
            self.es[i] = particle.e
            self.ks[i] = particle.k

        valid = np.logical_and(np.isfinite(self.ks), np.isfinite(self.es))
        vs = vs[valid]
        self.es = self.es[valid]
        self.ks = self.ks[valid]
        # FIXME: This seems pointless?
        # self.Ninterp = np.sum(valid)

        self._initialize_e_of_k()

        target_data = np.zeros(
            (self.nchis, self.Nclusters, self.Necc, self.time_order + 2, self.Nnuk)
        )

        target_data_nuphase = np.zeros(
            (self.nchis, self.Nclusters, self.Necc, self.time_order + 2, self.Nnuk)
        )

        self.eclusters = np.linspace(0.05, 0.95, self.Nclusters)
        self.kclusters = np.zeros(self.Nclusters)
        for i, e in enumerate(self.eclusters):
            # find the closest k.
            ii = np.argmin(np.abs(self.es - e))
            self.kclusters[i] = self.ks[ii]
        self.nukclusters = 2.0 / self.kclusters - 1.0
        self.mukclusters = self.nukclusters - self.alpha / (2.0 * self.kclusters)

        self.chi_eval = np.linspace(0, 2.0 * np.pi, self.nchis)

        # This method should be equivalent and is much simpler. For John to test
        if self.use_multiprocessing:

            pool = multiprocessing.Pool()
            results = pool.map(compute_target_data,
                               [(j, i, jj, m, self.Nclusters, self.time_order, self.Necc, self.Nnuk, self.chi_eval, self.kclusters, self.eclusters, self.alpha)
                                for j in range(self.Nclusters)
                                for i in range(self.time_order + 2)
                                for jj in range(self.Necc)
                                for m in range(self.Nnuk)])
            pool.close()
            pool.join()

            for result in results:
                j, jj, i, m, y0, y1 = result
                target_data[:, j, jj, i, m] = y0
                target_data_nuphase[:, j, jj, i, m] = y1
            return target_data, target_data_nuphase

        else:

            for j in range(self.Nclusters):
                for i in range(self.time_order+ 2):
                    for jj in range(self.Necc):
                        for m in range(self.Nnuk):
                        # t_terms.append(res.y.flatten())
                            y0,y1 = evaluate_integrals( self.chi_eval, jj, self.kclusters[j], self.eclusters[j], i, m, self.alpha )
                            target_data[:, j, jj, i, m] = y0
                            target_data_nuphase[:, j, jj, i, m] = y1
            return target_data, target_data_nuphase

    def e_of_k(self, k):
        """The object's internal map between k and e0."""
        return 1.0 / (1.0 + np.exp(self.ek_logodds(k)))

    def _initialize_e_of_k(self):
        if self.logodds_initialized:
            raise Exception(
                "The e-k relationship has already been initialized and it is trying to be initialized again. This is extremely dangerous because the e-k relationship is baked in to the indexing scheme already."
            )
        to_sort_k = np.argsort(self.ks)
        x = self.ks[to_sort_k]
        y = np.clip(np.log(1.0 / self.es[to_sort_k] - 1.0), -1.0, 2.5)
        valid = np.logical_and(np.isfinite(x), np.isfinite(y))
        self.ek_logodds = scipy.interpolate.InterpolatedUnivariateSpline(
            x[valid], y[valid], k=1
        )
        self.logodds_initialized = True

    def invert(self, order_shape):
        if self.Warrs is not None and self.shape_zeros is not None:
            if (
                order_shape in self.Warrs.keys()
                and order_shape in self.shape_zeros.keys()
            ):
                return self.Warrs[order_shape], self.shape_zeros[order_shape]
        shape_zeroes = cos_zeros(order_shape)
        W_arr = np.zeros((order_shape, order_shape))
        for i in range(order_shape):
            # coefficient for W0, W1, ... for this zero
            coeffs = np.zeros(order_shape)
            # evaluate equation 4.34 of LB15 to gather relevant W's
            # sin^2 eta W[eta] = (1/4) ( (W0-W2) + (2W1-W3) cos eta + sum_2^infty (2 Wn - W(n-2) - W(n+2)) cos (n eta) )
            coeffs[0] = 0.5
            for j in np.arange(1, len(coeffs)):
                coeffs[j] = np.cos(j * shape_zeroes[i])

            # so these coeffs are for the equation W[first0] = a*W0 + b*W1 + ...
            # We are constructing a system of such equations so that we can solve for W0, W1, ... in terms of
            # the W[zeroes]. So these coefficients are the /rows/ of such a matrix.
            W_arr[i, :] = coeffs[:]
        # This matrix when multiplied by W[zeros] should give the W0, W1, ...
        W_inv_arr = np.linalg.inv(W_arr)
        return W_inv_arr, shape_zeroes

    def get_chi_index_arr(self, N):
        """
        The precomputer evaluates several integrals on a grid uniformly-spaced (in chi) from 0 to 2pi. By default this grid has 1000 entries.
        Each particleLB generates its map between t and chi by querying lbprecomputer::get_t_terms, which returns an array of values at an array of chi values.
        The process of finding the values to return is often the most expensive part of initializing a particle. The particle may not need all 1000 entries.
        In this case, we need to find the points in the original 1000-element array that we would like to use in the new smaller array.
        This method returns that mapping. Given a new N (number of points to evaluate chi from 0 to 2pi), returns an array of size N with the indices into the full array 0,1,2,...self.nchis-1.
        The corresponding chi's can be found by evaluating self.chi_eval[arr] where arr is the array returned by this method. This is done in the method lbprecomputer::get_chi_arr.
        """
        assert N <= self.nchis
        return np.linspace(0, self.nchis - 1, N, dtype=int)

    def get_chi_arr(self, N):
        """
        A convenience function to find the points where chi is evaluated if N points are requested.
        See documentation for lbprecomputer::get_chi_index_arr.
        """
        inds = self.get_chi_index_arr(N)
        return self.chi_eval[inds]

    def get_t_terms(
        self,
        kIn,
        eIn,
        Necc=None,
        nchis=None,
        includeNu=True,
        maxorder=None,
        debug=False,
    ):
        if maxorder is None:
            ordermax = self.time_order + 2
        else:
            ordermax = maxorder
        if ordermax > self.time_order + 2:
            raise Exception(
                "More orders requested than have been pre-computed in lbprecomputer::get_t_terms"
            )

        if nchis is None:
            nchiEval = self.nchis
        elif nchis <= self.nchis:
            nchiEval = nchis
        else:
            raise Exception(
                "More chi evaluation points requested than have been pre-computed in lbprecomputer::get_t_terms"
            )

        if Necc is None:
            Neccs = self.Necc
        elif Necc <= self.Necc:
            Neccs = Necc
        else:
            raise Exception(
                "More terms in the eccentricity series requested than have been pre-computed in lbprecomputer::get_t_terms"
            )

        nuk = 2.0 / kIn - 1.0
        muk = nuk - self.alpha / (2.0 * kIn)

        # for now just find the closest cluster...
        dists = (nuk - self.nukclusters) ** 2 + (eIn - self.eclusters) ** 2
        ii = np.argmin(dists)

        dnuk = nuk - self.nukclusters[ii]

        # self.mukclusters = self.nukclusters - self.alpha/(2.0*self.kclusters)

        dmuk = muk - self.mukclusters[ii]
        de = eIn - self.eclusters[ii]
        chiarr = self.get_chi_index_arr(nchiEval)
        chivals = self.get_chi_arr(nchiEval)

        # ok, the thing we want is ordermax x (Necc) x nchi x (Nnuk)
        # The thing we have is:
        # ->>       nchis x Nek x NeccTaylor x Ntime x NkTaylor
        # self.target_data = np.zeros((nchis, self.Nclusters, self.Necc,
        #                             timeorder + 2, self.Nnuk))

        # Ts = self.target_data[chiarr, ii, :, :ordermax, :].squeeze() # chi x NeccTaylor x Ntime x NkTaylor ~~ chi x j x n x i

        # to_sum = np.transpose(self.target_data[chiarr, ii, :, :ordermax, :], (2,1,0,3) )
        # to_sum_nuphase = np.transpose(self.target_data_nuphase[chiarr, ii, :, :ordermax, :], (2,1,0,3) )

        # orders, eccs, chis, ems, peas, eyes = np.meshgrid( np.arange(ordermax), np.arange(Neccs), chiarr, np.arange(self.Nnuk), np.arange(self.Nnuk), np.arange(self.Nnuk), indexing='ij')
        orders, eccs, chis, ems, peas = np.meshgrid(
            np.arange(ordermax),
            np.arange(Neccs),
            chiarr,
            np.arange(self.Nnuk),
            np.arange(self.Nnuk),
            indexing="ij",
        )

        # all of this can be pre-computed for each cluster
        def AprimeNu(order, j):
            return scipy.special.polygamma(
                order, self.nukclusters[ii] + 1
            ) - scipy.special.polygamma(order, self.nukclusters[ii] + 1 - j)

        def AprimeMu(order, j):
            return scipy.special.polygamma(
                order, self.mukclusters[ii] + 1
            ) - scipy.special.polygamma(order, self.mukclusters[ii] + 1 - j)

        # ohkay
        j1d = np.arange(Neccs)
        m1d = np.arange(self.Nnuk)
        p1d = np.arange(self.Nnuk)
        i1d = np.arange(
            self.Nnuk
        )  # these are the same array but for my own sanity I've named them different things.

        p2d, i2d = np.meshgrid(p1d, i1d, indexing="ij")
        p2d_pm, m2d_pm = np.meshgrid(p1d, m1d, indexing="ij")

        _, eccs5d, _, ems5d, peas5d = np.meshgrid(
            np.arange(ordermax),
            np.arange(Neccs),
            chiarr,
            np.arange(self.Nnuk),
            np.arange(self.Nnuk),
            indexing="ij",
        )
        _, _, _, ems4d = np.meshgrid(
            np.arange(ordermax),
            np.arange(Neccs),
            chiarr,
            np.arange(self.Nnuk),
            indexing="ij",
        )
        _, eccs3d, _ = np.meshgrid(
            np.arange(ordermax), np.arange(Neccs), chiarr, indexing="ij"
        )

        ap0nu = AprimeNu(0, j1d)
        ap1nu = AprimeNu(1, j1d)
        ap2nu = AprimeNu(2, j1d)
        ap3nu = AprimeNu(3, j1d)
        ap4nu = AprimeNu(4, j1d)

        if includeNu:
            ap0mu = AprimeMu(0, j1d)
            ap1mu = AprimeMu(1, j1d)
            ap2mu = AprimeMu(2, j1d)
            ap3mu = AprimeMu(3, j1d)
            ap4mu = AprimeMu(4, j1d)

            apmjmu = np.zeros((self.Nnuk + 1, self.Nnuk + 1, Neccs))
            apmjmu[0, 0, :] = 1  # for 0 derivatives, g(A) = 1
            apmjmu[1, 1, :] = 1  # for 1 derivative, g(A) = A

            apmjmu[0, 2, :] = ap1mu
            apmjmu[2, 2, :] = 1

            apmjmu[0, 3, :] = ap2mu
            apmjmu[1, 3, :] = 3 * ap1mu
            apmjmu[3, 3, :] = 1

            apmjmu[0, 4, :] = ap3mu + 3 * (ap1mu) ** 2
            apmjmu[1, 4, :] = 4 * ap2mu
            apmjmu[2, 4, :] = 6 * ap1mu
            apmjmu[4, 4, :] = 1

            apmjmu[0, 5, :] = ap4mu + 10 * ap1mu * ap2mu
            apmjmu[1, 5, :] = 5 * ap3mu + 15 * ap1mu**2
            apmjmu[2, 5, :] = 10 * ap2mu
            apmjmu[3, 5, :] = 10 * ap1mu
            apmjmu[5, 5, :] = 1

        apmjnu = np.zeros((self.Nnuk + 1, self.Nnuk + 1, Neccs))
        apmjnu[0, 0, :] = 1  # for 0 derivatives, g(A) = 1
        apmjnu[1, 1, :] = 1  # for 1 derivative, g(A) = A

        apmjnu[0, 2, :] = ap1nu
        apmjnu[2, 2, :] = 1

        apmjnu[0, 3, :] = ap2nu
        apmjnu[1, 3, :] = 3 * ap1nu
        apmjnu[3, 3, :] = 1

        apmjnu[0, 4, :] = ap3nu + 3 * (ap1nu) ** 2
        apmjnu[1, 4, :] = 4 * ap2nu
        apmjnu[2, 4, :] = 6 * ap1nu
        apmjnu[4, 4, :] = 1

        apmjnu[0, 5, :] = ap4nu + 10 * ap1nu * ap2nu
        apmjnu[1, 5, :] = 5 * ap3nu + 15 * ap1nu**2
        apmjnu[2, 5, :] = 10 * ap2nu
        apmjnu[3, 5, :] = 10 * ap1nu
        apmjnu[5, 5, :] = 1

        kcl = self.kclusters[ii]

        # set up some arrays that require a little bit of computation, then we'll expand them via indexing later
        efac = (
            (-de) ** j1d
            / scipy.special.factorial(j1d)
            * scipy.special.gamma(self.nukclusters[ii] + 1)
            / scipy.special.gamma(self.nukclusters[ii] + 1 - j1d)
        )
        nukfac = dnuk**m1d / scipy.special.factorial(m1d)

        # this is still kind of expensive because there are a lot of multiplications, i.e. we've expanding the matrix to an enormous size before multiplying (6d)
        # to_sum = efac[eccs] * nukfac[ems] * apmjnu[peas,ems,eccs] * ( pfact[peas] / ifact[eyes] ) * ipfact[peas,eyes]  * pmfac[peas,ems] * ap0nu[eccs]**ippower[peas,eyes] * self.target_data[chis, ii, eccs, orders, eyes]

        # to_sum = ( ipfact[peas,eyes] / ifact[eyes] ) *  ap0nu[eccs]**ippower[peas,eyes] * self.target_data[chis, ii, eccs, orders, eyes]
        to_sum = (
            apmjnu[peas5d, ems5d, eccs5d]
            * self.target_data[chis, ii, eccs, orders, peas]
        )

        # to_sum_marg1 = np.sum(to_sum, axis=-1) # do the sum over i
        # to_sum_marg2 = np.sum(to_sum_marg1 * pfact[peas5d] * apmjnu[peas5d,ems5d,eccs5d], axis=-1 ) # do the sum over p
        to_sum_marg2 = np.sum(to_sum, axis=-1)  # do the sum over p
        to_sum_marg3 = np.sum(to_sum_marg2 * nukfac[ems4d], axis=-1)  # sum over m
        res = np.sum(to_sum_marg3 * efac[eccs3d], axis=1)  # sum over j.

        if includeNu:
            efacmu = (
                (-de) ** j1d
                / scipy.special.factorial(j1d)
                * scipy.special.gamma(self.mukclusters[ii] + 1)
                / scipy.special.gamma(self.mukclusters[ii] + 1 - j1d)
            )
            mukfac = dmuk**m1d / scipy.special.factorial(m1d)

            to_sum = (
                apmjmu[peas5d, ems5d, eccs5d]
                * self.target_data_nuphase[chis, ii, eccs, orders, peas]
            )

            # to_sum_marg1 = np.sum(to_sum, axis=-1) # do the sum over i
            # to_sum_marg2 = np.sum(to_sum_marg1 * pfact[peas5d] * apmjnu[peas5d,ems5d,eccs5d], axis=-1 ) # do the sum over p
            to_sum_marg2 = np.sum(to_sum, axis=-1)  # do the sum over p
            to_sum_marg3 = np.sum(to_sum_marg2 * mukfac[ems4d], axis=-1)  # sum over m
            res_nu = np.sum(to_sum_marg3 * efac[eccs3d], axis=1)  # sum over j.

            # to_sum_nuphase = efacmu[eccs] * mukfac[ems] * apmjmu[peas,ems,eccs] * scipy.special.factorial(peas)/np.clip(scipy.special.factorial(eyes)*scipy.special.factorial(peas-eyes),1.0,None) * (peas <= ems).astype(int) * (eyes <= peas).astype(int) * ap0mu[eccs]**np.clip(peas-eyes,0,None) * self.target_data_nuphase[chis, ii, eccs, orders, eyes]

        # res = np.sum(np.sum(np.sum(np.sum(to_sum, axis=-1), axis=-1), axis=-1), axis=1)
        if includeNu:
            pass
            # res_nu = np.sum(np.sum(np.sum(np.sum(to_sum_nuphase, axis=-1), axis=-1), axis=-1), axis=1)
        else:
            res_nu = np.zeros(res.shape)

        return res, res_nu

    def precompute_inverses_up_to(self, max_shape_order, hard_reset=False):
        if self.shape_zeros is None or hard_reset:
            self.shape_zeros = {}
        if self.Warrs is None or hard_reset:
            self.Warrs = {}

        for shape_order in range(max_shape_order + 1, -1, -1):
            W_inv_arr, shape_zeros = self.invert(shape_order)
            self.Warrs[shape_order] = W_inv_arr
            self.shape_zeros[shape_order] = shape_zeros

    @classmethod
    def load(cls, filename) -> Precomputer:
        with open(filename, "rb") as file:
            return pickle.load(file)

    def save(self) -> None:
        with open(f"{self.identifier}_lbpre.pickle", "wb") as file:
            pickle.dump(self, file)


def evaluate_integrals(chis, jj, kIn, eIn, n, m, alpha):
    nuk = 2.0 / kIn - 1.0
    muk = 2.0 / kIn - 1.0 - alpha / (2.0 * kIn)
    deltapsi = scipy.special.polygamma(0, nuk + 1) - scipy.special.polygamma(
        0, nuk + 1 - jj
    )

    def to_integrate(chi, val):
        return (
            (1.0 - eIn * np.cos(chi)) ** (nuk - jj)
            * np.cos(chi) ** jj
            * np.cos(n * chi)
            * (deltapsi + np.log(1.0 - eIn * np.cos(chi))) ** m
        )

    res = scipy.integrate.solve_ivp(
        to_integrate,
        [0, 2.0 * np.pi],
        [0],
        vectorized=True,
        rtol=2.4e-14,
        atol=1.0e-15,
        t_eval=chis,
        method="DOP853",
    )

    assert np.all(np.isclose(res.t, chis))
    y0 = res.y.flatten()

    deltapsi = scipy.special.polygamma(0, muk + 1) - scipy.special.polygamma(
        0, muk + 1 - jj
    )

    def to_integrate(chi, val):
        return (
            (1.0 - eIn * np.cos(chi)) ** (nuk - jj - alpha / (2.0 * kIn))
            * np.cos(n * chi)
            * np.cos(chi) ** jj
            * (deltapsi + np.log(1.0 - eIn * np.cos(chi))) ** m
        )

    res = scipy.integrate.solve_ivp(
        to_integrate,
        [0, 2.0 * np.pi],
        [0],
        vectorized=True,
        rtol=2.4e-14,
        atol=1.0e-15,
        t_eval=chis,
        method="DOP853",
    )
    y1 = res.y.flatten()

    return y0, y1


def compute_target_data(args):
    j, i, jj, m, Nclusters, time_order, Necc, Nnuk, chi_eval, kclusters, eclusters, alpha = args

    y0, y1 = evaluate_integrals(chi_eval, jj, kclusters[j], eclusters[j], i, m, alpha)
    return j, jj, i, m, y0, y1