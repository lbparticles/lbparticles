import numpy as np
import matplotlib.pyplot as plt
import corner
import pickle
import time
import copy
import pdb
import scipy.integrate
import scipy.fft
import scipy.spatial
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Units throughout are Msun - pc - Myr so that 1 pc/Myr ~ 1 km/s
G = 0.00449987  # pc^3 / (solar mass Myr^2)


class lbprecomputer:
    def __init__(self, timeorder, shapeorder, psir, etarget, nchis, nks, Neccs, alpha, vwidth=50):
        # enumerate possible k-e combinations.
        # compute integrals for a bunch of them.
        self.ordertime = timeorder
        self.ordershape = shapeorder

        self.identifier = ('big_'
                           + str(timeorder).zfill(2)
                           + '_' + str(nchis).zfill(4)
                           + '_' + str(Neccs).zfill(4)
                           + '_' + psir.name()
                           + '_alpha' + str(alpha).replace('.', 'p'))
        # surely timeorder is just the /maximum/ timeorder allowed?

        # do a quick survey of k-e.
        R = 8100.0
        eps = 1.0e-8
        # vc = np.sqrt(np.abs(R*(psir(R+eps)-psir(R-eps))/(2.0*eps)))
        self.psir = psir
        vc = self.psir.vc(R)
        self.Necc = Neccs
        #self.N = nks -- replace with 2 numbers: 
        #  number of clusters and number of points in initial e-k interpolation
        self.Ninterp = 1000
        self.Nclusters = nks
        self.Nnuk = 5

        self.nchis = nchis
        self.vwidth = vwidth
        self.alpha = alpha
        self.logodds_initialized=False

        # first pass
        self.ks = np.zeros(self.Ninterp)
        self.es = np.zeros(self.Ninterp)
        vs = np.linspace(vc / 10, vc * 2, self.Ninterp)
        for i in range(self.Ninterp):
            xCart = [R, 0, 0]
            vCart = [1.0, vs[i], 0]
            part = particleLB(xCart, vCart, self.psir, 1.0, None, quickreturn=True)
            self.es[i] = part.e
            self.ks[i] = part.k

        i = np.nanargmin(np.abs(self.es - etarget))
        vtarget = vs[i]

        # do a second pass targeted on a more useful range of velocities <--> e's <--> k's.
        vs = np.linspace(vtarget - vwidth, vtarget + vwidth, self.Ninterp - 22)
        vs = np.concatenate([vs, np.zeros(22) + 1000])
        vclose_ind = np.argmin(np.abs(vs - vc))
        vclose = np.abs(vs[vclose_ind] - vc)
        vs[-11:] = np.linspace(vc - 0.9 * vclose, vc + 0.9 * vclose,
                               11)  # add in more points close to zero eccentricity
        vclose_ind = np.argmin(np.abs(vs))
        vclose = np.abs(vs[vclose_ind])
        vs[-22:-11] = np.linspace(vclose / 10.0, vclose * 0.9, 11)
        for i in range(self.Ninterp):
            xCart = [R, 0, 0]
            vCart = [0.01, vs[i], 0]
            part = particleLB(xCart, vCart, self.psir, 1.0, None, quickreturn=True)
            self.es[i] = part.e
            self.ks[i] = part.k

        valid = np.logical_and(np.isfinite(self.ks), np.isfinite(self.es))
        vs = vs[valid]
        self.es = self.es[valid]
        self.ks = self.ks[valid]
        self.Ninterp = np.sum(valid)

        self.initialize_e_of_k()


        # currently nchis x Nk x Necc x Ntime
        # ->>       nchis x Nek x NeccTaylor x Ntime x NkTaylor
        self.target_data = np.zeros((nchis, self.Nclusters, self.Necc,
                                     timeorder + 2, self.Nnuk))  

        self.target_data_nuphase = np.zeros((nchis, self.Nclusters, self.Necc,
                                             timeorder + 2, self.Nnuk))  


        self.eclusters = np.linspace(0.05, 0.95, self.Nclusters)
        self.kclusters = np.zeros(self.Nclusters)
        for i,e in enumerate(self.eclusters):
            # find the closest k.
            ii = np.argmin( np.abs( self.es - e ))
            self.kclusters[i] = self.ks[ii]
        self.nukclusters = 2.0/self.kclusters - 1.0
        self.mukclusters = self.nukclusters - self.alpha/(2.0*self.kclusters)

        self.chi_eval = np.linspace(0, 2.0 * np.pi, self.nchis)
        for j in tqdm(range(self.Nclusters)):
            for i in range(self.ordertime + 2):
                for jj in range(self.Necc):
                    for m in range(self.Nnuk):
                    # t_terms.append(res.y.flatten())
                        y0,y1 = self.evaluate_integrals( self.chi_eval, jj, self.kclusters[j], self.eclusters[j], i, m )
                        self.target_data[:, j, jj, i, m] = y0
                        self.target_data_nuphase[:, j, jj, i, m] = y1 
                    
        #self.generate_interpolators()

    def evaluate_integrals( self, chis, jj, kIn, eIn, n, m ):
        nuk = 2.0 / kIn - 1.0
        muk = 2.0 / kIn - 1.0 - self.alpha/(2.0*kIn)
        deltapsi = scipy.special.polygamma(0,nuk+1) - scipy.special.polygamma(0,nuk+1-jj)
        
        def to_integrate(chi, val):
            return (1.0 - eIn * np.cos(chi)) ** (nuk-jj) * np.cos(chi)**jj * np.cos(n * chi) * (deltapsi + np.log(1.0-eIn*np.cos(chi)))**m

        res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi ], [0], vectorized=True,
                                        rtol=1.0e-14, atol=1.0e-14, t_eval=chis, method='DOP853')
        try:
            assert np.all(np.isclose(res.t, self.chi_eval))
        except:
            pdb.set_trace()
        y0 = res.y.flatten()

        deltapsi = scipy.special.polygamma(0,muk+1) - scipy.special.polygamma(0,muk+1-jj)
        def to_integrate(chi, val):
            return (1.0 - eIn * np.cos(chi)) ** (nuk - jj - self.alpha / (2.0 * kIn)) * np.cos(n * chi) * np.cos(chi)**jj * (deltapsi + np.log(1.0-eIn*np.cos(chi)))**m

        res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi], [0], vectorized=True,
                                        rtol=1.0e-14, atol=1.0e-14, t_eval=chis, method='DOP853')
        y1 = res.y.flatten()

        return y0,y1

    def e_of_k(self, k):
        ''' The object's internal map between k and e0.'''
        return 1.0/(1.0 + np.exp(self.ek_logodds(k)))
    def initialize_e_of_k(self):
        if self.logodds_initialized:
            raise Exception("The e-k relationship has already been initialized and it is trying to be initialized again. This is extremely dangerous because the e-k relationship is baked in to the indexing scheme already.")
        to_sort_k = np.argsort(self.ks)
        x = self.ks[to_sort_k]
        y = np.clip(np.log(1.0/self.es[to_sort_k] - 1.0),-1.0,2.5)
        #self.ek_logodds = scipy.interpolate.make_smoothing_spline(x,y) # note that this function will NOT be updated with more data (since then the e values at a given k computed below might change)
        valid = np.logical_and(np.isfinite(x), np.isfinite(y))
        #self.ek_logodds = scipy.interpolate.CubicSpline(x[valid],y[valid])
        self.ek_logodds = scipy.interpolate.InterpolatedUnivariateSpline(x[valid],y[valid], k=1)
        self.logodds_initialized=True


    def invert(self, ordershape):
        # do matrix inversions too, because why not.

        if hasattr(self, 'Warrs') and hasattr(self, 'shapezeros'):
            if ordershape in self.Warrs.keys() and ordershape in self.shapezeros.keys():
                return self.Warrs[ordershape], self.shapezeros[ordershape]
        shapezeroes = coszeros(ordershape)
        W_arr = np.zeros((ordershape, ordershape))
        for i in range(ordershape):
            coeffs = np.zeros(ordershape)  # coefficient for W0, W1, ... for this zero
            # evaluate equation 4.34 of LB15 to gather relevant W's
            # sin^2 eta W[eta] = (1/4) ( (W0-W2) + (2W1-W3) cos eta + sum_2^infty (2 Wn - W(n-2) - W(n+2)) cos (n eta) )
            coeffs[0] = 0.5
            for j in np.arange(1, len(coeffs)):
                coeffs[j] = np.cos(j * shapezeroes[i])

            # so these coeffs are for the equation W[first0] = a*W0 + b*W1 + ...
            # We are constructing a system of such equations so that we can solve for W0, W1, ... in terms of
            # the W[zeroes]. So these coefficients are the /rows/ of such a matrix.
            W_arr[i, :] = coeffs[:]
        W_inv_arr = np.linalg.inv(W_arr)  # This matrix when multiplied by W[zeros] should give the W0, W1, ...
        return W_inv_arr, shapezeroes

    def generate_interpolators(self):
        assert False
        sort_k = np.argsort(self.ks)
        sorted_k = self.ks[sort_k]
        self.interpolators = np.zeros((self.ordertime + 2, self.Necc, self.nchis), dtype=object)
        self.interpolators_nuphase = np.zeros((self.ordertime + 2, self.Necc, self.nchis), dtype=object)
        for i in range(self.ordertime + 2):
            for j in range(self.Necc):
                for k in range(self.nchis):
                    self.interpolators[i, j, k] = scipy.interpolate.CubicSpline(sorted_k, self.target_data[
                        k, sort_k, j, i], )  # here target_data are assumed to be nchi x k x ecc x time

                    self.interpolators_nuphase[i, j, k] = scipy.interpolate.CubicSpline(sorted_k, self.target_data_nuphase[
                        k, sort_k, j, i])  

        self.interpolatorsND = scipy.interpolate.CubicSpline( sorted_k, self.target_data[:,sort_k, :, :], axis=1 )
        self.interpolatorsND_nuphase = scipy.interpolate.CubicSpline( sorted_k, self.target_data_nuphase[:,sort_k, :, :], axis=1 )

        #self.interpolatorsND =  scipy.interpolate.RegularGridInterpolator( (self.chi_eval,sorted_k, np.arange(self.Necc), np.arange(self.ordertime+2)), self.target_data[:,sort_k,:,:], method='cubic' )
        #self.interpolatorsND_nuphase =  scipy.interpolate.RegularGridInterpolator( (self.chi_eval,sorted_k, np.arange(self.Necc), np.arange(self.ordertime+2)), self.target_data_nuphase[:,sort_k,:,:], method='cubic' )


    def get_k_given_e(self, ein):
        assert False # should never be called
        xCart0 = [8100.0, 0, 0]
        vCart0 = [0.0003, 220, 0]

        def to_zero(vin):
            vCart = vCart0[:]
            vCart[1] = vin
            xCart = xCart0[:]
            part = particleLB(xCart, vCart, self.psir, 1.0, None, quickreturn=True)

            return part.e - ein

        a = 1.0
        b = 220.0
        if to_zero(a) * to_zero(b) > 0:
            print("Initial guess for bounds failed - trying fine sampling")
            trial_x = np.linspace(b - 10, b + 1, 10000)
            trial_y = np.array([to_zero(trial_x[i]) for i in range(len(trial_x))])
            switches = trial_y[1:] * trial_y[:-1] < 0
            if np.any(switches):
                inds = np.ones(len(trial_x) - 1)[switches]
                a = trial_x[inds[0]]
                b = trial_x[inds[0] + 1]
            else:
                pdb.set_trace()

        res = scipy.optimize.brentq(to_zero, a, b, xtol=1.0e-14)
        vCart = vCart0[:]
        vCart[1] = res
        part = particleLB(xCart0, vCart, self.psir, 1.0, None, quickreturn=True)
        return part.k

    def add_new_data(self, nnew):
        assert False
        Nstart = len(self.ks)

        # we could choose random values of k, but this is not ideal from a testing/reproducibility perspective or from the perspective of choosing more optimal points.
        # not necessarily efficient but the main cost is doing the integrals themselves...
        new_ks = []
        for i in tqdm(range(nnew)):
            sorted_ks = np.array( sorted(self.ks) )
            diffs = sorted_ks[1:] - sorted_ks[:-1]
            ii = np.argmax(diffs)
            new_ks.append( 0.5*(sorted_ks[ii+1]+sorted_ks[ii]) )
            self.ks = np.concatenate((self.ks, [new_ks[-1]]))

        # pretty sure we don't ever use self.es again.


        #self.ks = np.concatenate((self.ks, new_ks))
        news_shape = np.array(self.target_data.shape)
        news_shape[1] = nnew
        assert news_shape[0] == self.nchis
        assert news_shape[3] == self.ordertime + 2

        self.target_data = np.concatenate((self.target_data, np.zeros(news_shape)), axis=1)
        self.target_data_nuphase = np.concatenate((self.target_data_nuphase, np.zeros(news_shape)), axis=1)


        for j in tqdm(range(Nstart, self.N + nnew)):
            nuk = 2.0 / self.ks[j] - 1.0
            for jj in range(self.Necc):
                for i in range(self.ordertime + 2):
                    y0, y1 = self.evaluate_integrals( self.chi_eval, jj, self.ks[j], i )
                    self.target_data[:,j,jj,i] = y0
                    self.target_data_nuphase[:,j,jj,i] = y1

        self.N = self.N + nnew
        self.generate_interpolators()

        return self

    def save(self):
        with open(self.identifier + '_lbpre.pickle', 'wb') as f:
            # could delete interpolators here to save disk space.
            pickle.dump(self, f)

    @classmethod
    def load(cls, fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)

    def get_chi_index_arr(self, N):
        ''' 
            The precomputer evaluates several integrals on a grid uniformly-spaced (in chi) from 0 to 2pi. By default this grid has 1000 entries.
            Each particleLB generates its map between t and chi by querying lbprecomputer::get_t_terms, which returns an array of values at an array of chi values.
            The process of finding the values to return is often the most expensive part of initializing a particle. The particle may not need all 1000 entries.
            In this case, we need to find the points in the original 1000-element array that we would like to use in the new smaller array.
            This method returns that mapping. Given a new N (number of points to evaluate chi from 0 to 2pi), returns an array of size N with the indices into the full array 0,1,2,...self.nchis-1.
            The corresponding chi's can be found by evaluating self.chi_eval[arr] where arr is the array returned by this method. This is done in the method lbprecomputer::get_chi_arr.
        '''
        assert N<=self.nchis
        return np.linspace(0,self.nchis-1,N, dtype=int)
    def get_chi_arr(self,N):
        ''' A convenience function to find the points where chi is evaluated if N points are requested. See documentation for lbprecomputer::get_chi_index_arr.'''
        inds = self.get_chi_index_arr(N)
        return self.chi_eval[inds]

    def get_t_terms(self, kIn, eIn, Necc=None, nchis=None, includeNu=True, maxorder=None, debug=False):
        if maxorder is None:
            ordermax = self.ordertime+2
        else:
            ordermax = maxorder
        if ordermax>self.ordertime+2:
            raise Exception("More orders requested than have been pre-computed in lbprecomputer::get_t_terms")

        if nchis is None:
            nchiEval = self.nchis
        elif nchis <= self.nchis:
            nchiEval = nchis
        else:
            raise Exception("More chi evaluation points requested than have been pre-computed in lbprecomputer::get_t_terms")

        if Necc is None:
            Neccs = self.Necc
        elif Necc <= self.Necc:
            Neccs = Necc
        else:
            raise Exception("More terms in the eccentricity series requested than have been pre-computed in lbprecomputer::get_t_terms")

        nuk = 2.0/kIn - 1.0
        muk = nuk - self.alpha/(2.0*kIn)

        # for now just find the closest cluster...
        dists = (nuk-self.nukclusters)**2 + (eIn-self.eclusters)**2
        ii = np.argmin(dists)

        dnuk = nuk - self.nukclusters[ii]
        
        self.mukclusters = self.nukclusters - self.alpha/(2.0*self.kclusters)
        
        dmuk = muk - self.mukclusters[ii]
        de = eIn - self.eclusters[ii]
        chiarr = self.get_chi_index_arr(nchiEval)
        chivals = self.get_chi_arr(nchiEval)
        
        # ok, the thing we want is ordermax x (Necc) x nchi x (Nnuk)
        # The thing we have is:
        # ->>       nchis x Nek x NeccTaylor x Ntime x NkTaylor
        #self.target_data = np.zeros((nchis, self.Nclusters, self.Necc,
        #                             timeorder + 2, self.Nnuk))  

        #Ts = self.target_data[chiarr, ii, :, :ordermax, :].squeeze() # chi x NeccTaylor x Ntime x NkTaylor ~~ chi x j x n x i

        #to_sum = np.transpose(self.target_data[chiarr, ii, :, :ordermax, :], (2,1,0,3) )
        #to_sum_nuphase = np.transpose(self.target_data_nuphase[chiarr, ii, :, :ordermax, :], (2,1,0,3) )

        #orders, eccs, chis, ems, peas, eyes = np.meshgrid( np.arange(ordermax), np.arange(Neccs), chiarr, np.arange(self.Nnuk), np.arange(self.Nnuk), np.arange(self.Nnuk), indexing='ij')
        orders, eccs, chis, ems, peas = np.meshgrid( np.arange(ordermax), np.arange(Neccs), chiarr, np.arange(self.Nnuk), np.arange(self.Nnuk), indexing='ij')

        # all of this can be pre-computed for each cluster
        def AprimeNu(order, j):
            return (scipy.special.polygamma(order,self.nukclusters[ii]+1) - scipy.special.polygamma(order,self.nukclusters[ii]+1-j))
        def AprimeMu(order, j):
            return (scipy.special.polygamma(order,self.mukclusters[ii]+1) - scipy.special.polygamma(order,self.mukclusters[ii]+1-j))

        # ohkay
        j1d = np.arange(Neccs)
        m1d = np.arange(self.Nnuk)
        p1d = np.arange(self.Nnuk)
        i1d = np.arange(self.Nnuk) # these are the same array but for my own sanity I've named them different things.

        p2d,i2d = np.meshgrid( p1d, i1d, indexing='ij')
        p2d_pm, m2d_pm = np.meshgrid( p1d, m1d, indexing='ij')


        _, eccs5d, _, ems5d, peas5d = np.meshgrid( np.arange(ordermax), np.arange(Neccs), chiarr, np.arange(self.Nnuk), np.arange(self.Nnuk), indexing='ij')
        _, _, _, ems4d  = np.meshgrid( np.arange(ordermax), np.arange(Neccs), chiarr, np.arange(self.Nnuk), indexing='ij')
        _, eccs3d, _ = np.meshgrid( np.arange(ordermax), np.arange(Neccs), chiarr, indexing='ij')





        ap0nu = AprimeNu(0,j1d)
        ap1nu = AprimeNu(1,j1d)
        ap2nu = AprimeNu(2,j1d)
        ap3nu = AprimeNu(3,j1d)
        ap4nu = AprimeNu(4,j1d)

        if includeNu:
            ap0mu = AprimeMu(0,j1d)
            ap1mu = AprimeMu(1,j1d)
            ap2mu = AprimeMu(2,j1d)
            ap3mu = AprimeMu(3,j1d)
            ap4mu = AprimeMu(4,j1d)

            apmjmu = np.zeros((self.Nnuk+1, self.Nnuk+1, Neccs))
            apmjmu[0,0,:] = 1 # for 0 derivatives, g(A) = 1
            apmjmu[1,1,:] = 1 # for 1 derivative, g(A) = A

            apmjmu[0,2,:] = ap1mu
            apmjmu[2,2,:] = 1

            apmjmu[0,3,:] = ap2mu 
            apmjmu[1,3,:] = 3*ap1mu
            apmjmu[3,3,:] = 1

            apmjmu[0,4,:] = ap3mu + 3*(ap1mu)**2
            apmjmu[1,4,:] = 4*ap2mu
            apmjmu[2,4,:] = 6*ap1mu
            apmjmu[4,4,:] = 1

            apmjmu[0,5,:] = ap4mu + 10*ap1mu*ap2mu
            apmjmu[1,5,:] = 5*ap3mu + 15*ap1mu**2
            apmjmu[2,5,:] = 10*ap2mu
            apmjmu[3,5,:] = 10*ap1mu
            apmjmu[5,5,:] = 1

        apmjnu = np.zeros((self.Nnuk+1, self.Nnuk+1, Neccs))
        apmjnu[0,0,:] = 1 # for 0 derivatives, g(A) = 1
        apmjnu[1,1,:] = 1 # for 1 derivative, g(A) = A

        apmjnu[0,2,:] = ap1nu 
        apmjnu[2,2,:] = 1

        apmjnu[0,3,:] = ap2nu 
        apmjnu[1,3,:] = 3*ap1nu
        apmjnu[3,3,:] = 1

        apmjnu[0,4,:] = ap3nu + 3*(ap1nu)**2
        apmjnu[1,4,:] = 4*ap2nu
        apmjnu[2,4,:] = 6*ap1nu
        apmjnu[4,4,:] = 1

        apmjnu[0,5,:] = ap4nu + 10*ap1nu*ap2nu
        apmjnu[1,5,:] = 5*ap3nu + 15*ap1nu**2
        apmjnu[2,5,:] = 10*ap2nu
        apmjnu[3,5,:] = 10*ap1nu
        apmjnu[5,5,:] = 1


        kcl = self.kclusters[ii]

        # set up some arrays that require a little bit of computation, then we'll expand them via indexing later
        efac = (-de)**j1d / scipy.special.factorial(j1d) * scipy.special.gamma(self.nukclusters[ii]+1)/scipy.special.gamma(self.nukclusters[ii]+1-j1d)
        nukfac = dnuk**m1d/scipy.special.factorial(m1d)
        pfact = scipy.special.factorial(p1d)
        ifact = scipy.special.factorial(i1d)
        ipfact = (i2d<=p2d).astype(int) / np.clip(scipy.special.factorial(p2d-i2d), 1.0, None)
        ippower = np.clip(p2d-i2d,0,None)
        pmfac = (p2d_pm <= m2d_pm).astype(int)

        # this is still kind of expensive because there are a lot of multiplications, i.e. we've expanding the matrix to an enormous size before multiplying (6d)
        #to_sum = efac[eccs] * nukfac[ems] * apmjnu[peas,ems,eccs] * ( pfact[peas] / ifact[eyes] ) * ipfact[peas,eyes]  * pmfac[peas,ems] * ap0nu[eccs]**ippower[peas,eyes] * self.target_data[chis, ii, eccs, orders, eyes]


        #to_sum = ( ipfact[peas,eyes] / ifact[eyes] ) *  ap0nu[eccs]**ippower[peas,eyes] * self.target_data[chis, ii, eccs, orders, eyes] 
        to_sum =   apmjnu[peas5d, ems5d, eccs5d] * self.target_data[chis, ii, eccs, orders, peas] 

        #to_sum_marg1 = np.sum(to_sum, axis=-1) # do the sum over i
        #to_sum_marg2 = np.sum(to_sum_marg1 * pfact[peas5d] * apmjnu[peas5d,ems5d,eccs5d], axis=-1 ) # do the sum over p
        to_sum_marg2 = np.sum(to_sum, axis=-1 ) # do the sum over p
        to_sum_marg3 = np.sum(to_sum_marg2 * nukfac[ems4d], axis=-1) # sum over m
        res = np.sum(to_sum_marg3 * efac[eccs3d], axis=1) # sum over j.


        if includeNu:
            efacmu = (-de)**j1d / scipy.special.factorial(j1d) * scipy.special.gamma(self.mukclusters[ii]+1)/scipy.special.gamma(self.mukclusters[ii]+1-j1d)
            mukfac = dmuk**m1d/scipy.special.factorial(m1d)

            to_sum =   apmjmu[peas5d, ems5d, eccs5d] * self.target_data_nuphase[chis, ii, eccs, orders, peas] 

            #to_sum_marg1 = np.sum(to_sum, axis=-1) # do the sum over i
            #to_sum_marg2 = np.sum(to_sum_marg1 * pfact[peas5d] * apmjnu[peas5d,ems5d,eccs5d], axis=-1 ) # do the sum over p
            to_sum_marg2 = np.sum(to_sum, axis=-1 ) # do the sum over p
            to_sum_marg3 = np.sum(to_sum_marg2 * mukfac[ems4d], axis=-1) # sum over m
            res_nu = np.sum(to_sum_marg3 * efac[eccs3d], axis=1) # sum over j.



            #to_sum_nuphase = efacmu[eccs] * mukfac[ems] * apmjmu[peas,ems,eccs] * scipy.special.factorial(peas)/np.clip(scipy.special.factorial(eyes)*scipy.special.factorial(peas-eyes),1.0,None) * (peas <= ems).astype(int) * (eyes <= peas).astype(int) * ap0mu[eccs]**np.clip(peas-eyes,0,None) * self.target_data_nuphase[chis, ii, eccs, orders, eyes]


        
        #res = np.sum(np.sum(np.sum(np.sum(to_sum, axis=-1), axis=-1), axis=-1), axis=1)
        if includeNu:
            pass
            #res_nu = np.sum(np.sum(np.sum(np.sum(to_sum_nuphase, axis=-1), axis=-1), axis=-1), axis=1)
        else:
            res_nu = np.zeros(res.shape)
        
        return res, res_nu



    def get_t_terms_interp(self, kIn, eIn, maxorder=None, includeNu=True, nchis=None, Necc=None, debug=False):
        # interpolate target data to the actual k<-->e.
        # note that this will be called (once) by each particle, so ideally it should be reasonably fast.

        if maxorder is None:
            ordermax = self.ordertime+2
        else:
            ordermax = maxorder
        if ordermax>self.ordertime+2:
            raise Exception("More orders requested than have been pre-computed in lbprecomputer::get_t_terms")

        if nchis is None:
            nchiEval = self.nchis
        elif nchis <= self.nchis:
            nchiEval = nchis
        else:
            raise Exception("More chi evaluation points requested than have been pre-computed in lbprecomputer::get_t_terms")

        if Necc is None:
            Neccs = self.Necc
        elif Necc <= self.Necc:
            Neccs = Necc
        else:
            raise Exception("More terms in the eccentricity series requested than have been pre-computed in lbprecomputer::get_t_terms")

        #delta_e = self.e_of_k(kIn) - eIn
        delta_e = eIn - self.e_of_k(kIn) 
        chiarr = self.get_chi_index_arr(nchiEval)
        chivals = self.get_chi_arr(nchiEval)

        nuk = 2.0 / kIn - 1.0

        # time to vectorize!
        # the things we want to return are summed~ordermax x nchieval and ret_nu~ ordermax x nchieval. May also want ordermax x ecc x nchieval

        # meshgrid should be in the same shape we want the returned stuff to be. So..
        orders, eccs, chis = np.meshgrid( np.arange(ordermax), np.arange(Neccs), chivals, indexing='ij' )

        #pts = np.stack((chis.flatten(), kIn*np.ones(chis.shape).flatten(), eccs.flatten(), orders.flatten()), axis=1)

        nukfacvec = scipy.special.gamma(nuk+1)/scipy.special.gamma(nuk+1-eccs) # may need to use gammaln and gammasgn
        facvec = delta_e**eccs / scipy.special.factorial(eccs) * (-1.0)**eccs
        #pdb.set_trace()
        #retvec = facvec * nukfacvec * self.interpolatorsND( pts ).reshape(chis.shape)
        retvec = facvec * nukfacvec * np.transpose( self.interpolatorsND( kIn )[chiarr], ( 2,1,0 ))[:ordermax,:Necc,:]
        #self.interpolatorsND =  scipy.interpolate.RegularGridInterpolator( (self.chi_eval,sorted_k, np.arange(self.Necc), np.arange(self.ordertime+2)), self.target_data[:,sort_k,:,:], method='cubic' )

        if includeNu:
            nukalphafacvec = scipy.special.gamma(nuk+1 -self.alpha/(2.0*kIn))/scipy.special.gamma(nuk+1-eccs - self.alpha/(2.0*kIn)) 
            facalphavec = delta_e**eccs / scipy.special.factorial(eccs) * (-1.0)**eccs
            #retalphavec = facalphavec * nukalphafacvec * self.interpolatorsND_nuphase( pts ).reshape(chis.shape)
            retalphavec = facalphavec * nukalphafacvec * np.transpose(self.interpolatorsND_nuphase( kIn )[chiarr], (2,1,0))[:ordermax,:Necc,:]
        else:
            retalphavec = np.zeros(retvec.shape)

        if debug:

            retTrue = np.zeros( (ordermax, Neccs, nchiEval) )
            for i in range(ordermax):
                for j in range(Neccs):

                    def to_integrate(chi, dummy):
                        return (1.0 - self.e_of_k(kIn)*np.cos(chi))**(nuk-j) * np.cos(i*chi) * np.cos(chi)**j
                    res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi + 0.001], [0], vectorized=True, \
                            rtol=1.0e-13, atol=1.0e-14, t_eval=self.get_chi_arr(nchiEval), method='DOP853')
                    retTrue[i,j,:] = nukfacvec[i,j,:] * delta_e**j / scipy.special.factorial(j) * (-1.0)**j * res.y[:]
            pdb.set_trace()
        return np.sum(retvec,axis=1), np.sum(retalphavec,axis=1)

        ret = np.zeros( (ordermax, Neccs, nchiEval) )
        retTrue = np.zeros( (ordermax, Neccs, nchiEval) )
        ret_nu = np.zeros( (ordermax, nchiEval) )
        for i in range(ordermax):
            for j in range(Neccs):
                nukfac = 1.0
                if j>0:
                    for jj in range(1,j+1):
                        nukfac *= nuk-jj+1
                if j==1:
                    assert nukfac == nuk
                if j==2:
                    assert nukfac == nuk*(nuk-1)

                nukalphafac = 1.0
                if j>0:
                    for jj in range(1, j+1):
                        nukalphafac *= (nuk-jj+1-self.alpha/(2.0*kIn))
                if j==1:
                    assert nukalphafac == nuk - self.alpha/(2.0*kIn)
                if j==2:
                    assert nukalphafac == (nuk-self.alpha/(2.0*kIn))*(nuk-1-self.alpha/(2.0*kIn))

                for k in range(nchiEval):
                    keval = chiarr[k]
                    ret[i,j,k] =  nukfac * delta_e**j / scipy.special.factorial(j) * (-1.0)**j * self.interpolators[i,j,keval](kIn) # here interpolators are time x ecc x chi
                    if includeNu:
                        ret_nu[i,k] += nukalphafac * delta_e**j / scipy.special.factorial(j) * (-1.0)**j *self.interpolators_nuphase[i,j,keval](kIn)

                if debug:
                    def to_integrate(chi, dummy):
                        return (1.0 - self.e_of_k(kIn)*np.cos(chi))**(nuk-j) * np.cos(i*chi) * np.cos(chi)**j
                    res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi + 0.001], [0], vectorized=True, \
                            rtol=1.0e-13, atol=1.0e-14, t_eval=self.get_chi_arr(nchiEval), method='DOP853')
                    retTrue[i,j,:] =nukfac * delta_e**j / scipy.special.factorial(j) * (-1.0)**j * res.y[:]
                    #fig,ax = plt.subplots()
                    #ks = np.linspace(kIn-0.003, kIn+0.003, 1000)
                    #ax.plot( ks, self.interpolators[i,j,chiarr[nchiEval-10]](ks), c='k' )
                    #ax.scatter( [kIn], [res.y[0][nchiEval-10]], c='r', marker='s', zorder=2, s=5)
                    ##self.target_data[:, j, jj, i] = res.y.flatten() # target data is chis x k x ecc x time
                    #ax.scatter( self.ks, self.target_data[chiarr[nchiEval-10],:,j,i], c='b', marker='x', zorder=3, s=8)
                    ##print(self.target_data[chiarr[nchiEval-10],:,j,i])
                    #ax.set_xlabel(r'$k$')
                    #ax.set_ylabel(r'Integral at $\chi='+str(self.get_chi_arr(nchiEval)[nchiEval-10])+r'$')
                    #for ii in range(len(self.ks)):
                    #    if ii<97:
                    #        ax.axvline( self.ks[ii], c='pink' )
                    #    else:
                    #        ax.axvline( self.ks[ii], c='r' )
                    #ax.set_xlim(np.min(ks), np.max(ks))
                    #ax.set_ylim( res.y[0][nchiEval-10]-0.01, res.y[0][nchiEval-10]+0.01 )
                    #fig.savefig('dbgecc.png',dpi=300)

                    #pdb.set_trace()

                    #plt.close(fig)
        
        summed = np.sum(ret,axis=1)
        if debug:
            summedTrue = np.sum(retTrue,axis=1)
            dbg_ek(self,kIn,eIn)
            pdb.set_trace()


#        def to_integrate(chi, dummy, enn):
#            return (1.0 - eIn * np.cos(chi))**nuk * np.cos(enn*chi)
#        for n in range(ordermax):
#            
#            res = scipy.integrate.solve_ivp(to_integrate, [0, 2.0 * np.pi + 0.001], [0], vectorized=True, \
#                    rtol=1.0e-13, atol=1.0e-14, t_eval=self.get_chi_arr(nchiEval), args=(n,), method='DOP853')
#            print("n=",n,": ",summed[n,:]/res.y[:])
#
#            pdb.set_trace()
        return summed, ret_nu

def add_orders(lbpre, N):

    Nstart = lbpre.target_data.shape[3]


    #self.ks = np.concatenate((self.ks, new_ks))
    news_shape = np.array(lbpre.target_data.shape)
    news_shape[3] = N
    assert news_shape[0] == lbpre.nchis

    lbpre.target_data = np.concatenate((lbpre.target_data, np.zeros(news_shape)), axis=3)
    lbpre.target_data_nuphase = np.concatenate((lbpre.target_data_nuphase, np.zeros(news_shape)), axis=3)


    for j in tqdm(range(len(lbpre.ks))):
        nuk = 2.0 / lbpre.ks[j] - 1.0
        for jj in range(lbpre.Necc):
            for i in range(lbpre.ordertime + 2, lbpre.ordertime+2+N):
                y0, y1 = lbpre.evaluate_integrals( lbpre.chi_eval, jj, lbpre.ks[j], i )
                lbpre.target_data[:,j,jj,i] = y0
                lbpre.target_data_nuphase[:,j,jj,i] = y1

    lbpre.ordertime = lbpre.ordertime + N

    lbpre.generate_interpolators()

    return lbpre 


def dbg_ek(lbpre, kIn, eIn):
    # make a quick figure...
    fig,ax = plt.subplots()
    ks = np.linspace(kIn-0.01, kIn+0.01)
    ax.plot( ks, lbpre.e_of_k(ks), c='k' )
    ax.scatter( lbpre.ks[:len(lbpre.es)], lbpre.es, c='k' )
    ax.scatter([kIn], [eIn], c='blue')
    ax.set_xlabel('k')
    ax.set_ylabel('e')
    ax.set_xlim(kIn-0.01, kIn+0.01)
    plt.savefig('dbgeofk.png')
    pdb.set_trace()
    plt.close(fig)




class hernquistpotential:
    def __init__(self, scale, mass=None, vcirc=None):
        ''' Create a hernquistpotential object.
        Parameters:
            scale - the scale radius of the potential in parsecs
            mass - the mass of the material producing the potential, in solar masses
            vcirc - the circular velocity at r=scale, in pc/Myr (close to km/s)
        Exactly one of mass or vcirc must be specified (not both)
        '''
        # vcirc^2 = G*mass*r/(r+a)^2.
        # at r=a, vcirc^2 = G*mass*a/4a^2 = G*mass/(4*a)
        if (mass is None and vcirc is None) or (not mass is None and not vcirc is None):
            raise Exception("Need to specify exactly one of mass, or vcirc.")
        if mass is None:
            self.mass = vcirc*vcirc*4.0*scale/G
        else:
            self.mass = mass
        self.scale = scale

    def __call__(self,r):
        return G*self.mass/(r+self.scale)

    def vc(self, r):
        return np.sqrt(G*self.mass*r)/(r+self.scale)

    def Omega(self, r): #TODO CREATE ISSUE
        return vc(r)/r

    def gamma(self,r):
        return np.sqrt(3.0 - 2.0*r/(r+self.scale))

    def kappa(self,r):
        return self.Omega(r)*self.gamma(r)

    def ddr(self, r):
        return -G*self.mass/(r+self.scale)**2

    def ddr2(self, r):
        return 2.0*G*self.mass/(r+self.scale)**3

    def ddr3(self, r):
        return -6.0*G*self.mass/(r+self.scale)**4
    def name(self):
        '''A unique name for the object'''
        return 'hernquistpotential_scale'+str(self.scale).replace('.','p')+'_mass'+str(self.mass).replace('.','p')

class logpotential:
    def __init__(self, vcirc, nur=None):
        self.vcirc = vcirc

        self.nur = nur
        if not nur is None:
            pass


    def __call__(self, r):
        return -self.vcirc ** 2 * np.log(r)

    def Omega(self, r):
        return self.vcirc / r

    def gamma(self, r):
        return np.sqrt(2.0)

    def kappa(self, r):
        return self.Omega(r) * self.gamma(r)

    def vc(self, r):
        return self.vcirc

    def ddr(self, r, IZ=0):
        return -self.vcirc ** 2 / r

    def ddr2(self, r, IZ=0):
        return self.vcirc ** 2 / (r * r)

    def ddr3(self, r):
        return -2.0 * self.vcirc ** 2 / (r * r * r)

    def deltanusq(self, r, IZ=0):
        pass

    def name(self):
        '''A unique name for the object'''
        return 'logpotential'+str(self.vcirc).replace('.','p')

def particle_ivp2(t, y, psir, alpha, nu0 ):
    '''The derivative function for scipy's solve_ivp - used to compare integrated particle trajectories to our semi-analytic versions'''
    #vcirc = 220.0
    #nu0 = np.sqrt(4.0*np.pi*G*0.2)
    #alpha = 2.2
    # y assumed to be a 6 x N particle array of positions and velocities.
    xx = y[0,:]
    yy = y[1,:]
    zz = y[2,:]
    vx = y[3,:]
    vy = y[4,:]
    vz = y[5,:]

    r = np.sqrt(xx*xx + yy*yy)# + zz*zz)
    g = psir.ddr(r)
    #g = -vcirc*vcirc / r
    nu = nu0 * (r/8100.0)**(-alpha/2.0)

    res = np.zeros( y.shape )
    res[0,:] = vx
    res[1,:] = vy
    res[2,:] = vz
    res[3,:] = g*xx/r
    res[4,:] = g*yy/r
    #res[5,:] = g*zz/r
    res[5,:] = - zz*nu*nu


    return res


class particleLB:
    # use orbits from Lynden-Bell 2015.
    # def __init__(self, xCart, vCart, vcirc, vcircBeta, nu):
    def __init__(self, xCartIn, vCartIn, psir, nunought, lbpre, rnought=8100.0, ordershape=1, ordertime=1, tcorr=True,
                 emcorr=1.0, Vcorr=1.0, wcorrs=None, wtwcorrs=None, debug=False, quickreturn=False, profile=False,
                 tilt=False, alpha=2.2, adhoc=None, nchis=300, Nevalz=1000, atolz=1.0e-7, rtolz=1.0e-7, zopt='integrate',
                 Necc=10):
        self.adhoc = adhoc
        # psir is psi(r), the potential as a function of radius.
        # don't understand things well enough yet, but let's start writing in some of the basic equations

        # primary variables are ra and rp. These are set as follows....
        # 2*epsilon + 2 *psi - h^2 / r^2 = 0 at pericenter and apocenter
        # self.vcircBeta = vcircBeta
        # self.vcirc = vcirc
        # self.beta = 2.0 /np.sqrt(2.0*(vcircBeta+1.0)) ### beta = 2 Omega/kappa;   kappa = 2 Omega/beta
        self.nunought = nunought
        self.alpha = alpha
        if not lbpre is None:
            if hasattr(lbpre, 'alpha'):
                self.alpha = lbpre.alpha
        self.rnought = rnought
        self.psi = psir
        self.ordershape = ordershape
        self.ordertime = ordertime
        self.Necc = Necc
        self.nchis = nchis
        self.xCart0 = copy.deepcopy(xCartIn)  # in the global coordinate system
        self.vCart0 = copy.deepcopy(vCartIn)

        # everything internal should be rotated into this frame in such a way that it can be un-rotated!
        self.hvec = np.cross(xCartIn, vCartIn)
        self.hhat = self.hvec / np.sqrt(np.sum(self.hvec * self.hvec))

        v = np.cross(np.array([0, 0, 1.0]), self.hhat)
        sine = np.sqrt(np.sum(v * v))  # wait you don't actually use this lol
        cose = np.dot(self.hhat, np.array([0, 0, 1.0]))

        vcross = np.zeros((3, 3))
        vcross[0, 1] = -v[2]
        vcross[1, 0] = v[2]
        vcross[0, 2] = v[1]
        vcross[2, 0] = -v[1]
        vcross[1, 2] = -v[0]
        vcross[2, 1] = v[0]

        self.zopt = zopt
        tilt = zopt=='tilt'

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
        # z already called z :-)

        u = (x * vx + y * vy) / R
        v = ((x * vy - vx * y) / R)  # used to subtract vcirc but unnecessarily
        # thetadot = (x*vy - vx*y)/(R*R)
        w = vz

        if profile:
            tim = timer()

        # Treat vertical motions with epicyclic approximation (3 hydra heads meme lol)
        self.Ez = 0.5 * (w * w + (nunought * (R / self.rnought) ** (-alpha / 2.0)) ** 2 * z * z)
        # related quantity:
        # https://arxiv.org/pdf/2205.01781.pdf#:~:text=We%20re%2Dexamine%20the%20time,%CF%892q%20under%20various%20regularity%20assumptions.
        self.IzIC = self.Ez / (nunought * (R / self.rnought) ** -(alpha / 2.0))
        self.psiIC = np.arctan2(z * (nunought * (R / self.rnought) ** -(alpha / 2.0)), w)
        # self.phi0 = np.arctan2( -w, z*nu )

        # so here's what's going on:

        self.h = R * v
        self.epsilon = 0.5 * (vCart[0] ** 2 + vCart[1] ** 2) - self.psi(
            R)  # deal with vertical motion separately -- convention is that psi positive

        #        def extrema( r ):
        #            return 2.0*self.epsilon + 2.0*self.psi(r) - self.h*self.h/(r*r)

        def fpp(r, epsi, hi):
            return 2.0 * epsi + 2.0 * self.psi(r) - hi * hi / (r * r), 2.0 * (
                    self.psi.ddr(r) + hi * hi / (r * r * r)), 2.0 * (self.psi.ddr2(r) - hi * hi / (r * r * r * r))

        # approximate rdot^2 (i.e. extrema(r) defined above) as a parabola.
        rcirc = self.h / self.psi.vc(R)
        eff, effprime, effpp = fpp(rcirc, self.epsilon, self.h)
        curvature = -0.5 * effpp
        # peri_zero = rcirc - np.sqrt( eff/curvature )# very rough initial guess under parabola approximation
        # apo_zero = rcirc - np.sqrt( eff/curvature )
        peri_zero = np.min([rcirc / 2.0, R])
        apo_zero = np.max([rcirc * 2.0, R])

        res_peri = scipy.optimize.root_scalar(fpp, args=(self.epsilon, self.h), fprime=True, fprime2=True, x0=peri_zero,
                                              method='halley', rtol=1.0e-8, xtol=1.0e-10)
        res_apo = scipy.optimize.root_scalar(fpp, args=(self.epsilon, self.h), fprime=True, fprime2=True, x0=apo_zero,
                                             method='halley', rtol=1.0e-8, xtol=1.0e-10)

        if res_peri.converged:
            self.peri = res_peri.root
        else:
            pdb.set_trace()

        if res_apo.converged:
            self.apo = res_apo.root
        else:
            pdb.set_trace()


        if profile:
            tim.tick('Find peri apo')

        self.X = self.apo / self.peri

        dr = 0.00001
        self.cRa = self.apo * self.apo * self.apo / (self.h * self.h) * self.psi.ddr(
            self.apo)  # centrifugal ratio at apocenter
        self.cRp = self.peri * self.peri * self.peri / (self.h * self.h) * self.psi.ddr(
            self.peri)  # centrifugal ratio at pericenter

        if not np.isfinite(self.cRa):
            pdb.set_trace()
        if not np.isfinite(self.X):
            pdb.set_trace()
        if not np.isfinite(self.cRp):
            pdb.set_trace()

        self.k = np.log((-self.cRa - 1) / (self.cRp + 1)) / np.log(self.X)

        self.m0sq = 2 * self.k * (1.0 + self.cRp) / (1.0 - self.X ** -self.k) / (emcorr ** 2)
        self.m0 = np.sqrt(self.m0sq)

        self.perU = self.peri ** -self.k
        self.apoU = self.apo ** -self.k

        self.e = (self.perU - self.apoU) / (self.perU + self.apoU)
        if quickreturn:
            return
        self.ubar = 0.5 * (self.perU + self.apoU)
        self.Ubar = 0.5 * (1.0 / self.perU + 1.0 / self.apoU)

        self.ell = self.ubar ** (-1.0 / self.k)

        chi_eval = lbpre.get_chi_arr(nchis)

        

        nuk = 2.0 / self.k - 1.0
        tfac = self.ell * self.ell / (self.h * self.m0 * (1.0 - self.e * self.e) ** (nuk + 0.5)) / tcorr
        #mytfac = self.Ubar ** nuk / (self.h * self.m0 * self.ubar * np.sqrt(1 - self.e * self.e))
        nufac = nunought * tfac * self.Ubar ** (-self.alpha / (2.0 * self.k)) / self.rnought ** (-self.alpha / 2.0)
        if self.ordertime>=0:
            timezeroes = coszeros(self.ordertime)
            wt_arr = np.zeros((self.ordertime, self.ordertime))
            wtzeroes = np.zeros(self.ordertime)  # the values of w[chi] at the zeroes of cos((n+1)chi)
            for i in range(self.ordertime):
                coeffs = np.zeros(self.ordertime)  # coefficient for w0, w1, ... for this zero
                coeffs[0] = 0.5
                for j in np.arange(1, len(coeffs)):
                    coeffs[j] = np.cos(j * timezeroes[i])
                wt_arr[i, :] = coeffs[:] * (self.e * self.e * np.sin(timezeroes[i]) ** 2)

                ui = self.ubar * (1.0 + self.e * np.cos(self.eta_given_chi(timezeroes[i]))) # should never need this.
                ui2 = 1.0 / (0.5 * (1.0 / self.perU + 1.0 / self.apoU) * (1.0 - self.e * np.cos(timezeroes[i])))
                assert np.isclose(ui, ui2)
                wtzeroes[i] = (np.sqrt(self.essq(ui) / self.ess(ui)) - 1.0)
                # pdb.set_trace()

            if profile:
                tim.tick('Set up matrix')

            wt_inv_arr = np.linalg.inv(wt_arr)
            if profile:
                tim.tick('Invert')
            self.wts = np.dot(wt_inv_arr, wtzeroes)
            if profile:
                tim.tick('Dot with zeros')

            if wtwcorrs is None:
                pass
            else:
                self.wts = self.wts + wtwcorrs

            self.wts_padded = list(self.wts) + list([0, 0, 0, 0])
            # now that we know the coefficients we can put together t(chi) and then chi(t).
            # tee needs to be multiplied by l^2/(h*m0*(1-e^2)^(nu+1/2)) before it's in units of time.

            t_terms, nu_terms = lbpre.get_t_terms(self.k, self.e, maxorder=self.ordertime+2, \
                    includeNu=(zopt=='first' or zopt=='zero'), nchis=nchis, Necc=self.Necc )
            if profile:
                tim.tick('Obtain tee terms')

            tee = (1.0 + 0.25 * self.e * self.e * (self.wts_padded[0] - self.wts_padded[2])) * t_terms[0]
            nuu = (1.0 + 0.25 * self.e * self.e * (self.wts_padded[0] - self.wts_padded[2])) * nu_terms[0]
            if profile:
                tim.tick('0th order tee terms')
            if self.ordertime > 0:
                for i in np.arange(1, self.ordertime + 2):
                    prefac = -self.wts_padded[i - 2] + 2 * self.wts_padded[i] - self.wts_padded[i + 2]  # usual case
                    if i == 1:
                        prefac = self.wts_padded[i] - self.wts_padded[
                            i + 2]  # w[i-2] would give you w[-1] or something, but this term should just be zero.
                    prefac = prefac * 0.25 * self.e * self.e
                    tee = tee + prefac * t_terms[i]
                    nuu = nuu + prefac * nu_terms[i]
                    if profile:
                        tim.tick('ith order tee terms ' + str(i))
            if profile:
                tim.tick('set up tee')

            tee = tee.flatten()
            nuu = nuu.flatten()
            #        self.t_of_chi = scipy.interpolate.interp1d( chi_eval, tee * tfac )
            #        self.chi_of_t = scipy.interpolate.interp1d( tee*tfac, chi_eval)

            # nuu * nufac is the integral term in phi(t), evaluated at a sequence of chi's from 0 to 2pi.
            # The following approximate integrals need to include the initial phase, which isn't evaluated unitl later.

            if zopt=='first':
                dchi = chi_eval[1] - chi_eval[0]
                integrands = np.sin(chi_eval) / (1.0 - self.e * np.cos(chi_eval)) * np.cos(2.0 * nuu * nufac)
                to_integrate = scipy.interpolate.CubicSpline(chi_eval, integrands)
                lefts = integrands[:-1]
                rights = integrands[1:]
                self.cosine_integral = np.zeros(len(chi_eval))
                # self.cosine_integral[1:] = np.cumsum((lefts+rights)/2.0 * dchi)
                self.cosine_integral[1:] = np.cumsum(
                    [to_integrate.integrate(chi_eval[k], chi_eval[k + 1]) for k in range(len(chi_eval) - 1)])

                integrands = np.sin(chi_eval) / (1.0 - self.e * np.cos(chi_eval)) * np.sin(2.0 * nuu * nufac)
                to_integrate = scipy.interpolate.CubicSpline(chi_eval, integrands)
                # lefts = integrands[:-1]
                # rights = integrands[1:]
                self.sine_integral = np.zeros(len(chi_eval))
                # self.sine_integral[1:] = np.cumsum( (lefts+rights)/2.0 * dchi )
                self.sine_integral[1:] = np.cumsum(
                    [to_integrate.integrate(chi_eval[k], chi_eval[k + 1]) for k in range(len(chi_eval) - 1)])



            try:
                self.t_of_chi = scipy.interpolate.CubicSpline(chi_eval, tee * tfac)
                self.chi_of_t = scipy.interpolate.CubicSpline(tee * tfac, chi_eval)
                self.nut_of_chi = scipy.interpolate.CubicSpline(chi_eval, nuu * nufac)
                self.nut_of_t = scipy.interpolate.CubicSpline(tee * tfac, nuu * nufac)
                # self.nut_of_chi = scipy.interpolate.CubicSpline( chi_eval, nu2 )
                # self.nut_of_t= scipy.interpolate.CubicSpline( tee*tfac, nu2 )
                if zopt=='first':
                    self.sine_integral_of_chi = scipy.interpolate.CubicSpline(chi_eval, self.sine_integral)
                    self.cosine_integral_of_chi = scipy.interpolate.CubicSpline(chi_eval, self.cosine_integral)
            except:
                print("chi doesn't seem to be monotonic!!")
                # raise ValueError
                pdb.set_trace()
            self.Tr = tee[-1] * tfac
            self.phase_per_Tr = nuu[
                                    -1] * nufac  # the phase of the vertical oscillation advanced by the particle over 1 radial oscillation
            # self.phase_per_Tr = nu2[-1]
        else:
            # exact.
            def to_integrate(chi, dummy):
                ui = self.ubar * (1.0 + self.e * np.cos(self.eta_given_chi(chi))) # should never need this.
                #ui2 = 1.0 / (0.5 * (1.0 / self.perU + 1.0 / self.apoU) * (1.0 - self.e * np.cos(chi)))
                ret = (1.0 - self.e*np.cos(chi))**nuk * np.sqrt(self.essq(ui) / self.ess(ui))
                if np.isnan(ret) or not np.isfinite(ret):
                    return 0.0
                return ret
            chis = lbpre.get_chi_arr(nchis)
            res = scipy.integrate.solve_ivp(to_integrate, [1.0e-6, 2.0 * np.pi + 0.001], [0.0], vectorized=True, \
                    rtol=10**ordertime, atol=1.0e-14, t_eval=chis[1:], method='DOP853')
            ys = np.zeros(nchis)
            if res.success:
                ys[1:] = res.y.flatten()*tfac
                self.chi_of_t = scipy.interpolate.CubicSpline( ys, chi_eval )
                self.t_of_chi = scipy.interpolate.CubicSpline( chi_eval, ys )
                self.Tr = self.t_of_chi(2.0*np.pi)
            else:
                pdb.set_trace()

        if profile:
            tim.tick('set up t/chi interpolators')

        if debug:
            fig, ax = plt.subplots()
            # plot the true wobble function and our approximations to it
            chiArray = np.linspace(0, 2 * np.pi, 1000)
            ui = np.array([self.ubar * (1.0 + self.e * np.cos(self.eta_given_chi(chi))) for chi in chiArray])
            ax.plot(chiArray, np.sqrt(self.essq(ui) / self.ess(ui)) - 1.0, c='k', lw=4, zorder=-10)
            accum = self.e * self.e * np.sin(chiArray) ** 2 * 0.5 * self.wts[0]
            accum2 = 0.25 * self.e * self.e * (self.wts_padded[0] - self.wts_padded[2]) * np.ones(len(chiArray))
            ax.plot(chiArray, accum, c='r')
            ax.plot(chiArray, accum2, c='b')
            for i in np.arange(1, self.ordertime + 2):
                accum = accum + self.e * self.e * np.sin(chiArray) * np.sin(chiArray) * self.wts_padded[i] * np.cos(
                    i * chiArray)
                ax.plot(chiArray, accum, c='r', alpha=i / (self.ordertime + 3), zorder=i)
                prefac = -self.wts_padded[i - 2] + 2 * self.wts_padded[i] - self.wts_padded[i + 2]  # usual case
                if i == 1:
                    prefac = self.wts_padded[i] - self.wts_padded[
                        i + 2]  # w[i-2] would give you w[-1] or something, but this term should just be zero.
                accum2 = accum2 + 0.25 * self.e * self.e * prefac * np.cos(i * chiArray)
                ax.plot(chiArray, accum2, c='b', alpha=i / (self.ordertime + 3), zorder=i + 1, ls='--')

            ax.scatter(timezeroes, wtzeroes)
            plt.savefig('testlb_wobble_' + str(self.ordertime).zfill(2) + '.png', dpi=300)
            plt.close(fig)
            if profile:
                tim.tick('wobble plot')

        # pdb.set_trace()

        # set up perturbation theory coefficients - note that this matrix inversion can be pre-computed for each order
        # instead of for each particle - worry about that later I guess.
        # shapezeroes = lbpre.shapezeroes    #coszeros( self.ordershape )
        Wzeroes = np.zeros(self.ordershape)  # the values of W[eta] at the zeroes of cos((n+1)eta)
        W_inv_arr, shapezeroes = lbpre.invert(self.ordershape)
        for i in range(self.ordershape):
            ui = self.ubar * (1.0 + self.e * np.cos(shapezeroes[i]))
            Wzeroes[i] = (np.sqrt(self.essq(ui) / self.ess(ui)) - 1.0) * self.ubar * self.ubar / (
                    (self.perU - ui) * (ui - self.apoU))

        self.Ws = np.dot(W_inv_arr,
                         Wzeroes)  # The array of W0, W1,... which will be helpful later! Possibly in the next line.

        if wcorrs is None:
            pass
        else:
            assert len(wcorrs) == len(self.Ws)
            self.Ws = self.Ws + wcorrs

        self.Wpadded = np.array(list(self.Ws) + [0, 0, 0, 0])
        if profile:
            tim.tick('evaluate Ws')

        if debug:
            fig, ax = plt.subplots()
            etaArray = np.linspace(0, 2.0 * np.pi, 1000)
            ui = self.ubar * (1.0 + self.e * np.cos(etaArray))
            ax.plot(etaArray,
                    np.sin(etaArray) ** 2 * (np.sqrt(self.essq(ui) / self.ess(ui)) - 1.0) * self.ubar * self.ubar / (
                            (self.perU - ui) * (ui - self.apoU)), c='k', lw=4, zorder=-10)
            # accum = 0.5*self.Ws[0]*np.ones(len(etaArray))
            accum = 0.5 * self.Ws[0] * np.sin(etaArray) ** 2
            ax.plot(etaArray, accum, c='r', lw=1, zorder=0)

            accum2 = 0.25 * (self.Wpadded[0] * (1.0 - 2 * np.cos(2 * etaArray)) + self.Wpadded[1] * (
                    -np.cos(3 * etaArray) + np.cos(etaArray)) + self.Wpadded[2] * (
                                     -1.0 + 2 * np.cos(2 * etaArray) - -np.cos(4 * etaArray)))
            ax.plot(etaArray, accum2, c='green', lw=1, zorder=0)

            accum3 = 0.25 * (self.Wpadded[0] - self.Wpadded[2]) * np.ones(len(etaArray))

            for i in np.arange(1, self.ordershape):
                accum = accum + self.Wpadded[i] * np.cos(i * etaArray) * np.sin(etaArray) ** 2
                ax.plot(etaArray, accum, c='r', lw=1, zorder=i)
                if i > 2:
                    accum2 = accum2 + 0.25 * self.Wpadded[i] * (
                            2.0 * np.cos(i * etaArray) - (1.0 / (i + 2)) * np.cos((i + 2) * etaArray) - (
                            1.0 / (i - 2.0)) * np.cos((i - 2) * etaArray))
                    ax.plot(etaArray, accum2, c='green', lw=1, zorder=i + 1, ls='--')

                prefac = -self.Wpadded[i - 2] + 2 * self.Wpadded[i] - self.Wpadded[i + 2]
                if i == 1:
                    prefac = self.Wpadded[i] - self.Wpadded[i + 2]

                accum3 = accum3 + 0.25 * prefac * np.cos(i * etaArray)
                ax.plot(etaArray, accum3, c='b', lw=1, zorder=i + 2, ls='-.')
            ax.scatter(shapezeroes, Wzeroes * np.sin(shapezeroes) ** 2)
            plt.savefig('testlb_wobbleshape_' + str(self.ordershape).zfill(2) + '.png', dpi=300)
            plt.close(fig)

            if profile:
                tim.tick('another wobble plot')

        ustar = 2.0 / (self.peri ** self.k + self.apo ** self.k)
        self.half_esq_w0 = np.sqrt(self.essq(ustar) / self.ess(ustar)) - 1.0

        nulg = 2.0 / self.k - 1.0  # beware of name collision with midplane density used for vertical oscillations
        zlg = 1.0 / np.sqrt(1 - self.e * self.e)  # z Legendre to distinguish from the cylindrical/cartesian coordinate
        # scipy.special.lpmv(2, nulg, zlg) - series expansion used internally doesn't work for z>1
        dz = zlg * 1.0e-5
        # lpmv = (z*z-1) * ( scipy.special.eval_legendre(nulg,zlg+dz)  - 2 * scipy.special.eval_legendre(nulg,zlg) + scipy.special.eval_legendre(nulg,zlg-dz) )/(dz*dz)
        #        lpmv = -nulg*(nulg-1)*(1.0-zlg)  / (1.0+zlg) * scipy.special.eval_legendre(nulg,zlg) # try something
        #        Tr_old = 2.0*np.pi*self.ell*self.ell/(self.m0*self.h) * (1.0-self.e*self.e)**(-nulg-0.5)*zlg**(-nulg) * scipy.special.eval_legendre(nulg, zlg) * (1.0 + 0.5 * self.half_esq_w0 *(1.0 - lpmv/((nulg+1)*(nulg+2)*scipy.special.eval_legendre(nulg, zlg) ) ) )

        # self.Tr = self.Tr/tcorr # ad hoc to try to debug

        self.V1 = 0.5 * (1 - self.e * self.e) * (
                (1.0 - self.e) ** (1.0 - 2.0 / self.k) + (1 + self.e) ** (1.0 - 2.0 / self.k))
        self.V2 = (1 - self.e * self.e) ** 2 / (2.0 * self.e) * (
                (1.0 - self.e) ** (-2.0 / self.k) - (1 + self.e) ** (-2.0 / self.k)) - self.V1

        self.Ve = (self.V2 / self.V1) * self.e / Vcorr

        # at initial:  u = ubar (1+e cos eta)
        etaIC = np.arccos((R ** -self.k / self.ubar - 1.0) / self.e)
        if u > 0:
            # particle moving outward, from pericenter to apocenter. Since pericenter <-> eta=0, this means we can just take the main arccos branch:
            self.etaIC = etaIC
        else:
            # particle moving inwards, from apocenter to pericenter, so we should take eta>pi.
            self.etaIC = np.pi + (np.pi - etaIC)

        # self.phiIC = (self.etaIC - (1.0/8.0)*self.e*self.e*self.W0tw*np.sin(self.etaIC))/self.em # eq. 2.14 of L-B15
        self.phiIC = self.phi(self.etaIC)  # should probably just call mphi instead of doing this

        # do some quick checks:
        condA = np.isclose(self.ubar * (1.0 + self.e * np.cos(self.etaIC)), R ** -self.k)

        if not condA:
            pdb.set_trace()

        assert condA

        self.thetaIC = theta  # where the particle is in the "global" cylindrical coordinate system. I think we just have to move to add or subtract phiIC and thetaIC where appropriate.

        # wlog I think we can just start the "clock" at 
        # self.chiIC = np.arcsin( np.sqrt(1-self.e*self.e)*np.sin(self.etaIC)/(1.0+self.e*np.cos(self.etaIC)))
        # self.chiIC = np.arctan2( np.sqrt(1-self.e*self.e)*np.sin(self.etaIC)/(1.0+self.e*np.cos(self.etaIC)), (1.0 - (1.0-self.e**2)/(1.0+self.e*np.cos(self.etaIC)))/self.e ) # just try this out for a sec...
        # U/Ubar = 1 - e cos chi => (1 - U/Ubar)/e = cos chi
        chiIC = np.arccos((1.0 - R ** self.k / (0.5 * (1.0 / self.apoU + 1.0 / self.perU))) / self.e)
        if u > 0:
            self.chiIC = chiIC
        else:
            self.chiIC = np.pi + (np.pi - chiIC)
        # assert np.isclose( 1.0-self.e*np.cos(self.chiIC),

        # self.tperiIC = self.Tr/(2.0*np.pi) * ( self.chiIC - self.Ve*np.sin(self.chiIC))
        self.tperiIC = self.t_of_chi(self.chiIC)

        #        tPeri = t + self.tperiIC # time since reference pericenter. When the input t is 0, we want this to be the same as the tPeri we found in init()
        #        nu_t = self.zphase_given_tperi(tPeri) + self.nu_t_0 # nu times t
        #
        #        w = -np.sqrt(2*self.Ez)*np.sin(nu_t)

        # self.nu_t_0  = np.arcsin(-self.vCart0[2]/np.sqrt(2.0*self.Ez)) - self.zphase_given_tperi(self.tperiIC)
        if zopt=='first' or zopt=='zero':
            self.nu_t_0 = np.arctan2(-w, z * self.nu(0)) - self.zphase_given_tperi(self.tperiIC)
        # self.phi0 = np.arctan2( -w, z*nu )

        if zopt=='fourier':
            self.initialize_z_fourier(40, profile=profile)
            if profile:
                tim.tick('initialize z (fourier)')

        if zopt=='integrate':
            self.initialize_z_numerical(rtol=rtolz, atol=atolz, Neval=Nevalz)
            if profile:
                tim.tick('initialize z')


        if np.isnan(self.tperiIC):
            pdb.set_trace()

        if profile:
            tim.tick('Finish up')
            tim.report()

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
        # now ambiguous what R to return... currently returns cylindrical radius in the plane of the orbit.
        r, _, _, _ = self.rphi(t)
        return r

    def vabs(self, t):
        return self.xvabs(t)[1]

    def uvwinclined(self, t):
        # not aligned!
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

        # use cosine formula
        thetarel = np.arccos((rsq + rrefsq - distsq) / (2.0 * np.sqrt(rsq * rrefsq)))

        # need to recover the sign somehow.
        th = np.arctan2(y, x)
        thref = np.arctan2(yref, xref)

        dtheta = th - thref
        ang = np.nan
        if np.isclose(np.abs(dtheta), thetarel, atol=1.0e-6):
            ang = dtheta
        else:
            # dtheta is affected by wrapping so we have to be a bit more careful about using it to assign the sign for thetarel
            # pdb.set_trace()
            if dtheta > np.pi:
                ang = dtheta - 2.0 * np.pi
            elif dtheta < -np.pi:
                ang = dtheta + 2.0 * np.pi

            if not np.isclose(np.abs(ang), thetarel):
                pdb.set_trace()

        resX = r * np.cos(ang) - np.sqrt(rrefsq)
        resY = r * np.sin(ang)

        return resX, resY, z - zref

    def nu(self, t):
        #r, phiabs, rdot, vphi = self.rphi(t)
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
        # the first step is to compute the Fourier components of nu^2.
        if profile:
            tim = timer()
        matr = np.zeros((zorder,zorder))
        coszeroes  = coszeros(zorder)
        for i in range(zorder):
            row = np.zeros(zorder)
            row[0] = 0.5
            js = np.arange(1,zorder)
            row[1:] = np.cos( js*coszeroes[i] )
            matr[i,:] = row[:]

        tPeri = coszeroes * self.Tr/(2.0*np.pi)
        chi = self.chi_given_tperi(tPeri)
        rs = self.r_given_chi(chi)

        # Myr^-2
        nusqs = self.nunought**2 * (rs / self.rnought) ** (-self.alpha )

        # Need to convert nusq to the right units, namely tau-units.
        nusqs = nusqs * (self.Tr/np.pi)**2 # I think this is right..

        if profile:
            tim.tick('set up nusq matrix')
        thetans = np.linalg.inv(matr) @ nusqs # note that this is the same matrix used for computing phi, so if this is expensive we can swap this with a quick call to the precomputer.
        # bns may be off by a factor of 2, e.g. we may need bns = bns/2.0 because of the pi- rather than 2pi- periodic setup in these old papers.
        thetans = thetans/2.0
        if profile:
            tim.tick('invert matrix matrix')

        thetans_padded = np.zeros(8*zorder+1)
        thetans_padded[:zorder] = thetans[:]

        # next, we construct the B_mp matrix: 1's on the diagonal, and theta_{m-p}/(theta_0-4m^2) 
        def get_bmp(bmpsize=2*zorder+1):
            Bmp = np.zeros( (bmpsize, bmpsize) )
            diag = zip( np.arange(bmpsize), np.arange(bmpsize))
            ms, ps = np.meshgrid( np.arange(bmpsize), np.arange(bmpsize), indexing='ij')  # should double-check this probably
            mneg = ms - (bmpsize-1)/2 #zorder
            pneg = ps - (bmpsize-1)/2 #zorder

            ms = ms.flatten()
            ps = ps.flatten()
            mneg = mneg.flatten()
            pneg = pneg.flatten()
            diffs = np.abs(mneg - pneg).astype(int) # because we have a cosine series the negative bns are the same as their positive counterparts.

            vals = thetans_padded[diffs]/(thetans[0] - 4*mneg*mneg) 
            Bmp[ms,ps] = vals

            Bmp[np.arange(bmpsize),np.arange(bmpsize)] = 1.0
            return Bmp
        

        # next, we take the determinant of the B_mp matrix.
        #Bmp = get_bmp()
        #det0 = np.linalg.det(Bmp)

        if profile:
            tim.tick('set up get_bmp')
        Bmp = get_bmp(4*zorder+1)
        if profile:
            tim.tick('run get_bmp')
        det = np.linalg.det(Bmp)
        if profile:
            tim.tick('compute determinant')

        # next, we find mu from said deterimant
        rhs = np.array([-det * np.sin( np.pi/2.0 * np.sqrt(thetans[0]))**2]).astype(complex)
        mu = np.arcsinh(np.sqrt(rhs)) * 2.0/np.pi
        #mu1 = np.arcsinh(np.sqrt(np.array([-det*np.sin(np.pi/2.0 * np.sqrt(thetans[0]))**2]).astype(complex))) * 2.0/np.pi # likely to be complex

        # then we solve the linear equation for b_n (likely need to construct the corresponding matrix first). Keep in mind for both this matrix and the B_mp matrix, the indices used in the literature are symmetric about zero!

        bmatr = np.zeros((4*zorder+1, 4*zorder+1)).astype(complex)


        rowind = np.arange(-2*zorder, 2*zorder+1)
        for i,nn in enumerate(np.arange(-2*zorder, 2*zorder+1)):
            row = thetans_padded[np.abs(rowind-nn)].astype(complex)
            bnfac = (mu+2*nn*1j)**2
            row[i] += bnfac
            row = row/bnfac

            bmatr[i,:] = row[:]
        #qr = np.linalg.qr(bmatr)
        if profile:
            tim.tick('set up bmatr')
        U,ess,Vt = np.linalg.svd(bmatr)
        if profile:
            tim.tick('compute svd of bmatr')
        assert np.argmin(np.abs(ess)) == len(ess)-1
        assert np.min(np.abs(ess))<1.0e-3
        bvec = Vt[-1,:]


        # Here tau is a pi-periodic rescaling of t, i.e. tau = t*pi/Tr
        # so at this point f(x) = e^(mu x) Phi(x) = exp( mu tau) sum( b_n exp(2ni tau)), where n is rowind ^^.
        # The general solution is z = D1 exp( mu tau) sum( b_n exp(2ni tau)) + D2 exp( - mu tau) sum( b_n exp(- 2ni tau))
        # It follows that vz = D1 sum( (mu+2ni) b_n exp((mu+2ni)tau) ) + D2 sum( -(mu+2ni) b_n exp(-(mu+2ni)tau) )
        # As usual we can cast this as a matrix equation (the thing I'm worried about though is given that z and vz are real, I guess D1 and D2 will generally be complex, possibly imaginary; will z remain real? Not sure!

        tau = self.tperiIC * np.pi/self.Tr

        # the following matrix will be used in the equation icmatr @ [D1, D2].T = [z, vz].T
        icmatr = np.zeros((2,2)).astype(complex)
        # D1's prefactor for z
        icmatr[0,0] = np.sum( bvec * np.exp((mu+2*rowind*1j)* tau)) # what's the zero-point of tau btw? I assume it's the same as tPeri.
        # D2's prefactor for z
        icmatr[0,1] = np.sum( bvec * np.exp(-(mu+2*rowind*1j)* tau)) 
        # D1's prefactor for vz
        icmatr[1,0] = np.sum( bvec * (mu+2*rowind*1j) * np.exp((mu+2*rowind*1j)* tau)) # what's the zero-point of tau btw? I assume it's the same as tPeri.
        # D2's prefactor for vz
        icmatr[1,1] = np.sum( bvec *-(mu+2*rowind*1j) * np.exp(-(mu+2*rowind*1j)* tau)) 

        
        ics = np.array([self.xCart0[2],self.vCart0[2]]).reshape((2,1))
        ics[1] = ics[1] * self.Tr/np.pi # v units need to be dz/dtau.

        self.zDs = np.linalg.inv(icmatr) @ ics
        self.bvec = bvec
        self.zrowind = rowind
        self.zmu = mu
        if profile:
            tim.tick('Compute coefficients of general soln')
            tim.report()


        debug = False
        if debug:
            # first initialize a scheme we know works:
            self.initialize_z_numerical()

            ts = np.linspace(0,2*self.Tr,300)
            fig,ax = plt.subplots()
            four = [self.zvz_fourier(t)[0] for t in ts]
            num = [self.zvz_floquet(t)[0] for t in ts]
            ax.plot(ts,num,c='k',label='Numerical-1')
            ax.plot(ts,four,c='r', label='Fourier')
            ax.set_xlabel('t (Myr)')
            ax.axvline(self.Tr)
            ax.set_ylabel('z (pc)')
            ax.legend()
            plt.savefig('dbg_fourierfloquet.png',dpi=300)
            plt.close(fig)
            pdb.set_trace()

        return True

    def zvz_fourier(self, t):

        tau = (t + self.tperiIC)*np.pi/self.Tr
        z = self.zDs[0] * np.sum( self.bvec * np.exp((self.zmu+2*self.zrowind*1j)* tau)) + self.zDs[1] * np.sum( self.bvec * np.exp(-(self.zmu+2*self.zrowind*1j)* tau))
        vz = self.zDs[0] * np.sum( self.bvec * (self.zmu+2*self.zrowind*1j) * np.exp((self.zmu+2*self.zrowind*1j)* tau)) + self.zDs[1] * np.sum( self.bvec *-(self.zmu+2*self.zrowind*1j)* np.exp(-(self.zmu+2*self.zrowind*1j)* tau))

        return z.real.flatten()[0],vz.real.flatten()[0] * np.pi/self.Tr

        # Once we know the b_n, we need to solve for the coefficients of each solution given the ICs. Should be 2 eq's, 2 unknowns.


    def initialize_z_numerical( self, atol=1.0e-8, rtol=1.0e-8, Neval=1000 ):
        def to_integrate(tt, y):
            zz = y[0]
            vz = y[1]

            nu = self.nu(tt)

            res = np.zeros( y.shape )
            res[0] = vz
            res[1] = -zz*nu*nu
            return res
        ic0 = [1.0, 0.0]
        ic1 = [0.0, 1.0]
        ts = np.linspace( 0, self.Tr, Neval )
        res0 = scipy.integrate.solve_ivp( to_integrate, [np.min(ts),np.max(ts)], ic0, t_eval=ts, atol=atol, rtol=rtol, method='DOP853', vectorized=True)
        res1 = scipy.integrate.solve_ivp( to_integrate, [np.min(ts),np.max(ts)], ic1, t_eval=ts, atol=atol, rtol=rtol, method='DOP853', vectorized=True)

        z0s = res0.y[0,:]
        vz0s = res0.y[1,:]
        z1s = res1.y[0,:]
        vz1s = res1.y[1,:]


        self.z0_interp = scipy.interpolate.CubicSpline(  ts, z0s )
        self.z1_interp = scipy.interpolate.CubicSpline(  ts, z1s )
        self.vz0_interp = scipy.interpolate.CubicSpline( ts, vz0s )
        self.vz1_interp = scipy.interpolate.CubicSpline( ts, vz1s )


        self.monodromy = np.zeros((2,2))
        self.monodromy[0,0] = self.z0_interp(self.Tr)
        self.monodromy[1,0] = self.vz0_interp(self.Tr)
        self.monodromy[0,1] = self.z1_interp(self.Tr)
        self.monodromy[1,1] = self.vz1_interp(self.Tr)

        self.zICvec = np.array([self.xCart0[2], self.vCart0[2]]).reshape((2,1))



    def zvz_floquet(self, t):
        texcess = t % self.Tr
        norb = round( (t-texcess)/self.Tr )
        to_mult = np.zeros( (2,2) )
        to_mult[0,0] = self.z0_interp( texcess)
        to_mult[1,0] = self.vz0_interp(texcess)
        to_mult[0,1] = self.z1_interp( texcess)
        to_mult[1,1] = self.vz1_interp(texcess)
        ics = self.zICvec
        ret = to_mult @ np.linalg.matrix_power( self.monodromy, norb ) @ ics
        return ret[0][0], ret[1][0]


    def zvz(self, t):
        if self.zopt=='tilt':
            return 0.0, 0.0

        tPeri = t + self.tperiIC # time since reference pericenter. When the input t is 0, we want this to be the same as the tPeri we found in init()
        nu_now  =self.nu(t)

        # for backwards compatibility
        if hasattr(self,'IzIC'):
            pass
        else:
            r,_,_,_ = self.rphi(0)
            self.IzIC = self.Ez/( self.nunought*(r/self.rnought)**-(self.alpha/2.0) )

        IZ = self.IzIC
        if self.zopt=='zero':
            nu_t = self.zphase_given_tperi(tPeri) + self.nu_t_0 # nu times t THIS is phi(t), eq. 27 of Fiore 2022.
            w = -np.sqrt(2*IZ*nu_now) * np.sin(nu_t)
            z = np.sqrt(2*IZ/nu_now) * np.cos(nu_t)

            # this is all we need for the zeroeth order approximation from Fiore 2022.
            return z,w

        elif self.zopt=='first':

            nu_t = self.zphase_given_tperi(tPeri) + self.nu_t_0 # nu times t THIS is phi(t), eq. 27 of Fiore 2022.
            phiconst = self.nu_t_0
            chi_excess = self.chi_excess_given_tperi(tPeri)

            Norb = self.Norb(tPeri)

            if Norb==0:
                cosine_integral = self.effcos(chi_excess)
                sine_integral = self.effsin(chi_excess)
            else:
                cosine_integral = self.effcos(2*np.pi) \
                        - self.alpha*self.e/(2.0*self.k) * \
                        ( np.cos(2*(self.nu_t_0 + Norb*self.phase_per_Tr))*self.cosine_integral_of_chi(chi_excess) \
                        - np.sin(2*(self.nu_t_0 + Norb*self.phase_per_Tr))*self.sine_integral_of_chi(chi_excess) )
                sine_integral = self.effsin(2*np.pi) \
                        - self.alpha*self.e/(2.0*self.k) * \
                        ( np.sin(2*(self.nu_t_0 + Norb*self.phase_per_Tr))*self.cosine_integral_of_chi(chi_excess) \
                        + np.cos(2*(self.nu_t_0 + Norb*self.phase_per_Tr))*self.sine_integral_of_chi(chi_excess) )

                # so far we've included the initial component and the "current" component, but if Norb>1 there is an additional set of terms from each radial orbit.
                if Norb>1:
                    arrCos = [np.cos(2.0*(self.nu_t_0 + (i+1)*self.phase_per_Tr)) for i in range(Norb-1)]
                    arrSin = [np.sin(2.0*(self.nu_t_0 + (i+1)*self.phase_per_Tr)) for i in range(Norb-1)]
                    to_add_cosine = -self.alpha*self.e/(2.0*self.k) * ( self.cosine_integral_of_chi(2.0*np.pi) * np.sum( arrCos ) -   self.sine_integral_of_chi(2.0*np.pi) * np.sum( arrSin ) )
                    to_add_sine =   -self.alpha*self.e/(2.0*self.k) * (   self.sine_integral_of_chi(2.0*np.pi) * np.sum( arrCos ) + self.cosine_integral_of_chi(2.0*np.pi) * np.sum( arrSin ) )
                    #pdb.set_trace()


                    cosine_integral = cosine_integral + to_add_cosine
                    sine_integral = sine_integral + to_add_sine



            # only necessary for debugging.
            #def to_integrate(tpr):
            #    tperi = tpr + self.tperiIC
            #    nu_tpr = self.zphase_given_tperi(tperi) + self.nu_t_0
            #    return self.nudot(tpr)/self.nu(tpr) * np.cos( 2.0 * nu_tpr )
            
            # compare the numerical integral to the machinations above to approximate it.
            #res = scipy.integrate.quad( to_integrate, 0, t)
            #tbc = cosine_integral  


            nu_t = nu_t  - 0.5*sine_integral 


            IZ = self.IzIC * np.exp( cosine_integral  )

            w = -np.sqrt(2*IZ*nu_now) * np.sin(nu_t)
            z = np.sqrt(2*IZ/nu_now) * np.cos(nu_t)
            return z,w

        elif self.zopt=='integrate':
            return self.zvz_floquet(t)
#            IZ = self.IzIntegrated(t)
#            psi = self.psi(t)
#            w = -np.sqrt(2*IZ*nu_now) * np.sin(psi)
#            z = np.sqrt(2*IZ/nu_now) * np.cos(psi)
#            return z,w 
        elif self.zopt=='fourier':
            return self.zvz_fourier(t)
        else:
            raise Exception("Need to specify an implemented zopt. Options: fourier, integrate, first, zeroeth, tilt")




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

        ret = eta + 0.25 * self.e * self.e * (self.Wpadded[0] - self.Wpadded[2]) * eta
        for i in np.arange(1, self.ordershape + 2):
            prefac = -self.Wpadded[i - 2] + 2 * self.Wpadded[i] - self.Wpadded[i + 2]
            if i == 1:
                prefac = self.Wpadded[i] - self.Wpadded[i + 2]

            ret = ret + 0.25 * prefac * self.e * self.e * (1.0 / i) * np.sin(i * eta)

        # look, this is an approximation to an integral!
        # let's try doing the integral here and see what happens.
        def to_integrate(etaIn):
            ui = self.ubar * (1.0 + self.e * np.cos(etaIn))
            W = (np.sqrt(self.essq(ui) / self.ess(ui)) - 1.0) * self.ubar * self.ubar / (
                    (self.perU - ui) * (ui - self.apoU))
            return 1.0 + self.e * self.e * np.sin(etaIn) * np.sin(etaIn) * W

        return ret / self.m0

    def essq(self, u):
        return self.m0sq * (self.perU - u) * (u - self.apoU)

    def ess(self, u):
        # u = r**(-k) => du/dr = -k r^-(k+1) => r du/dr = -k r^-k = -k u
        r = u ** (-1.0 / self.k)
        return (2.0 * self.epsilon + 2.0 * self.psi(r) - self.h * self.h / (r * r)) * r * r / (self.h * self.h) * (
                u * u * self.k * self.k)

    def t(self, chi):
        return self.Tr / (2.0 * np.pi) * (chi - (self.V2 / self.V1 * self.e * np.sin(chi)))
    def r_given_chi(self, chi):
        return (self.Ubar * (1.0 - self.e*np.cos(chi)))**(1.0/self.k)
    def u_given_chi(self, chi):
        return 1.0/(self.Ubar * (1.0 - self.e*np.cos(chi)))
    def eta_given_chi(self, chi):
        # we need to know where we are in the orbit in order to evaluate r and phi.
        sinchi = np.asarray(np.sin(chi))
        coschi = np.asarray(np.cos(chi))
        sqrte = np.sqrt(1 - self.e * self.e)

        eta_from_arccos = np.arccos(
            ((1.0 - self.e * self.e) / (1.0 - self.e * coschi) - 1.0) / self.e)  # returns a number between 0 and pi
        # hang on if you have this then you don't need to do anything numeric.
        eta_ret = None
        ret = np.where( sinchi>0, eta_from_arccos, 2*np.pi - eta_from_arccos )
        return ret

        # defunct, replaced by vectorized version above
#        if sinchi > 0:
#            # assert np.isclose( to_zero(eta_from_arccos), 0.0)
#            eta_ret = eta_from_arccos
#        else:
#            eta2 = np.pi + (np.pi - eta_from_arccos)
#            # assert np.isclose( to_zero(eta2), 0.0)
#            eta_ret = eta2
#
#        # at this point we are free to add 2pi as we see fit. Given corrections to phi we can't just leave eta between 0 and 2pi!
#        # We need eta and chi to be 1-to-1. 
#        nrot = (chi - (chi % (2.0 * np.pi))) / (2.0 * np.pi)
#        eta_ret = eta_ret + 2 * np.pi * nrot  
#
#        return eta_ret

    #

    def emphi(self, eta):
        ''' m*phi = eta - (1/8)*e^2 W0twiddle * sin(2 eta)'''

        phi = self.phi(eta)

        u = self.ubar * (1.0 + self.e * np.cos(eta))  # ez
        r = u ** (-1.0 / self.k)

        return r, self.m0 * phi
        # r given eta.
        # return self.ell / (1.0 + self.e*np.cos( mphi  + 1.0/8 * self.e*self.e*self.W0tw*np.sin(2*mphi)))**(1.0/self.k), mphi
    def rvectorized(self, t):
        tPeri = t + self.tperiIC  # time since reference pericenter. When the input t is 0, we want this to be the same as the tPeri we found in init()
        chi = self.chi_given_tperi(tPeri)
        return self.r_given_chi(chi)
        eta = self.eta_given_chi(chi)
        u = self.ubar * (1.0 + self.e * np.cos(eta)) 
        r = u ** (-1.0 / self.k)
        return r
        

    def rphi(self, t):
        # This is where we do all the nontrivial stuff. t,eta,chi, and phi are all defined to be zero at pericenter.
        # t throughout this class is time relative to the reference pericenter. The particle has many different passages through r=rPeri, but one close to its initial conditions is chosen for reference. The t input into this function though needs to be adjusted because it's t in our absolute coordinate system. 
        # eta is kind of the mean anomaly - where the particle is along its orbit, u = ubar( 1 + e cos eta), where u = r^-k. Only need values between 0 and 2 pi
        # chi is used in the integral of the time elapsed, so it has a 1-to-1 mapping to t and increases without bound
        # phi is the actual angle the particle is away from (the reference) pericenter

        tPeri = t + self.tperiIC  # time since reference pericenter. When the input t is 0, we want this to be the same as the tPeri we found in init()
        chi = self.chi_given_tperi(tPeri)
        eta = self.eta_given_chi(chi)

        r, mphi = self.emphi(eta)
        phiabs = mphi / self.m0 + (self.thetaIC - self.phiIC)  # mphi/em is the angle to the reference pericenter.

        rdot = np.sqrt(2 * self.epsilon - self.h * self.h / (r * r) + 2.0 * self.psi(r)) * np.sign(
            np.sin(chi))  # this seems to be wrong!
        vphi = self.h / r

        return r, phiabs, rdot, vphi


class timer:
    def __init__(self):
        self.ticks = [time.time()]
        self.labels = []

    def tick(self, label):
        self.ticks.append(time.time())
        self.labels.append(label)

    def timeto(self, label):
        if label in self.labels:
            i = self.labels.index(label)
            return self.ticks[i + 1] - self.ticks[i]
        else:
            return np.nan

    def report(self):
        arr = np.array(self.ticks)
        deltas = arr[1:] - arr[:-1]
        print("Timing report:")
        for i in range(len(self.labels)):
            print(self.labels[i], deltas[i], 100 * deltas[i] / np.sum(deltas), r'%')


def precompute_inverses_up_to(lbpre, maxshapeorder, hardreset=False):
    if not hasattr(lbpre, 'shapezeros') or hardreset:
        lbpre.shapezeros = {}
    if not hasattr(lbpre, 'Warrs') or hardreset:
        lbpre.Warrs = {}

    for shapeorder in tqdm(range(maxshapeorder + 1, -1, -1)):
        W_inv_arr, shapezeroes = lbpre.invert(shapeorder)
        lbpre.Warrs[shapeorder] = W_inv_arr
        lbpre.shapezeros[shapeorder] = shapezeroes
    lbpre.save()


def buildlbpre(nchis=1000, nks=100, Neccs=10, etarget=0.08, psir=logpotential(220.0), shapeorder=100, timeorder=10, alpha=2.2,
               filename=None, vwidth=50):
    if filename == None:
        lbpre = lbprecomputer(timeorder, shapeorder, psir, etarget, nchis, nks, Neccs, alpha, vwidth=vwidth)
        return lbpre
    lbpre = lbprecomputer.load(filename)
    lbpre.add_new_data(1000)
    return lbpre


def coszeros(ordN):
    ''' Finds the first ordN zeros of cos(ordN theta).'''
    # cos x = 0 for x=pi/2 + k pi for k any integer
    # so is it just as simple as...
    theZeros = np.zeros(ordN)
    for i in range(ordN):
        theZeros[i] = (np.pi / 2.0 + i * np.pi) / ordN
    return theZeros


def survey_lb():
    ## quickly get a sense of what values of e and k need to be tabulated.
    xCart0 = np.array([8100.0, 1.0, 0.0])
    vcirc = 220.0
    vCart0 = np.array([1.0, vcirc, 0.0])

    nu = np.sqrt(4.0 * np.pi * G * 1.0)

    ordertime = 10
    ordershape = 10

    nsamp = 10000
    results = np.zeros((nsamp, 3))
    # xArr = np.random.randn( nsamp, 3 ) * 1000
    # vArr = np.random.randn( nsamp, 3 ) * 40
    for i in range(10000):
        vCart = vCart0  # + vArr[i,:]
        rf = np.random.random()
        vCart[1] = rf * 300 + 1.0
        vCart[2] = 0  # for now

        xCart = xCart0  # + xArr[i,:]
        xCart[2] = 0  # for now

        vc = 220 + np.random.randn() * 10

        def psir(r):
            return - vc * vc * np.log(r)

        if not vCart[0] ** 2 + vCart[1] ** 2 + vCart[2] ** 2 > 0:
            pdb.set_trace()
        part = particleLB(xCart, vCart, psir, nu, ordershape=ordershape, ordertime=ordertime)

        results[i, 0] = part.k
        results[i, 1] = part.e
        results[i, 2] = part.X

        if np.any(np.isnan(results[i, :])):
            pdb.set_trace()

    fig = corner.corner(results, labels=['k', 'e', 'X'])
    plt.savefig('testlb_ke_range.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(results[:, 0], results[:, 1], c='k', lw=0, s=2, alpha=0.1)
    ax.set_yscale('log')
    plt.savefig('testlb_ke_scatter.png', dpi=300)
    plt.close(fig)


def rms(arr):
    return np.sqrt( np.mean( arr*arr ) )


class benchmark_groundtruth:
    def __init__(self, ts, xcart, vcart, psir, alpha, nu0):
        self.ts = ts
        self.xcart = copy.deepcopy(xcart)
        self.vcart = copy.deepcopy(vcart)
        self.psir = psir
        self.alpha = alpha
        self.nu0 = nu0
    def run(self):
        ics = np.zeros(6)
        ics[:3] = self.xcart[:]
        ics[3:] = self.vcart[:]

        tprev=self.ts[0]
        self.partarray = np.zeros( (6, len(self.ts) ) )
        self.partarray[ :, 0 ] = ics[:] 


        tim = timer()
        for i,t in enumerate(self.ts):
            if i>0:
                res = scipy.integrate.solve_ivp( particle_ivp2, [tprev,t], self.partarray[:,i-1], vectorized=True, rtol=1.0e-14, atol=1.0e-14, method='DOP853', args=(self.psir, self.alpha, self.nu0))
                self.partarray[:,i] = res.y[:,-1]
                tprev = t
        tim.tick('run')
        self.runtime = tim.timeto('run')


class particle_benchmarker:
    def __init__(self, groundtruth,identifier,args,kwargs):
        self.groundtruth = groundtruth
        self.args = args

        # make sure the ground truth has been run with the same underlying assumptions as we are using to construct the model
        assert self.args[0] == self.groundtruth.psir
        assert self.args[1] == self.groundtruth.nu0

        self.kwargs = kwargs
        self.identifier = identifier
        self.rerrs = {}
        self.phierrs = {}
        self.zerrs = {}
    def initialize_particle(self):
        tim = timer()
        self.part = particleLB(copy.deepcopy(self.groundtruth.xcart), copy.deepcopy(self.groundtruth.vcart), *self.args, **self.kwargs)
        tim.tick('init')
        self.inittime = tim.timeto('init')
    def evaluate_particle(self):
        self.history= np.zeros( (6, len(self.groundtruth.ts) ) )
        tim = timer()
        for i,t in enumerate(self.groundtruth.ts):
            r,phi, rdot, vphi = self.part.rphi(t)
            z,vz = self.part.zvz(t)
            self.history[:,i] = [r, phi, rdot, vphi, z,vz]
        tim.tick('eval')
        self.evaltime = tim.timeto('eval')

    def runtime(self, neval=10):
        return self.inittime + self.evaltime*float(neval)/float(len(self.groundtruth.ts))
    def isparticle(self):
        return True
    def estimate_errors(self, tr, identifier):
        timerangeA = np.argmin(np.abs(tr[0] - self.groundtruth.ts))
        timerangeB = np.argmin(np.abs(tr[1] - self.groundtruth.ts))
        #timerange = [ self.groundtruth.ts[timerangeA], self.groundtruth.ts[timerangeB] ]
        timerange = [timerangeA, timerangeB]
        if tr[1] > np.max(self.groundtruth.ts):
            timerange[1] = None

        #print("timerange: ",timerange)

        resid_r = np.sqrt( self.groundtruth.partarray[0,:]**2+ self.groundtruth.partarray[1,:]**2) - self.history[0,:]
        dphis_zero = (np.arctan2( self.groundtruth.partarray[1,:], self.groundtruth.partarray[0,:]) - self.history[1,:]) % (2.0*np.pi)
        dphis_zero[dphis_zero>np.pi] = dphis_zero[dphis_zero>np.pi] - 2.0*np.pi
        dphis_zero = dphis_zero * self.history[0,:] # unwrap to dy.
        resid_z = self.groundtruth.partarray[2,:] - self.history[4,:]

        self.rerrs[identifier] = rms( resid_r[timerange[0]:timerange[1]] )
        self.phierrs[identifier] = rms( dphis_zero[timerange[0]:timerange[1]] )
        self.zerrs[identifier] = rms( resid_z[timerange[0]:timerange[1]] )


class integration_benchmarker:
    def __init__(self,groundtruth,identifier,args,kwargs):
        self.groundtruth = groundtruth
        self.args = args
        self.kwargs = kwargs
        self.identifier = identifier
        self.rerrs = {}
        self.zerrs = {}
        self.runtimes = {}
        self.herrs = {}
        self.epserrs = {}
        self.apoerrs = {}
        self.perierrs = {}
        self.fixed_time_ages = {}
    def run(self):
        tim = timer()
        #res = scipy.integrate.solve_ivp( particle_ivp2, [0,np.max(self.ts)], ics, vectorized=True, atol=1.0e-10, rtol=1.0e-10, t_eval=ts ) # RK4 vs DOP853 vs BDF.
        ics = np.zeros(6)
        ics[:3] = self.groundtruth.xcart[:]
        ics[3:] = self.groundtruth.vcart[:]
        res = scipy.integrate.solve_ivp(particle_ivp2, [0,np.max(self.groundtruth.ts)], ics, *self.args, args=(self.groundtruth.psir,self.groundtruth.alpha,self.groundtruth.nu0), t_eval=self.groundtruth.ts, **self.kwargs ) # RK4 vs DOP853 vs BDF.
        tim.tick('run')
        self.partarray = res.y
        self.runtim= tim.timeto('run')
    def runtime(self):
        return self.runtim
    def isparticle(self):
        return False
    def errs_at_fixed_time(self, t_target, identifier, rtol=0.2): 
        def to_zero(t):
            self.estimate_errors( [0.5*t, t], 'test' )
            return self.runtimes['test'] - t_target
        if to_zero(20)*to_zero(10000) <=0:
            res = scipy.optimize.root_scalar( to_zero, method='brentq', bracket=[20, 10000], rtol=rtol )
            self.estimate_errors( [0.5*res.root, res.root], identifier )
            self.fixed_time_ages[identifier] = res.root
        elif to_zero(20)>0:
            self.estimate_errors( [0,20], identifier )
            self.fixed_time_ages[identifier] = 20.0 
        else:
            self.estimate_errors( [9000,10000], identifier )
            self.fixed_time_ages[identifier] = 10000.0 
            



        
    def estimate_errors(self, tr, identifier, psir=None, nu0=None):
        timerangeA = np.argmin(np.abs(tr[0] - self.groundtruth.ts))
        timerangeB = np.argmin(np.abs(tr[1] - self.groundtruth.ts))
        timerange = [timerangeA, timerangeB]
        if tr[1] > np.max(self.groundtruth.ts):
            timerange[1] = None

        self.rerrs[identifier] = rms( np.sqrt(self.partarray[0,timerange[0]:timerange[1]]**2 + self.partarray[1,timerange[0]:timerange[1]]**2) - np.sqrt(self.groundtruth.partarray[0,timerange[0]:timerange[1]]**2 + self.groundtruth.partarray[1,timerange[0]:timerange[1]]**2) )
        self.zerrs[identifier] = rms( self.partarray[2,timerange[0]:timerange[1]]-self.groundtruth.partarray[2,timerange[0]:timerange[1]] )

        tim = timer()
        ics = np.zeros(6)
        ics[:3] = self.groundtruth.xcart[:]
        ics[3:] = self.groundtruth.vcart[:]
        res = scipy.integrate.solve_ivp(particle_ivp2, [0,np.max(self.groundtruth.ts[timerange[0]:timerange[1]])], ics, *self.args, args=(self.groundtruth.psir,self.groundtruth.alpha,self.groundtruth.nu0), t_eval=self.groundtruth.ts[timerange[0]:timerange[1]], **self.kwargs ) 
        tim.tick('run')
        self.runtimes[identifier] = tim.timeto('run')

        if not psir is None:
            partStart = particleLB(self.partarray[:3,0], self.partarray[3:,0], psir, nu0, None, quickreturn=True, zopt='integrate')
            partEnd = particleLB(self.partarray[:3,timerange[0]], self.partarray[3:,timerange[0]], psir, nu0, None, quickreturn=True, zopt='integrate')
            self.herrs[identifier] = (partEnd.h - partStart.h)/partStart.h
            self.epserrs[identifier] = (partEnd.epsilon - partStart.epsilon)/partStart.epsilon
            self.apoerrs[identifier] = (partEnd.apo - partStart.apo)/partStart.apo
            self.perierrs[identifier] = (partEnd.peri - partStart.peri)/partStart.peri




def benchmark():
    #psir = logpotential(220.0)
    psir = hernquistpotential(20000, vcirc=220)
    nu0 = np.sqrt(4*np.pi*G*0.2)
    alpha = 2.2
    xCart = np.array([8100.0, 0.0, 21.0])
    vCart = np.array([10.0, 230.0, 10.0])
    ordertime=5
    ordershape=14
    ts = np.linspace( 0, 10000.0, 1000) # 10 Gyr

    #lbpre = lbprecomputer.load( 'big_30_1000_alpha2p2_lbpre.pickle' )
    #lbpre = lbprecomputer.load('big_09_1000_0010_hernquistpotential_scale20000_mass868309528941p9471_alpha2p2_lbpre.pickle')
    #lbpre = lbprecomputer.load('big_06_0300_0010_hernquistpotential_scale20000_mass876185312020p125_alpha2p2_lbpre.pickle')
    lbpre = lbprecomputer.load('big_10_0300_0010_hernquistpotential_scale20000_mass852664632533p8286_alpha2p2_lbpre.pickle')
    #lbpre.generate_interpolators()
    #lbpre.save()

    Npart = 12
    #results = np.zeros((49,Npart))


    argslist = []
    kwargslist = []
    ids = []
    simpleids = []
    colors = []


#    argslist.append( (psir, nu0, lbpre) ) 
#    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':'integrate', 'nchis':100} )
#    ids.append( r'2 Int - $n_\chi=100$' )
#    simpleids.append('zintchi100')
#    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':'fourier', 'nchis':100, 'profile':True} )
    ids.append( r'Fourier - $n_\chi=100$' )
    simpleids.append('zintchi100')
    colors.append('orange')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':0, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot0')
    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':1, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot1')
    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':2, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot2')
    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':3, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot3')
    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':4, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot4')
    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':5, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot5')
    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':6, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot6')
    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':7, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot7')
    colors.append('r')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':8, 'zopt':'integrate', 'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot8')
    colors.append('r')

#    argslist.append( (psir, nu0, lbpre) ) 
#    kwargslist.append( {'ordershape':ordershape, 'ordertime':-5, 'zopt':'fourier', 'nchis':100} )
#    ids.append( r'Fourier - ot-5 - $n_\chi=100$' )
#    simpleids.append('zfourierchi100otm5')
#    colors.append('gray')
#
#    argslist.append( (psir, nu0, lbpre) ) 
#    kwargslist.append( {'ordershape':ordershape, 'ordertime':-10, 'zopt':'fourier', 'nchis':100} )
#    ids.append( r'Fourier - ot-10 - $n_\chi=100$' )
#    simpleids.append('zfourierchi100otm10')
#    colors.append('lightblue')


    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'zopt':'fourier', 'nchis':100} )
    ids.append( r'Fourier - $n_\chi=100$' )
    simpleids.append('zfourierchi100ot8')
    colors.append('yellow')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'zopt':'fourier', 'nchis':300} )
    ids.append( r'Fourier - $n_\chi=1000$' )
    simpleids.append('zfourierchi1000ot8')
    colors.append('green')


    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':'integrate', 'nchis':300} )
    ids.append( r'2 Int - $n_\chi=300$' )
    simpleids.append('zintchi300')
    colors.append('k')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':'integrate', 'nchis':20} )
    ids.append( r'2 Int - $n_\chi=20$' )
    simpleids.append('zintchi20')
    colors.append('pink')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':'first', 'nchis':100} )
    ids.append( r'1st Order - $n_\chi=100$' )
    simpleids.append('fiore1chi20')
    colors.append('maroon')

    argslist.append( (psir, nu0, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':'zero', 'nchis':100} )
    ids.append( r'Fiore 0th - $n_\chi=100$' )
    simpleids.append('fiore0chi100')
    colors.append('blue')

#    argslist.append( () )
#    kwargslist.append( {'vectorized':True, 'atol':1.0e-10, 'rtol':1.0e-10, 'method':'RK45'} )
#    ids.append(r'RK4 $\epsilon\sim 10^{-10}$')
#    simpleids.append('rk4epsm10')
#    colors.append('gray')
#
#    argslist.append( () )
#    kwargslist.append( {'vectorized':True, 'atol':1.0e-10, 'rtol':1.0e-10, 'method':'DOP853'} )
#    ids.append(r'DOP853 $\epsilon\sim 10^{-10}$')
#    simpleids.append('dop853epsm10')
#    colors.append('maroon')
#
#    argslist.append( () )
#    kwargslist.append( {'vectorized':True, 'atol':1.0e-10, 'rtol':1.0e-10, 'method':'BDF'} )
#    ids.append(r'BDF $\epsilon\sim 10^{-10}$')
#    simpleids.append('bdfepsm10')
#    colors.append('darkblue')
#
#    argslist.append( () )
#    kwargslist.append( {'vectorized':True, 'atol':1.0e-5, 'rtol':1.0e-5, 'method':'RK45'} )
#    ids.append(r'RK4 $\epsilon\sim 10^{-5}$')
#    simpleids.append('rk4epsm5')
#    colors.append('orange')

    argslist.append( () )
    kwargslist.append( {'vectorized':True, 'atol':1.0e-5, 'rtol':1.0e-5, 'method':'DOP853'} )
    ids.append(r'DOP853 $\epsilon\sim 10^{-5}$')
    simpleids.append('dop853epsm5')
    colors.append('purple')

    argslist.append( () )
    kwargslist.append( {'vectorized':True, 'atol':1.0e-6, 'rtol':1.0e-6, 'method':'DOP853'} )
    ids.append(r'DOP853 $\epsilon\sim 10^{-6}$')
    simpleids.append('dop853epsm6')
    colors.append('pink')

    argslist.append( () )
    kwargslist.append( {'vectorized':True, 'atol':1.0e-7, 'rtol':1.0e-7, 'method':'DOP853'} )
    ids.append(r'DOP853 $\epsilon\sim 10^{-7}$')
    simpleids.append('dop853epsm7')
    colors.append('red')


    results = np.zeros( (len(argslist),Npart) ,dtype=object) # 10 configurations to test, with Npart orbits.

    for ii in tqdm(range(Npart)):
        stri = str(ii).zfill(3)

        dv = np.random.randn(3)*25.0
        vCartThis = vCart + dv
        
        gt = benchmark_groundtruth( ts, xCart, vCartThis, psir, alpha, nu0 )
        gt.run()

        for j in range(len(argslist)):
            if 'ordershape' in kwargslist[j].keys():
                # these are the options for a particlelb, so we need to use a particle_benchmarker. This logic could probably be put into benchmarker_factory class.
                results[j,ii] = particle_benchmarker(gt,ids[j],argslist[j],kwargslist[j])
                results[j,ii].initialize_particle()
                results[j,ii].evaluate_particle()
                results[j,ii].estimate_errors([9000,10000],'lastgyr')
                results[j,ii].estimate_errors([200,300],'200myr')
                results[j,ii].estimate_errors([900,1000],'1gyr')

                if results[j,ii].rerrs['lastgyr'] > 0.1:

                    pass
#                    part = results[j,ii].part
#                    if part.ordertime==ordertime and part.nchis>30:
#                        # now it is surprising
#                        t_terms, nu_terms = lbpre.get_t_terms(part.k, part.e, maxorder=part.ordertime+2, \
#                                includeNu=False, nchis=part.nchis, Necc=part.Necc, debug=True )

            else:
                results[j,ii] = integration_benchmarker(gt,ids[j],argslist[j],kwargslist[j])
                results[j,ii].run()
                results[j,ii].estimate_errors([9000,10000],'lastgyr', psir=psir,nu0=nu0)
                results[j,ii].estimate_errors([200,300],'200myr')
                results[j,ii].estimate_errors([900,1000],'1gyr', psir=psir,nu0=nu0)
                results[j,ii].errs_at_fixed_time(0.02, 'fixedtime', rtol=0.2)


    alpha = 0.9
    siz = 80
    fig,ax = plt.subplots(figsize=(12,12))
    for j in range(len(argslist)):
        if not results[j,0].isparticle():
            ax.scatter( [results[j,ii].runtimes['lastgyr'] for ii in range(Npart)], [results[j,ii].rerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker='o', lw=1, alpha=alpha, edgecolors='silver' )
            ax.scatter( [results[j,ii].runtimes['200myr'] for ii in range(Npart)], [results[j,ii].rerrs['200myr'] for ii in range(Npart)], c=colors[j], marker='P', lw=1, alpha=alpha, edgecolors='silver' )
            ax.scatter( [results[j,ii].runtimes['1gyr'] for ii in range(Npart)], [results[j,ii].rerrs['1gyr'] for ii in range(Npart)], c=colors[j], marker='s', lw=1, alpha=alpha, edgecolors='silver' )
        else:
            if 'ordertime' in kwargslist[j].keys():
                if kwargslist[j]['ordertime'] == ordertime or kwargslist[j]['ordertime']<0 :
                    ax.scatter( [results[j,ii].runtime() for ii in range(Npart)], [results[j,ii].rerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker='o', lw=1, alpha=alpha,edgecolors='k', s=siz )
                    ax.scatter( [results[j,ii].runtime() for ii in range(Npart)], [results[j,ii].rerrs['200myr'] for ii in range(Npart)], c=colors[j], marker='P', lw=1, alpha=alpha, edgecolors='k', s=siz )
                    ax.scatter( [results[j,ii].runtime() for ii in range(Npart)], [results[j,ii].rerrs['1gyr'] for ii in range(Npart)], c=colors[j], marker='s', lw=1, alpha=alpha, edgecolors='k', s=siz )
    ax.set_xlabel(r'Evaluation Time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'RMS Error in r (pc)')
    ax.legend()
    ax2 = ax.twinx()
    ax2.scatter([0],[0], c='k', marker='o', label='10 Gyr')
    ax2.scatter([0],[0], c='k', marker='s', label='1 Gyr')
    ax2.scatter([0],[0], c='k', marker='P', label='0.2 Gyr')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc=(0.85,0.04))

    ax3 = ax.twinx()
    ax3.scatter([0],[0], c='r', marker='o', edgecolors='silver', label='Integrator')
    ax3.scatter([0],[0], c='r', marker='o', edgecolors='k', label=r'LBParticle', s=siz)
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.03,0.05))

    fig.savefig('benchmark_rt.png', dpi=300)
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(12,12))
    for j in range(len(argslist)):
        if not results[j,0].isparticle():
            ax.scatter( [results[j,ii].runtimes['lastgyr'] for ii in range(Npart)], [results[j,ii].zerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker='o', lw=1, alpha=alpha, edgecolors='silver' )
            ax.scatter( [results[j,ii].runtimes['200myr'] for ii in range(Npart)], [results[j,ii].zerrs['200myr'] for ii in range(Npart)], c=colors[j], marker='P', lw=1, alpha=alpha, edgecolors='silver' )
            ax.scatter( [results[j,ii].runtimes['1gyr'] for ii in range(Npart)], [results[j,ii].zerrs['1gyr'] for ii in range(Npart)], c=colors[j], marker='s', lw=1, alpha=alpha, edgecolors='silver' )
        else:
            if 'ordertime' in kwargslist[j].keys():
                if kwargslist[j]['ordertime'] == ordertime or kwargslist[j]['ordertime']<0 :
                    ax.scatter( [results[j,ii].runtime() for ii in range(Npart)], [results[j,ii].zerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker='o', lw=1, alpha=alpha,edgecolors='k', s=siz )
                    ax.scatter( [results[j,ii].runtime() for ii in range(Npart)], [results[j,ii].zerrs['200myr'] for ii in range(Npart)], c=colors[j], marker='P', lw=1, alpha=alpha, edgecolors='k', s=siz )
                    ax.scatter( [results[j,ii].runtime() for ii in range(Npart)], [results[j,ii].zerrs['1gyr'] for ii in range(Npart)], c=colors[j], marker='s', lw=1, alpha=alpha, edgecolors='k', s=siz )
    ax.set_xlabel(r'Evaluation Time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'RMS Error in z (pc)')
    ax.legend()
    ax2 = ax.twinx()
    ax2.scatter([0],[0], c='k', marker='o', label='10 Gyr')
    ax2.scatter([0],[0], c='k', marker='s', label='1 Gyr')
    ax2.scatter([0],[0], c='k', marker='P', label='0.2 Gyr')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc=(0.85,0.04))

    ax3 = ax.twinx()
    ax3.scatter([0],[0], c='r', marker='o', edgecolors='silver', label='Integrator')
    ax3.scatter([0],[0], c='r', marker='o', edgecolors='k', label=r'LBParticle', s=80)
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.03,0.05))

    fig.savefig('benchmark_zt.png', dpi=300)
    plt.close(fig)


    fig,ax = plt.subplots(figsize=(8,8))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            pass
        else:
            ax.scatter([results[j,ii].zerrs['lastgyr'] for ii in range(Npart)], [np.abs(results[j,ii].herrs['lastgyr']) for ii in range(Npart)], c=colors[j], marker='o', lw=0, alpha=alpha, label=ids[j] )
            ax.scatter([results[j,ii].zerrs['lastgyr'] for ii in range(Npart)], [np.abs(results[j,ii].epserrs['lastgyr']) for ii in range(Npart)], c=colors[j], marker='s', lw=0, alpha=alpha )
            ax.scatter([results[j,ii].zerrs['lastgyr'] for ii in range(Npart)], [np.abs(results[j,ii].apoerrs['lastgyr']) for ii in range(Npart)], c=colors[j], marker='<', lw=0, alpha=alpha )
            ax.scatter([results[j,ii].zerrs['lastgyr'] for ii in range(Npart)], [np.abs(results[j,ii].perierrs['lastgyr']) for ii in range(Npart)], c=colors[j], marker='>', lw=0, alpha=alpha )
    ax.set_xlabel(r'RMS Error in z (pc)')
    ax.set_ylabel(r'Errors in conserved quantities')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right')
    ax2 = ax.twinx()
    ax2.scatter([0],[0], c='k', marker='o', label='h')
    ax2.scatter([0],[0], c='k', marker='s', label=r'$\epsilon$')
    ax2.scatter([0],[0], c='k', marker='>', label='Peri')
    ax2.scatter([0],[0], c='k', marker='<', label='Apo')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='upper left')
    fig.savefig('benchmark_zh.png')
    plt.close(fig)


    fig,ax = plt.subplots(figsize=(8,8))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            pass
        else:
            ax.scatter([results[j,ii].fixed_time_ages['fixedtime'] for ii in range(Npart)], [results[j,ii].zerrs['fixedtime'] for ii in range(Npart)], c=colors[j], label=ids[j], marker='o', lw=0, alpha=alpha)
    ax.set_xlabel('Time Integrated at Fixed Cost (Myr)')
    ax.set_ylabel('RMS Error in z (pc)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    zep = [results[0,ii].zerrs['lastgyr'] for ii in range(Npart)]
    ax.axhline( np.mean(zep)+np.std(zep) )
    ax.axhline( np.mean(zep)-np.std(zep) )
    ax.legend()
    fig.savefig('benchmark_fixedt.png')
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(12,12))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            if 'ordertime' in kwargslist[j].keys():
                if kwargslist[j]['ordertime'] == ordertime or kwargslist[j]['ordertime']<0 :
                    dks = np.array([dist_to_nearest_k(lbpre, results[j,ii].part.k) for ii in range(Npart)]) # typically will be of order 0.0005
                    #sizes = 90+np.log10(sizes)*10
                    if results[j,0].part.nchis == 20:
                        marker='P'
                    elif results[j,0].part.nchis == 100:
                        marker='D'
                    elif results[j,0].part.nchis == 300:
                        marker='s'
                    else:
                        marker='o'
                    sizes = np.array([results[j,ii].part.e for ii in range(Npart)])*100 + 30
                    ax.scatter( dks , [results[j,ii].rerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker=marker, lw=1, alpha=alpha,edgecolors='k', s=sizes )
        else:
            pass
    ax.set_xlabel(r'$\Delta k$')
    ax.set_ylabel('RMS Error in r (pc)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()

    ax3 = ax.twinx()
    ax3.scatter([0],[0], c='k', marker='P', edgecolors='k', label=r'$n_\chi=20$')
    ax3.scatter([0],[0], c='k', marker='D', edgecolors='k', label=r'$n_\chi=100$')
    ax3.scatter([0],[0], c='k', marker='s', edgecolors='k', label=r'$n_\chi=300$')
    ax3.scatter([0],[0], c='k', marker='o', edgecolors='k', label=r'Other')
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.05,0.8))

    fig.savefig('benchmark_ekr.png')
    plt.close(fig)


    fig,ax = plt.subplots(figsize=(12,12))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            if 'ordertime' in kwargslist[j].keys():
                if results[j,ii].part.zopt == 'integrate':
                    des = np.array([lbpre.e_of_k(results[j,ii].part.k) - results[j,ii].part.e for ii in range(Npart)] )
                    sizes =  np.abs(kwargslist[j]['ordertime']) *10 + 40
                    dks = np.array([dist_to_nearest_k(lbpre, results[j,ii].part.k) for ii in range(Npart)]) # typically will be of order 0.0005
                    #sizes = 90+np.log10(sizes)*10
                    if results[j,0].part.nchis == 20:
                        marker='P'
                    elif results[j,0].part.nchis == 100:
                        marker='D'
                    elif results[j,0].part.nchis == 1000:
                        marker='s'
                    else:
                        marker='o'
                    ax.scatter( np.abs(des), [results[j,ii].rerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker=marker, lw=1, alpha=alpha, edgecolors='k', s=sizes)
        else:
            pass
    ax.set_xlabel(r'$|\Delta e|$')
    ax.set_ylabel('RMS Error in r (pc)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()

    ax3 = ax.twinx()
    ax3.scatter([0],[0], c='k', marker='P', edgecolors='k', label=r'$n_\chi=20$')
    ax3.scatter([0],[0], c='k', marker='D', edgecolors='k', label=r'$n_\chi=100$')
    ax3.scatter([0],[0], c='k', marker='s', edgecolors='k', label=r'$n_\chi=1000$')
    ax3.scatter([0],[0], c='k', marker='o', edgecolors='k', label=r'Other')
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.03,0.9))

    fig.savefig('benchmark_der.png')
    plt.close(fig)



    fig,ax = plt.subplots(figsize=(12,12))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            if 'ordertime' in kwargslist[j].keys():
                if results[j,ii].part.zopt == 'integrate':
                    des = np.array([lbpre.e_of_k(results[j,ii].part.k) - results[j,ii].part.e for ii in range(Npart)] )
                    sizes =  np.abs(kwargslist[j]['ordertime']) *10 + 40
                    dks = np.array([dist_to_nearest_k(lbpre, results[j,ii].part.k) for ii in range(Npart)]) # typically will be of order 0.0005
                    #sizes = 90+np.log10(sizes)*10
                    if results[j,0].part.nchis == 20:
                        marker='P'
                    elif results[j,0].part.nchis == 100:
                        marker='D'
                    elif results[j,0].part.nchis == 300:
                        marker='s'
                    else:
                        marker='o'
                    ax.scatter( [kwargslist[j]['ordertime']]*Npart, [results[j,ii].rerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker=marker, lw=1, alpha=alpha, edgecolors='k', s=sizes)
        else:
            pass
    ax.set_xlabel(r'Ordertime')
    ax.set_ylabel('RMS Error in r (pc)')
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.legend()

    ax3 = ax.twinx()
    ax3.scatter([0],[0], c='k', marker='P', edgecolors='k', label=r'$n_\chi=20$')
    ax3.scatter([0],[0], c='k', marker='D', edgecolors='k', label=r'$n_\chi=100$')
    ax3.scatter([0],[0], c='k', marker='s', edgecolors='k', label=r'$n_\chi=1000$')
    ax3.scatter([0],[0], c='k', marker='o', edgecolors='k', label=r'Other')
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.03,0.9))

    fig.savefig('benchmark_order.png')
    plt.close(fig)



def dist_to_nearest_k(lbpre, k):
    i = np.argmin( np.abs(lbpre.ks - k) )
    return np.abs(lbpre.ks[i] - k)

def getPolarFromCartesianXV(xv):
    x = xv[0, :]
    y = xv[1, :]
    z = xv[2, :]
    vx = xv[3, :]
    vy = xv[4, :]
    vz = xv[5, :]

    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    u = (x * vx + y * vy) / r
    v = (x * vy - vx * y) / r

    return np.vstack([r, theta, z, u, v, vz])


def getPolarFromCartesian(xcart, vcart):
    x, y, z = xcart
    vx, vy, vz = vcart

    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    u = (x * vx + y * vy) / r
    v = (x * vy - vx * y) / r

    return (r, theta, z), (u, v, vz)


def getCartesianFromPolar(xpol, vpol):
    r, theta, z = xpol
    u, vincl, w = vpol  # vincl includes the circular velocity

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    vx = u * np.cos(theta) - vincl * np.sin(theta)
    vy = u * np.sin(theta) + vincl * np.cos(theta)

    return (x, y, z), (vx, vy, w)


# handle time/accounting - refer the querier to the correct unperturbedParticle solution, i.e. where the particle is on its epicycle.
class perturbedParticle:
    def __init__(self):
        self.stitchedSolutions = []  # each element i is the particle's trajectory from time ts[i] to ts[i+1]
        self.ts = []

    def add(self, t, part):
        self.ts.append(t)
        self.stitchedSolutions.append(part)

    def exists(self, t):
        # check whether the particle has been produced yet
        return t > self.ts[0]

    def getpart(self, t):
        i = np.searchsorted(self.ts, t, side='left')
        return self.ts[i - 1], self.stitchedSolutions[i - 1]

    def xabs(self, t):
        if self.exists(t):
            tref, part = self.getpart(t)
            return part.xabs(t - tref)
        else:
            assert False




if __name__ == '__main__':
    #lbpre = buildlbpre(timeorder=9, psir=hernquistpotential(20000, vcirc=221), vwidth=200, filename='big_09_1000_0010_hernquistpotential_scale20000_mass868309528941p9471_alpha2p2_lbpre.pickle')
#    lbpre = buildlbpre(timeorder=6, psir=hernquistpotential(20000, vcirc=222), vwidth=200, nchis=300)
#    lbpre.save()
#    lbpre = buildlbpre(timeorder=6, psir=hernquistpotential(20000, vcirc=222), vwidth=200, nchis=300, filename='big_06_0300_0010_hernquistpotential_scale20000_mass876185312020p125_alpha2p2_lbpre.pickle')
#    lbpre.save()
#    lbpre = lbprecomputer.load('big_06_0300_0010_hernquistpotential_scale20000_mass876185312020p125_alpha2p2_lbpre.pickle')
#    lbpre = add_orders(lbpre, 4)
#    lbpre.save()

    #lbpre = buildlbpre(timeorder=10, psir=hernquistpotential(20000, vcirc=219), vwidth=200, nchis=300, nks=10)
    #lbpre.save()

    benchmark()

