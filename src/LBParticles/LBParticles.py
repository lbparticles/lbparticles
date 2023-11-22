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


# I would suggest for units Msun - pc - Myr so that 1 pc/Myr ~ 1 km/s
# caution is warranted about the definition of t. Is this t relative to the most recent perturbation or since the start of the "simulation"?
G= 0.00449987 # pc^3 / (solar mass Myr^2)

class lbprecomputer:
    def __init__(self, timeorder, shapeorder, psir, etarget, nchis, nks, alpha, vwidth=50 ):
        # enumerate possible k-e combinations.
        # compute integrals for a bunch of them.
        self.ordertime=timeorder
        self.ordershape=shapeorder

        self.identifier = 'big_'+str(timeorder).zfill(2)+'_'+str(nchis).zfill(4)+'_alpha'+str(alpha).replace('.','p') # surely timeorder is just the /maximum/ timeorder allowed?

        # do a quick survey of k-e.
        R=8100.0
        eps = 1.0e-8
        #vc = np.sqrt(np.abs(R*(psir(R+eps)-psir(R-eps))/(2.0*eps)))
        self.psir = psir
        vc = self.psir.vc(R)
        self.N = nks
        self.nchis = nchis
        self.vwidth=vwidth
        self.alpha = alpha
        

        # first pass
        self.ks = np.zeros(self.N)
        self.es = np.zeros(self.N)
        vs = np.linspace(vc/10,vc*2, self.N)
        for i in range(self.N):
            xCart = [R,0,0]
            vCart = [1.0, vs[i], 0]
            part = particleLB( xCart, vCart, self.psir, 1.0, None, quickreturn=True)
            self.es[i] = part.e
            self.ks[i] = part.k

        i = np.nanargmin( np.abs(self.es - etarget) )
        vtarget = vs[i]


        # do a second pass targeted on a more useful range of velocities <--> e's <--> k's.
        vs = np.linspace( vtarget-vwidth, vtarget+vwidth, self.N-22)
        vs = np.concatenate( [vs, np.zeros(22)+1000])
        vclose_ind = np.argmin( np.abs(vs-vc) )
        vclose = np.abs( vs[vclose_ind] - vc )
        vs[-11:] = np.linspace( vc - 0.9*vclose, vc + 0.9*vclose, 11 ) # add in more points close to zero eccentricity
        vclose_ind = np.argmin( np.abs(vs) )
        vclose = np.abs( vs[vclose_ind] )
        vs[-22:-11] = np.linspace( vclose/10.0,vclose*0.9,11 )
        for i in range(self.N):
            xCart = [R,0,0]
            vCart = [0.01, vs[i], 0]
            part = particleLB( xCart, vCart, self.psir, 1.0, None, quickreturn=True)
            self.es[i] = part.e
            self.ks[i] = part.k



        self.target_data = np.zeros( (nchis, self.N, timeorder+2) ) # tbd how to interpolate this. rn I'm thinking of doing a 1D interpolation at each chi, then leave the particle to decide how it will interpolate that.

        self.target_data_nuphase = np.zeros( (nchis, self.N, timeorder+2) ) # tbd how to interpolate this. rn I'm thinking of doing a 1D interpolation at each chi, then leave the particle to decide how it will interpolate that.
        


        self.chi_eval = np.linspace(0,2.0*np.pi, self.nchis)
        for j in range(self.N):
            nuk = 2.0/self.ks[j] - 1.0
            for i in range(self.ordertime+2):
                def to_integrate(chi, val):
                    return (1.0 - self.es[j]*np.cos(chi))**nuk * np.cos( i * chi)
                res = scipy.integrate.solve_ivp( to_integrate, [0,2.0*np.pi+0.001], [0], vectorized=True, rtol=1.0e-13, atol=1.0e-14, t_eval=self.chi_eval )
                assert np.all(np.isclose(res.t,self.chi_eval))
                self.target_data[:,j,i] = res.y.flatten()

                def to_integrate(chi, val):
                    return (1.0 - self.es[j]*np.cos(chi))**(nuk-self.alpha/(2.0*self.ks[j])) * np.cos( i * chi)
                res = scipy.integrate.solve_ivp( to_integrate, [0,2.0*np.pi+0.001], [0], vectorized=True, rtol=1.0e-13, atol=1.0e-14, t_eval=self.chi_eval )
                self.target_data_nuphase[:,j,i] = res.y.flatten()
                #t_terms.append(res.y.flatten())


            
        self.generate_interpolators()




        self.save()



    def invert(self, ordershape ):
        # do matrix inversions too, because why not.

        if hasattr(self, 'Warrs') and hasattr(self,'shapezeros'):
            if ordershape in self.Warrs.keys() and ordershape in self.shapezeros.keys():
                return self.Warrs[ordershape], self.shapezeros[ordershape]
        shapezeroes = coszeros( ordershape )
        W_arr = np.zeros( (ordershape, ordershape) )
        for i in range(ordershape):
            coeffs = np.zeros( ordershape ) # coefficient for W0, W1, ... for this zero
            # evaluate equation 4.34 of LB15 to gather relevant W's
            # sin^2 eta W[eta] = (1/4) ( (W0-W2) + (2W1-W3) cos eta + sum_2^infty (2 Wn - W(n-2) - W(n+2)) cos (n eta) )
            coeffs[0] = 0.5
            for j in np.arange(1,len(coeffs)):
                coeffs[j] = np.cos( j*shapezeroes[i] )

            # so these coeffs are for the equation W[first0] = a*W0 + b*W1 + ...
            # We are constructing a system of such equations so that we can solve for W0, W1, ... in terms of the W[zeroes]. So these coefficients are the /rows/ of such a matrix.
            W_arr[i,:] = coeffs[:]
        W_inv_arr = np.linalg.inv( W_arr ) # This matrix when multiplied by W[zeros] should give the W0, W1, ...
        return W_inv_arr, shapezeroes

    def generate_interpolators(self):
        sort_e = np.argsort( self.es )
        sorted_e = self.es[sort_e]
        self.interpolators = np.zeros( (self.ordertime+2, self.nchis), dtype=object )
        self.interpolators_nuphase = np.zeros( (self.ordertime+2, self.nchis), dtype=object )
        for i in range(self.ordertime+2):
            for k in range(self.nchis):

                self.interpolators[i,k] = scipy.interpolate.CubicSpline( sorted_e, self.target_data[k,sort_e,i] ) # not obvious if we should pick es, ks, or do something fancier

                self.interpolators_nuphase[i,k] = scipy.interpolate.CubicSpline( sorted_e, self.target_data_nuphase[k,sort_e,i] ) # not obvious if we should pick es, ks, or do something fancier

    def get_k_given_e(self, ein):
        # very hacky I'm sorry
        xCart0 = [8100.0,0,0]
        vCart0 = [0.0003, 220, 0]
        def to_zero(vin):
            vCart = vCart0[:]
            vCart[1] = vin
            xCart = xCart0[:]
            part = particleLB( xCart, vCart, self.psir, 1.0, None, quickreturn=True)
            
            return part.e - ein

        a = 1.0
        b = 220.0
        if to_zero(a) * to_zero(b)>0:
            print("Initial guess for bounds failed - trying fine sampling")
            trial_x = np.linspace(b-10, b+1, 10000)
            trial_y = np.array([to_zero(trial_x[i]) for i in range(len(trial_x))])
            switches = trial_y[1:]*trial_y[:-1] < 0
            if np.any(switches):
                inds = np.ones(len(trial_x)-1)[switches]
                a = trial_x[inds[0]]
                b = trial_x[inds[0]+1]
            else:
                pdb.set_trace()
            

        res = scipy.optimize.brentq( to_zero, a, b, xtol=1.0e-14)
        vCart = vCart0[:]
        vCart[1] = res
        part = particleLB( xCart0, vCart, self.psir, 1.0, None, quickreturn=True)
        return part.k

    def add_new_data(self, nnew):
        Nstart = len(self.es)

        #new_es = np.random.random(size=nnew)
        new_es = sorted(np.random.beta( 0.9, 2.0, size=nnew) ) # do the hardest ones first
        new_ks = [self.get_k_given_e(new_es[i]) for i in tqdm(range(nnew))]

        self.ks = np.concatenate( (self.ks, new_ks) )
        self.es = np.concatenate( (self.es, new_es) )
        news_shape = np.array(self.target_data.shape)
        news_shape[1] = nnew
        assert news_shape[0] == self.nchis
        assert news_shape[2] == self.ordertime+2

        self.target_data = np.concatenate( (self.target_data, np.zeros(news_shape)), axis=1 )
        self.target_data_nuphase = np.concatenate( (self.target_data_nuphase, np.zeros(news_shape)), axis=1 )


        for j in tqdm(range(Nstart,self.N+nnew)):
            nuk = 2.0/self.ks[j] - 1.0
            for i in range(self.ordertime+2):
                def to_integrate(chi, val):
                    return (1.0 - self.es[j]*np.cos(chi))**nuk * np.cos( i * chi)
                res = scipy.integrate.solve_ivp( to_integrate, [0,2.0*np.pi+0.001], [0], vectorized=True, rtol=1.0e-13, atol=1.0e-14, t_eval=self.chi_eval )
                assert np.all(np.isclose(res.t,self.chi_eval))
                self.target_data[:,j,i] = res.y.flatten()

                def to_integrate(chi, val):
                    return (1.0 - self.es[j]*np.cos(chi))**(nuk - self.alpha/(2.0*self.ks[j])) * np.cos( i * chi)
                res = scipy.integrate.solve_ivp( to_integrate, [0,2.0*np.pi+0.001], [0], vectorized=True, rtol=1.0e-13, atol=1.0e-14, t_eval=self.chi_eval )
                assert np.all(np.isclose(res.t,self.chi_eval))
                self.target_data_nuphase[:,j,i] = res.y.flatten()

        self.N = self.N + nnew
        self.generate_interpolators()

        self.save()

    def save(self):
        with open(self.identifier+'_lbpre.pickle', 'wb') as f:
            pickle.dump(self, f)
    @classmethod
    def load(cls, fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)


    def get_t_terms(self, e, maxorder=None):
        # interpolate target data to the actual k<-->e.
        # note that this will be called (once) by each particle, so ideally it should be reasonably fast.

        if maxorder is None:
            ordermax = self.ordertime+2
        else:
            ordermax = maxorder
        ret = np.zeros( (ordermax, self.nchis) )
        ret_nu = np.zeros( (ordermax, self.nchis) )
        for i in range(ordermax):
            for k in range(self.nchis):
                ret[i,k] = self.interpolators[i,k](e)
                ret_nu[i,k] = self.interpolators_nuphase[i,k](e)
        
        return ret, ret_nu


class logpotential:
    def __init__(self, vcirc):
        self.vcirc = vcirc
    def __call__(self, r):
        return -self.vcirc**2 * np.log(r)
    def Omega(self,r):
        return self.vcirc/r
    def gamma(self,r):
        return np.sqrt(2.0)
    def kappa(self,r):
        return self.Omega(r)*self.gamma(r)
    def vc(self, r):
        return self.vcirc
    def ddr(self, r):
        return -self.vcirc**2 / r
    def ddr2(self, r):
        return self.vcirc**2 / (r*r)


class particleLB:
    # use orbits from Lynden-Bell 2015.
    #def __init__(self, xCart, vCart, vcirc, vcircBeta, nu):
    def __init__(self, xCartIn, vCartIn, psir, nunought, lbpre, rnought=8100.0, ordershape=1, ordertime=1, tcorr=True, emcorr=1.0, Vcorr=1.0, wcorrs=None, wtwcorrs=None, debug=False, quickreturn=False, profile=False, tilt=False, alpha=2.2, adhoc=None):
        self.adhoc=adhoc
        # psir is psi(r), the potential as a function of radius.
        # don't understand things well enough yet, but let's start writing in some of the basic equations

        # primary variables are ra and rp. These are set as follows....
        # 2*epsilon + 2 *psi - h^2 / r^2 = 0 at pericenter and apocenter
        #self.vcircBeta = vcircBeta
        #self.vcirc = vcirc
        #self.beta = 2.0 /np.sqrt(2.0*(vcircBeta+1.0)) ### beta = 2 Omega/kappa;   kappa = 2 Omega/beta
        self.nunought = nunought
        self.alpha = alpha
        if not lbpre is None:
            if hasattr(lbpre, 'alpha'):
                self.alpha = lbpre.alpha
        self.rnought = rnought
        self.psi = psir
        self.ordershape = ordershape
        self.ordertime = ordertime
        self.xCart0 = copy.deepcopy(xCartIn) # in the global coordinate system
        self.vCart0 = copy.deepcopy(vCartIn)

        self.hvec = np.cross( xCartIn, vCartIn ) # everything internal should be rotated into this frame in such a way that it can be un-rotated!
        self.hhat = self.hvec/np.sqrt(np.sum(self.hvec*self.hvec))

        v = np.cross( np.array([0,0,1.0]), self.hhat )
        sine = np.sqrt(np.sum(v*v)) # wait you don't actually use this lol
        cose = np.dot( self.hhat, np.array([0,0,1.0]) )

        vcross = np.zeros((3,3))
        vcross[0,1] = -v[2]
        vcross[1,0] = v[2]
        vcross[0,2] = v[1]
        vcross[2,0] = -v[1]
        vcross[1,2] = -v[0]
        vcross[2,1] = v[0]

        if tilt:
            rot = np.eye(3) + vcross  + vcross@vcross * 1.0/(1.0+cose) 
        else:
            rot = np.eye(3)

        self.rot = Rotation.from_matrix( rot )

        xCart = self.rot.apply(xCartIn,inverse=True)
        vCart = self.rot.apply(vCartIn,inverse=True)

        x,y,z = xCart
        vx,vy,vz = vCart
        
        if tilt:
            assert np.isclose( z,0 )
            assert np.isclose( vz, 0)
    
        R = np.sqrt(x*x + y*y)
        theta = np.arctan2(y,x)
        # z already called z :-)

        u = (x*vx+y*vy)/R
        v = ((x*vy - vx*y)/R ) # used to subtract vcirc but unnecessarily
        #thetadot = (x*vy - vx*y)/(R*R)
        w = vz
        
        if profile:
            tim = timer()


        # Treat vertical motions with epicyclic approximation (3 hydra heads meme lol)
        self.Ez = 0.5 * (w*w + (nunought*(R/self.rnought)**(-alpha/2.0))**2*z*z)
        # related quantity:
        # https://arxiv.org/pdf/2205.01781.pdf#:~:text=We%20re%2Dexamine%20the%20time,%CF%892q%20under%20various%20regularity%20assumptions.
        self.IzIC = self.Ez/( nunought*(R/self.rnought)**-(alpha/2.0) )
        self.psiIC = np.arctan2( z*( nunought*(R/self.rnought)**-(alpha/2.0) ) , w )
        #self.phi0 = np.arctan2( -w, z*nu )


        # so here's what's going on:
        
        self.h = R*v
        self.epsilon = 0.5 * (vCart[0]**2 + vCart[1]**2 )  - self.psi(R) # deal with vertical motion separately -- convention is that psi positive

#        def extrema( r ):
#            return 2.0*self.epsilon + 2.0*self.psi(r) - self.h*self.h/(r*r)

        def fpp( r, epsi, hi ):
            return 2.0*epsi + 2.0*self.psi(r) - hi*hi/(r*r), 2.0*( self.psi.ddr(r) + hi*hi / (r*r*r) ), 2.0*(self.psi.ddr2(r) - hi*hi/(r*r*r*r))

        # approximate rdot^2 (i.e. extrema(r) defined above) as a parabola.
        rcirc = self.h/self.psi.vc(R)
        eff, effprime, effpp = fpp(rcirc, self.epsilon, self.h )
        curvature = -0.5 * effpp
        #peri_zero = rcirc - np.sqrt( eff/curvature )# very rough initial guess under parabola approximation
        #apo_zero = rcirc - np.sqrt( eff/curvature )
        peri_zero = np.min([rcirc/2.0, R])
        apo_zero = np.max([rcirc*2.0, R])

        res_peri = scipy.optimize.root_scalar( fpp, args=(self.epsilon, self.h), fprime=True, fprime2=True, x0=peri_zero, method='halley', rtol=1.0e-8, xtol=1.0e-10)
        res_apo = scipy.optimize.root_scalar( fpp, args=(self.epsilon, self.h), fprime=True, fprime2=True, x0=apo_zero, method='halley', rtol=1.0e-8, xtol=1.0e-10)

        if res_peri.converged:
            self.peri = res_peri.root
        else:
            pdb.set_trace()

        if res_apo.converged:
            self.apo = res_apo.root
        else:
            pdb.set_trace()


        tcorrs = []
        if tcorr is True and not tilt:
            epsi = self.epsilon
            for i in range(10):
                #tcorr = 1.0 - self.Ez/(nunought* (self.peri/self.rnought)**-(alpha/2.0))**2 * 1.0/self.peri**2
                tcorr = 1.0 - self.Ez/(nunought* (R/self.rnought)**-(alpha/2.0))**2 * 1.0/self.peri**2
                tcorrs.append(tcorr)

                #epsi = self.epsilon/tcorr #**0.25
                hi = self.h/tcorr
                #epsi = -( self.apo*self.apo * self.psi(self.apo) - self.peri*self.peri * self.psi(self.peri) ) / (self.apo*self.apo - self.peri*self.peri)  

                #res_peri = scipy.optimize.root_scalar( fpp, args=(epsi,hi), fprime=True, fprime2=True, x0=peri_zero, method='halley', rtol=1.0e-8, xtol=1.0e-10)
                #self.peri = res_peri.root

                #res_apo = scipy.optimize.root_scalar( fpp, args=(epsi, self.h), fprime=True, fprime2=True, x0=apo_zero, method='halley', rtol=1.0e-8, xtol=1.0e-10)
                #self.apo = res_apo.root


            #self.epsilon = self.epsilon #/tcorr**0.25
            #self.h = self.h/tcorr
            #epsi = -( self.apo*self.apo * self.psi(self.apo) - self.peri*self.peri * self.psi(self.peri) ) / (self.apo*self.apo - self.peri*self.peri)  
            #self.epsilon = epsi

            tcorrs = np.array(tcorrs)
            #print("tcorr delta sequence:", tcorrs[1:]-tcorrs[:-1] )

            #if 2.0*self.epsilon + 2.0*self.psi(self.peri) - self.h*self.h/(self.peri*self.peri) <=0:
            #    pdb.set_trace()

            #pdb.set_trace()

        else:
            tcorr = 1.0





#
        if profile:
            tim.tick('Find peri apo')

        self.X = self.apo/self.peri

        dr = 0.00001
        self.cRa = self.apo *self.apo* self.apo /(self.h*self.h) * self.psi.ddr(self.apo)  # centrifugal ratio at apocenter
        self.cRp = self.peri*self.peri*self.peri/(self.h*self.h) * self.psi.ddr(self.peri)  # centrifugal ratio at pericenter

        if not np.isfinite(self.cRa):
            pdb.set_trace()
        if not np.isfinite(self.X):
            pdb.set_trace()
        if not np.isfinite(self.cRp):
            pdb.set_trace()

        self.k = np.log( (-self.cRa -1)/ (self.cRp + 1) )/np.log(self.X)

        self.m0sq = 2*self.k*(1.0+self.cRp)/(1.0 - self.X**-self.k) / (emcorr**2)
        self.m0 = np.sqrt(self.m0sq)

        self.perU = self.peri**-self.k
        self.apoU = self.apo**-self.k

        self.e = (self.perU - self.apoU)/(self.perU+self.apoU)
        if quickreturn:
            return
        self.ubar = 0.5*(self.perU+self.apoU)
        self.Ubar = 0.5*(1.0/self.perU + 1.0/self.apoU)

        self.ell = self.ubar**(-1.0/self.k)

        try:
            chi_eval = lbpre.chi_eval
        except:
            chi_eval = np.linspace(0,2.0*np.pi, lbpre.nchis) # try it out

        timezeroes = coszeros( self.ordertime )
        wt_arr = np.zeros( (self.ordertime, self.ordertime) )
        wtzeroes = np.zeros( self.ordertime) # the values of w[chi] at the zeroes of cos((n+1)chi)
        nuk = 2.0/self.k - 1.0
        for i in range(self.ordertime):
            coeffs = np.zeros(self.ordertime) # coefficient for w0, w1, ... for this zero
            coeffs[0] = 0.5
            for j in np.arange(1,len(coeffs)):
                coeffs[j] = np.cos( j*timezeroes[i] )
            wt_arr[i,:] = coeffs[:] * (self.e*self.e*np.sin(timezeroes[i])**2)

            ui = self.ubar*(1.0 + self.e*np.cos( self.eta_given_chi( timezeroes[i] )) )
            ui2 = 1.0/ (0.5*(1.0/self.perU + 1.0/self.apoU) * (1.0 - self.e*np.cos( timezeroes[i] ) ) )
            assert np.isclose(ui,ui2)
            wtzeroes[i] = (np.sqrt(self.essq(ui)/self.ess(ui)) - 1.0)
            #pdb.set_trace()

        if profile:
            tim.tick('Set up matrix')


        wt_inv_arr = np.linalg.inv( wt_arr )
        if profile:
            tim.tick('Invert')
        self.wts = np.dot( wt_inv_arr, wtzeroes )
        if profile:
            tim.tick('Dot with zeros')

        if wtwcorrs is None:
            pass
        else:
            self.wts = self.wts + wtwcorrs

        self.wts_padded = list(self.wts)+list([0,0,0,0])
        # now that we know the coefficients we can put together t(chi) and then chi(t).
        # tee needs to be multiplied by l^2/(h*m0*(1-e^2)^(nu+1/2)) before it's in units of time.


        t_terms, nu_terms = lbpre.get_t_terms(self.e, maxorder=self.ordertime+2)
        if profile:
            tim.tick('Obtain tee terms')

        tee = (1.0+ 0.25*self.e*self.e*(self.wts_padded[0]-self.wts_padded[2])) * t_terms[0]
        nuu = (1.0+ 0.25*self.e*self.e*(self.wts_padded[0]-self.wts_padded[2])) * nu_terms[0]
        if profile:
            tim.tick('0th order tee terms')
        if self.ordertime>0:
            for i in np.arange(1,self.ordertime+2):
                prefac = -self.wts_padded[i-2] + 2*self.wts_padded[i] - self.wts_padded[i+2]# usual case
                if i==1:
                    prefac = self.wts_padded[i] - self.wts_padded[i+2] # w[i-2] would give you w[-1] or something, but this term should just be zero.
                prefac = prefac*0.25*self.e*self.e
                tee = tee + prefac*t_terms[i]
                nuu = nuu + prefac*nu_terms[i]
                if profile:
                    tim.tick('ith order tee terms '+str(i))
        if profile:
            tim.tick('set up tee')

        tfac = self.ell*self.ell/(self.h * self.m0 * (1.0-self.e*self.e)**(nuk+0.5) )/tcorr
        mytfac = self.Ubar**nuk / (self.h*self.m0 * self.ubar * np.sqrt(1-self.e*self.e))
        nufac = nunought * tfac * self.Ubar**(-self.alpha/(2.0*self.k)) / self.rnought**(-self.alpha/2.0)
        tee = tee.flatten()
        nuu = nuu.flatten()
#        self.t_of_chi = scipy.interpolate.interp1d( chi_eval, tee * tfac )
#        self.chi_of_t = scipy.interpolate.interp1d( tee*tfac, chi_eval)

        # nuu * nufac is the integral term in phi(t), evaluated at a sequence of chi's from 0 to 2pi.
        # The following approximate integrals need to include the initial phase, which isn't evaluated unitl later.


        dchi = chi_eval[1]-chi_eval[0]
        integrands = np.sin(chi_eval)/(1.0-self.e*np.cos(chi_eval)) * np.cos(2.0*nuu*nufac)
        to_integrate = scipy.interpolate.CubicSpline( chi_eval, integrands)
        lefts = integrands[:-1]
        rights = integrands[1:]
        self.cosine_integral = np.zeros(len(chi_eval))
        #self.cosine_integral[1:] = np.cumsum((lefts+rights)/2.0 * dchi)
        self.cosine_integral[1:] = np.cumsum([to_integrate.integrate(chi_eval[k],chi_eval[k+1]) for k in range(len(chi_eval)-1)])


        integrands = np.sin(chi_eval)/(1.0-self.e*np.cos(chi_eval)) * np.sin(2.0*nuu*nufac)
        to_integrate = scipy.interpolate.CubicSpline( chi_eval, integrands)
        #lefts = integrands[:-1]
        #rights = integrands[1:]
        self.sine_integral = np.zeros(len(chi_eval))
        #self.sine_integral[1:] = np.cumsum( (lefts+rights)/2.0 * dchi )
        self.sine_integral[1:] = np.cumsum( [to_integrate.integrate(chi_eval[k],chi_eval[k+1]) for k in range(len(chi_eval)-1)] )


#        dchi = chi_eval[1]-chi_eval[0]
#        self.cosine_integral = np.cumsum( dchi*np.sin(chi_eval)/(1.0-self.e*np.cos(chi_eval)) * np.cos(2.0*nuu*nufac) )
#        self.sine_integral = np.cumsum( dchi*np.sin(chi_eval)/(1.0-self.e*np.cos(chi_eval)) * np.sin(2.0*nuu*nufac) )

        #nu1 = nuu*nufac + self.alpha*self.e/(4.0*self.k) * np.cumsum( dchi * np.sin(chi_eval) / (1.0 - self.e*np.cos(chi_eval)) * np.sin(2.0* nuu*nufac) ) 
        #nu2 = nuu*nufac + self.alpha*self.e/(4.0*self.k) * np.cumsum( dchi * np.sin(chi_eval) / (1.0 - self.e*np.cos(chi_eval)) * np.sin(2.0* nu1) ) 

        try:
            self.t_of_chi = scipy.interpolate.CubicSpline( chi_eval, tee * tfac )
            self.chi_of_t = scipy.interpolate.CubicSpline( tee*tfac, chi_eval)
            self.nut_of_chi = scipy.interpolate.CubicSpline( chi_eval, nuu*nufac )
            self.nut_of_t= scipy.interpolate.CubicSpline( tee*tfac, nuu*nufac )
            #self.nut_of_chi = scipy.interpolate.CubicSpline( chi_eval, nu2 )
            #self.nut_of_t= scipy.interpolate.CubicSpline( tee*tfac, nu2 )
            self.sine_integral_of_chi = scipy.interpolate.CubicSpline( chi_eval, self.sine_integral)
            self.cosine_integral_of_chi = scipy.interpolate.CubicSpline( chi_eval, self.cosine_integral)
        except:
            print("chi doesn't seem to be monotonic!!")
            #raise ValueError
            pdb.set_trace()
        self.Tr = tee[-1]*tfac
        self.phase_per_Tr = nuu[-1] * nufac # the phase of the vertical oscillation advanced by the particle over 1 radial oscillation
        #self.phase_per_Tr = nu2[-1]

        if profile:
            tim.tick('set up t/chi interpolators')



        if debug:
            fig,ax = plt.subplots()
            # plot the true wobble function and our approximations to it
            chiArray = np.linspace(0,2*np.pi,1000)
            ui = np.array([self.ubar*(1.0 + self.e*np.cos( self.eta_given_chi( chi) ) ) for chi in chiArray])
            ax.plot( chiArray, np.sqrt(self.essq(ui)/self.ess(ui)) - 1.0, c='k', lw=4, zorder=-10)
            accum = self.e*self.e * np.sin(chiArray)**2 * 0.5 * self.wts[0]
            accum2 = 0.25*self.e*self.e*(self.wts_padded[0] - self.wts_padded[2])*np.ones(len(chiArray))
            ax.plot( chiArray, accum, c='r' )
            ax.plot( chiArray, accum2, c='b' )
            for i in np.arange(1,self.ordertime+2):
                accum = accum + self.e*self.e * np.sin(chiArray)*np.sin(chiArray) * self.wts_padded[i]*np.cos(i*chiArray)
                ax.plot( chiArray, accum, c='r', alpha=i/(self.ordertime+3), zorder=i)
                prefac = -self.wts_padded[i-2] + 2*self.wts_padded[i] - self.wts_padded[i+2]# usual case
                if i==1:
                    prefac =  self.wts_padded[i] - self.wts_padded[i+2] # w[i-2] would give you w[-1] or something, but this term should just be zero.
                accum2 = accum2 + 0.25*self.e*self.e*prefac*np.cos(i*chiArray)
                ax.plot( chiArray, accum2, c='b', alpha=i/(self.ordertime+3), zorder=i+1, ls='--' )

            ax.scatter( timezeroes, wtzeroes )
            plt.savefig('testlb_wobble_'+str(self.ordertime).zfill(2)+'.png', dpi=300)
            plt.close(fig)
            if profile:
                tim.tick('wobble plot')

        #pdb.set_trace()

        # set up perturbation theory coefficients - note that this matrix inversion can be pre-computed for each order instead of for each particle - worry about that later I guess.
        #shapezeroes = lbpre.shapezeroes    #coszeros( self.ordershape )
        Wzeroes = np.zeros( self.ordershape) # the values of W[eta] at the zeroes of cos((n+1)eta)
        W_inv_arr, shapezeroes = lbpre.invert( self.ordershape )
        for i in range(self.ordershape):

            ui = self.ubar*(1.0 + self.e*np.cos( shapezeroes[i] ) )
            Wzeroes[i] = ( np.sqrt(self.essq(ui)/self.ess(ui)) - 1.0)*self.ubar*self.ubar / ( ( self.perU - ui )*(ui - self.apoU) )


        self.Ws = np.dot( W_inv_arr, Wzeroes) # The array of W0, W1,... which will be helpful later! Possibly in the next line.


        if wcorrs is None:
            pass
        else:
            assert len(wcorrs)==len(self.Ws)
            self.Ws = self.Ws + wcorrs


        self.Wpadded = np.array( list(self.Ws) + [0,0,0,0] )
        if profile:
            tim.tick('evaluate Ws')

        if debug:
            fig,ax = plt.subplots()
            etaArray = np.linspace(0,2.0*np.pi,1000)
            ui = self.ubar*(1.0+self.e*np.cos(etaArray) )
            ax.plot( etaArray, np.sin(etaArray)**2 * (np.sqrt(self.essq(ui)/self.ess(ui))-1.0)*self.ubar*self.ubar/((self.perU - ui)*(ui-self.apoU)), c='k', lw=4, zorder=-10)
            #accum = 0.5*self.Ws[0]*np.ones(len(etaArray))
            accum = 0.5*self.Ws[0]*np.sin(etaArray)**2
            ax.plot(etaArray, accum, c='r', lw=1, zorder=0)


            accum2 = 0.25*( self.Wpadded[0]*(1.0 - 2*np.cos(2*etaArray)) + self.Wpadded[1]*( -np.cos(3*etaArray) + np.cos(etaArray) ) + self.Wpadded[2]*(-1.0 + 2*np.cos(2*etaArray) - -np.cos(4*etaArray)))
            ax.plot(etaArray, accum2, c='green', lw=1, zorder=0)

            accum3 = 0.25*(self.Wpadded[0]-self.Wpadded[2])*np.ones(len(etaArray))

            for i in np.arange(1, self.ordershape):
                accum = accum + self.Wpadded[i]*np.cos(i*etaArray)*np.sin(etaArray)**2
                ax.plot(etaArray, accum, c='r', lw=1, zorder=i)
                if i>2:
                    accum2 = accum2 + 0.25* self.Wpadded[i]*( 2.0 * np.cos(i*etaArray) - (1.0/(i+2))*np.cos((i+2)*etaArray) - (1.0/(i-2.0))*np.cos((i-2)*etaArray))
                    ax.plot(etaArray, accum2, c='green', lw=1, zorder=i+1, ls='--')

                prefac = -self.Wpadded[i-2] + 2*self.Wpadded[i] - self.Wpadded[i+2]
                if i==1:
                    prefac = self.Wpadded[i]-self.Wpadded[i+2]

                accum3 = accum3 + 0.25*prefac*np.cos(i*etaArray)
                ax.plot(etaArray, accum3, c='b', lw=1, zorder=i+2, ls='-.')
            ax.scatter( shapezeroes, Wzeroes*np.sin(shapezeroes)**2 )
            plt.savefig('testlb_wobbleshape_'+str(self.ordershape).zfill(2)+'.png', dpi=300)
            plt.close(fig)

            if profile:
                tim.tick('another wobble plot')
        


        ustar = 2.0/(self.peri**self.k + self.apo**self.k)
        self.half_esq_w0 = np.sqrt( self.essq(ustar)/self.ess(ustar) ) - 1.0


        nulg = 2.0/self.k - 1.0 # beware of name collision with midplane density used for vertical oscillations
        zlg = 1.0/np.sqrt(1-self.e*self.e) # z Legendre to distinguish from the cylindrical/cartesian coordinate
        #scipy.special.lpmv(2, nulg, zlg) - series expansion used internally doesn't work for z>1
        dz = zlg*1.0e-5
        #lpmv = (z*z-1) * ( scipy.special.eval_legendre(nulg,zlg+dz)  - 2 * scipy.special.eval_legendre(nulg,zlg) + scipy.special.eval_legendre(nulg,zlg-dz) )/(dz*dz)
#        lpmv = -nulg*(nulg-1)*(1.0-zlg)  / (1.0+zlg) * scipy.special.eval_legendre(nulg,zlg) # try something
#        Tr_old = 2.0*np.pi*self.ell*self.ell/(self.m0*self.h) * (1.0-self.e*self.e)**(-nulg-0.5)*zlg**(-nulg) * scipy.special.eval_legendre(nulg, zlg) * (1.0 + 0.5 * self.half_esq_w0 *(1.0 - lpmv/((nulg+1)*(nulg+2)*scipy.special.eval_legendre(nulg, zlg) ) ) ) 

        #self.Tr = self.Tr/tcorr # ad hoc to try to debug

        self.V1 = 0.5*(1-self.e*self.e)*( (1.0-self.e)**(1.0-2.0/self.k) + (1+self.e)**(1.0-2.0/self.k))
        self.V2 = (1-self.e*self.e)**2/(2.0*self.e) * ((1.0-self.e)**(-2.0/self.k) - (1+self.e)**(-2.0/self.k)) - self.V1

        self.Ve = (self.V2/self.V1) * self.e / Vcorr

        # at initial:  u = ubar (1+e cos eta)
        etaIC = np.arccos( ( R**-self.k / self.ubar - 1.0)/self.e )
        if u>0:
            # particle moving outward, from pericenter to apocenter. Since pericenter <-> eta=0, this means we can just take the main arccos branch:
            self.etaIC = etaIC
        else:
            # particle moving inwards, from apocenter to pericenter, so we should take eta>pi.
            self.etaIC = np.pi + (np.pi - etaIC)

        #self.phiIC = (self.etaIC - (1.0/8.0)*self.e*self.e*self.W0tw*np.sin(self.etaIC))/self.em # eq. 2.14 of L-B15
        self.phiIC =  self.phi( self.etaIC) # should probably just call mphi instead of doing this

        # do some quick checks:
        condA = np.isclose( self.ubar*(1.0+self.e*np.cos(self.etaIC) ), R**-self.k )

        if not condA:
            pdb.set_trace()

        assert condA


        self.thetaIC = theta # where the particle is in the "global" cylindrical coordinate system. I think we just have to move to add or subtract phiIC and thetaIC where appropriate.

        # wlog I think we can just start the "clock" at 
        #self.chiIC = np.arcsin( np.sqrt(1-self.e*self.e)*np.sin(self.etaIC)/(1.0+self.e*np.cos(self.etaIC)))
        #self.chiIC = np.arctan2( np.sqrt(1-self.e*self.e)*np.sin(self.etaIC)/(1.0+self.e*np.cos(self.etaIC)), (1.0 - (1.0-self.e**2)/(1.0+self.e*np.cos(self.etaIC)))/self.e ) # just try this out for a sec...
        # U/Ubar = 1 - e cos chi => (1 - U/Ubar)/e = cos chi
        chiIC = np.arccos( (1.0 - R**self.k / (0.5*(1.0/self.apoU + 1.0/self.perU) ) )/self.e )
        if u>0:
            self.chiIC = chiIC
        else:
            self.chiIC = np.pi + (np.pi - chiIC)
        #assert np.isclose( 1.0-self.e*np.cos(self.chiIC), 

        #self.tperiIC = self.Tr/(2.0*np.pi) * ( self.chiIC - self.Ve*np.sin(self.chiIC))
        self.tperiIC = self.t_of_chi(self.chiIC)



#        tPeri = t + self.tperiIC # time since reference pericenter. When the input t is 0, we want this to be the same as the tPeri we found in init()
#        nu_t = self.zphase_given_tperi(tPeri) + self.nu_t_0 # nu times t
#
#        w = -np.sqrt(2*self.Ez)*np.sin(nu_t)

        #self.nu_t_0  = np.arcsin(-self.vCart0[2]/np.sqrt(2.0*self.Ez)) - self.zphase_given_tperi(self.tperiIC) 
        self.nu_t_0 = np.arctan2( -w, z*self.nu(0) ) - self.zphase_given_tperi(self.tperiIC)
        #self.phi0 = np.arctan2( -w, z*nu )


        if np.isnan(self.tperiIC):
            pdb.set_trace()

        if profile:
            tim.tick('Finish up')
            tim.report()


    def getpart(self, t):
        return 0.0, self

    def xvinclined(self, t):
        r,phiabs,rdot,vphi = self.rphi(t)
        z,vz = self.zvz(t)
        vx = rdot*np.cos(phiabs) - vphi*np.sin(phiabs)
        vy = rdot*np.sin(phiabs) + vphi*np.cos(phiabs)

        return r*np.cos(phiabs), r*np.sin(phiabs), z, vx,vy,vz

    def xvabs(self, t):
        res = self.xvinclined(t)
        if hasattr(self,'rot'):
            rotated = self.rot.apply( [res[:3], res[3:]], inverse=False )
        else:
            rotated = (res[:3], res[3:])
        return rotated[0], rotated[1]

    def xabs(self, t):
        
        return self.xvabs(t)[0]

    def getR(self, t):
        # now ambiguous what R to return... currently returns cylindrical radius in the plane of the orbit.
        r,_,_,_ = self.rphi(t)
        return r

    def vabs(self, t):
        return self.xvabs(t)[1]

    def uvwinclined(self, t):
        # not aligned!
        r,phiabs,rdot,vphi = self.rphi(t)
        z,vz = self.zvz(t)
        return rdot, vphi, vz


    def xrelPart(self, t, part, tpart):
        x,y,z = self.xabs(t)
        rsq = x*x + y*y
        r = np.sqrt(rsq)

        xref,yref,zref = part.xabs(tpart)
        rrefsq = xref*xref + yref*yref

        distsq = (xref-x)*(xref-x) + (yref-y)*(yref-y)
        
        # use cosine formula
        thetarel = np.arccos( (rsq + rrefsq - distsq) / (2.0*np.sqrt(rsq*rrefsq)) )
        
        # need to recover the sign somehow.
        th= np.arctan2(y,x)
        thref = np.arctan2(yref,xref)

        dtheta = th-thref
        ang = np.nan
        if np.isclose(np.abs(dtheta), thetarel, atol=1.0e-6):
            ang = dtheta
        else:
            # dtheta is affected by wrapping so we have to be a bit more careful about using it to assign the sign for thetarel
            #pdb.set_trace()
            if dtheta>np.pi:
                ang = dtheta-2.0*np.pi
            elif dtheta<-np.pi:
                ang = dtheta+2.0*np.pi

            if not np.isclose(np.abs(ang),thetarel):
                pdb.set_trace()
            

        resX = r*np.cos(ang) - np.sqrt(rrefsq)
        resY = r*np.sin(ang)

        return resX,resY, z-zref

    def nu(self, t):
        r,phiabs,rdot,vphi = self.rphi(t)
        return self.nunought * (r/self.rnought)**(-self.alpha/2.0)

    def Norb(self, t):
        past_peri = t % self.Tr
        NOrb = (t-past_peri)/self.Tr
        return int(NOrb)

    def effcos(self,chi):
        return -self.alpha*self.e/(2.0*self.k) * (np.cos(2.0*self.nu_t_0)*(self.cosine_integral_of_chi(chi)-self.cosine_integral_of_chi(self.chiIC)) - np.sin(2.0*self.nu_t_0)*(self.sine_integral_of_chi(chi)-self.sine_integral_of_chi(self.chiIC)) )
    def effsin(self,chi):
        return -self.alpha*self.e/(2.0*self.k) * (np.sin(2.0*self.nu_t_0)*(self.cosine_integral_of_chi(chi)-self.cosine_integral_of_chi(self.chiIC)) + np.cos(2.0*self.nu_t_0)*(self.sine_integral_of_chi(chi)-self.sine_integral_of_chi(self.chiIC)) )


    def zvz(self, t):
        tPeri = t + self.tperiIC # time since reference pericenter. When the input t is 0, we want this to be the same as the tPeri we found in init()
        nu_t = self.zphase_given_tperi(tPeri) + self.nu_t_0 # nu times t THIS is phi(t), eq. 27 of Fiore 2022.

        #phiconst = self.phiconst(t)
        chi_excess = self.chi_excess_given_tperi(tPeri)

        Norb = self.Norb(tPeri)
        #Norb = self.Norb(t)
        #cosine_integral = self.cosine_integral_of_chi(chi) + self.adhoc[1]*Norb*self.cosine_integral_of_chi(2.0*np.pi) - self.adhoc[2]*self.cosine_integral_of_chi(self.chiIC)
        #sine_integral = self.sine_integral_of_chi(chi) + self.adhoc[4]* Norb*self.sine_integral_of_chi(2.0*np.pi) - self.adhoc[5]*self.sine_integral_of_chi(self.chiIC)

        if Norb==0:
            cosine_integral = self.effcos(chi_excess)
            sine_integral = self.effsin(chi_excess)
        else:
            cosine_integral = self.effcos(2*np.pi) - self.alpha*self.e/(2.0*self.k) * ( np.cos(2*(self.nu_t_0 + Norb*self.phase_per_Tr))*self.cosine_integral_of_chi(chi_excess) - np.sin(2*(self.nu_t_0 + Norb*self.phase_per_Tr))*self.sine_integral_of_chi(chi_excess) )
            sine_integral = self.effsin(2*np.pi) - self.alpha*self.e/(2.0*self.k) * ( np.sin(2*(self.nu_t_0 + Norb*self.phase_per_Tr))*self.cosine_integral_of_chi(chi_excess) + np.cos(2*(self.nu_t_0 + Norb*self.phase_per_Tr))*self.sine_integral_of_chi(chi_excess) )

            # so far we've included the initial component and the "current" component, but if Norb>1 there is an additional set of terms from each radial orbit.
            if Norb>1:
                arrCos = [np.cos(2.0*(self.nu_t_0 + (i+1)*self.phase_per_Tr)) for i in range(Norb-1)]
                arrSin = [np.sin(2.0*(self.nu_t_0 + (i+1)*self.phase_per_Tr)) for i in range(Norb-1)]
                to_add_cosine = -self.alpha*self.e/(2.0*self.k) * ( self.cosine_integral_of_chi(2.0*np.pi) * np.sum( arrCos ) -   self.sine_integral_of_chi(2.0*np.pi) * np.sum( arrSin ) )
                to_add_sine =   -self.alpha*self.e/(2.0*self.k) * (   self.sine_integral_of_chi(2.0*np.pi) * np.sum( arrCos ) + self.cosine_integral_of_chi(2.0*np.pi) * np.sum( arrSin ) )
                #pdb.set_trace()


                cosine_integral = cosine_integral + to_add_cosine
                sine_integral = sine_integral + to_add_sine


        #cosine_integral =  self.cosine_integral_of_chi(chi) + (self.adhoc[4]+Norb)*self.cosine_integral_of_chi(2.0*np.pi) - self.cosine_integral_of_chi(self.chiIC)
        #sine_integral =  self.sine_integral_of_chi(chi) +  (self.adhoc[4]+Norb)*self.sine_integral_of_chi(2.0*np.pi) - self.sine_integral_of_chi(self.chiIC)

        def to_integrate(tpr):
            tperi = tpr + self.tperiIC
            nu_tpr = self.zphase_given_tperi(tperi) + self.nu_t_0
            return self.nudot(tpr)/self.nu(tpr) * np.cos( 2.0 * nu_tpr )
        
        #res = scipy.integrate.quad( to_integrate, 0, t)
#        #tbc = -( cosine_integral * 0.5*np.cos(2.0*phiconst)*self.alpha*self.e/self.k - 0.5*sine_integral * np.sin(2.0*phiconst)*self.alpha*self.e/self.k )
        #tbc = cosine_integral  
#        if Norb>0:
#            pdb.set_trace()


        #nu_t = nu_t + 0.25*np.sin(2.0*phiconst)*self.alpha*self.e/self.k * cosine_integral + 0.25*np.cos(2.0*phiconst)*self.alpha*self.e/self.k * sine_integral
        #nu_t = nu_t + self.adhoc[0] * ( 0.25*np.cos(2.0*phiconst)*self.alpha*self.e/self.k * cosine_integral - 0.25*np.sin(2.0*phiconst)*self.alpha*self.e/self.k * sine_integral) + self.adhoc[1]* ( 0.25*np.sin(2.0*phiconst)*self.alpha*self.e/self.k * cosine_integral +  0.25*np.cos(2.0*phiconst)*self.alpha*self.e/self.k * sine_integral )
        #nu_t = nu_t + self.adhoc[0] * 0.5 * cosine_integral - self.adhoc[1]*0.5*sine_integral 
        nu_t = nu_t  - 0.5*sine_integral 

        nu_now  =self.nu(t)


        if hasattr(self,'IzIC'):
            pass
        else:
            r,_,_,_ = self.rphi(0)
            self.IzIC = self.Ez/( self.nunought*(r/self.rnought)**-(self.alpha/2.0) )

        #IZ = self.IzIC * np.exp( -np.cos(2.0*phiconst) * cosine_integral * self.alpha*self.e/(2.0*self.k) + np.sin(2.0*phiconst) * sine_integral * self.alpha*self.e/(2.0*self.k) )
        #IZ = self.IzIC * np.exp( -np.sin(2.0*phiconst) * cosine_integral * self.alpha*self.e/(2.0*self.k) - np.cos(2.0*phiconst) * sine_integral * self.alpha*self.e/(2.0*self.k) )
        #IZ = self.IzIC  * np.exp( np.sin(2.0*phiconst) * cosine_integral * self.alpha*self.e/(2.0*self.k) + np.cos(2.0*phiconst) * sine_integral * self.alpha*self.e/(2.0*self.k) )
        #IZ = self.IzIC  * np.exp( np.sin(2.0*phiconst) * cosine_integral * self.alpha*self.e/(2.0*self.k) + np.cos(2.0*phiconst) * sine_integral * self.alpha*self.e/(2.0*self.k) )
        #IZ = self.IzIC * np.exp( self.adhoc[2]*cosine_integral + self.adhoc[3]*sine_integral )
        IZ = self.IzIC * np.exp( cosine_integral  )

        w = -np.sqrt(2*IZ*nu_now) * np.sin(nu_t)
        z = np.sqrt(2*IZ/nu_now) * np.cos(nu_t)
        return z,w #, res[0], tbc


    def zphase_given_tperi(self, t):
        past_peri = t%self.Tr
        return (t-past_peri)/self.Tr * self.phase_per_Tr + self.nut_of_t(past_peri)

    def chi_excess_given_tperi(self, t):
        past_peri = t % self.Tr
        return self.chi_of_t(past_peri)

    def chi_given_tperi(self, t):

        past_peri = t % self.Tr
        return (t-past_peri)/self.Tr * 2.0*np.pi + self.chi_of_t(past_peri)


        assert not np.isnan(t)
        def to_zero(chi):
            return (self.Tr/(2.0*np.pi)) * (chi - self.Ve*np.sin(chi) ) - t
        def to_zero_prime(chi):
            return (self.Tr/(2.0*np.pi)) * (1.0 - self.Ve*np.cos(chi) )
        res = scipy.optimize.newton( to_zero, t*2*np.pi/self.Tr, fprime=to_zero_prime, tol=1.0e-12 )
        return res


    def phi(self, eta):


        ret = eta + 0.25*self.e*self.e*( self.Wpadded[0] - self.Wpadded[2] ) * eta
        for i in np.arange(1,self.ordershape+2):
            prefac = -self.Wpadded[i-2] + 2*self.Wpadded[i] - self.Wpadded[i+2]
            if i==1:
                prefac = self.Wpadded[i]-self.Wpadded[i+2]

            ret = ret + 0.25*prefac*self.e*self.e*(1.0/i)*np.sin(i*eta)

        # look, this is an approximation to an integral!
        # let's try doing the integral here and see what happens.
        def to_integrate(etaIn):
            ui = self.ubar * (1.0 + self.e*np.cos(etaIn))
            W = (np.sqrt(self.essq(ui)/self.ess(ui)) - 1.0)*self.ubar*self.ubar / ((self.perU-ui)*(ui - self.apoU))
            return 1.0+self.e*self.e*np.sin(etaIn)*np.sin(etaIn)*W

        return ret/self.m0


    def essq(self, u):
        return self.m0sq*(self.perU-u)*(u-self.apoU)

    def ess(self, u):
        # u = r**(-k) => du/dr = -k r^-(k+1) => r du/dr = -k r^-k = -k u
        r = u**(-1.0/self.k)
        return (2.0*self.epsilon + 2.0*self.psi(r) - self.h*self.h/(r*r)) * r*r / (self.h*self.h) * (u*u*self.k*self.k)

    def t(self, chi):
        return self.Tr/(2.0*np.pi) * (chi - (self.V2/self.V1 * self.e * np.sin(chi) ) )

    def eta_given_chi(self, chi):
        # we need to know where we are in the orbit in order to evaluate r and phi.
        sinchi = np.sin(chi)
        coschi = np.cos(chi)
        sqrte = np.sqrt(1-self.e*self.e)

        eta_from_arccos = np.arccos(  ((1.0-self.e*self.e)/(1.0-self.e*coschi) - 1.0)/self.e ) # returns a number between 0 and pi
        # hang on if you have this then you don't need to do anything numeric.
        eta_ret = None
        if sinchi>0:
            #assert np.isclose( to_zero(eta_from_arccos), 0.0)
            eta_ret = eta_from_arccos
        else:
            eta2 = np.pi + (np.pi - eta_from_arccos)
            #assert np.isclose( to_zero(eta2), 0.0)
            eta_ret = eta2

        # at this point we are free to add 2pi as we see fit. Given corrections to phi we can't just leave eta between 0 and 2pi!
        # We need eta and chi to be 1-to-1. 
        nrot =  (chi - (chi % (2.0*np.pi))) / (2.0*np.pi)
        eta_ret = eta_ret + 2*np.pi*nrot # mayybe?

        return eta_ret

#

    def emphi(self, eta):
        ''' m*phi = eta - (1/8)*e^2 W0twiddle * sin(2 eta)'''

        phi = self.phi(eta)

        u = self.ubar* (1.0 + self.e* np.cos(eta)) # ez
        r = u**(-1.0/self.k)

        return r, self.m0*phi
        # r given eta.
        #return self.ell / (1.0 + self.e*np.cos( mphi  + 1.0/8 * self.e*self.e*self.W0tw*np.sin(2*mphi)))**(1.0/self.k), mphi

    def rphi(self, t):
        # This is where we do all the nontrivial stuff. t,eta,chi, and phi are all defined to be zero at pericenter.
        # t throughout this class is time relative to the reference pericenter. The particle has many different passages through r=rPeri, but one close to its initial conditions is chosen for reference. The t input into this function though needs to be adjusted because it's t in our absolute coordinate system. 
        # eta is kind of the mean anomaly - where the particle is along its orbit, u = ubar( 1 + e cos eta), where u = r^-k. Only need values between 0 and 2 pi
        # chi is used in the integral of the time elapsed, so it has a 1-to-1 mapping to t and increases without bound
        # phi is the actual angle the particle is away from (the reference) pericenter

        tPeri = t + self.tperiIC # time since reference pericenter. When the input t is 0, we want this to be the same as the tPeri we found in init()
        chi = self.chi_given_tperi(tPeri)
        eta = self.eta_given_chi( chi )

        
        r,mphi = self.emphi(eta)
        phiabs = mphi/self.m0 + (self.thetaIC - self.phiIC) # mphi/em is the angle to the reference pericenter. 

        rdot = np.sqrt(2*self.epsilon - self.h*self.h/(r*r) + 2.0*self.psi(r)) * np.sign( np.sin(chi) ) # this seems to be wrong!
        vphi = self.h/r

        return r,phiabs, rdot,vphi


class timer:
    def __init__(self):
        self.ticks = [time.time()]
        self.labels = []
    def tick(self,label):
        self.ticks.append(time.time())
        self.labels.append(label)
    def timeto(self, label):
        if label in self.labels:
            i = self.labels.index(label)
            return self.ticks[i+1]-self.ticks[i]
        else:
            return np.nan
    def report(self):
        arr = np.array(self.ticks)
        deltas = arr[1:]-arr[:-1]
        print("Timing report:")
        for i in range(len(self.labels)):
            print(self.labels[i], deltas[i], 100*deltas[i]/np.sum(deltas),r'%')


def precompute_inverses_up_to(lbpre, maxshapeorder, hardreset=False):
    if not hasattr(lbpre, 'shapezeros') or hardreset:
        lbpre.shapezeros = {}
    if not hasattr(lbpre, 'Warrs') or hardreset:
        lbpre.Warrs = {}

    for shapeorder in tqdm(range(maxshapeorder+1,-1,-1)):
        W_inv_arr, shapezeroes = lbpre.invert(shapeorder)
        lbpre.Warrs[shapeorder] = W_inv_arr
        lbpre.shapezeros[shapeorder] = shapezeroes
    lbpre.save()


def buildlbpre(nchis = 1000, nks = 100, etarget = 0.08, psir = logpotential(220.0), shapeorder = 100, timeorder=10,alpha=2.2,filename=None):
    if filename == None:    
        lbpre = lbprecomputer( timeorder, shapeorder, psir, etarget, nchis, nks, alpha, vwidth=20 )
        return 0
    lbpre = lbprecomputer.load(filename)
    lbpre.add_new_data( 1000 )
    return 0


def coszeros(ordN):
    ''' Finds the first ordN zeros of cos(ordN theta).'''
    # cos x = 0 for x=pi/2 + k pi for k any integer
    # so is it just as simple as...
    theZeros = np.zeros(ordN)
    for i in range(ordN):
        theZeros[i] = (np.pi/2.0 + i*np.pi)/ordN
    return theZeros


def survey_lb():
    ## quickly get a sense of what values of e and k need to be tabulated.
    xCart0 = np.array([8100.0, 1.0, 0.0])
    vcirc = 220.0
    vCart0 = np.array([1.0, vcirc, 0.0])



    nu = np.sqrt( 4.0*np.pi*G * 1.0 )

    ordertime = 10
    ordershape = 10

    nsamp = 10000
    results = np.zeros( (nsamp, 3) )
    #xArr = np.random.randn( nsamp, 3 ) * 1000
    #vArr = np.random.randn( nsamp, 3 ) * 40
    for i in range(10000):
        vCart = vCart0 #+ vArr[i,:]
        rf = np.random.random()
        vCart[1] = rf*300+1.0
        vCart[2] = 0 # for now

        xCart = xCart0 # + xArr[i,:]
        xCart[2] = 0 # for now

        vc = 220 + np.random.randn()*10
        def psir(r):
            return - vc*vc* np.log(r)


        if not vCart[0]**2 + vCart[1]**2 + vCart[2]**2 > 0:
            pdb.set_trace()
        part = particleLB(xCart, vCart, psir, nu, ordershape=ordershape, ordertime=ordertime)

        results[i,0] = part.k
        results[i,1] = part.e
        results[i,2] = part.X

        if np.any(np.isnan(results[i,:])):
            pdb.set_trace()

    fig = corner.corner(results, labels=['k','e','X'])
    plt.savefig('testlb_ke_range.png', dpi=300)
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.scatter( results[:,0], results[:,1], c='k', lw=0, s=2, alpha=0.1)
    ax.set_yscale('log')
    plt.savefig('testlb_ke_scatter.png', dpi=300)
    plt.close(fig)


def benchmark():
    psir = logpotential(220.0)
    nu = np.sqrt( 4.0*np.pi*G*0.2 )

    xCart = np.array([8100.0, 0.0, 21.0])
    vCart = np.array([10.0, 230.0, 10.0])

    ordertime=8
    ordershape=8
    tim = timer()



    # things to vary: orders, nchi, (nk, velocity distance from progenitor)
    # things to check: error in r (purely dependent on time treatment!)
    # error in phi (may depend on ordershape in addition to time errors)
    # also record time so we can meaingfully trade expense with accuracy.

    partQuick = particleLB(xCart, vCart, psir, nu, None, quickreturn=True)
    tim.tick('Initialize progenitor')
#    lbpre = lbprecomputer(ordertime, ordershape, psir, partQuick.e, 1000, 1000, vwidth=10 )
#    tim.tick('precomputer')
#    part = particleLB(xCart, vCart, psir, nu, lbpre, ordershape=ordershape, ordertime=ordertime, profile=True )
#    tim.tick('full initialize with profiler')


    lbpre = lbprecomputer.load( 'big_10_1000_lbpre.pickle' )
    tim.tick('load precomputer')

    #precompute_inverses_up_to(lbpre, 1001)
    #tim.tick('precompute inverses')

    print('****')
    tim.report()

    Nboot = 100
    results = np.zeros( (14, Nboot*100) )
    for ii in tqdm(range(Nboot)):
        timmy = timer()
        #nchi = int( 10**(np.random.random()*2+1) ) # minimum of 10 maximum of 10^4?
        nchi = 1000
        #ordershape = int(np.random.random() * 7 +1)
        ordershape = int( 10** (1+np.random.random()*2) )
        ordertime = int( np.random.random() * 7 +1)
        #nk = int( 10**(np.random.random()*1.5+1.5) )
        nk = lbpre.N
        
        #print("computing lbpre with orders: ",ordertime,ordershape," and nchi and nk: ",nchi,nk)
        #lbpre = lbprecomputer(ordertime, ordershape, psir, partQuick.e, nchi, nk, vwidth=50 )
        timmy.tick('lbpre')


        # subsample velocities so we don't spend 1000% of our time recomputing lbpre


        # one thing I'd like to do here is compare /relative/ errors. In principle 
        for j in range(100): 
            dvcart = np.random.randn(3)*10.0

            if j==0:
                dvcart = np.zeros(3) # try something 

            vCartThis = vCart + dvcart



            part = particleLB(xCart, vCartThis, psir, nu, lbpre, ordershape=ordershape, ordertime=ordertime)
            timmy.tick('initialize'+str(j))


            ts = np.linspace(0, 10000.0, 1000) # 10 Gyr
            tprev=ts[0]
            partarray = np.zeros( (6, len(ts) ) )
            partarray[ :, 0 ] = [ xCart[0], xCart[1], 0.0, vCartThis[0], vCartThis[1], 0.0 ]

            # integrate
            for i,t in enumerate(ts):
                if i>0:
                    res = scipy.integrate.solve_ivp( particle_ivp, [tprev,t], partarray[:,i-1], vectorized=True, rtol=1.0e-12, atol=1.0e-8 )
                    partarray[:,i] = res.y[:,-1]
                    tprev = t
            timmy.tick('integrate'+str(j))


            history = np.zeros( (4, len(ts) ) )
            for i,t in enumerate(ts):
                r,phi, rdot, vphi = part.rphi(t)
                history[:,i] = [r, phi, rdot, vphi]
            timmy.tick('evaluate'+str(j))


            resid_r = np.sqrt( partarray[0,:]**2+ partarray[1,:]**2) - history[0,:]

            dphis = (np.arctan2( partarray[1,:], partarray[0,:]) - history[1,:]) % (2.0*np.pi)
            dphis[dphis>np.pi] = dphis[dphis>np.pi] - 2.0*np.pi
            dphis = dphis * history[0,:] # unwrap to dy.


            results[0,ii*100+j] = ordertime
            results[1,ii*100+j] = ordershape
            results[2,ii*100+j] = nchi
            results[3,ii*100+j] = nk
            results[4,ii*100+j] = dvcart[0]
            results[5,ii*100+j] = dvcart[1]
            results[6,ii*100+j] = dvcart[2]
            results[7,ii*100+j] = part.e
            results[8,ii*100+j] = np.sqrt(np.mean(resid_r[-100:]**2))
            results[9,ii*100+j] = np.sqrt(np.mean(dphis[-100:]**2))
            results[10,ii*100+j] = timmy.timeto('initialize'+str(j))
            results[11,ii*100+j] = timmy.timeto('integrate'+str(j))
            results[12,ii*100+j] = timmy.timeto('evaluate'+str(j)) # keep in mind this is 100 evaluations, not just one
            results[13,ii*100+j] = np.sqrt(np.sum(dvcart*dvcart)) #timmy.timeto('lbpre')

            if results[8,ii*100+j] > 3.0:
                # huuuuge error or so I thought.
                #pdb.set_trace()
                #pass # yikes!
                print('rms r:', results[8,ii*100+j])

        # save at intermediate steps too
        np.savetxt('benchmark.txt', results[:,:ii*100+j].T, header='ordertime,ordershape,nchi,nk,dvx,dvy,dvz,e,rmsR,rmsPhi,timetoInitialize,timetoIntegrate,timetoEvaluate,vmag')

        timmy.report() # have to see what the heck is taking so long


def getPolarFromCartesianXV( xv ):
    x = xv[0,:]
    y = xv[1,:]
    z = xv[2,:]
    vx = xv[3,:]
    vy = xv[4,:]
    vz = xv[5,:]

    r = np.sqrt(x*x+y*y)
    theta = np.arctan2(y,x)
    u = (x*vx + y*vy)/r
    v = (x*vy- vx*y)/r

    return np.vstack([r,theta,z,u,v,vz]) 


def getPolarFromCartesian( xcart, vcart ):
    x,y,z = xcart
    vx,vy,vz = vcart

    r = np.sqrt(x*x+y*y)
    theta = np.arctan2(y,x)
    u = (x*vx + y*vy)/r
    v = (x*vy- vx*y)/r

    return (r,theta,z), (u,v,vz)


def getCartesianFromPolar( xpol, vpol ):
    r, theta, z = xpol
    u,vincl,w = vpol # vincl includes the circular velocity

    x = r*np.cos(theta)
    y = r*np.sin(theta)
    vx = u*np.cos(theta) - vincl*np.sin(theta)
    vy = u*np.sin(theta) + vincl*np.cos(theta)

    return (x,y,z), (vx,vy,w)


# handle time/accounting - refer the querier to the correct unperturbedParticle solution, i.e. where the particle is on its epicycle.
class perturbedParticle:
    def __init__(self):
        self.stitchedSolutions = [] # each element i is the particle's trajectory from time ts[i] to ts[i+1]
        self.ts = []
    def add(self, t, part):
        self.ts.append( t )
        self.stitchedSolutions.append( part )
    def exists(self, t):
        # check whether the particle has been produced yet
        return t>self.ts[0]
    def getpart(self, t):
        i = np.searchsorted( self.ts, t, side='left' ) 
        return self.ts[i-1], self.stitchedSolutions[i-1]
    def xabs(self, t):
        if self.exists(t):
            tref, part = self.getpart(t)
            return part.xabs(t-tref)
        else:
            assert False


def findClosestApproach( orb1, orb2, tmin ):
    # Find the /first/ local minimum in the distance between orb1 and orb2 /after/ tmin.
    # orb1 and orb2 
    def dist2(t):
        assert t>=tmin
        trefOrb1, solnOrb1 = orb1.getpart( t )
        trefOrb2, solnOrb2 = orb2.getpart( t )

        xcart1 = np.array(solnOrb1.xabs(t-trefOrb1))
        xcart2 = np.array(solnOrb2.xabs(t-trefOrb2))
        return np.sum((xcart1-xcart2)**2)

    # this may be the stupidest algorithm I've ever implemented

    ts = np.linspace( tmin+0.001, tmin + 1.0/orb1.stitchedSolutions[0].Omega + 1.0/orb2.stitchedSolutions[0].Omega, 1000)
    ds = np.array(ts.shape)
    for i,t in enumerate(ts):
        ds[i] = dist2(t)

    ii = np.argmin(ds)
    assert ii!=0 and ii!=len(ts)-1 # pathological cases that I'm sure we'll encounter

    res = scipy.optimize.minimize_scalar( dist2, bracket=(ts[ii-1], ts[ii+1]) )
    return res.x


def applyPerturbation( orbISO, orbStar, tPerturb, mstar ):
    # Assume tPerturb is the relevant time of closest approach. The ISO, whose 'perturbedParticle' orbit
    
    trefISO, solnISO = orbISO.getpart( tPerturb )
    trefStar, solnStar = orbStar.getpart( tPerturb )

    xcartStar = solnStar.xabs(tPerturb-trefStar)
    vcartStar = solnStar.vabs(tPerturb-trefStar)

    xcartISO = solnISO.xabs(tPerturb-trefISO)
    vcartISO = np.array(solnISO.vabs(tPerturb-trefISO))

    # initial
    vrel = np.array(vcartISO) - np.array(vcartStar)
    xrel = np.array(xcartISO) - np.array(xcartStar)

    b = np.sqrt( np.sum( xrel**2 ) )
    V0 = np.sqrt( np.sum( vrel**2 ) )


    dvperp = 2*b*V0**3 / (G*mstar) * 1.0/( 1.0 + b*b*V0**4/(G*G*mstar*mstar))
    dvpar = 2.0*V0 / ( 1.0 + b*b*V0**4/(G*G*mstar*mstar))
    
    vnew = vcartISO - dvpar * vrel/V0  - dvperp * xrel/b

    return vnew


if __name__=='__main__':
    benchmark()
    



