from lbparticles import Precomputer, Particle, HernquistPotential, VertOptionEnum, PotentialWrapper, galpyPotential, galpyFreq, numericalFreqDeriv

import time
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import matplotlib.cm
import scipy.integrate
import scipy.optimize
import numpy as np
from tqdm import tqdm as tqdm
import copy

G = 0.00449987


def rms(arr):
    return np.sqrt( np.mean( arr*arr ) )



def particle_ivp2(t, y, psir):
    # y assumed to be a 6 x N particle array of positions and velocities.
    xx = y[0,:]
    yy = y[1,:]
    zz = y[2,:]
    vx = y[3,:]
    vy = y[4,:]
    vz = y[5,:]

    r = np.sqrt(xx*xx + yy*yy)
    g = psir.ddr(r)
    #nu = nu0 * (r/8100.0)**(-alpha/2.0)
    nu = psir.nu(r)

    res = np.zeros( y.shape )
    res[0,:] = vx
    res[1,:] = vy
    res[2,:] = vz
    res[3,:] = g*xx/r
    res[4,:] = g*yy/r
    res[5,:] = - zz*nu*nu


    return res

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

class benchmark_groundtruth:
    def __init__(self, ts, xcart, vcart, psir):
        self.ts = ts
        self.xcart = copy.deepcopy(xcart)
        self.vcart = copy.deepcopy(vcart)
        self.psir = psir
    def run(self):
        ics = np.zeros(6)
        ics[:3] = self.xcart[:]
        ics[3:] = self.vcart[:]

        tprev=self.ts[0]
        self.partarray = np.zeros( (6, len(self.ts) ) )
        self.partarray[ :, 0 ] = ics[:] 


        tim = timer()

        res = scipy.integrate.solve_ivp( particle_ivp2, [np.min(self.ts),np.max(self.ts)], self.partarray[:,0], vectorized=True, rtol=2.3e-13, atol=1.0e-11, method='DOP853', args=(self.psir,), t_eval=self.ts)
        self.partarray[:,:] = res.y[:,:]

#        for i,t in enumerate(self.ts):
#            if i>0:
#                res = scipy.integrate.solve_ivp( particle_ivp2, [tprev,t], self.partarray[:,i-1], vectorized=True, rtol=2.3e-13, atol=1.0e-11, method='DOP853', args=(self.psir,))
#                self.partarray[:,i] = res.y[:,-1]
#                tprev = t
        tim.tick('run')
        self.runtime = tim.timeto('run')

class particle_benchmarker:
    def __init__(self, groundtruth,identifier,args,kwargs):
        self.groundtruth = groundtruth
        self.args = args

        # make sure the ground truth has been run with the same underlying assumptions as we are using to construct the model
        assert self.args[0] == self.groundtruth.psir
        #assert self.args[1] == self.groundtruth.nu0

        self.kwargs = kwargs
        self.identifier = identifier
        self.rerrs = {}
        self.phierrs = {}
        self.zerrs = {}
        self.zmax = np.max(np.abs(groundtruth.partarray[2,:])) # rough estimate of the maximum height the particle reaches.
    def initialize_particle(self):
        tim = timer()
        self.part = Particle(copy.deepcopy(self.groundtruth.xcart), copy.deepcopy(self.groundtruth.vcart), *self.args, **self.kwargs)
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
        res = scipy.integrate.solve_ivp(particle_ivp2, [0,np.max(self.groundtruth.ts)], ics, *self.args, args=(self.groundtruth.psir,), t_eval=self.groundtruth.ts, **self.kwargs ) # RK4 vs DOP853 vs BDF.
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
        res = scipy.integrate.solve_ivp(particle_ivp2, [0,np.max(self.groundtruth.ts[timerange[0]:timerange[1]])], ics, *self.args, args=(self.groundtruth.psir,), t_eval=self.groundtruth.ts[timerange[0]:timerange[1]], **self.kwargs ) 
        tim.tick('run')
        self.runtimes[identifier] = tim.timeto('run')

        if not psir is None:
            partStart = Particle(self.partarray[:3,0], self.partarray[3:,0], psir, None, quickreturn=True, zopt='integrate')
            partEnd = Particle(self.partarray[:3,timerange[0]], self.partarray[3:,timerange[0]], psir, None, quickreturn=True, zopt='integrate')
            self.herrs[identifier] = (partEnd.h - partStart.h)/partStart.h
            self.epserrs[identifier] = (partEnd.epsilon - partStart.epsilon)/partStart.epsilon
            self.apoerrs[identifier] = (partEnd.apo - partStart.apo)/partStart.apo
            self.perierrs[identifier] = (partEnd.peri - partStart.peri)/partStart.peri







def benchmark2():
    import galpy.potential
    import astropy.units as units
    mw2014 = galpy.potential.MWPotential2014
    psirr = galpyPotential(mw2014, units, galpy.potential)
    nu = galpyFreq(mw2014, units, galpy.potential)
    dlnnudr = numericalFreqDeriv(nu)
    print("potential set up 1")
    psir = PotentialWrapper( psirr, nur=nu, dlnnudr=dlnnudr)
    print("potential set up 2")
    xCart = np.array([8100.0, 0.0, 21.0])
    vCart = np.array([0.0, 219.0, 10.0])
    #ordertime=5
    #ordershape=14
    ts = np.linspace( 0, 10000.0, 1000) # 10 Gyr

    lbpre = Precomputer.load('../big_10_1000_alpha2p2_lbpre.pickle')
    print("done precomputing")

    Npart = 40

    argslist = []
    kwargslist = []
    ids = []
    simpleids = []
    colors = []
    markers=[]


    # Things to vary:
    # nchis -- number of points in the interpolation
    # ordertime
    # ordershape


    nchis_default = 300
    ordertime_default = 8
    ordershape_default = 15

    nchis_var = np.array([ 10, 30, 100, 300, 1000])
    ordertimes_var = np.arange(1,10,1)
    ordershapes_var = np.array([1,3,10,31,100])

    cmap_chi_base = plt.get_cmap('Purples')
    cmap_chi = mpcolors.LinearSegmentedColormap.from_list('cmap_chi', cmap_chi_base(np.linspace(0.1,1.0,len(nchis_var))), N=len(nchis_var) ) 

    cmap_ordertimes_base = plt.get_cmap('Blues')
    cmap_ordertimes = mpcolors.LinearSegmentedColormap.from_list('cmap_ordertimes', cmap_ordertimes_base(np.linspace(0.1,1.0,len(ordertimes_var))), N=len(ordertimes_var) ) 

    cmap_ordershapes_base = plt.get_cmap('Reds')
    cmap_ordershapes = mpcolors.LinearSegmentedColormap.from_list('cmap_ordershapes', cmap_ordershapes_base(np.linspace(0.1,1.0,len(ordershapes_var))), N=len(ordershapes_var) ) 

    

    for i,nchi in enumerate(nchis_var):
        argslist.append( (psir, lbpre) ) 
        kwargslist.append( {'ordershape':ordershape_default, 'ordertime':ordertime_default, 'zopt':VertOptionEnum.FOURIER, 'nchis':nchi} )
        ids.append( r'$n_\chi='+str(nchi)+r'$' )
        simpleids.append('nchi'+str(nchi))
        colors.append(cmap_chi((0.5+float(i))/(len(nchis_var))))
        markers.append( 's' )

    for i,ordertime in enumerate(ordertimes_var):
        argslist.append( (psir, lbpre) ) 
        kwargslist.append( {'ordershape':ordershape_default, 'ordertime':ordertime, 'zopt':VertOptionEnum.FOURIER, 'nchis':nchis_default} )
        ids.append( r'$N_\mathrm{time}='+str(ordertime)+r'$' )
        simpleids.append('ordertime'+str(ordertime))
        colors.append(cmap_ordertimes((0.5+float(i))/(len(ordertimes_var))))
        markers.append( 'o' )

    for i,ordershape in enumerate(ordershapes_var):
        argslist.append( (psir, lbpre) ) 
        kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime_default, 'zopt':VertOptionEnum.FOURIER, 'nchis':nchis_default} )
        ids.append( r'$N_\mathrm{shape}='+str(ordershape)+r'$' )
        simpleids.append('ordershape'+str(ordershape))
        colors.append(cmap_ordershapes((0.5+float(i))/(len(ordershapes_var))))
        markers.append( 'd' )

    import pdb




    results = np.zeros( (len(argslist),Npart) ,dtype=object) # 10 configurations to test, with Npart orbits.
    dvphis = np.linspace( - 200.0, 0.0, Npart)

    for ii in tqdm(range(Npart)):
        stri = str(ii).zfill(3)

        #dv = np.random.randn(3)*25.0
        dv = np.array([1,dvphis[ii],0])
        vCartThis = vCart + dv
        
        gt = benchmark_groundtruth( ts, xCart, vCartThis, psir )
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
                results[j,ii].estimate_errors([9000,10000],'lastgyr', psir=psir)
                results[j,ii].estimate_errors([200,300],'200myr')
                results[j,ii].estimate_errors([900,1000],'1gyr', psir=psir)
                results[j,ii].errs_at_fixed_time(0.025, 'fixedtime', rtol=0.2)



    fig,ax = plt.subplots(ncols=3, figsize=(9,5.))
    for j in range(len(argslist)):
        print("eccentricities: ", [results[j,ii].part.e for ii in range(Npart)] )
        x= np.array([results[j,ii].part.e for ii in range(Npart)])
        y = [results[j,ii].rerrs['lastgyr']/(results[j,ii].part.apo-results[j,ii].part.peri) for ii in range(Npart)]
        ax[0].plot( x, y, c=colors[j], label=ids[j], marker=markers[j], lw=1, alpha=0.95, markeredgecolor='k' )
        ax[1].plot( x, [results[j,ii].phierrs['lastgyr']/(results[j,ii].part.apo-results[j,ii].part.peri) for ii in range(Npart)], c=colors[j], label=ids[j], marker=markers[j], lw=1, alpha=0.95, markeredgecolor='k' )
        ax[2].plot( x, [results[j,ii].zerrs['lastgyr']/results[j,ii].zmax for ii in range(Npart)], c=colors[j], label=ids[j], marker=markers[j], lw=1, alpha=0.95, markeredgecolor='k' )

    ax[0].set_xlabel('e')
    ax[1].set_xlabel('e')
    ax[2].set_xlabel('e')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    ax[0].set_ylim(1.0e-11, 10.0)
    ax[1].set_ylim(1.0e-11, 10.0)
    ax[2].set_ylim(1.0e-11, 10.0)
    ax[0].set_ylabel(r'$\mathrm{RMS}(r-r_\mathrm{truth})/(r_\mathrm{apo}-r_\mathrm{peri})$')
    ax[1].set_ylabel(r'$\mathrm{RMS}(r\Delta\phi)/(r_\mathrm{apo}-r_\mathrm{peri})$')
    ax[2].set_ylabel(r'$\mathrm{RMS}(z-z_\mathrm{truth})/z_\mathrm{max}$')
    ax[0].set_title('r')
    ax[1].set_title(r'$\phi$')
    ax[2].set_title('z')


    fig.subplots_adjust(bottom=0.42, wspace=0.48)
    cax1 = fig.add_axes([0.1, 0.25, 0.77, 0.02])
    cax2 = fig.add_axes([0.1, 0.16, 0.77, 0.02])
    cax3 = fig.add_axes([0.1, 0.07, 0.77, 0.02])

    sc1 = cax1.scatter([1.05],[0.5],c=[0.7], vmin=0, vmax=1, marker='s', edgecolors='k', cmap=cmap_chi, clip_on=False)
    cbar1 = plt.colorbar( sc1, cax=cax1, orientation='horizontal') 
    cbar1.set_ticks( (np.arange(len(nchis_var))+0.5)/float(len(nchis_var)-1+0.5+0.5), labels=[str(nchi) for nchi in nchis_var] )
    #cbar1.set_label(r'$N_\chi$', x=1.065, y=0)
    cax1.text(1.065, 0.5, r'$N_\chi$', va='center')

    sc2 = cax2.scatter([1.05],[0.5],c=[0.7], vmin=0, vmax=1, marker='o', edgecolors='k',cmap=cmap_ordertimes, clip_on=False)
    cbar2 = plt.colorbar( sc2, cax=cax2, orientation='horizontal') 
    cbar2.set_ticks( (np.arange(len(ordertimes_var))+0.5)/float(len(ordertimes_var)-1+0.5+0.5), labels=[str(ordertime) for ordertime in ordertimes_var] )
    #cbar2.set_label(r'$N_\mathrm{time}$', x=1.065, y=2)
    cax2.text(1.065, 0.5, r'$N_\mathrm{time}$', va='center')

    sc3 = cax3.scatter([1.05],[0.5],c=[0.7], vmin=0, vmax=1, marker='d', edgecolors='k',cmap=cmap_ordershapes, clip_on=False)
    cbar3 = plt.colorbar( sc3, cax=cax3, orientation='horizontal') 
    cbar3.set_ticks( (np.arange(len(ordershapes_var))+0.5)/float(len(ordershapes_var)-1+0.5+0.5), labels=[str(ordershape) for ordershape in ordershapes_var] )
    #cbar3.set_label(r'$N_\mathrm{shape}$', x=1.065, y=10)
    cax3.text(1.065, 0.5, r'$N_\mathrm{shape}$', va='center')

    fig.savefig('all_errors.png', dpi=300)
    plt.close(fig)

    return



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
                if (kwargslist[j]['ordertime'] == ordertime or kwargslist[j]['ordertime']<0) and results[j,ii].part.ordershape==ordershape :
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
                if (kwargslist[j]['ordertime'] == ordertime or kwargslist[j]['ordertime']<0) and (kwargslist[j]['ordershape'] == ordershape):
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
                if results[j,ii].part.zopt == VertOptionEnum.INTEGRATE:
                    des = np.array([dist_to_nearest_e(lbpre, results[j,ii].part.e) for ii in range(Npart)] )
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



    fig,ax = plt.subplots(figsize=(8,8))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            if 'ordertime' in kwargslist[j].keys():
                if results[j,ii].part.zopt == VertOptionEnum.INTEGRATE and results[j,ii].part.ordershape==ordershape:
                    des = np.array([dist_to_nearest_e(lbpre, results[j,ii].part.e) for ii in range(Npart)] )
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
                    ax.scatter( np.array([kwargslist[j]['ordertime']]*Npart)+np.random.randn(Npart)*0.05, [results[j,ii].rerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker=marker, lw=1, alpha=alpha, edgecolors='k', s=sizes)
                    print("Making benchmark_order - ", kwargslist[j], colors[j], simpleids[j] )
        else:
            pass
    ax.set_xlabel(r'Time Order')
    ax.set_ylabel('RMS Error in r (pc)')
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.legend()

    ax3 = ax.twinx()
    ax3.scatter([-100],[-100], c='k', marker='P', edgecolors='k', label=r'$n_\chi=20$')
    ax3.scatter([-100],[-100], c='k', marker='D', edgecolors='k', label=r'$n_\chi=100$')
    ax3.scatter([-100],[-100], c='k', marker='s', edgecolors='k', label=r'$n_\chi=300$')
    ax3.scatter([-100],[-100], c='k', marker='o', edgecolors='k', label=r'Other')
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.03,0.04))

    ax.set_xlim(-0.3, 8.3)

    fig.savefig('benchmark_order.png')
    plt.close(fig)

    
    fig,ax = plt.subplots(figsize=(8,8))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            if 'ordershape' in kwargslist[j].keys():
                if results[j,ii].part.zopt == VertOptionEnum.INTEGRATE and results[j,ii].part.ordertime==ordertime:
                    des = np.array([dist_to_nearest_e(lbpre, results[j,ii].part.e) for ii in range(Npart)] )
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
                    ax.scatter( np.array([kwargslist[j]['ordershape']]*Npart)+np.random.randn(Npart)*0.05, [results[j,ii].phierrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker=marker, lw=1, alpha=alpha, edgecolors='k', s=sizes)
                    #print("Making benchmark_order - ", kwargslist[j], colors[j], simpleids[j] )
        else:
            pass
    ax.set_xlabel(r'Shape Order')
    ax.set_ylabel('RMS Error in $r\phi$ (pc)')
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.legend()

    ax3 = ax.twinx()
    ax3.scatter([-100],[-100], c='k', marker='P', edgecolors='k', label=r'$n_\chi=20$')
    ax3.scatter([-100],[-100], c='k', marker='D', edgecolors='k', label=r'$n_\chi=100$')
    ax3.scatter([-100],[-100], c='k', marker='s', edgecolors='k', label=r'$n_\chi=300$')
    ax3.scatter([-100],[-100], c='k', marker='o', edgecolors='k', label=r'Other')
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.03,0.04))

    ax.set_xlim(-0.3, 30.3)

    fig.savefig('benchmark_ordershape.png')
    plt.close(fig)






def benchmark():
    import galpy.potential
    import astropy.units as units
    mw2014 = galpy.potential.MWPotential2014
    psirr = galpyPotential(mw2014, units, galpy.potential)
    nu = galpyFreq(mw2014, units, galpy.potential)
    dlnnudr = numericalFreqDeriv(nu)
    print("potential set up 1")
#    nu0 = np.sqrt(4*np.pi*G*0.2)
#    alpha = 2.2
#    def nu(r):
#        return nu0 * (r/8100.0)**(-alpha/2.0)
#    def dlnnudr(r):
#        return -alpha/(2.0*r)
    psir = PotentialWrapper( psirr, nur=nu, dlnnudr=dlnnudr)
    print("potential set up 2")
    xCart = np.array([8100.0, 0.0, 21.0])
    vCart = np.array([10.0, 230.0, 10.0])
    ordertime=5
    ordershape=14
    ts = np.linspace( 0, 10000.0, 1000) # 10 Gyr

    #lbpre = Precomputer(psir, Nclusters=10, use_multiprocessing=True)
    #lbpre.save()
    lbpre = Precomputer.load('../big_10_1000_alpha2p2_lbpre.pickle')
    #lbpre = Precomputer.load('big_10_1000_alpha2p2_lbpre.pickle')
    print("done precomputing")


    #lbpre.generate_interpolators()
    #lbpre.save()

    Npart = 3
    #results = np.zeros((49,Npart))


    argslist = []
    kwargslist = []
    ids = []
    simpleids = []
    colors = []


    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':VertOptionEnum.INTEGRATE, 'nchis':100} )
    ids.append( r'$2\ \mathrm{Int }- n_\chi=100$' )
    simpleids.append('zintchi100')
    colors.append('r')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':VertOptionEnum.FOURIER, 'nchis':100} )
    ids.append( r'Fourier - $n_\chi=100$' )
    simpleids.append('zintchi100')
    colors.append('orange')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':0,  'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot0')
    colors.append('r')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':1,  'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot1')
    colors.append('r')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':2,  'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot2')
    colors.append('r')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':3,  'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot3')
    colors.append('r')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':4,  'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot4')
    colors.append('r')

#    argslist.append( (psir, nu0, lbpre) ) 
#    kwargslist.append( {'ordershape':ordershape, 'ordertime':5,, 'nchis':100} )
#    ids.append( r'$2 Int - n_\chi=100$' )
#    simpleids.append('zintchi100ot5')
#    colors.append('r')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':6,  'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot6')
    colors.append('r')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':7,  'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot7')
    colors.append('r')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':8,  'nchis':100} )
    ids.append( None )
    simpleids.append('zintchi100ot8')
    colors.append('r')


    for ii in range(30):

        argslist.append( (psir, lbpre) ) 
        kwargslist.append( {'ordershape':ii, 'ordertime':ordertime, 'nchis':100} )
        ids.append( None )
        simpleids.append('zintchi100os'+str(ii).zfill(2))
        colors.append('b')


#    argslist.append( (psir, lbpre) ) 
#    kwargslist.append( {'ordershape':ordershape, 'ordertime':-5, 'zopt':'fourier', 'nchis':100} )
#    ids.append( r'Fourier - ot-5 - $n_\chi=100$' )
#    simpleids.append('zfourierchi100otm5')
#    colors.append('gray')
#
#    argslist.append( (psir, lbpre) ) 
#    kwargslist.append( {'ordershape':ordershape, 'ordertime':-10, 'zopt':'fourier', 'nchis':100} )
#    ids.append( r'Fourier - ot-10 - $n_\chi=100$' )
#    simpleids.append('zfourierchi100otm10')
#    colors.append('lightblue')


    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape,'zopt':VertOptionEnum.FOURIER, 'nchis':100} )
    ids.append( r'Fourier - $n_\chi=100$' )
    simpleids.append('zfourierchi100ot8')
    colors.append('yellow')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'zopt':VertOptionEnum.FOURIER, 'nchis':300} )
    ids.append( r'Fourier - $n_\chi=1000$' )
    simpleids.append('zfourierchi1000ot8')
    colors.append('green')


    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime,  'nchis':300} )
    ids.append( r'2 Int - $n_\chi=300$' )
    simpleids.append('zintchi300')
    colors.append('k')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime,  'nchis':20} )
    ids.append( r'2 Int - $n_\chi=20$' )
    simpleids.append('zintchi20')
    colors.append('pink')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':VertOptionEnum.FIRST, 'nchis':100} )
    ids.append( r'1st Order - $n_\chi=100$' )
    simpleids.append('fiore1chi20')
    colors.append('maroon')

    argslist.append( (psir, lbpre) ) 
    kwargslist.append( {'ordershape':ordershape, 'ordertime':ordertime, 'zopt':VertOptionEnum.ZERO, 'nchis':100} )
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
        
        gt = benchmark_groundtruth( ts, xCart, vCartThis, psir )
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
                results[j,ii].estimate_errors([9000,10000],'lastgyr', psir=psir)
                results[j,ii].estimate_errors([200,300],'200myr')
                results[j,ii].estimate_errors([900,1000],'1gyr', psir=psir)
                results[j,ii].errs_at_fixed_time(0.025, 'fixedtime', rtol=0.2)


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
                if (kwargslist[j]['ordertime'] == ordertime or kwargslist[j]['ordertime']<0) and results[j,ii].part.ordershape==ordershape :
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
                if (kwargslist[j]['ordertime'] == ordertime or kwargslist[j]['ordertime']<0) and (kwargslist[j]['ordershape'] == ordershape):
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
                if results[j,ii].part.zopt == VertOptionEnum.INTEGRATE:
                    des = np.array([dist_to_nearest_e(lbpre, results[j,ii].part.e) for ii in range(Npart)] )
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



    fig,ax = plt.subplots(figsize=(8,8))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            if 'ordertime' in kwargslist[j].keys():
                if results[j,ii].part.zopt == VertOptionEnum.INTEGRATE and results[j,ii].part.ordershape==ordershape:
                    des = np.array([dist_to_nearest_e(lbpre, results[j,ii].part.e) for ii in range(Npart)] )
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
                    ax.scatter( np.array([kwargslist[j]['ordertime']]*Npart)+np.random.randn(Npart)*0.05, [results[j,ii].rerrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker=marker, lw=1, alpha=alpha, edgecolors='k', s=sizes)
                    print("Making benchmark_order - ", kwargslist[j], colors[j], simpleids[j] )
        else:
            pass
    ax.set_xlabel(r'Time Order')
    ax.set_ylabel('RMS Error in r (pc)')
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.legend()

    ax3 = ax.twinx()
    ax3.scatter([-100],[-100], c='k', marker='P', edgecolors='k', label=r'$n_\chi=20$')
    ax3.scatter([-100],[-100], c='k', marker='D', edgecolors='k', label=r'$n_\chi=100$')
    ax3.scatter([-100],[-100], c='k', marker='s', edgecolors='k', label=r'$n_\chi=300$')
    ax3.scatter([-100],[-100], c='k', marker='o', edgecolors='k', label=r'Other')
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.03,0.04))

    ax.set_xlim(-0.3, 8.3)

    fig.savefig('benchmark_order.png')
    plt.close(fig)

    
    fig,ax = plt.subplots(figsize=(8,8))
    for j in range(len(argslist)):
        if results[j,0].isparticle():
            if 'ordershape' in kwargslist[j].keys():
                if results[j,ii].part.zopt == VertOptionEnum.INTEGRATE and results[j,ii].part.ordertime==ordertime:
                    des = np.array([dist_to_nearest_e(lbpre, results[j,ii].part.e) for ii in range(Npart)] )
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
                    ax.scatter( np.array([kwargslist[j]['ordershape']]*Npart)+np.random.randn(Npart)*0.05, [results[j,ii].phierrs['lastgyr'] for ii in range(Npart)], c=colors[j], label=ids[j], marker=marker, lw=1, alpha=alpha, edgecolors='k', s=sizes)
                    #print("Making benchmark_order - ", kwargslist[j], colors[j], simpleids[j] )
        else:
            pass
    ax.set_xlabel(r'Shape Order')
    ax.set_ylabel('RMS Error in $r\phi$ (pc)')
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.legend()

    ax3 = ax.twinx()
    ax3.scatter([-100],[-100], c='k', marker='P', edgecolors='k', label=r'$n_\chi=20$')
    ax3.scatter([-100],[-100], c='k', marker='D', edgecolors='k', label=r'$n_\chi=100$')
    ax3.scatter([-100],[-100], c='k', marker='s', edgecolors='k', label=r'$n_\chi=300$')
    ax3.scatter([-100],[-100], c='k', marker='o', edgecolors='k', label=r'Other')
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc=(0.03,0.04))

    ax.set_xlim(-0.3, 30.3)

    fig.savefig('benchmark_ordershape.png')
    plt.close(fig)



def dist_to_nearest_k(lbpre, k):
    i = np.argmin( np.abs(lbpre.kclusters - k) )
    return np.abs(lbpre.kclusters[i] - k)

def dist_to_nearest_e(lbpre, e):
    i = np.argmin( np.abs(lbpre.eclusters - e) )
    return np.abs(lbpre.eclusters[i] - e)



if __name__=='__main__':
    benchmark2()

