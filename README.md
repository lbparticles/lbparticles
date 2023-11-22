# LBParticles

This code describes the orbits of particles in central potentials following the work of Lynden-Bell (2015).

Once initialized with a 3D position and velocity, and a specified potential, the particle's position and velocity at an arbitrary point in time can be computed in constant time.

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

# Specifying a potential 

The code includes a simple class called logpotential, corresponding to a circular velocity that is constant in radius. Other potentials may be implemented in the future or by the user, provided they implement the same methods present in the logpotential class.

The logpotential or similar specifies the spherically-symmetric part of the potential. For particles orbiting near a Milky Way-like disk, the particle is subject to vertical oscillations at a characteristic frequency sqrt(4 pi G rho0), where G is Newton's constant and rho0 is the midplane density, of order 0.2 solar masses per cubic parsec for the Solar neighborhood. The code additionally allows this midplane density to vary as a powerlaw, so that the frequency of vertical oscillation will depend on the particle's position.

For particles whose orbits rarely take them near the disk because of high inclinations or large semi-major axes, the orbital plane of the particle will change very little as the result of gravitational interactions with the disk. To adopt this assumption, one can pass `tilt=True` in the constructor for particleLB. The particle's position and velocity will still be in the usual disk-centered coordinate system (see below), but internally the particle's coordinate system will be rotated such that it has no "z" motion, and when the particle's position is requested at a different time, the particle's internal coordinates will be rotated back to the global coordinate system.

# Loading a precomputer 

In order to evaluate the particle's position and velocity at arbitrary times, a series of definite integrals need to be computed. This is handled by the class lbprecomputer. The integrals are functions of the potential, the eccentricity of the particle's orbit, and an integer specifying the order in a series of cosines; including higher orders generally improves the accuracy of the estimate. At initialization, the particle uses a precomputer object to look up the values of these integrals. The precomputer provides these values interpolated from a grid (which is irregular in eccentricity, but regular in orbital phase and order), where points on the grid are evaluated ahead of time. The precomputer can be initialized and more points added in the eccentricity dimension with the member method add\_new\_data (see the method buildlbpre). For a precomputer set up with a 220 km/s logarithmic potential, and an alpha (powerlaw index of midplane density) of 2.2, you can download this [file](https://www.dropbox.com/scl/fi/do318kg26e80mxqdehq5d/big_10_1000_alpha2p2_lbpre.pickle?rlkey=k1i9m5p9bs2obs2co2rwyqt8d&dl=1).

# Initializing a particle 

The particle's position and velocity at the particle's t=0
```
from LBParticles import lbprecomputer, particleLB, logpotential, G
import numpy as np

xcart = [8100.0, 0.0, 90.0] # the Sun's location in x,y,z cartesian coordinates (in parsecs)
vcart = [-11.1, 12.24 + 220.0, 7.25] # similar to the Sun's velocity in vx, vy, vz (given the position xcart) in units of pc/Myr.
nu0 = np.sqrt( 4.0*np.pi * G * 0.2) # vertical oscillation frequency at r=8100 pc.
alpha = 2.2 # powerlaw slope of the midplane density with radius, so that nu = nu0 (r/r0)^(-alpha/2)
psir = logpotential(220.0) # a logarithmic potential with a circular velocity of 220 pc/Myr.
lbpre = lbprecomputer.load('big_10_1000_alpha2p2_lbpre.pickle') # load the precomputer.
ordershape = 10 # number of terms used in the series to find the tangential position of the particle
ordertime = 5 # number of terms used in the series to find the relationship between the particle's phase in its radial oscillation and the current time.

part = particleLB( xcart, vcart, psir, nu0, lbpre, ordershape=ordershape, ordertime=ordertime, alpha=2.2 )

X,V = part.xvabs(100) # find the particle's position and velocity 100 Myr later.

``` 


# Units 

Internally the code uses the parsec-solar mass-Myr system. The main advantage of this system is that a km/s is very close to a parsec/Myr, though they are different by a few percent.

The code's cartesian coordinate system is such that the x-y plane at z=0 is the midplane of the disk. The origin (x=y=z=0) is the center of the disk and of the spherical potential. 
