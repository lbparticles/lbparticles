# LBParticles

_This code describes the orbits of particles in central potentials following the work of Lynden-Bell (2015)._

Once initialized with a 3D position and velocity, and a specified potential, 
the particle's position and velocity at an arbitrary point in time can be 
computed in constant time.

## Documentation

```{toctree}
:maxdepth: 2

pages/install
pages/quickstart
pages/implement_a_potential
pages/api
```


## Units 

Internally the code uses the parsec-solar mass-Myr system. The main advantage of this system is that a km/s is very close to a parsec/Myr, though they are different by a few percent.

The code's cartesian coordinate system is such that the x-y plane at z=0 is the midplane of the disk. The origin (x=y=z=0) is the center of the disk and of the spherical potential. 


## Attribution

If you make use of this code, please cite (TODO):

```tex
    @article{lbparticles,
      .
      .
      .
    }
```

## Authors & License

Copyright 2023 John Forbes

Built by [John Forbes](https://github.com/jcforbes) and contributors (see
[the contribution graph](https://github.com/LBParticles/LBParticles/graphs/contributors) for the most
up-to-date list). Licensed under the MIT license (see `LICENSE`).