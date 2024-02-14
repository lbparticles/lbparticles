---
title: "lbparticles: Test Particle Orbits in Spherical Potentials with Disks"
tags:
  - Python
  - astronomy
  - interstellar objects
  - particle orbit model
  - galactic dynamics
authors:
  - name: John C. Forbes
    orcid: 0000-0002-1975-4449
    equal-contrib: false
    affiliation: 1
  - name: Jack Patterson
    orcid: 0009-0001-1692-4676
    equal-contrib: false
    affiliation: 1
  - name: Angus Forrest
    orcid: 0009-0008-0355-5809
    equal-contrib: false
    affiliation: 1
affiliations:
  - name: University of Canterbury, New Zealand
    index: 1
date: 07 December 2023
bibliography: paper.bib
---

# Summary

![lbparticles logo](logo.png)


The `lbparticles` python package is a tool to model semianalytic orbits of point mass particles in explicit static potentials with spherical symmetry but arbitrary radial profiles, with the option of including a disk with an arbitrary surface density profile. The model conserves linear and angular momentum where numerical integration methods would not due to numerical inaccuracies. The model has a faster computation speed than integration methods due to the use of approximate analytical equations derived at initialisation [@Forbes:2024]. The package is a direct descendant of Lynden-Bell's previous work in 2D point mass orbit approximate solutions [@Lynden:2015]; this was developed over multiple papers [@Lynden:2008] [@Lynden:2010] [@Lynden:2015_2].

The lbparticles package allows for the investigation into the dynamics of point mass particles. This has been done in the past with direct integration of the equations of motion with `galpy`, `SPARTA`, and `SciPy` [@Bovy:2015] [@Diemer:2017] [@Virtanen:2020]. The main benefits of `lbparticles` compared to these methods is calculation time, conservation of quantities, and compactness. The speed of calculation of particles is faster than Scipy which can be seen in the companion paper [@Forbes:2024]. The conservation of important quantities that enables large time scale analysis of dynamics as methods that rely on integration can acrue floating point error in conserved quantities causing them to not be constant. lbparticles is more compact as the solution only requires a 6D position where as in a blind integration you have to store the 6D position of the particle at many different points in time.

This paper describes the `lbparticles` package, the software architecture and its implementation in python. For information on the algorithms used in this package see the sibling paper [@Forbes:2024].

# lbparticles package

The `lbparticles` is a python package that can calculate orbits of point mass particles in central potentials. The package can be easily integrated into another python project as it has a small footprint; only depending on `numpy` [@Harris:2020] and `scipy` [@Virtanen:2020].

Example code and API Documentation can be found on the [readthedocs](https://lbparticles.readthedocs.io/en/latest/) page in the form of embedded jupyter notebooks and embedded class and function DOCSTRINGs [@Kluyver:2016].

The lbparticles was designed for investigating the dynamics of interstellar objects orbits around the milky way galaxy.

# Classes

`lbparticles` has four main Classes in its implementation; `Potential`, `PotentialWrapper`, `Precomputer`, and `Particle`.


## `Potential`

The Potential class is where the user defines the static central potential. The user gives the explicit function, explicit first derivative, and explicit second derivative in the given class methods,`__call__`, `ddr`, and `ddr2` respectively.

## `PotentialWrapper`

The PotentialWrapper class provides numerous functions for calculating needed quantities for using the Precomputer class. The PotentialWrapper allows for modulation of the given Potential class (`__call__`, `ddr`, and `ddr2`) through adding terms porportional to the vertical moment of inertia ($I_z$). 

## `Precomputer`

The Precomputer class is where the analytic part of the model is computed. It is initialised with a given Potential and then passed to Particle objects that will move within the potential. A benefit of the Precomputer class' design is that 

## `Particle`

The Particle class has multiple different vertical options for z-motion. These methods are called 2Int, Fourier, Volterra (Zeroth and First), and Tilt.

### 2Int

The 2Int vertical motion option numerically integrates the equation of motion for $z$ over a single radial period.

### Fourier

The Fourier motion option decomposes $\nu(t)$ into a Fourier series, from which $\mu$ and $f(t)$ may be estimated, where $\nu$ is the typical oscillation frequency the vertical motion, $\mu$ is the purely imagninary exponential constant, and $f$ is the function that solves $z = A e^{\mu t}f(t) + Be^{-\mu t}f(-t)$.

### Volterra (Zeroth and First Order)

The Volterra motion option uses Fiore's prescription to estimate $z$ based on a series of integrals of the phase of the oscillation with a choice between a zeroth order and a first order approximation [@Fiore:2022].

### Tilt

The Tilt motion option ignores the disk potential and treats the particle as orbiting in a purely spherically-symmetric potential.


# Design Principles

We aim for `lbparticles` to be accessible to the vast majority of astronomers and comosolgist. `lbparticles` package is written in python as it is widely used in the field, allowing for researchers to interface with it and also be able to  contribute and maintain the code in python [@Rossum:2009]. The main alternative choice we considered was writing the core of lbparticles in rust and using LAPACK with python bindings for researcher interaction [@Matsakis:2014] [@Anderson:1992]. This would be the fastest and maintainable code base but doesn't leverage the fields knowledge of python. We decided that implementing lbparticles solely in python would be the best choice so that all current contributors can contribute to the whole codebase.

The design and implementation of lbparticles was affected by the choice of python as the main language. This is because python utilises classes heavily, a object oriented paradigm, rather than a functional paradigm. This has lead to a heriarchy of importance in the implementation where a potential needs to be specific for the precomputer to be built and only then can a particle's orbit be determined. This is in contrast to a implementation where particles could be instatiated with their initial conditions and then place inside different potential to determine orbit.

The lbparticles package is supported by the University of Canterbury interstellar objects working group. For support and advice either reach out using email or the github repo.

# Acknowledgments

Rutherford Discovery Fellowships

University of Canterbury Scholarships

# References
