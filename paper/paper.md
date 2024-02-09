---
title: "lbparticles: Semi-analytic Particle Model Python Package"
tags:
  - Python
  - astronomy
  - interstellar objects
  - particle orbit model
  - galactic dynamics
authors:
  - name: John C. Forbes
    orcid: 0000-0002-1975-4449
    equal-contrib: true
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

The `lbparticles` python package is a tool to model semianalytic orbits of point mass particles in explicit static central potentials. The model conserves linear and angular momentum where numerical integration methods would not due to numerical inaccuracies. The model has a faster computation speed than integration methods due to the use of approximate analytical equations derived at initialisation [@Forbes:2024]. The package is a direct descendant of Lynden-Bell's previous work in 2D point mass orbit approximate solutions [@Lynden:2015].

The lbparticles package allows for the investigation into the dynamics of point mass particles. This has been done in the past with integrating methods such as `galpy`, `SPARTA`, and `SciPy` [@Bovy:2015] [@Virtanen:2020] [@Diemer:2017]. The main benefit of `lbparticles` compared to these methods is the speed of calculation and the conservation of important quantities that enables long term analysis of dynamics.

This paper describes the `lbparticles` package, the software architecture and its implementation in python. For information on the algorithms used in this package see the sibling paper [@Forbes:2024].

# Statement of Intent - lbparticles Package

The `lbparticles` is a python package that can calculate orbits of point mass particles in central potentials. The package can be easily integrated into another python project as it has a small footprint; only depending on `numpy` [@Harris:2020] and `scipy` [@Virtanen:2020].

Example code and API Documentation can be found on the [readthedocs](https://lbparticles.readthedocs.io/en/latest/) page in the form of embedded jupyter notebooks and embedded class and function DOCSTRINGs [@Kluyver:2016].

The lbparticles was designed for investigating interstellar objects orbits around the milky way galaxy.

# Classes

`lbparticles` has three main Classes in its implementation; `Potential`, `Precomputer`, and `Particle`.


## `Potential`

The Potential class is where the user defines the static central potential. The user gives the explicit function, explicit first derivative, and explicit second derivative in the given class methods,`__call__`, `ddr`, and `ddr2` respectively.

Uses [@Fiore:2022]

## `Precomputer`

The Precomputer class is where the analytic part of the model is computed. It is initialised with a given Potential and then passed to Particle objects that will move within the potential.

## `Particle`

The Particle class has multiple different vertical options for z-motion. These methods are called 2Int, Fourier, Volterra (Zeroth and First), and Tilt.

### 2Int

### Fourier

### Volterra

Zeroth and First.

### Tilt

# Design Principles

We have chosen to construct lbparticles for accessability ...

This is important for python, because it's a language built with object oriented coding in mind...

The lbparticles package is supported by the University of Canterbury interstellar objects working group ...

# Acknowledgments

Rutherford Discovery Fellowships

University of Canterbury Scholarships

# References
