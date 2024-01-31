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

The `lbparticles` python package is a tool to model semianalytic orbits of point mass particles in a static central potential at a galaxy scale. The model conserves linear and angular momentum where numerical integration methods would not due to numerical inaccuracies. The model has a faster computation speed than integration methods due to approximate analytical equations [@Forbes:2024]. The package is built upon Lynden-Bell's previous work [@Lynden:2015].

The lbparticles package allows for the investigation into population dynamics of interstellar objects. 

Are there similar projects in the past? Is this competely new application?
The main benefit of the package compared to using off the shelf integration is the speed of calculation as well as conserving important quantities that may enable 
[@Bovy:2015] [@Virtanen:2020] [@Diemer:2017]

This paper describes the lbparticles package, the available classes, and the underlying architecture.

# Statement of Intent - lbparticles Package

The lbparticles is a python package that can calculate semianalytic orbits can be easily integrated into a python project with a small footprint; only depending on `numpy` [@Harris:2020] and `scipy` [@Virtanen:2020].

Example code can be found on the [readthedocs](https://lbparticles.readthedocs.io/en/latest/) page in the form of embedded jupyter notebooks [@Kluyver:2016].  ...

The lbparticles is designed for interstellar object research...

# Classes

How do the classes of the lbparticles operate?

## precomputer

## particle

Vertical Options for Z-motion.

### 2Int

### Fourier

### Volterra

### Tilt

Uses [@Fiore:2022]

## Potential

# Design Principles

We have chosen to construct lbparticles for accessability ...

This is important for python, because it's a language built with object oriented coding in mind...

The lbparticles package is supported by the University of Canterbury interstellar objects working group ...

# Acknowledgments

Rutherford Discovery Fellowships
University of Canterbury Scholarships

# References
