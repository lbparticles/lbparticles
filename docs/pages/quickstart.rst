.. _quickstart:

Getting Started
===============

Loading a precomputer
---------------------

In order to evaluate the particle's position and velocity at arbitrary times, a series of definite integrals need
to be computed. This is handled by the class lbprecomputer. The integrals are functions of the potential, the
eccentricity of the particle's orbit, and an integer specifying the order in a series of cosines; including higher
orders generally improves the accuracy of the estimate. At initialization, the particle uses a precomputer object to
look up the values of these integrals. The precomputer provides these values interpolated from a grid (which is
irregular in eccentricity, but regular in orbital phase and order), where points on the grid are evaluated ahead of
time. The precomputer can be initialized and more points added in the eccentricity dimension with the member method
add\_new\_data (see the method buildlbpre). For a precomputer set up with a 220 km/s logarithmic potential, and an
alpha (powerlaw index of midplane density) of 2.2, you can download this
`file <https://www.dropbox.com/scl/fi/do318kg26e80mxqdehq5d/big_10_1000_alpha2p2_lbpre.pickle?rlkey=k1i9m5p9bs2obs2co2rwyqt8d&dl=1>`_.