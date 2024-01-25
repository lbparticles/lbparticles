from lbparticles.potentials import LogPotential, HernquistPotential
import numpy as np
import pytest


@pytest.mark.parametrize("r", np.linspace(0.1, 400, 100))
class Test_LogPotential:
    def test_call(self, r):
        psir = LogPotential(220.0)
        psir(r)

    def test_ddr(self, r):
        psir = LogPotential(220.0)
        psir.ddr(r)

    def test_ddr2(self, r):
        psir = LogPotential(220.0)
        psir.ddr2(r)


@pytest.mark.parametrize("r", np.linspace(0.1, 400, 100))
class Test_HernquistPotential:
    def test_call(self, r):
        psir = HernquistPotential.vcirc(2.2, 220.0)
        psir(r)

    def test_ddr(self, r):
        psir = HernquistPotential.vcirc(2.2, 220.0)
        psir.ddr(r)

    def test_ddr2(self, r):
        psir = HernquistPotential.vcirc(2.2, 220.0)
        psir.ddr2(r)
