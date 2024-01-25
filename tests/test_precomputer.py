from src.lbparticles.lbparticles import Precomputer, PotentialWrapper, LogPotential


class Test_Precomputer:
    def test_default_creation(self):
        lbdata = Precomputer(PotentialWrapper(LogPotential(220.0)))
