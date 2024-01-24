from src.lbparticles.lbparticles import CylindVec, CartVec


class test_cartVec:
    def test_vector_cart_to_cylind():
        assert type(CartVec(0, 0, 0, 0, 0, 0).cart_to_cylind) == CylindVec


class test_cylindVec:
    def test_vector_cylind_to_cart():
        assert type(CylindVec(0, 0, 0, 0, 0, 0).cylind_to_cart) == CartVec
