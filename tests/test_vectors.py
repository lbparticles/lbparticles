from src.lbparticles.lbparticles import CylindVec, CartVec


class Test_cartVec:
    def test_vector_cart_to_cylind(self):
        assert type(CartVec(1, 1, 1, 0, 0, 0).cart_to_cylind()) == CylindVec


class Test_cylindVec:
    def test_vector_cylind_to_cart(self):
        assert type(CylindVec(1, 1, 1, 0, 0, 0).cylind_to_cart()) == CartVec
