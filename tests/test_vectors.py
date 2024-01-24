import lbparticles as lb


class test_cartVec():
    def test_vector_cart_to_cylind():
        assert type(lb.CartVec(0, 0, 0, 0, 0,
                    0).cart_to_cylind) == lb.CylindVec


class test_cylindVec():
    def test_vector_cylind_to_cart():
        assert type(lb.CylindVec(0, 0, 0, 0, 0,
                    0).cart_to_cylind) == lb.CartVec
