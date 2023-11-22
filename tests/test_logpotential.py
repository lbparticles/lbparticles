from LBParticles import logpotential, particleLB, lbprecomputer, getCartesianFromPolar, G
import numpy as np
from tqdm import tqdm
import pdb


def test_lb2():
    vcirc = 220.0
    psir = logpotential(vcirc)
    nu = np.sqrt(4.0 * np.pi * G * 0.2)

    R0 = 8100.0
    ordertime = 8
    ordershape = 8
    part = particleLB([8100.0, 0, 0], [1.0, 221.0, 0.0], psir, nu, None, quickreturn=True, ordertime=ordertime,
                      ordershape=ordershape)
    # lbpre = lbprecomputer.load( 'big_10_1000_lbpre.pickle' )
    lbpre = lbprecomputer.load('big_10_1000_alpha2p2_lbpre.pickle')

    for i in tqdm(range(100)):
        # xvec = np.random.randn(3)*1.0
        # vvec = np.random.randn(3)*0.1

        R = R0 + np.random.randn() * 2.0
        theta = np.random.random() * 2.0 * np.pi
        z = np.random.randn() * 2.0

        u = np.random.randn() * 10.0
        v = vcirc + np.random.randn() * 10.0
        w = np.random.randn() * 10.0

        xcart, vcart = getCartesianFromPolar((R, theta, z), (u, v, w))
        part = particleLB(xcart, vcart, psir, nu, lbpre, quickreturn=False, ordertime=ordertime, ordershape=ordershape)

        # part = unperturbedParticle( xcart, vcart, vcirc, vcircBeta, nu)

        xabs = part.xabs(0)
        vabs = part.vabs(0)

        x, y, z = xabs
        vx, vy, vz = vabs

        # pdb.set_trace()

        xc = np.isclose(xcart[0], x)
        yc = np.isclose(xcart[1], y)
        zc = np.isclose(xcart[2], z)

        vxc = np.isclose(vcart[0], vx)
        vyc = np.isclose(vcart[1], vy)
        vzc = np.isclose(vcart[2], vz)

        if not np.all([xc, yc, zc, vxc, vyc, vzc]):
            pdb.set_trace()
