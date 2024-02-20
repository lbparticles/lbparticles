from __future__ import annotations

from enum import Enum

import numpy as np
from dataclasses import dataclass

def cos_zeros(ordN):
    """
    Finds the first ordN zeros of cos(ordN theta)

    cos x = 0 for x=pi/2 + k pi for k any integer
    """
    the_zeros = np.zeros(ordN)
    for i in range(ordN):
        the_zeros[i] = (np.pi / 2.0 + i * np.pi) / ordN
    return the_zeros

@dataclass(frozen=True)
class CartVec:
    """DOCSTRING"""

    x: float = 0
    y: float = 0
    z: float = 0
    vx: float = 0
    vy: float = 0
    vz: float = 0

    def cart_to_cylind(self) -> CylindVec:
        r = np.sqrt(self.x * self.x + self.y * self.y)
        return CylindVec(
            r,
            np.arctan2(self.y, self.x),
            self.z,
            (self.x * self.vx + self.y * self.vy) / r,
            (self.x * self.vy - self.vx * self.y) / r,
            self.vz,
        )


@dataclass(frozen=True)
class CylindVec:
    """DOCSTRING"""

    r: float = 0
    theta: float = 0
    z: float = 0
    vr: float = 0
    vtheta: float = 0
    vz: float = 0

    def cylind_to_cart(self) -> CartVec:
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)
        vx = self.vr * np.cos(self.theta) - self.vtheta * np.sin(self.theta)
        vy = self.vr * np.sin(self.theta) + self.vtheta * np.cos(self.theta)
        return CartVec(x, y, self.z, vx, vy, self.vz)


class VertOptionEnum(Enum):
    """DOCSTRING"""

    INTEGRATE = 1
    FOURIER = 2
    TILT = 3
    FIRST = 4
    ZERO = 5