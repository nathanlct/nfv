import numpy as np

from nfv.initial_conditions import PiecewiseConstant


class Riemann(PiecewiseConstant):
    def __init__(self, k1, k2, **kwargs):
        super().__init__([k1, k2], **kwargs)

    # def __init__(self, k1, k2):
    #     self.k1 = k1
    #     self.k2 = k2
    #     self.ks = np.array([k1, k2])

    # def discretize(self, nx):
    #     """See parent class."""
    #     nx2 = nx // 2
    #     kleft = [self.k1] * nx2
    #     kright = [self.k2] * nx2
    #     if nx % 2 == 0:
    #         return np.array(kleft + kright)
    #     else:
    #         return np.array(kleft + [0.5 * (self.k1 + self.k2)] + kright)
