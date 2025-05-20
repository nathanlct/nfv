import math

import torch

from nfv.flows import Flow
from nfv.utils.tensor import ensure_tensor


class Greenberg(Flow):
    def __init__(self, kmax=1.0, c0=2.0, safe_mode=True):
        """
        kmax: jam density, typically 1.0
        c0: parameter scaling the function
        """
        self.kmax = kmax
        if safe_mode:
            assert abs(self.kmax - 1.0) < 1e-9  # hardcoded for kmax = 1 right now
        self.c0 = c0
        self.k_crits = [self.kmax * math.exp(-1)]
        self.k_crit = self.k_crits[0]

        self.vmax = self.qp(0.0)
        self.w = self.qp(self.kmax)

    def __repr__(self):
        return "greenberg"

    @ensure_tensor
    def q(self, k):
        k = torch.clamp(k, 0, self.kmax)
        return self.c0 * k * torch.log(self.kmax / (k + 1e-7))  # k/kmax ?

    @ensure_tensor
    def qp(self, k):
        return -self.c0 * (torch.log(k + 1e-7) + 1)

    @ensure_tensor
    def R(self, u):
        # derivative of f(k)-ku wrt k is > 0 iff k < exp(-u/c0-1)
        x = torch.exp(-u / self.c0 - 1)
        return torch.where(self.kmax < x, -self.kmax * (self.c0 * math.log(self.kmax) + u), self.c0 * x)

    @ensure_tensor
    def Rp(self, u):
        x = torch.exp(-u / self.c0 - 1)
        return torch.where(self.kmax < x, -self.kmax, -x)
