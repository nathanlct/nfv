import torch

from nfv.flows import Flow
from nfv.utils.tensor import ensure_tensor


class Greenshield(Flow):
    def __init__(self, vmax=1.0, kmax=1.0):
        """
        vmax: free flow speed
        kmax: jam density
        """
        self.vmax = vmax
        self.kmax = kmax
        self.k_crit = self.kmax / 2.0
        self.w = -self.vmax

    def __repr__(self):
        return "greenshield"

    @ensure_tensor
    def q(self, k):
        k = torch.clamp(k, 0, self.kmax)
        return self.vmax * k * (1.0 - k / self.kmax)

    @ensure_tensor
    def qp(self, k):
        return self.vmax * (1 - 2 * k / self.kmax)

    # def qp_inv(self, xt):
    #     return 0.5 * self.kmax * (1 - xt / self.vmax)

    @ensure_tensor
    def R(self, u):
        left = -self.kmax * u
        mid = self.vmax * self.kmax * (u / self.vmax - 1) ** 2 / 4
        right = 0
        return torch.where(u >= self.vmax, right, torch.where(u <= -self.vmax, left, mid))

    @ensure_tensor
    def Rp(self, u):
        left = -self.kmax
        mid = self.kmax / 2 * (u / self.vmax - 1)
        right = 0
        return torch.where(u >= self.vmax, right, torch.where(u <= -self.vmax, left, mid))

    # def rarefaction_k(self, x_over_t):
    #     return 0.5 * self.kmax * (1 - x_over_t / self.vmax)
