import torch

from nfv.flows import Flow
from nfv.utils.tensor import ensure_tensor


class Triangular(Flow):
    def __init__(self, vmax=1.0, w=-1.0, kmax=1.0):
        """
        vmax: free flow speeds
        w: congestion flow speed
        kmax: jam density
        """
        # if w >= 0:
        #     raise ValueError("Triangular flow: w must be negative")
        self.vmax = vmax
        self.w = w
        self.kmax = kmax
        self.k_crit = -self.kmax * self.w / (self.vmax - self.w)

    def __repr__(self):
        return "triangular"

    @ensure_tensor
    def q(self, k):
        k = torch.clamp(k, 0, self.kmax)
        return torch.where(k < self.k_crit, self.vmax * k, self.w * (k - self.kmax))

    @ensure_tensor
    def qp(self, k):
        return torch.where(k < self.k_crit, self.vmax, self.w)

    @ensure_tensor
    def R(self, u):
        return torch.where(u >= self.vmax, 0, torch.where(u <= self.w, -self.kmax * u, self.k_crit * (self.vmax - u)))

    @ensure_tensor
    def Rp(self, u):
        return torch.where(u >= self.vmax, 0, torch.where(u <= self.w, -self.kmax, -self.k_crit))

    @ensure_tensor
    def rarefaction_k(self, x_over_t):
        return self.k_crit
