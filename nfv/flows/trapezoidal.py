import torch

from nfv.flows import Flow
from nfv.utils.tensor import ensure_tensor


class Trapezoidal(Flow):
    def __init__(self, vmax=1.0, w=-1.5, kmax=1.0, kcrit0=0.2, kcrit1=0.8):
        """
        vmax: free flow speed
        w: congestion flow speed
        kmax: jam density
        q_max: maximum flow
        """
        self.vmax = vmax
        self.w = w
        self.kmax = kmax
        self.q_max1 = vmax * kcrit0
        self.q_max2 = w * (kcrit1 - kmax)
        self.kcrit0 = kcrit0
        self.kcrit1 = kcrit1
        self.k_crits = [self.kcrit0, self.kcrit1]
        self.k_crit = kcrit0 if self.q_max1 > self.q_max2 else kcrit1

    def __repr__(self):
        return "trapezoidal"

    @ensure_tensor
    def q(self, k):
        k = torch.clamp(k, 0, self.kmax)
        return torch.where(
            k < self.kcrit0,
            self.vmax * k,
            torch.where(
                k < self.kcrit1,
                (self.q_max1 - self.q_max2) / (self.kcrit0 - self.kcrit1) * (k - self.kcrit0) + self.q_max1,
                self.w * (k - self.kmax),
            ),
        )

    @ensure_tensor
    def qp(self, k):
        return torch.where(
            k < self.kcrit0,
            self.vmax,
            torch.where(
                k < self.kcrit1,
                (self.q_max1 - self.q_max2) / (self.kcrit0 - self.kcrit1),
                self.w,
            ),
        )

    @ensure_tensor
    def R(self, u):
        alpha = (self.q_max1 - self.q_max2) / (self.kcrit0 - self.kcrit1)
        return torch.where(
            u >= self.vmax,
            0,
            torch.where(
                u >= alpha,
                (self.vmax - u) * self.kcrit0,
                torch.where(u <= self.w, -self.kmax * u, alpha * (self.kcrit1 - self.kcrit0) + self.q_max1 - u * self.kcrit1),
            ),
        )

    @ensure_tensor
    def Rp(self, u):
        alpha = (self.q_max1 - self.q_max2) / (self.kcrit0 - self.kcrit1)
        return torch.where(u >= self.vmax, 0, torch.where(u >= alpha, -self.kcrit0, torch.where(u <= self.w, -self.kmax, -self.kcrit1)))


class TrapezoidalFlat(Trapezoidal):
    def __init__(self, vmax=1.0, w=-1.0, qmax=0.2, kmax=1.0):
        kcrit0 = qmax / vmax
        kcrit1 = qmax / w + kmax
        super().__init__(vmax, w, kmax, kcrit0, kcrit1)
