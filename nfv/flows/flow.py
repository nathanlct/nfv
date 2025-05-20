"""
Implementation of several common traffic flow functions.

Flow: q [veh/time]
Density: k (here rho) [veh/distance]
Speed: v [distance/time]

Fundamental equation of traffic flow: q = kv

Free flow speed (vmax): maximum speed of traffic flow
Max density (kmax): jam density
Critical density (kcrit): density that maximizes the flux
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Flow(ABC):
    @abstractmethod
    def q(self, k):
        """Get flow q as a function of density k"""
        pass

    def __call__(self, k):
        return self.q(k)

    @abstractmethod
    def qp(self, k):
        """Get derivative of flow wrt to density (dq/dk)"""
        pass

    # @abstractmethod
    # def qp_inv(self, xt):
    #     """Inverse of flow derivative"""
    #     pass

    @abstractmethod
    def R(self, u):
        """R(u) function for Lax Hopf: R(u) = sup_k (q(k) - uk)"""
        pass

    @abstractmethod
    def Rp(self, u):
        """Derivative of R(u)"""
        pass

    @property
    def qmax(self):
        return self.q(self.k_crit)

    def plot(self, path=None, dpi=300, lines=True, kmax=1.0):
        ks = np.linspace(0, self.kmax, 1000)
        qs = self.q(ks)
        plt.figure(figsize=(4, 3), dpi=dpi)
        plt.plot(ks * 1609, qs * 3600)
        plt.grid()
        plt.xlabel("Density")
        plt.ylabel("Flow")
        if lines:
            if hasattr(self, "k_crits"):
                for k_crit in self.k_crits:
                    plt.axvline(k_crit, color="red", linestyle="--", linewidth=0.5)
            plt.axvline(self.k_crit * 1609, color="red", linestyle="--", linewidth=1.0)
            plt.axhline(self.qmax * 3600, color="green", linestyle="--", linewidth=1.0)
        # plt.xlim(0.0, kmax)
        # plt.ylim(bottom=0.0)
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
        plt.close()
