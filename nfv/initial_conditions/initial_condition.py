from abc import ABC

import matplotlib.pyplot as plt
import numpy as np


class InitialCondition(ABC):
    def discretize(self, nx, points_per_cell=5):
        """Discretize the initial condition by integrating it over `nx` cells.

        Args:
            nx (int): Number of discrete cells (in space).

        Returns:
            np.ndarray: Array of length `nx` containing the integral average of `ks` over each cell.
        """
        raise NotImplementedError  # TODO approximate integral for arbitrary case

    def plot(self, nx=100):
        xs = np.arange(nx + 1)
        ks = self.discretize(nx=nx)  # cell averages
        plt.figure()
        plt.stairs(ks, xs, baseline=None, linewidth=2.0, zorder=2)
        plt.xlabel("cells")
        plt.ylabel("avg value in cell")
        plt.grid()
        plt.show()
