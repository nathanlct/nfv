import numpy as np

from nfv.initial_conditions import InitialCondition


class PiecewiseConstant(InitialCondition):
    def __init__(self, ks, x_noise=False):
        """Piecewise constant initial condition.

        Args:
            ks (list of float): Step values of the function, assuming uniform step size.
            x_noise (bool): If True, adds noise to the x-coordinates when discretizing.
        """
        self.ks = np.array(ks)
        self.xs = np.linspace(0, 1, len(self.ks) + 1)  # between 0 and 1 -- can multiply by kmax if needed

        if x_noise:
            delta = 2.0 * np.random.rand(len(self.xs) - 2) - 1.0  # [-1, 1]
            delta *= (1 / len(self.ks)) * 0.25  # 1/4 or piece size (assuming regular pieces)
            self.xs[1:-1] += delta  # don't move boundaries
            # assert list(self.xs) == list(self.xs).sorted()  # make sure noise isn't too large

    def discretize(self, nx):
        """See parent class."""
        # compute the x-coordinates of the step function's boundary points
        x_ks = self.xs  # assuming domain spans [0, 1]
        # compute the x-coordinates of the boundaries of nx cells
        x_cells = np.linspace(0, 1, nx + 1)

        # compute the overlap x cells and ks cells
        # overall, overlap[i, j] is the fraction of function step ks[j] that overlaps with cell i (x_cells[i] to x_cells[i+1])
        lower = np.maximum(x_cells[:-1][:, None], x_ks[:-1])
        upper = np.minimum(x_cells[1:][:, None], x_ks[1:])
        overlap = np.clip(upper - lower, 0, None)

        # compute the integral in each new cell by summing contributions from each piece (weighted average)
        # each row of `overlap` sums to 1/nx -> multiply by nx so that sum of weights is 1
        cell_integrals = (self.ks * overlap).sum(axis=1) * nx

        return cell_integrals
