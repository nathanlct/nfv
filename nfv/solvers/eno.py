# Essentially non-oscillatory scheme
# see https://www3.nd.edu/~yzhang10/WENO_ENO.pdf
# and https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf (original)
import torch

from nfv.solvers.godunov import Godunov


def ENO(k_BX, flow_fn):
    """
    Params
        k_BX: array of n densities at time t
        flux: flux function (k_BX -> flux)
    Returns
        array of n-1 flows that will then update the densities as follows:
        `k_BX[1:-1] = k_BX[1:-1] - dt / dx * (flows[1:] - flows[:-1])`
    """
    # precompute primitive of k_BX
    # this is defined at the boundary points x_{i +- 1/2}, so we should have len(V) == len(k_BX) + 1
    V = torch.cat((torch.zeros(k_BX.shape[0], 1).to(k_BX.dtype).to(k_BX.device), torch.cumsum(k_BX, dim=1)), dim=1)

    # precompute divided differences 3 -- we need one for each 2 cells, so should have len(dds) == len(k_BX) - 1
    # note: we don't divide by the dx term since dx is constant for us, thus doesn't change comparisons
    dds3 = torch.abs(V[:, 2:] - 2 * V[:, 1:-1] + V[:, :-2])

    # precompute divided differences 4 -- we need one for each 3 cells, so should have len(dds) == len(k_BX) - 2
    # note: we don't divide by the dx term since dx is constant for us, thus doesn't change comparisons
    dds4 = torch.abs(V[:, 3:] - 3 * V[:, 2:-1] + 3 * V[:, 1:-2] - V[:, :-3])

    # compute cases for each i in [2, len(k_BX)-2]: either left, mid or right (3 possible stencils of 3 cells each)
    case_left = -1
    case_mid = 0
    case_right = 1
    idx = torch.arange(2, k_BX.shape[1] - 2)
    cond = dds3[:, idx - 1] < dds3[:, idx]
    casesA = torch.where(dds4[:, idx - 2] < dds4[:, idx - 1], case_left, case_mid)
    casesB = torch.where(dds4[:, idx - 1] < dds4[:, idx], case_mid, case_right)
    cases = torch.where(cond, casesA, casesB)

    # compute u's for each case
    u_left = 1 / 3 * k_BX[:, :-4] - 7 / 6 * k_BX[:, 1:-3] + 11 / 6 * k_BX[:, 2:-2]
    u_mid = -1 / 6 * k_BX[:, 1:-3] + 5 / 6 * k_BX[:, 2:-2] + 1 / 3 * k_BX[:, 3:-1]
    u_right = 1 / 3 * k_BX[:, 2:-2] + 5 / 6 * k_BX[:, 3:-1] - 1 / 6 * k_BX[:, 4:]

    us = torch.where(cases == case_left, u_left, torch.where(cases == case_mid, u_mid, u_right))
    flows = Godunov(us, flow_fn)

    return flows
