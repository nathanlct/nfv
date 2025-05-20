# Weighted essentially non-oscillatory (WENO) scheme
import torch

from nfv.solvers.godunov import Godunov


def WENO(k_BX, flow_fn):
    u1 = 1 / 3 * k_BX[:, :-4] - 7 / 6 * k_BX[:, 1:-3] + 11 / 6 * k_BX[:, 2:-2]  # i-2 to i (left)
    u2 = -1 / 6 * k_BX[:, 1:-3] + 5 / 6 * k_BX[:, 2:-2] + 1 / 3 * k_BX[:, 3:-1]  # i-1 to i+1 (mid)
    u3 = 1 / 3 * k_BX[:, 2:-2] + 5 / 6 * k_BX[:, 3:-1] - 1 / 6 * k_BX[:, 4:]  # i to i+2 (right)

    beta1 = 13 / 12 * (k_BX[:, :-4] - 2 * k_BX[:, 1:-3] + k_BX[:, 2:-2]) ** 2 + 1 / 4 * (k_BX[:, :-4] - 4 * k_BX[:, 1:-3] + 3 * k_BX[:, 2:-2]) ** 2
    beta2 = 13 / 12 * (k_BX[:, 1:-3] - 2 * k_BX[:, 2:-2] + k_BX[:, 3:-1]) ** 2 + 1 / 4 * (k_BX[:, 1:-3] - k_BX[:, 3:-1]) ** 2
    beta3 = 13 / 12 * (k_BX[:, 2:-2] - 2 * k_BX[:, 3:-1] + k_BX[:, 4:]) ** 2 + 1 / 4 * (3 * k_BX[:, 2:-2] - 4 * k_BX[:, 3:-1] + k_BX[:, 4:]) ** 2

    alpha1 = (1 / 10) / (1e-9 + beta1) ** 2
    alpha2 = (3 / 5) / (1e-9 + beta2) ** 2
    alpha3 = (3 / 10) / (1e-9 + beta3) ** 2

    us = (alpha1 * u1 + alpha2 * u2 + alpha3 * u3) / (alpha1 + alpha2 + alpha3)

    flows = Godunov(us, flow_fn)

    return flows
