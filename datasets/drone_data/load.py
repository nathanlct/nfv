import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import interpolate

from nfv import Problem


def load_drone_data(normalize_density=False, dt=None):
    # see https://arxiv.org/pdf/2209.08763
    data_dx = 10.34385  # m
    data_dt = 1.0  # s
    density_TX = np.load("datasets/drone_data/hwy04_densities_c30.npy")  # (T, X)
    if dt is not None:
        # dt different than data_dt default -> resample density_TX across time accordingly
        t_max = (density_TX.shape[0] - 1) * data_dt
        t_old = np.arange(0, t_max + 1e-5, data_dt)
        t_new = np.arange(0, t_max + 1e-5, dt)
        print(f"{t_old.shape=}, {t_new.shape=}, {density_TX.shape=}, {t_old.max()=}, {t_new.max()=}")
        f = interpolate.interp1d(t_old, density_TX, axis=0)
        density_TX = f(t_new)
    if normalize_density:
        print(density_TX.max())
        density_TX /= density_TX.max()  # normalize so that k_max=1
    density_BTX = torch.tensor(density_TX.copy()).unsqueeze(0)  # add batch dimension B=1
    problem = Problem.create_from_data(density_BTX, dx=data_dx, dt=dt or data_dt, kmax=density_TX.max())
    return problem
