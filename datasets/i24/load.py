import torch

from nfv import Problem


def load_i24_data(name="11-29-2022"):
    data_dx = 64.3736  # m
    data_dt = 10.0  # s
    density_TX = torch.load(f"datasets/i24/train/{name}_post_TX.pt")
    density_BTX = density_TX.unsqueeze(0)  # batch dim
    density_BTX = density_BTX * 140  # veh/km/lane
    density_BTX = density_BTX / 1000  # veh/m/lane
    problem = Problem.create_from_data(density_BTX, dx=data_dx, dt=data_dt, kmax=density_BTX.max())
    return problem


if __name__ == "__main__":
    load_i24_data()
