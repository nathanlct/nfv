import torch


def Godunov(k_BX, flow_fn):
    flow_BX = flow_fn(k_BX)
    cond1 = k_BX[:, :-1] <= k_BX[:, 1:]
    cond2 = k_BX[:, 1:] > flow_fn.k_crit
    cond3 = flow_fn.k_crit > k_BX[:, :-1]
    flux_BX = torch.where(
        cond1,
        torch.minimum(flow_BX[:, :-1], flow_BX[:, 1:]),
        torch.where(cond2, flow_BX[:, 1:], torch.where(cond3, flow_BX[:, :-1], flow_fn(flow_fn.k_crit))),
    )
    return flux_BX
