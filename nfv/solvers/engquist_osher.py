import torch


def EngquistOsher(k_BX, flow_fn):
    flow_BX = flow_fn(k_BX)
    cond1 = torch.logical_and(k_BX[:, :-1] < flow_fn.k_crit, k_BX[:, 1:] < flow_fn.k_crit)
    cond2 = torch.logical_and(k_BX[:, :-1] > flow_fn.k_crit, k_BX[:, 1:] > flow_fn.k_crit)
    cond3 = torch.logical_and(k_BX[:, :-1] > flow_fn.k_crit, k_BX[:, 1:] <= flow_fn.k_crit)
    flows = torch.where(
        cond1,
        flow_BX[:, :-1],
        torch.where(
            cond2,
            flow_BX[:, 1:],
            torch.where(
                cond3,
                flow_fn(flow_fn.k_crit),
                flow_BX[:, :-1] + flow_BX[:, 1:] - flow_fn(flow_fn.k_crit),
            ),
        ),
    )
    return flows
