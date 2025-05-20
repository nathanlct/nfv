import numpy as np
import torch


class FVM:
    def __init__(self, flux_fn, **kwargs):
        self.flux_fn = flux_fn
        self.kwargs = kwargs

    def __str__(self):
        return f"FVM({self.flux_fn.__name__})"

    def __call__(self, *args, **kwargs):
        kwargs.update(self.kwargs)
        return _finite_volume(self.flux_fn, *args, **kwargs)

    def __name__(self):
        return f"FVM({self.flux_fn.__name__})"


def _finite_volume(
    flux_fn,
    ic,
    nx,
    nt,
    dx,
    dt,
    flow,
    kmax,
    boundaries_BTX,
    dtype,
    device,
    iterative=False,
    boundary_size=1,
    boundary_pad=2,
    T=1,
    flipX=False,
    ghost_cells=0,
):
    """
    Args
        solutions: specify ground truth solutions to use for boundary conditions
        iterative: builds solution iteratively without in-place modifications (for safe backpropagation)

        roughly, the densities are updated on x[boundary_size+boundary_pad:-(boundary_size+boundary_pad)]
        and the input of the scheme is x[boundary_pad:-boundary_pad]
    """
    # Start with initial condition
    # ic_discretized = np.array([ic.discretize(nx) for ic in ic])
    # ic_discretized = torch.from_numpy(ic_discretized).to(dtype).to(device)
    bst = boundary_size + boundary_pad
    # assert bst == 3

    if iterative:
        assert T == 1
        assert ghost_cells == 0
        k_BTX = boundaries_BTX[:, 0, :].unsqueeze(1)

        for t in range(nt - 1):
            # predict t+1 from t
            flows_BX = torch.max(0.0, flux_fn(k_BTX[:, -1, :], flow))  # JUST ADDED THIS CLIPPING
            new_k_BX = k_BTX[:, -1, 1:-1] + dt / dx * (flows_BX[:, :-1] - flows_BX[:, 1:])
            if boundaries_BTX is None:
                bc_left, bc_right = new_k_BX[:, 0], new_k_BX[:, -1]
            else:
                bc_left, bc_right = boundaries_BTX[:, t + 1, 0], boundaries_BTX[:, t + 1, -1]
            new_k_BX = torch.concat([bc_left.unsqueeze(1).unsqueeze(1), new_k_BX.unsqueeze(1), bc_right.unsqueeze(1).unsqueeze(1)], dim=2)
            new_k_BX = torch.clamp(new_k_BX, 0.0, flow.kmax)

            k_BTX = torch.concat([k_BTX, new_k_BX], dim=1)
        return k_BTX
    else:
        # k_BTX = torch.zeros((len(ic), nt, nx)).to(dtype).to(device)
        # k_BTX[:, 0, :] = ic_discretized

        k_BTX = torch.clone(boundaries_BTX)

        if ghost_cells > 0:
            # ghost cells that copy neighbouring cells
            cells_left = k_BTX[:, :, [0]]
            cells_right = k_BTX[:, :, [-1]]

            cells_left = cells_left.repeat(1, 1, ghost_cells)
            cells_right = cells_right.repeat(1, 1, ghost_cells)

            k_BTX = torch.concat([cells_left, k_BTX, cells_right], dim=2)

        if flipX:
            k_BTX = k_BTX.flip(2)

        for t in range(T, nt):  # we are predicting t from (t-1, ..., t-T)
            # compute flows (X cells => X-1 flows)

            slice_x_input = slice(boundary_pad, -boundary_pad) if boundary_pad != 0 else slice(None, None)

            if T == 1:
                flows_BX = flux_fn(k_BTX[:, t - 1, slice_x_input], flow)
            else:
                flows_BX = flux_fn(k_BTX[:, t - T : t, slice_x_input], flow)

            # Euler update: k(t+1) = k(t) + dt/dx * (flows_in - flows_out)
            k_BTX[:, t, bst:-bst] = torch.clamp(k_BTX[:, t - 1, bst:-bst] + dt / dx * (flows_BX[:, :-1] - flows_BX[:, 1:]), 0.0, kmax)

        if flipX:
            k_BTX = k_BTX.flip(2)

        if ghost_cells > 0:
            k_BTX = k_BTX[:, :, ghost_cells:-ghost_cells]
        return k_BTX
