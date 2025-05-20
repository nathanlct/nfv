import contextlib
import os

import numpy as np
import torch
from tqdm import tqdm

from nfv.flows import Flow
from nfv.initial_conditions import InitialCondition
from nfv.utils.tensor import batch_iterator


class Problem:
    def __init__(
        self,
        nx: int,
        nt: int,
        dx: float,
        dt: float,
        ic: InitialCondition | list[InitialCondition],
        flow: Flow,
        kmax: float = None,
    ):
        """
        nx: number of cells in space (including boundary conditions)
        nt: number of cells in time (including initial condition)
        dx: float cell size in space
        dt: float cell size in time
        ic: initial condition(s)
        flow: flow model
        device: device to use for computations (cpu by default)
        """
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.ic = ic if isinstance(ic, list) else [ic]
        self.flow = flow
        self.kmax = kmax or (flow and self.flow.kmax)

        self.solutions = {}

    @classmethod
    def create_from_data(cls, data, dx, dt, kmax):
        B, T, X = data.shape
        problem = cls(nx=X, nt=T, dx=dx, dt=dt, ic=None, flow=None, kmax=kmax)
        problem.solutions["ground_truth"] = data
        return problem

    def solve(
        self,
        solver,
        dtype=torch.float64,
        device="cpu",
        save=None,
        boundaries=None,
        batch_size=None,
        grad=False,
        load_maybe=None,
        force_recompute=False,
        flow=None,
        progressbar=False,
        **kwargs,
    ):
        """
        Args:
            save: save this solution under this name (if not None), eg for use with the boundaries argument
            boundaries: if None, use ghost cells. Otherwise, set to a previously-computed solution (eg "LaxHopf") to use that as ground truth boundaries.
        """
        context = torch.no_grad() if not grad else contextlib.nullcontext()
        if load_maybe is not None and save is not None:
            load_path = os.path.join(load_maybe, f"{save}.pt")
        else:
            load_path = None

        if load_path and os.path.exists(load_path) and not force_recompute:
            solution = torch.load(load_path).to(dtype).to(device)
        else:
            with context:
                solution = []
                iterator = enumerate(
                    batch_iterator(self.ic, self.solutions[boundaries].to(dtype).to(device) if boundaries else [], batch_size=batch_size)
                )
                if progressbar:
                    iterator = tqdm(iterator, total=len(self.ic) // batch_size)
                for i, (ic_batch, bounds) in iterator:
                    if boundaries and boundaries not in self.solutions:
                        raise ValueError(f"Boundaries {boundaries} not found in saved solutions")
                    solution.append(
                        solver(
                            ic=ic_batch,
                            nx=self.nx,
                            nt=self.nt,
                            dx=self.dx,
                            dt=self.dt,
                            flow=flow or self.flow,
                            kmax=self.kmax,
                            boundaries_BTX=bounds,  # boundaries and self.solutions[boundaries].to(dtype).to(device),
                            dtype=dtype,
                            device=device,
                            iterative=grad,
                            **kwargs,
                        )
                    )
                solution = torch.cat(solution).to(dtype).to(device) if batch_size is not None else solution[0]
                if load_path:
                    os.makedirs(os.path.dirname(load_path), exist_ok=True)
                    torch.save(solution.to(torch.float32), load_path)
        if save:
            self.solutions[save] = solution
        return solution

    @property
    def cfl(self) -> float:
        """Courant-Friedrichs-Lewy condition"""
        return self.dt / self.dx
