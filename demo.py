import numpy as np
import torch

from nfv import Problem
from nfv.flows import Greenberg, Greenshield, Trapezoidal, Triangular, TriangularSkewed, Underwood
from nfv.initial_conditions import PiecewiseConstant, Riemann
from nfv.solvers import DG, ENO, FVM, WENO, EngquistOsher, Godunov, LaxFriedrichs, LaxHopf
from nfv.utils.plotting import plot_heatmap

# 0. Plot a shock and a rarefaction for Lax hopf and greenshieds flow.
rarefaction_ic = Riemann(1, 0)
shockwave_ic = Riemann(0.1, 0.7)
flow = Greenshield()

problem = Problem(nx=200, nt=1000, dx=1e-3, dt=1e-4, ic=[rarefaction_ic, shockwave_ic], flow=flow)
sol_lh = problem.solve(LaxHopf, batch_size=1, save="lax_hopf", dtype=torch.float64).cpu().numpy()

plot_heatmap(
    [sol_lh],
    title_col=["Rarefaction wave", "Shock wave"],
    transpose=False,
    path="sol_simple.png",
    xlabel=True,
    ylabel=True,
    cbar=True,
)


# 1. Solve two riemann problems for each flow-scheme pair: a rarefaction and a shock
for ic_name, ic in [("Rarefaction", rarefaction_ic), ("Shock", shockwave_ic)]:
    sols = []
    for flow in [Greenshield(), Triangular(), TriangularSkewed(), Trapezoidal(), Greenberg(), Underwood()]:
        problem = Problem(nx=200, nt=1000, dx=1e-3, dt=1e-4, ic=ic, flow=flow)

        # Generate all solutions
        sol_lh = problem.solve(LaxHopf, batch_size=1, save="lax_hopf", dtype=torch.float64).cpu().numpy()
        sol_gd = problem.solve(FVM(Godunov), boundaries="lax_hopf").cpu().numpy()
        sol_lf = problem.solve(FVM(LaxFriedrichs(dx=problem.dx, dt=problem.dt)), boundaries="lax_hopf").cpu().numpy()
        sol_eo = problem.solve(FVM(EngquistOsher), boundaries="lax_hopf").cpu().numpy()
        sol_eno = problem.solve(FVM(ENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy()
        sol_weno = problem.solve(FVM(WENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy()
        sol_dg = problem.solve(DG, boundaries="lax_hopf", dtype=torch.float32).cpu().numpy()

        sols.append([sol_lh, sol_gd, sol_lf, sol_eo, sol_eno, sol_weno, sol_dg])

    plot_heatmap(
        sols,
        title_col=["Lax-Hopf", "Godunov", "Lax-Friedrichs", "Engquist-Osher", "ENO", "WENO", "DG"],
        title_row=["Greenshield", "Triangular", "Triangular 2", "Trapezoidal", "Greenberg", "Underwood"],
        transpose=False,
        path=f"sol_riemann_{ic_name}.png",
        xlabel=False,
        ylabel=False,
        cbar=False,
        fontsize=30,
    )


# 2. Generate several piecewise-constant conditions for each flow-scheme pair
for flow in [Greenshield(), Triangular(), TriangularSkewed(), Trapezoidal(), Greenberg(), Underwood()]:
    # Generate 5 random initial conditions with 10 pieces each
    np.random.seed(42)
    ics = [PiecewiseConstant(np.random.rand(10)) for _ in range(5)]

    problem = Problem(nx=200, nt=1000, dx=1e-3, dt=1e-4, ic=ics, flow=flow)

    # Generate all solutions
    sol_lh = problem.solve(LaxHopf, batch_size=1, save="lax_hopf", dtype=torch.float64).cpu().numpy()
    sol_gd = problem.solve(FVM(Godunov), boundaries="lax_hopf").cpu().numpy()
    sol_lf = problem.solve(FVM(LaxFriedrichs(dx=problem.dx, dt=problem.dt)), boundaries="lax_hopf").cpu().numpy()
    sol_eo = problem.solve(FVM(EngquistOsher), boundaries="lax_hopf").cpu().numpy()
    sol_eno = problem.solve(FVM(ENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy()
    sol_weno = problem.solve(FVM(WENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy()
    sol_dg = problem.solve(DG, boundaries="lax_hopf", dtype=torch.float32).cpu().numpy()

    # Accuracy of schemes
    print(f"\nFlow: {flow}")
    print("GD  ", np.mean(np.abs(sol_lh - sol_gd)), np.mean(np.square(sol_lh - sol_gd)))
    print("LF  ", np.mean(np.abs(sol_lh - sol_lf)), np.mean(np.square(sol_lh - sol_lf)))
    print("EO  ", np.mean(np.abs(sol_lh - sol_eo)), np.mean(np.square(sol_lh - sol_eo)))
    print("ENO ", np.mean(np.abs(sol_lh - sol_eno)), np.mean(np.square(sol_lh - sol_eno)))
    print("WENO", np.mean(np.abs(sol_lh - sol_weno)), np.mean(np.square(sol_lh - sol_weno)))
    print("DG  ", np.mean(np.abs(sol_lh - sol_dg)), np.mean(np.square(sol_lh - sol_dg)))

    plot_heatmap(
        [sol_lh, sol_gd, sol_lf, sol_eo, sol_eno, sol_weno, sol_dg],
        title_col=["Lax-Hopf", "Godunov", "Lax-Friedrichs", "Engquist-Osher", "ENO", "WENO", "DG"],
        transpose=True,
        path=f"sol_{str(flow)}.png",
        xlabel=False,
        ylabel=False,
        cbar=False,
        fontsize=30,
    )
