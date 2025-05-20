import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

from nfv import Problem
from nfv.flows import Greenberg, Greenshield, Trapezoidal, Triangular, TriangularSkewed, Underwood
from nfv.initial_conditions import PiecewiseConstant, Riemann
from nfv.solvers import DG, ENO, FVM, WENO, EngquistOsher, Godunov, LaxFriedrichs, LaxHopf
from nfv.utils.plotting import plot_heatmap


def create_solution_animation(sol_lh, dt, dx, save_path="solution_animation.mp4", duration=5, fps=30, curve_labels=None):
    dt = 100 * dt  # for prettiness
    dx = 100 * dx  # for prettiness
    start_time = time.time()

    print("Solution shape:", sol_lh.shape)

    # create figure with white background
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    ax.set_facecolor("white")

    # axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    # labels
    ax.set_xlabel("Space (x)", color="black", fontsize=12, fontfamily="serif")
    ax.set_ylabel("Solution u(t, x)", color="black", fontsize=12, fontfamily="serif")
    title = ax.set_title("Solution at t = 0.00", color="black", fontsize=14, fontfamily="serif", pad=20)

    n_frames = duration * fps
    frame_indices = np.linspace(0, sol_lh.shape[1] - 1, n_frames, dtype=int)
    times = frame_indices * dt

    x = np.linspace(0, dx * sol_lh.shape[2], sol_lh.shape[2])
    if curve_labels is None:
        curve_labels = [f"Curve {i + 1}" for i in range(sol_lh.shape[0])]

    lines = []
    # color palette
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
    ]

    if curve_labels and curve_labels[0] == "Exact solution":
        # plot numerical schemes
        for i in range(1, sol_lh.shape[0]):
            color = colors[(i - 1) % len(colors)]
            line = ax.plot(x, sol_lh[i, 0], label=curve_labels[i], color=color, alpha=0.9, linewidth=1.5)[0]
            lines.append(line)
        # plot exact solution
        lh_line = ax.plot(x, sol_lh[0, 0], label="Exact solution", color="black", linewidth=2.5, zorder=4)[0]
        lines = [lh_line] + lines
    else:
        # plot all curves
        for i in range(sol_lh.shape[0]):
            color = colors[i % len(colors)]
            line = ax.plot(x, sol_lh[i, 0], label=curve_labels[i] if curve_labels else f"Curve {i + 1}", color=color, alpha=0.9, linewidth=2)[0]
            lines.append(line)

    # legend
    legend = ax.legend(
        frameon=False,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    for text in legend.get_texts():
        text.set_color("black")
        text.set_fontfamily("serif")

    # axis limits
    ax.set_xlim(0, dx * sol_lh.shape[2])
    y_min = np.min(sol_lh)
    y_max = np.max(sol_lh)
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

    # tick styling
    ax.tick_params(colors="black", labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")

    # layout
    fig.tight_layout()
    plt.subplots_adjust(right=0.82)

    def update(frame_idx):
        frame = frame_indices[frame_idx]
        for i, line in enumerate(lines):
            line.set_ydata(sol_lh[i, frame])
        title.set_text(f"Solution at t = {times[frame_idx]:.2f}")
        return lines + [title]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=True,
    )

    anim.save(save_path, writer="ffmpeg", fps=fps)
    plt.close()

    end_time = time.time()
    compute_time = end_time - start_time
    file_size = os.path.getsize(save_path) / (1024 * 1024)

    print(f"Video saved to: {os.path.abspath(save_path)}")
    print(f"Computation time: {compute_time:.2f} seconds")
    print(f"File size: {file_size:.2f} MB\n")


# rarefaction and shock waves
for ic_name, ic in [("rarefaction", Riemann(1, 0)), ("shock", Riemann(0.1, 0.7))]:
    file_name = f"vid_{ic_name}_comparison.mp4"
    if os.path.exists(file_name):
        continue

    flow = Greenshield()
    problem = Problem(nx=200, nt=2002, dx=1e-3, dt=1e-4, ic=ic, flow=flow)

    sol_lh = problem.solve(LaxHopf, batch_size=1, save="lax_hopf", dtype=torch.float64).cpu().numpy().squeeze()
    sol_gd = problem.solve(FVM(Godunov), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_lf = problem.solve(FVM(LaxFriedrichs(dx=problem.dx, dt=problem.dt)), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_eo = problem.solve(FVM(EngquistOsher), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_eno = problem.solve(FVM(ENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy().squeeze()
    sol_weno = problem.solve(FVM(WENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy().squeeze()
    sol_dg = problem.solve(DG, boundaries="lax_hopf", dtype=torch.float32).cpu().numpy().squeeze()

    solutions = np.stack([sol_lh, sol_gd, sol_lf, sol_eo, sol_eno, sol_weno, sol_dg])
    solutions[:, 0, :] = sol_gd[0, :]
    solutions = solutions[:, :-1, :]

    create_solution_animation(
        solutions,
        dt=problem.dt,
        dx=problem.dx,
        save_path=file_name,
        duration=5,
        fps=30,
        curve_labels=["Exact solution", "Godunov", "Lax-Friedrichs", "Engquist-Osher", "ENO", "WENO", "DG"],
    )


# compare flows
file_name = "vid_flows_comparison.mp4"
if not os.path.exists(file_name):
    piecewise_ic = PiecewiseConstant(np.random.uniform(0.0, 1.0, 10))

    flows = [Greenshield(), Triangular(), TriangularSkewed(), Trapezoidal(), Underwood(), Greenberg()]

    solutions = []
    for flow in flows:
        problem = Problem(nx=200, nt=1001, dx=1e-3, dt=1e-4, ic=piecewise_ic, flow=flow)
        sol = problem.solve(LaxHopf, batch_size=1, save="lax_hopf", dtype=torch.float64).cpu().numpy().squeeze()
        solutions.append(sol)

    solutions = np.stack(solutions)

    create_solution_animation(
        solutions,
        dt=problem.dt,
        dx=problem.dx,
        save_path=file_name,
        duration=10,
        fps=30,
        curve_labels=["Greenshields", "Triangular 1", "Triangular 2", "Trapezoidal", "Underwood", "Greenberg"],
    )


# compare schemes for each flow
flows = [Greenshield(), Triangular(), TriangularSkewed(), Trapezoidal(), Underwood(), Greenberg()]
flow_names = ["Greenshields", "Triangular 1", "Triangular 2", "Trapezoidal", "Underwood", "Greenberg"]

piecewise_ic2 = PiecewiseConstant(np.random.uniform(0.0, 1.0, 10))

for flow, flow_name in zip(flows, flow_names):
    file_name = f"vid_piecewise_{flow_name.lower().replace(' ', '_')}_schemes.mp4"
    if os.path.exists(file_name):
        continue

    problem = Problem(nx=200, nt=2002, dx=1e-3, dt=1e-4, ic=piecewise_ic2, flow=flow)

    sol_lh = problem.solve(LaxHopf, batch_size=1, save="lax_hopf", dtype=torch.float64).cpu().numpy().squeeze()
    sol_gd = problem.solve(FVM(Godunov), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_lf = problem.solve(FVM(LaxFriedrichs(dx=problem.dx, dt=problem.dt)), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_eo = problem.solve(FVM(EngquistOsher), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_eno = problem.solve(FVM(ENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy().squeeze()
    sol_weno = problem.solve(FVM(WENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy().squeeze()
    sol_dg = problem.solve(DG, boundaries="lax_hopf", dtype=torch.float32).cpu().numpy().squeeze()

    solutions = np.stack([sol_lh, sol_gd, sol_lf, sol_eo, sol_eno, sol_weno, sol_dg])
    solutions[:, 0, :] = sol_gd[0, :]
    solutions = solutions[:, :-1, :]

    create_solution_animation(
        solutions,
        dt=problem.dt,
        dx=problem.dx,
        save_path=file_name,
        duration=10,
        fps=30,
        curve_labels=["Exact solution", "Godunov", "Lax-Friedrichs", "Engquist-Osher", "ENO", "WENO", "DG"],
    )


# generate heatmaps
for flow, flow_name in zip(flows, flow_names):
    file_name = f"heatmap_{flow_name.lower().replace(' ', '_')}_schemes.png"

    problem = Problem(nx=200, nt=2002, dx=1e-3, dt=1e-4, ic=piecewise_ic2, flow=flow)

    sol_lh = problem.solve(LaxHopf, batch_size=1, save="lax_hopf", dtype=torch.float64).cpu().numpy().squeeze()
    sol_gd = problem.solve(FVM(Godunov), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_lf = problem.solve(FVM(LaxFriedrichs(dx=problem.dx, dt=problem.dt)), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_eo = problem.solve(FVM(EngquistOsher), boundaries="lax_hopf").cpu().numpy().squeeze()
    sol_eno = problem.solve(FVM(ENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy().squeeze()
    sol_weno = problem.solve(FVM(WENO), boundaries="lax_hopf", boundary_size=3, boundary_pad=0).cpu().numpy().squeeze()
    sol_dg = problem.solve(DG, boundaries="lax_hopf", dtype=torch.float32).cpu().numpy().squeeze()

    solutions = np.stack([sol_lh, sol_gd, sol_lf, sol_eo, sol_eno, sol_weno, sol_dg])
    solutions = solutions[:, :-1, :]

    # l2 differences
    print(f"\nL2 differences for {flow_name}:")
    print("Godunov:      ", np.mean(np.square(sol_lh - sol_gd)))
    print("Lax-Friedrichs:", np.mean(np.square(sol_lh - sol_lf)))
    print("Engquist-Osher:", np.mean(np.square(sol_lh - sol_eo)))
    print("ENO:          ", np.mean(np.square(sol_lh - sol_eno)))
    print("WENO:         ", np.mean(np.square(sol_lh - sol_weno)))
    print("DG:           ", np.mean(np.square(sol_lh - sol_dg)))

    # plot heatmap
    plot_heatmap(
        [solutions],
        title_col=["Lax-Hopf", "Godunov", "Lax-Friedrichs", "Engquist-Osher", "ENO", "WENO", "DG"],
        transpose=False,
        path=file_name,
        xlabel=False,
        ylabel=False,
        cbar=False,
        fontsize=30,
    )
