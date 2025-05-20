import argparse
import datetime
import gc
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from rich.console import Console
from rich.table import Table

from models.load_model import load_model, model_list
from nfv import Problem
from nfv.flows import Greenberg, Greenshield, Trapezoidal, Triangular, TriangularSkewed, Underwood
from nfv.initial_conditions import PiecewiseConstant
from nfv.solvers import DG, ENO, FVM, WENO, EngquistOsher, Godunov, LaxFriedrichs, LaxHopf
from nfv.utils.device import get_device
from nfv.utils.formatting import format, str_beautify
from nfv.utils.plotting import plot_agg, plot_heatmap


def parse_args():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--logdir", type=str, default="logs/", help="Base path where to save eval logs.")
    parser.add_argument("--datadir", type=str, default="data/", help="Base path where to save precomputed data.")

    # computations
    parser.add_argument("--device", type=str, default="default", help="Device to use (default will use CUDA if available, otherwise CPU)")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])

    # data
    parser.add_argument("--data", default=False, action="store_true", help="If set, generate new data based on the parameters below.")
    parser.add_argument("--nx", type=int, default=200, help="Number of space cells.")
    parser.add_argument("--nt", type=int, default=1000, help="Number of time cells.")
    parser.add_argument("--dx", type=float, default=1e-3, help="Space discretization.")
    parser.add_argument("--dt", type=float, default=1e-4, help="Time discretization.")
    parser.add_argument("--n", type=int, default=100, help="Number of initial conditions to generate.")
    parser.add_argument("--pieces", type=int, default=10, help="Number of pieces in each piecewise-constant initial condition.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to generate data with.")
    force_help_str = "Data to regenerate even if it's already pre-computed (e.g. all/all, greenshield/all, all/supervised)"
    parser.add_argument("--force", nargs="+", type=str, default=None, help=force_help_str)
    parser.add_argument("--flow", nargs="+", type=str, default=None, help="Flows to use (by default, use all).")

    # plots
    plot_help_str = (
        "If set, list of plots to compute. Set to 'all' to generate all plots available."
        "Available plots include: flux, table, err_time, winrates, box, heatmap, density."
    )
    parser.add_argument("--plot", nargs="+", type=str, default=None, help=plot_help_str)
    parser.add_argument("--tmp", action="store_true", help="If set, save plots to tmp directory")

    return parser.parse_args()


def match(flow, scheme, forces):
    if forces:
        for force in forces:
            f1, f2 = force.split("/")
            if f1 == flow or f1 == "all":
                if f2 == scheme or f2 == "all":
                    return True
    return False


def print_memory_usage(prefix=""):
    # RAM
    mem = psutil.virtual_memory()
    ram_used_gb = mem.used / 1e9
    ram_total_gb = mem.total / 1e9
    ram_percent = mem.percent
    print(f"{prefix}RAM: {ram_percent:.1f}% used ({ram_used_gb:.2f}GB / {ram_total_gb:.2f}GB)", end="")

    # Try VRAM (GPU)
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used_gb = meminfo.used / 1e9
        vram_total_gb = meminfo.total / 1e9
        vram_percent = 100 * meminfo.used / meminfo.total
        print(f", VRAM: {vram_percent:.1f}% used ({vram_used_gb:.2f}GB / {vram_total_gb:.2f}GB)")
        pynvml.nvmlShutdown()
    except ModuleNotFoundError:
        print(", pynvml not installed")
    except Exception:
        print("")  # just newline if no GPU


def generate_data(data_dir, flows, models, args):
    # dir
    print(f"Saving data at {data_dir}")

    # schemes
    baselines = {
        "dg": DG,
        "godunov": FVM(Godunov, boundary_size=1, boundary_pad=2),
        "lax_friedrichs": FVM(LaxFriedrichs(dx=args.dx, dt=args.dt), boundary_size=1, boundary_pad=2),
        "engquist_osher": FVM(EngquistOsher, boundary_size=1, boundary_pad=2),
        "eno": FVM(ENO, boundary_size=3, boundary_pad=0),
        "weno": FVM(WENO, boundary_size=3, boundary_pad=0),
    }

    # initial conditions
    np.random.seed(args.seed)
    ics = [PiecewiseConstant(np.random.rand(args.pieces)) for _ in range(args.n)]

    # compute (or load precomputed) solutions
    for flow in flows:
        problem = Problem(nx=args.nx, nt=args.nt, dx=args.dx, dt=args.dt, ic=ics, flow=flow)
        print(f"Generating eval data for {flow}...")
        print_memory_usage(prefix="\t")
        t0 = time.time()
        schemes = baselines | {k: FVM(v) for k, v in models[flow].items()}
        data_dir_flow = os.path.join(data_dir, str(flow))
        print(f"{data_dir_flow=}, {len(problem.ic)}")
        print("\tlax hopf...")
        problem.solve(
            LaxHopf,
            batch_size=1,
            save="lax_hopf",
            load_maybe=data_dir_flow,
            device=args.device,
            dtype=torch.float64,
            force_recompute=match(str(flow), "lax_hopf", args.force),
            progressbar=True,
        )
        print_memory_usage(prefix="\t")
        for name, scheme in schemes.items():
            print(f"\t{name}...")
            problem.solve(
                scheme,
                boundaries="lax_hopf",
                save=name,
                load_maybe=data_dir_flow,
                device=args.device,
                dtype=args.dtype,
                force_recompute=match(str(flow), name, args.force),
                batch_size=50 if name == "dg" else None,
            )
            problem.solutions[name] = problem.solutions[name].to("cpu")
            print_memory_usage(prefix="\t")
        problem.solutions["lax_hopf"] = problem.solutions["lax_hopf"].to("cpu")
        print(f"\t{time.time() - t0:.3f}s, {sum(x.numel() * x.element_size() for x in problem.solutions.values()) / 1e9} GB")

        # precompute things that benefit from GPU
        # metrics
        solutions = {k: v[:, 1:, 3:-3] for k, v in problem.solutions.items()}  # remove boundary conditions from metrics
        abs_diff = {scheme: torch.abs(solutions["lax_hopf"] - solutions[scheme]) for scheme in schemes}
        errors_l1 = {scheme: torch.mean(abs_diff[scheme], axis=(1, 2)) for scheme in schemes}
        errors_l2 = {scheme: torch.mean(torch.square(solutions["lax_hopf"] - solutions[scheme]), axis=(1, 2)) for scheme in schemes}
        errors_rel = {
            scheme: torch.mean(abs_diff[scheme] / torch.maximum(torch.abs(solutions["lax_hopf"]), torch.tensor(1e-2)), axis=(1, 2))
            for scheme in schemes
        }
        torch.save(errors_l1, os.path.join(data_dir_flow, "errors_l1.pt"))
        torch.save(errors_l2, os.path.join(data_dir_flow, "errors_l2.pt"))
        torch.save(errors_rel, os.path.join(data_dir_flow, "errors_rel.pt"))

        # nn output
        for name, model in models[flow].items():
            ks = torch.linspace(0.0, 1.0, 1001).to(args.dtype)
            inputs = torch.cartesian_prod(ks, ks).to("cpu")  # (B, 2)
            outputs = model.to("cpu")(inputs)  # (B, 1)
            torch.save({"inputs": inputs, "outputs": outputs}, os.path.join(data_dir_flow, f"nn_{name}_outputs.pt"))

        del solutions
        del abs_diff
        del errors_l1
        del errors_l2
        del errors_rel
        del ks
        del inputs
        del outputs
        del problem

        gc.collect()
        torch.cuda.empty_cache()


def plot_numerical_flux(data_dir, log_dir, flows, models, args):
    # load model outputs
    s_out = ""
    for flow in flows:
        s_out += f"{str_beautify(flow)}\n"
        for name, model in models[flow].items():
            x = torch.load(os.path.join(data_dir, str(flow), f"nn_{name}_outputs.pt"))
            outputs = x["outputs"].to(args.dtype).to(args.device)
            s_out += f"\t{str_beautify(name)}: min {outputs.min():.3f}, max {outputs.max():.3f}, amplitude {outputs.max() - outputs.min():.3f}, qmax {flow.qmax:.3f}\n"
    with open(os.path.join(log_dir, "model_outputs.txt"), "w") as f:
        f.write(s_out)

    # compute F(x, x) (which should be = f(x) after shifting the model)
    ks = torch.linspace(0.0, 1.0, 10_000).to(args.dtype).to(args.device)
    for flow in flows:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=500)
        ax.plot(ks, flow.q(ks), label=str_beautify(flow))
        for name, model in models[flow].items():
            outputs = model(torch.stack([ks, ks], dim=1))
            # outputs -= outputs.min()
            ax.plot(ks, outputs, label=str_beautify(name))
        ax.legend()
        ax.grid()
        ax.set_ylim(bottom=0)
        plt.savefig(os.path.join(log_dir, f"flux_Fxx_{flow}.png"))
        plt.close(fig)

    # compute F(a, x) and F(x, b) for different fixed a and b (cross sections)
    for flow in flows:
        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(5, 10), dpi=500)
        for i, fixed_val in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
            fixed_val_vec = torch.full_like(ks, fixed_val)
            for name, model in ({"godunov": lambda x: Godunov(x, flow)} | models[flow]).items():
                outputs_left = model(torch.stack([fixed_val_vec, ks], dim=1))
                ax[i, 0].plot(ks, outputs_left, label=str_beautify(name), alpha=0.8)
                outputs_right = model(torch.stack([ks, fixed_val_vec], dim=1))
                ax[i, 1].plot(ks, outputs_right, label=str_beautify(name), alpha=0.8)
                for k in [0, 1]:
                    ax[i, k].axvline(fixed_val, color="red", linestyle="--", linewidth=0.5)
                # write both outputs to csv
                np.savetxt(os.path.join(log_dir, f"flux_cross_{flow}_{name}_{fixed_val:.2f}_x.csv"), outputs_left.numpy(), delimiter=",")
                np.savetxt(os.path.join(log_dir, f"flux_cross_{flow}_{name}_x_{fixed_val:.2f}.csv"), outputs_right.numpy(), delimiter=",")
            ax[i, 0].set_title(rf"$\mathcal{{F}}({fixed_val:.2f}, x)$")
            ax[i, 1].set_title(rf"$\mathcal{{F}}(x, {fixed_val:.2f})$")
            for k in [0, 1]:
                ax[i, k].legend(fontsize="x-small")
                ax[i, k].grid()
                ax[i, k].set_ylim(-0.05, flow.qmax * 1.5)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"flux_cross_{flow}.png"))
        plt.close(fig)


def plot_table(data_dir, log_dir, flows, models, args):
    table_tex = ""
    console = Console(record=True, file=open(os.devnull, "wt"), width=180)
    schemes = ["supervised", "unsupervised", "godunov", "lax_friedrichs", "engquist_osher", "eno", "weno", "dg"]

    for flow in flows:
        l1_errors = torch.load(os.path.join(data_dir, str(flow), "errors_l1.pt"))
        l2_errors = torch.load(os.path.join(data_dir, str(flow), "errors_l2.pt"))
        rel_errors = torch.load(os.path.join(data_dir, str(flow), "errors_rel.pt"))

        # TABLE: metrics (tex output)
        table_tex += "\\multirow{3}{*}{" + str_beautify(flow) + "}"
        for name, vals in zip(["L1", "L2", "Rel."], [l1_errors, l2_errors, rel_errors]):
            table_tex += f" & {name}"
            min_val_comp = min(torch.mean(vals[scheme]) for scheme in ["godunov", "lax_friedrichs", "engquist_osher", "supervised", "unsupervised"])
            for scheme in schemes:
                table_tex += f" & {format(vals[scheme], latex=True, bold_val=min_val_comp)}"
            table_tex += " \\\\\n"
        table_tex += "\t\\Xhline{1.pt}\n"

        # TABLE: metrics (html output)
        winrates_l1 = [torch.sum(l1_errors[scheme] < l1_errors["godunov"]) / len(l1_errors[scheme]) for scheme in schemes]
        winrates_l2 = [torch.sum(l2_errors[scheme] < l2_errors["godunov"]) / len(l2_errors[scheme]) for scheme in schemes]
        improvements_l1 = [torch.mean(l1_errors["godunov"]) / torch.mean(l1_errors[scheme]) for scheme in schemes]
        improvements_l2 = [torch.mean(l2_errors["godunov"]) / torch.mean(l2_errors[scheme]) for scheme in schemes]

        table = Table(title=str_beautify(flow))
        table.add_column("Metric", justify="right", style="cyan")
        for scheme in schemes:
            table.add_column(str_beautify(scheme), style="magenta" if "supervised" in scheme else "green", justify="center")
        table.add_row("L1 error", *[format(l1_errors[scheme]) for scheme in schemes])
        table.add_row("L2 error", *[format(l2_errors[scheme]) for scheme in schemes])
        table.add_row("Relative error", *[format(rel_errors[scheme]) for scheme in schemes])
        table.add_row("Winrates (L1 against GD)", *[f"{100 * x:.1f}%" for x in winrates_l1])
        table.add_row("Winrates (L2 against GD)", *[f"{100 * x:.1f}%" for x in winrates_l2])
        table.add_row(r"improve ratio over GD (L1)", *[f"{x:.2f}" for x in improvements_l1])
        table.add_row(r"improve ratio over GD (L2)", *[f"{x:.2f}" for x in improvements_l2])
        console.print()
        console.print(table, justify="center")
        console.print()

    table_tex.replace("Triangular Skewed", "\\shortstack{Triangular\\\\Skewed}")
    with open(os.path.join(log_dir, "metrics_table.tex"), "w") as f:
        f.write(table_tex)
    console.save_html(os.path.join(log_dir, "metrics_table.html"))


def plot_err_time(data_dir, log_dir, flows, models, args):
    for flow in flows:
        # BTX
        supervised = torch.load(os.path.join(data_dir, str(flow), "supervised.pt"))[:, 1:, 3:-3].to(args.device)
        unsupervised = torch.load(os.path.join(data_dir, str(flow), "unsupervised.pt"))[:, 1:, 3:-3].to(args.device)
        godunov = torch.load(os.path.join(data_dir, str(flow), "godunov.pt"))[:, 1:, 3:-3].to(args.device)
        lax_hopf = torch.load(os.path.join(data_dir, str(flow), "lax_hopf.pt"))[:, 1:, 3:-3].to(args.device)

        fig, axes = plt.subplots(ncols=2, figsize=(12, 4), dpi=300)
        for name, solutions in [("supervised", supervised), ("unsupervised", unsupervised), ("godunov", godunov)]:
            # BTX
            errs_l1 = torch.abs(lax_hopf - solutions)
            errs_l2 = torch.square(lax_hopf - solutions)
            for i, errs in enumerate([errs_l1, errs_l2]):
                axes[i].plot(torch.mean(errs, axis=(0, 2)), label=name)
                # axes[i].plot(np.max(errs.numpy(), axis=(0, 2)), label=name)
        for k in [0, 1]:
            axes[k].set_title(f"L{k + 1}")
            axes[k].set_xlabel("timestep")
            axes[k].legend()
            axes[k].grid()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"{flow}.png"))
        plt.close()


def plot_winrates(data_dir, log_dir, flows, models, args):
    schemes = ["supervised", "unsupervised", "godunov", "lax_friedrichs", "engquist_osher", "eno", "weno", "dg"]
    console = Console(record=True, file=open(os.devnull, "wt"), width=180)

    for flow in flows:
        l2_errors = torch.load(os.path.join(data_dir, str(flow), "errors_l2.pt"))

        # tex output
        tex = "\\def\\mydata{\n"
        for row in schemes:
            vals = []
            for col in schemes:
                if row == col:
                    vals.append("-")
                else:
                    # winrate of row against col
                    winrate = torch.sum(l2_errors[row] < l2_errors[col]) / len(l2_errors[row]) * 100
                    vals.append(str(round(winrate.item())))
            tex += "{" + ", ".join(vals) + "},\n"
        tex += "}"
        with open(os.path.join(log_dir, f"winrates_{flow}.tex"), "w") as f:
            f.write(tex)

        # HTML output with Rich table
        table = Table(title=str_beautify(flow))
        table.add_column("↓ winrate against →", justify="right", style="cyan")
        for col in schemes:
            table.add_column(str_beautify(col), justify="center")

        for row in schemes:
            row_vals = []
            for col in schemes:
                if row == col:
                    row_vals.append("-")
                else:
                    winrate = torch.sum(l2_errors[row] < l2_errors[col]) / len(l2_errors[row]) * 100
                    row_vals.append(f"{round(winrate.item())}%")
            table.add_row(str_beautify(row), *row_vals)

        console.print()
        console.print(table)
    console.save_html(os.path.join(log_dir, "winrates.html"))


def plot_box_plots(data_dir, log_dir, flows, models, args):
    schemes = ["supervised", "unsupervised", "godunov", "lax_friedrichs", "engquist_osher", "eno", "weno", "dg"]

    for flow in flows:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4), dpi=300)

        l1_errors = torch.load(os.path.join(data_dir, str(flow), "errors_l1.pt"))
        l2_errors = torch.load(os.path.join(data_dir, str(flow), "errors_l2.pt"))
        rel_errors = torch.load(os.path.join(data_dir, str(flow), "errors_rel.pt"))

        for ax, name, values in zip(axes, ["L1", "L2", "Rel."], [l1_errors, l2_errors, rel_errors]):
            scheme2str = {
                "godunov": "GOD",
                "lax_friedrichs": "LF",
                "engquist_osher": "EO",
                "supervised": "NN S",
                "unsupervised": "NN U",
                "eno": "ENO",
                "weno": "WENO",
                "dg": "DG",
            }
            vals = [values[scheme].to(args.device) for scheme in schemes]
            boxplot = ax.boxplot(vals, patch_artist=True, zorder=2, showfliers=False)
            colors = ["#FFB6C1", "#FFD580", "#AFEEEE", "#90EE90", "#FF77FF", "#ADD8E6", "#D2B48C", "#FFA07A"]
            for patch, color in zip(boxplot["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            for median in boxplot["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)
            for i, points in enumerate(vals):
                x_jitter = np.random.uniform(-0.14, 0.14, size=len(points))
                ax.scatter(np.full(len(points), i) + x_jitter + 1.0, points, color="black", alpha=0.3, s=2, zorder=1)
            ax.set_xticks(range(1, 1 + len(schemes)), [scheme2str[x] for x in schemes], rotation=90)
            ax.set_yscale("log")
            ax.grid(color="grey", linestyle="--", linewidth=0.5, axis="y", zorder=0)
            ax.minorticks_on()
            ax.grid(color="lightgrey", linestyle=":", linewidth=0.5, axis="y", which="minor", zorder=0)
            ax.set_title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"{flow}.png"))


def normalize(x):
    return x / x.max()


def plot_heatmaps(data_dir, log_dir, flows, models, args):
    data_plot = []
    data_plot_diff = []
    idx_plot = None  # (plot a specific ic; paper: 14 and 18)
    for flow in flows:
        # BTX
        lax_hopf = torch.load(os.path.join(data_dir, str(flow), "lax_hopf.pt"))[:, 1:, 3:-3].to(args.device)
        supervised = torch.load(os.path.join(data_dir, str(flow), "supervised.pt"))[:, 1:, 3:-3].to(args.device)
        unsupervised = torch.load(os.path.join(data_dir, str(flow), "unsupervised.pt"))[:, 1:, 3:-3].to(args.device)
        godunov = torch.load(os.path.join(data_dir, str(flow), "godunov.pt"))[:, 1:, 3:-3].to(args.device)
        eo = torch.load(os.path.join(data_dir, str(flow), "engquist_osher.pt"))[:, 1:, 3:-3].to(args.device)
        lax_friedrichs = torch.load(os.path.join(data_dir, str(flow), "lax_friedrichs.pt"))[:, 1:, 3:-3].to(args.device)
        eno = torch.load(os.path.join(data_dir, str(flow), "eno.pt"))[:, 1:, 3:-3].to(args.device)
        weno = torch.load(os.path.join(data_dir, str(flow), "weno.pt"))[:, 1:, 3:-3].to(args.device)
        dg = torch.load(os.path.join(data_dir, str(flow), "dg.pt"))[:, 1:, 3:-3].to(args.device)

        if idx_plot is None:
            idx_plot = np.random.randint(0, lax_hopf.shape[0])
        plot_heatmap(lax_hopf[idx_plot], path=os.path.join(log_dir, f"{idx_plot}_lxh_{str(flow)}.png"), xlabel=False, ylabel=False, cbar=False)
        plot_heatmap(supervised[idx_plot], path=os.path.join(log_dir, f"{idx_plot}_nfv_{str(flow)}.png"), xlabel=False, ylabel=False, cbar=False)

        data_plot.append(
            [
                lax_hopf[idx_plot],
                supervised[idx_plot],
                unsupervised[idx_plot],
                godunov[idx_plot],
                eo[idx_plot],
                lax_friedrichs[idx_plot],
                eno[idx_plot],
                weno[idx_plot],
                dg[idx_plot],
            ]
        )
        data_plot_diff.append(
            [
                lax_hopf[idx_plot],
                normalize(torch.abs(lax_hopf[idx_plot] - supervised[idx_plot])),
                normalize(torch.abs(lax_hopf[idx_plot] - unsupervised[idx_plot])),
                normalize(torch.abs(lax_hopf[idx_plot] - godunov[idx_plot])),
            ]
        )
    for name, data in zip(["complex", "complex_diff"], [data_plot, data_plot_diff]):
        plot_heatmap(
            data,
            title_col=["Lax Hopf", "Supervised", "Unsupervised", "Godunov", "EO", "Lax-Friedrichs", "ENO", "WENO", "DG"],
            title_row=[str_beautify(flow) for flow in flows],
            transpose=False,
            path=os.path.join(log_dir, f"{name}.png"),
            xlabel=False,
            ylabel=False,
            cbar=False,
            fontsize=16,
        )


def plot_density_time(data_dir, log_dir, flows, models, args):
    data_plot = []
    idx_plot = None
    for flow in flows:
        # BTX
        lax_hopf = torch.load(os.path.join(data_dir, str(flow), "lax_hopf.pt"))[:, 1:, 3:-3].to(args.device)
        supervised = torch.load(os.path.join(data_dir, str(flow), "supervised.pt"))[:, 1:, 3:-3].to(args.device)
        unsupervised = torch.load(os.path.join(data_dir, str(flow), "unsupervised.pt"))[:, 1:, 3:-3].to(args.device)
        godunov = torch.load(os.path.join(data_dir, str(flow), "godunov.pt"))[:, 1:, 3:-3].to(args.device)

        if idx_plot is None:
            idx_plot = np.random.randint(0, lax_hopf.shape[0])
        data_plot.append([lax_hopf[idx_plot], supervised[idx_plot], unsupervised[idx_plot], godunov[idx_plot]])

    schemes = ["Lax Hopf", "Supervised", "Unsupervised", "Godunov"]
    title = [str_beautify(flow) for flow in flows]

    plot_agg(data_plot, lambda x_TX: x_TX[0], transpose=True, legend=schemes, title=title, path=os.path.join(log_dir, "t0.png"))
    plot_agg(data_plot, lambda x_TX: x_TX[x_TX.shape[0] // 2], transpose=True, legend=schemes, title=title, path=os.path.join(log_dir, "tmid.png"))
    plot_agg(data_plot, lambda x_TX: x_TX[-1], transpose=True, legend=schemes, title=title, path=os.path.join(log_dir, "tf.png"))


if __name__ == "__main__":
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = "serif"

    # args
    args = parse_args()
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.device == "default":
        args.device = get_device()
    print(f"Using device {args.device}")

    data_dir = os.path.join(args.datadir, f"eval_nx{args.nx}_nt{args.nt}_dx{args.dx:.1e}_dx{args.dt:.1e}_ics{args.n}p{args.pieces}_seed{args.seed}")

    flows = [Greenshield(), Triangular(), TriangularSkewed(), Trapezoidal(), Greenberg(), Underwood()]
    if args.flow is not None:
        flows = [flow for flow in flows if str(flow) in args.flow]

    models = {flow: {} for flow in flows}
    for flow in flows:
        models[flow] = {k: load_model(str(flow), k).to(args.dtype).to(args.device) for k, v in model_list[str(flow)].items() if v is not None}

    # dirs
    with torch.no_grad():
        if args.data:
            generate_data(data_dir, flows, models, args)

        if args.plot:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            log_dir = os.path.join(args.logdir, f"{timestamp}_eval") if not args.tmp else "tmp/"

            print(f"Saving plots at {log_dir}")
            plot2fn = {
                "flux": plot_numerical_flux,
                "table": plot_table,
                "err_time": plot_err_time,
                "winrates": plot_winrates,
                "box": plot_box_plots,
                "heatmap": plot_heatmaps,
                "density": plot_density_time,
            }

            plots_to_plot = list(plot2fn.keys()) if args.plot[0] == "all" else args.plot
            for plot in plots_to_plot:
                print(f"Plotting {plot}...")
                t0 = time.time()
                plot_log_dir = os.path.join(log_dir, plot)
                os.makedirs(plot_log_dir, exist_ok=True)
                plot2fn[plot](data_dir, plot_log_dir, flows, models, args)
                print(f"\t{time.time() - t0:.3f}s, {plot_log_dir}")
