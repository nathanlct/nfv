import argparse
import datetime
import json
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import nn
from torch.nn import functional as F

from nfv import Problem
from nfv.flows import Greenberg, Greenshield, Trapezoidal, Triangular, TriangularSkewed, Underwood
from nfv.initial_conditions import PiecewiseConstant, Riemann
from nfv.models import CNNStencilModel
from nfv.solvers import FVM, Godunov, LaxHopf
from nfv.utils.device import get_device
from nfv.utils.plotting import plot_agg, plot_heatmap
from nfv.utils.tensor import set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    # wandb logging
    parser.add_argument("--track", action="store_true", default=False, help="If set, tracks the experiment using Wandb.")
    parser.add_argument("--wandb_entity", type=str, default="nathanlct")
    parser.add_argument("--wandb_project", type=str, default="numerical_schemes")
    parser.add_argument("--wandb_group", type=str, default=None)

    # experiment
    parser.add_argument("--device", type=str, default="default", choices=["default", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible experiments.")
    parser.add_argument("--logdir", type=str, default="logs/", help="Base path where to save experiment data.")

    # model
    parser.add_argument("--depth", type=int, default=6, help="Depth of the model (number of layers).")
    parser.add_argument("--hidden", type=int, default=15, help="Number of hidden units in each layer.")
    parser.add_argument("--act", type=str, default="ReLU", help="Activation function.")
    parser.add_argument("--last_act", type=str, default=None, help="Activation function after the last layer (none by default).")
    parser.add_argument("--checkpoint", type=str, default=None, help="If set to a model checkpoint path, loads it to resume training.")
    parser.add_argument(
        "--clip", type=float, default=None, help="If set, the model's output is clipped to this max value instead of the maximum flow."
    )

    # data
    parser.add_argument(
        "--flow",
        type=str,
        default="greenshield",
        help="Flow model to use.",
        choices=["greenshield", "triangular", "triangular_skewed", "trapezoidal", "greenberg", "underwood"],
    )
    parser.add_argument("--N_train", type=int, default=3, help="Number of riemann problems for training data (N for random, N*(N-1) for uniform)")
    parser.add_argument("--N_train_mode", type=str, default="random", choices=["random", "uniform", "complex"])
    parser.add_argument("--N_eval", type=int, default=5, help="Number of complex initial conditions to eval on.")
    parser.add_argument("--N_eval_pieces", type=int, default=10, help="Number of pieces in the piecewise constant initial condition.")
    parser.add_argument("--lh_precision", type=int, default=10, help="Precision of lax hopf solution.")
    parser.add_argument("--lh_batch_size", type=int, default=1, help="Batch size for lax hopf solution.")
    parser.add_argument("--x_noise", default=False, action="store_true")
    parser.add_argument("--train_complex", default=False, action="store_true")

    # training
    parser.add_argument("--loss_fn", type=str, default="l2", choices=["l1", "l2"], help="Loss function to use.")
    parser.add_argument("--schedule", type=str, help="Training schedule", default="[dict(epochs=5_000, lr=1e-4, nt=10, nx=20)]")
    parser.add_argument("--grad_norm_clip", type=float, default=1.0, help="Gradient norm clipping value")
    parser.add_argument("--train_dx", type=float, default=1e-3)
    parser.add_argument("--train_dt", type=float, default=1e-3)
    parser.add_argument("--loss_coef", type=float, default=1.0)
    # parser.add_argument("--k1", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    # eval
    parser.add_argument("--eval_every", type=int, default=1000, help="Eval every n epochs")
    parser.add_argument("--eval_nt", type=int, default=1000, help="Number of time steps to eval on.")
    parser.add_argument("--eval_nx", type=int, default=200, help="Number of spatial steps to eval on.")
    parser.add_argument("--eval_dx", type=float, default=1e-3)
    parser.add_argument("--eval_dt", type=float, default=1e-4)

    return parser.parse_args()


def process_args(args):
    # device
    if args.device == "default":
        args.device = get_device()
    print(f"Using device {args.device}")

    # seed
    if args.seed:
        set_seed(args.seed)

    # logdir
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    logdir = os.path.join(args.logdir, f"{timestamp}_{args.flow}")
    os.makedirs(logdir, exist_ok=False)
    with open(os.path.join(logdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    args.logdir = logdir
    print(f'Logging at "{logdir}"')

    # convert torch objects
    args.act = eval(f"nn.{args.act}")
    if args.last_act is not None:
        args.last_act = eval(f"nn.{args.last_act}")
    # args.loss_fn = eval(f"F.{args.loss_fn}")

    # flow
    args.flow = {
        "greenshield": Greenshield,
        "triangular": Triangular,
        "triangular_skewed": TriangularSkewed,
        "trapezoidal": Trapezoidal,
        "greenberg": Greenberg,
        "underwood": Underwood,
    }[args.flow]()

    # schedule
    args.schedule = eval(args.schedule)

    # wandb
    if args.track:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group, name=args.logdir, config=args, save_code=True)
        wandb.define_metric("epoch")
        wandb.save(os.path.join(logdir, "config.json"))


def train(args, eval_data):
    # create model
    if args.clip is None:
        clip = args.flow.qmax
    elif args.clip < 0:
        clip = None
    else:
        clip = args.clip
    print("Clipping model output at", clip)
    model = CNNStencilModel(depth=args.depth, hidden=args.hidden, act=args.act, last_act=args.last_act, clip=clip, dtype=torch.float32)
    model = model.to(args.device).to(torch.float32)

    if args.checkpoint is not None:
        print(f'Loading model from "{args.checkpoint}"')
        model.load_checkpoint(args.checkpoint, device=args.device)

    # iterate over stages of training schedule
    total_epochs = 0
    for schedule in args.schedule:
        optimizer = torch.optim.Adam(model.parameters(), lr=schedule["lr"])

        # generate train data
        if args.N_train_mode == "random":
            ic_train = [
                Riemann(np.random.rand() * args.flow.kmax, np.random.rand() * args.flow.kmax, x_noise=args.x_noise) for _ in range(args.N_train)
            ]
        elif args.N_train_mode == "uniform":
            riemann_ks = np.linspace(0.0, args.flow.kmax, args.N_train)
            ic_train = [Riemann(k1, k2, x_noise=args.x_noise) for k1 in riemann_ks for k2 in riemann_ks if abs(k1 - k2) > 1e-5]
        elif args.N_train_mode == "complex":
            ic_train = [PiecewiseConstant(np.random.rand(5), x_noise=args.x_noise) for _ in range(args.N_train)]
        train_data = Problem(nx=schedule["nx"], nt=schedule["nt"], dx=args.train_dx, dt=args.train_dt, ic=ic_train, flow=args.flow)
        print(f"Generating train data (nx={train_data.nx}, nt={train_data.nt}, N={len(train_data.ic)})...")
        t0 = time.time()
        train_data.solve(LaxHopf, batch_size=args.lh_batch_size, device=args.device, save="ground_truth")
        train_data.solve(FVM(Godunov), boundaries="ground_truth", device=args.device, save="godunov")
        print(f"Done, took {time.time() - t0:.2f} seconds.")

        # training loop
        n_epochs = schedule["epochs"]

        ground_truth_train = train_data.solutions["ground_truth"].to(torch.float32).to(args.device)

        t0 = time.time()

        ic_discretized = np.array([ic.discretize(train_data.nx) for ic in train_data.ic])
        ic_discretized = torch.from_numpy(ic_discretized).to(torch.float32).to(args.device)

        for epoch in range(n_epochs):
            log = {}

            batch_idx = torch.randperm(len(train_data.ic))[: args.batch_size] if args.batch_size is not None else slice(None)

            k_BTX = torch.clone(ic_discretized[batch_idx]).unsqueeze(1)

            loss = 0

            log["grad_norm"] = -1
            for t in range(train_data.nt - 1):
                # predict t+1 from t
                flows_BX = model(k_BTX[:, -1, :], train_data.flow)
                new_k_BX = k_BTX[:, -1, 1:-1] + train_data.dt / train_data.dx * (flows_BX[:, :-1] - flows_BX[:, 1:])
                bc_left, bc_right = ground_truth_train[batch_idx, t + 1, 0], ground_truth_train[batch_idx, t + 1, -1]
                new_k_BX = torch.concat([bc_left.unsqueeze(1).unsqueeze(1), new_k_BX.unsqueeze(1), bc_right.unsqueeze(1).unsqueeze(1)], dim=2)
                new_k_BX = torch.clamp(new_k_BX, 0.0, train_data.flow.kmax)
                k_BTX = torch.concat([k_BTX, new_k_BX], dim=1)

                if args.loss_fn == "l1":
                    loss += torch.mean(torch.abs(new_k_BX[:, 0, 1:-1] - ground_truth_train[batch_idx, t + 1, 1:-1]))
                if args.loss_fn == "l2":
                    loss += torch.mean(torch.square(new_k_BX[:, 0, 1:-1] - ground_truth_train[batch_idx, t + 1, 1:-1]))
            loss = args.loss_coef * loss / train_data.nt

            # backward
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()

            if grad_norm < 1e-9 or np.isnan(loss.detach().cpu().item()):
                # NN died, reboot training
                print("Rebooting training due to zero or NaN gradients")
                return train(args, eval_data)

            log["grad_norm"] = grad_norm
            log["loss"] = loss.detach().cpu().item()

            total_epochs += 1

            # eval
            if epoch % args.eval_every == 0 or epoch == n_epochs - 1:
                t0_eval = time.time()
                # save model
                model_save_path = os.path.join(args.logdir, f"model_{total_epochs}.pth")
                torch.save(model.state_dict(), model_save_path)
                if epoch == 0 or epoch == n_epochs - 1:
                    print(">", os.path.join(args.logdir, f"model_{total_epochs}.pth"))
                if args.track:
                    wandb.save(os.path.join(args.logdir, f"model_{total_epochs}.pth"))

                # eval
                prediction = (
                    eval_data.solve(FVM(model), boundaries="ground_truth", device=args.device, dtype=torch.float32, grad=False).cpu().numpy()
                )
                ground_truth_eval = eval_data.solutions["ground_truth"].cpu().numpy()
                godunov = eval_data.solutions["godunov"].cpu().numpy()

                model_errors_l1 = np.mean(np.abs(prediction - ground_truth_eval), axis=(1, 2))
                model_errors_l2 = np.mean(np.square(prediction - ground_truth_eval), axis=(1, 2))
                godunov_errors_l1 = np.mean(np.abs(godunov - ground_truth_eval), axis=(1, 2))
                godunov_errors_l2 = np.mean(np.square(godunov - ground_truth_eval), axis=(1, 2))

                log["eval/l1_error"] = np.mean(model_errors_l1)
                log["eval/l2_error"] = np.mean(model_errors_l2)
                log["eval/l1_error_diff"] = np.mean(model_errors_l1) - np.mean(godunov_errors_l1)
                log["eval/l2_error_diff"] = np.mean(model_errors_l2) - np.mean(godunov_errors_l2)
                log["eval/winrate_l1"] = np.sum(model_errors_l1 < godunov_errors_l1) / len(model_errors_l1)
                log["eval/winrate_l2"] = np.sum(model_errors_l2 < godunov_errors_l2) / len(model_errors_l2)
                log["eval/winrate_l2"] = np.sum(model_errors_l2 < godunov_errors_l2) / len(model_errors_l2)
                log["epochs_per_second"] = (epoch + 1) / (time.time() - t0)

                print(
                    f"epoch {epoch + 1 if epoch == 0 or epoch == n_epochs - 1 else epoch:5}/{n_epochs} | loss={log['loss']:.3e} | gradnorm={log['grad_norm']:.3e} | "
                    f"l1_err={log['eval/l1_error']:<.3e} | l2_err={log['eval/l2_error']:<.3e} | "
                    f"wr_l1={log['eval/winrate_l1'] * 100.0:.1f}% | wr_l2={log['eval/winrate_l2'] * 100.0:.1f}% | eps={int(log['epochs_per_second'])}"
                )

                # plots
                train_data_sample = train_data.solutions["ground_truth"][torch.randperm(len(train_data.ic))[:8]].cpu().numpy()
                fig = plot_heatmap(train_data_sample, return_fig=True, vmax=args.flow.kmax)
                log["plot/train_data"] = wandb.Image(fig)
                plt.close(fig)

                eval_idx = torch.randperm(len(eval_data.ic))[:5]
                eval_data_sample = [ground_truth_eval[eval_idx], prediction[eval_idx], godunov[eval_idx]]
                fig = plot_heatmap(eval_data_sample, return_fig=True, vmax=args.flow.kmax)
                log["plot/eval"] = wandb.Image(fig)
                plt.close(fig)
                fig_agg = plot_agg(eval_data_sample, lambda x_TX: x_TX[-1, :], legend=["gt", "nn", "god"], return_fig=True)
                log["plot/eval_final"] = wandb.Image(fig_agg)
                plt.close(fig_agg)

                # print(f"Finished eval in {time.time() - t0_eval:.2f} seconds")

            if args.track:
                wandb.log(log)


if __name__ == "__main__":
    matplotlib.use("Agg")  # non-interactive backend for faster rendering

    # parse args
    args = parse_args()
    process_args(args)

    # generate eval data
    eval_ics = [PiecewiseConstant(np.random.rand(args.N_eval_pieces), x_noise=True) for _ in range(args.N_eval)]
    eval_data = Problem(nx=args.eval_nx, nt=args.eval_nt, dx=args.eval_dx, dt=args.eval_dt, ic=eval_ics, flow=args.flow)
    print(f"Generating eval data data (nx={eval_data.nx}, nt={eval_data.nt}, N={len(eval_data.ic)})...")
    t0 = time.time()
    eval_data.solve(LaxHopf, batch_size=args.lh_batch_size, device=args.device, save="ground_truth")
    eval_data.solve(FVM(Godunov), boundaries="ground_truth", device=args.device, save="godunov")
    print(f"Done, took {time.time() - t0:.2f} seconds.")

    train(args, eval_data)
