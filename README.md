# NFV

**Neural Finite Volume Method (NFV)** is a Python library for solving partial differential equations (PDEs) using neural networks. It focuses on learning hyperbolic conservation laws, such as the Lighthill-Whitham-Richards (LWR) model, by integrating neural networks with traditional finite volume methods (FVMs).

NFV provides:
- **Neural PDE solvers:** Models and training code for learning conservation laws.
- **Classical numerical solvers:** Efficient implementations of standard finite-volume and finite-element methods, including Godunov, Lax-Friedrichs, Engquist-Osher, ENO, WENO, and Discontinuous Galerkin.
- **Exact LWR solutions:** Explicit solutions for Riemann problems and arbitrary piecewise-constant initial conditions via the Lax-Hopf algorithm.

This library bridges deep learning and numerical methods, making it easier to develop data-driven solvers for hyperbolic PDEs.

## Installation

```
git clone https://github.com/nathanlct/nfvm
cd nfv
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage

### Training

See `python train.py --help`. For instance, to train a model on the Greenshields flow:

```bash
python train.py --track --device cuda --logdir logs/ --N_train 256 --N_train_mode random --N_eval 100 --act ELU --flow greenshield --loss_coef 0.1 --train_dx 1e-3 --train_dt 5e-4 --schedule [dict(epochs=10_000, lr=1e-4, nt=10, nx=10), dict(epochs=10_000, lr=1e-4, nt=20, nx=20), dict(epochs=20_000, lr=1e-5, nt=50, nx=50), dict(epochs=20_000, lr=5e-6, nt=100, nx=100), dict(epochs=20_000, lr=5e-7, nt=100, nx=100), dict(epochs=20_000, lr=1e-7, nt=200, nx=200)] --batch_size 256 --x_noise
```
 
Notes:
- `--track` is used for tracking with Wandb.
- `--device cuda`: remove this if you are testing on CPU.
- The training data consists of 256 random piecewise-constant initial conditions; the validation data used during training has 100.
- The model is a simple fully-connected neural network with ELU activations.
- The schedule consists of several stages, each characterized by different number of training epochs, learning rates, and number of spatial and temporal steps the neural network is predicting (nx and nt). The learning rate decreases throughout training, as the number of autoregressively-predicted steps increases.
- `--x_noise` adds some noise to the initial conditions, which we found important for generalization.

### Evaluation

See `python eval.py --help`. For instance, to evaluate all models in `models/`:

```bash
python eval.py --logdir logs/ --datadir data/ --device cpu --data --n 1000 --dx 1e-3 --dt 1e-4 --nx 200 --nt 1000 --plot all
```

Notes:
- This will use `--n 1000` initial conditions, each with `--nx 200` spatial cells and `--nt 1000` temporal cells, with discretization `--dx` and `--dt`. Furthermore, `--pieces` can be used to specify how many pieces each piecewise-constant initial condition will have.
- `--data` generates this data and saves it in `--datadir`. If generating new plots on the same data, remove `--data` to not regenerate the data which takes time (it generates not only the ground truth solution, but also the prediction of all considered numerical schemes and all trained models). Also see `--force` to only regenerate part of the data.
- `--flow` can be used to specify which flows data should be generated for. By default, all six flows are considered.
- `--plot all` generates all plots and saves them in `--logdir`. See the help for a list of available plots.

### Miscellanea

- `python demo.py` shows examples of running all implemented solvers and generating [heatmap plots](https://nathanlichtle.com/research/nfv/). This includes: finite-volume methods (Godunov, Lax-Friedrichs, Engquist-Osher, ENO, WENO), a finite-element method (Discontinuous Galerkin = DG), and a Lax-Hopf algorithm which enables exact computation of the solutions for piecewise-constant initial data.
- `python plot_flows.py` plots all six considered flow functions (Greenshields, Triangular, Triangular Skewed, Trapezoidal, Greenberg, Underwood).

## Code structure

```bash
nfv
├── flows               # Implementation of flow functions
├── initial_conditions  # Riemann and piecewise-constant initial conditions
├── models              # Neural network models
├── solvers             # Implementations of finite-volume schemes, DG and Lax-Hopf for exact solutions
└── utils
└── % problem.py        # Class defining a problem (initial condition, flow, discretization...) and storing solutions
models                  # Trained models
datasets                # I-24 MOTION density fields ; BDD Drone dataset density field
```