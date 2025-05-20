import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Polynomial(nn.Module):
    def __init__(self, coefficients, device="cpu"):
        super(Polynomial, self).__init__()

        self.coefficients = nn.Parameter(coefficients.to(device))
        self.device = device

    def to(self, device):
        self.device = device
        self.coefficients = nn.Parameter(self.coefficients.to(device))
        return self

    def update(self, coefficients):
        self.coefficients = nn.Parameter(coefficients.to(self.device))
        return self

    def __mul__(self, other):
        # Cauchy product
        coefficients_A = self.coefficients
        coefficients_B = other.coefficients.to(self.device)

        n = coefficients_A.shape[0]
        m = coefficients_B.shape[0]

        # Extend the coefficients to the same size
        coefficients_A = torch.cat([coefficients_A, torch.zeros(m, device=self.device)], dim=0)
        coefficients_B = torch.cat([coefficients_B, torch.zeros(n, device=self.device)], dim=0)

        result = torch.zeros(n + m - 1, device=self.device)
        for i in range(n + m - 1):
            result[i] = torch.sum(coefficients_A[: i + 1] * coefficients_B[: i + 1].flip(0))

        return Polynomial(result, device=self.device)

    def __call__(self, x):
        x = x.unsqueeze(1)
        return x ** torch.arange(self.coefficients.shape[0], device=self.device) @ self.coefficients

    def __repr__(self):
        str = "Polynomial("
        for i, coefficient in enumerate(self.coefficients):
            if coefficient > 0:
                str += f" + {coefficient.item()}X^{i}"
            elif coefficient < 0:
                str += f" - {-coefficient.item()}X^{i}"
        str += f", device={self.device})"
        return str


class BatchPolynomial(nn.Module):
    def __init__(self, coefficients, device="cpu", requires_grad=False):
        super(BatchPolynomial, self).__init__()
        self.requires_grad = requires_grad
        if requires_grad:
            self.coefficients = nn.Parameter(coefficients.to(device))
        else:
            self.coefficients = coefficients.to(device)
        self.device = device

    def to(self, device):
        self.device = device

        if self.requires_grad:
            self.coefficients = nn.Parameter(self.coefficients.to(device))
        else:
            self.coefficients = self.coefficients.to(device)
        return self

    def sum(self, weights=None):
        if weights is None:
            return Polynomial(torch.sum(self.coefficients, dim=0), device=self.device)
        return Polynomial(torch.sum(self.coefficients * weights.unsqueeze(1), dim=0), device=self.device)

    def prime(self):
        return BatchPolynomial(self.coefficients[:, 1:] * torch.arange(1, self.coefficients.shape[1], device=self.device), device=self.device)

    def antiderivative(self):
        return BatchPolynomial(
            torch.cat(
                [
                    torch.zeros(self.coefficients.shape[0], 1, device=self.device),
                    self.coefficients / torch.arange(1, self.coefficients.shape[1] + 1, device=self.device),
                ],
                dim=1,
            ),
            device=self.device,
        )
        # return BatchPolynomial(torch.cat([torch.zeros(self.coefficients.shape[0], 1, device=self.device), self.coefficients / torch.arange(1, self.coefficients.shape[1], device=self.device)], dim=1), device=self.device)

    def __mul__(self, other):
        if isinstance(other, BatchPolynomial):
            # Cauchy product
            coefficients_A = self.coefficients
            coefficients_B = other.coefficients.to(self.device)
            if coefficients_A.shape[0] != coefficients_B.shape[0]:
                raise ValueError("The batch sizes must be the same.")

            n = coefficients_A.shape[1]
            m = coefficients_B.shape[1]
            batch_size = coefficients_A.shape[0]

            # Extend the coefficients to the same size
            coefficients_A = torch.cat([coefficients_A, torch.zeros((batch_size, m), device=self.device)], dim=1)
            coefficients_B = torch.cat([coefficients_B, torch.zeros((batch_size, n), device=self.device)], dim=1)

            result = torch.zeros((batch_size, n + m - 1), device=self.device)
            for i in range(n + m - 1):
                result[:, i] = torch.sum(coefficients_A[:, : i + 1] * coefficients_B[:, : i + 1].flip(1), dim=1)

            return BatchPolynomial(result, device=self.device)

    def __call__(self, x, max_degree=None):
        x = torch.stack([x] * self.coefficients.shape[0]).unsqueeze(-1).transpose(0, 1)

        power = torch.arange(self.coefficients.shape[1])
        power = torch.stack([power] * x.shape[1]).unsqueeze(0).to(device=self.device)

        # mask = (power <= max_degree if max_degree is not None else torch.ones_like(power))[0]
        return (((x**power) * self.coefficients.unsqueeze(0)).transpose(0, 1)).sum(dim=-1)

    def __repr__(self):
        str = ""
        for Polynomial in self.coefficients:
            str += "Polynomial("
            for i, coefficient in enumerate(Polynomial):
                if coefficient > 0:
                    str += f" + {coefficient.item()}X^{i}"
                elif coefficient < 0:
                    str += f" - {-coefficient.item()}X^{i}"
            str += f", device={self.device})"
            str += "\n"
        return str[:-1]


def legendreCoefficients(max_degree):
    # Initialize the coefficient matrix (max_degree + 1) x (max_degree + 1)
    coeffs = torch.zeros((max_degree + 1, max_degree + 1), dtype=torch.float32)

    # Initial conditions: P_0(x) = 1 and P_1(x) = x
    coeffs[0, 0] = 1  # P_0(x) = 1
    if max_degree == 0:
        return coeffs
    coeffs[1, 1] = 1  # P_1(x) = x

    # Recurrence relation to fill the matrix:
    # (n + 1)P_{n+1}(x) = (2n + 1)xP_n(x) - nP_{n-1}(x)
    for n in range(1, max_degree):
        coeffs[n + 1, 1:] += (2 * n + 1) * coeffs[n, :-1]  # Multiply P_n(x) by x
        coeffs[n + 1, :] -= n * coeffs[n - 1, :]  # Subtract n * P_{n-1}(x)
        coeffs[n + 1, :] /= n + 1  # Divide by (n + 1)

    return coeffs


class LegendreBasis(BatchPolynomial):
    def __init__(self, degree, requires_grad=False, device="cpu"):
        coeffs = legendreCoefficients(degree)
        self.degree = degree
        super().__init__(coefficients=coeffs, requires_grad=requires_grad, device=device)

    def mass_matrix(self):
        return torch.tensor([2 / (2 * i + 1) for i in range(self.degree + 1)]).to(self.device)


def basis(polynomials, number_of_polynomials, device="cpu"):
    if polynomials == "chebyshev":
        raise NotImplementedError
    elif polynomials == "bernstein":
        raise NotImplementedError
    elif polynomials == "legendre":
        return LegendreBasis(number_of_polynomials - 1, device=device)
    else:
        raise NotImplementedError


def legendre_roots_weights(n, device="cpu", dtype=torch.float):
    roots, weights = np.polynomial.legendre.leggauss(n)
    return torch.tensor(roots, dtype=dtype, device=device), torch.tensor(weights, dtype=dtype, device=device)


def DG_update(
    max_t,
    solution_DG,
    solution_BTX,
    weights_dg,
    left_boundary_indexes,
    right_boundary_indexes,
    godunov_flow_func,
    flow_model,
    mass_matrix_inv,
    half_cell_size,
    batch_size,
    cells,
    points_per_cell,
    weights_leggauss,
    polynomials,
    polynomials_prime,
    left_polynomials_value,
    dt,
):
    solution_BXT = solution_BTX.transpose(1, 2)
    for t in range(1, max_t):
        weights_dg_save = weights_dg.clone()
        for a in range(2):
            left_boundaries = solution_DG[:, left_boundary_indexes, t + a - 1]
            right_boundaries = solution_DG[:, right_boundary_indexes, t + a - 1]

            fluxes = godunov_flow_func(left_boundaries, right_boundaries, flow_model)
            # For legendre, right boundary is always 1 and left is (-1)^n
            fluxes = fluxes[:, 1:].unsqueeze(1) - fluxes[:, :-1].unsqueeze(1) * left_polynomials_value

            fluxes = mass_matrix_inv * fluxes

            f_u = flow_model(solution_DG[:, :, t + a - 1])

            residual = (
                mass_matrix_inv
                * (
                    weights_leggauss.unsqueeze(-1)
                    * (polynomials_prime.unsqueeze(-1) * f_u.view(batch_size, cells, points_per_cell).transpose(1, 2).unsqueeze(1))
                ).sum(dim=2)
                * half_cell_size
            ).float()

            weights_dg = weights_dg_save + (-dt * fluxes + dt * residual).squeeze(1) * (1 / 2 if a == 0 else 1.0)

            # TO ADD BOUNDARY CONDITIONS
            weights_dg[:, 1:, 0] = 0
            weights_dg[:, 0, 0] = solution_BXT[:, 0, t + a - 1]  # BXT
            weights_dg[:, 1:, -1] = 0
            weights_dg[:, 0, -1] = solution_BXT[:, -1, t + a - 1]

            solution_DG[:, :, t] = torch.einsum("ijk,ijl->ikl", polynomials, weights_dg).permute(0, 2, 1).reshape(batch_size, -1)


def godunov_flux(rho_l, rho_r, flow_model):
    fluxes_l = flow_model(rho_l)
    fluxes_r = flow_model(rho_r)

    flows = torch.where(
        rho_l <= rho_r,
        torch.minimum(fluxes_l, fluxes_r),
        torch.where(
            rho_r > flow_model.k_crit,
            fluxes_r,
            torch.where(
                flow_model.k_crit > rho_l,
                fluxes_l,
                torch.where(
                    rho_l < flow_model.k_crit,
                    fluxes_l,
                    flow_model(torch.tensor(flow_model.k_crit, device=rho_l.device)),
                ),
            ),
        ),
    )
    return flows


# def get_solution_DG(nx=500, nt=2000, cfl=0.1):
def DG(nx, nt, dx, dt, flow, boundaries_BTX, dtype, device, **kwargs):  # solution_TX, cfl=0.1, flow_model=None):
    # cells = 500
    # points_per_cell = 40
    # number_of_polynomials = 2
    # max_t = 2000
    # dx = 1e-2
    # dt = 1e-3

    boundaries_BTX = boundaries_BTX.to(torch.float32)

    # nt, nx = boundaries_BTX.shape

    cells = nx
    points_per_cell = 40  # 40
    number_of_polynomials = 2  # 2
    max_t = nt
    # dx = 1 / nx
    # dt = dx * cfl

    n_points = cells  # * args.points_per_cell
    x_max = dx * n_points / 2
    x = torch.linspace(-x_max, x_max, n_points)
    time = torch.arange(1, max_t + 1).unsqueeze(0) * dt

    ##### Variables for Discontinuous Galerkin
    half_cell_size = x_max / cells
    polynomials = basis(
        "legendre",
        number_of_polynomials,
        device=device,
    )

    # Pre-compute constants
    mass_matrix = polynomials.mass_matrix() * half_cell_size  # * h because of the rescale from [-1, 1] to [-x_max, x_max]/cells
    mass_matrix_inv = (mass_matrix ** (-1)).unsqueeze(0).unsqueeze(-1)

    x_polynomials, weights_leggauss = legendre_roots_weights(points_per_cell, device=device)
    weights_leggauss = weights_leggauss.unsqueeze(0).unsqueeze(0)

    polynomials_prime = polynomials.prime()(x_polynomials)
    polynomials_prime = polynomials_prime.unsqueeze(0) / half_cell_size  # rescale the derivative to the new domain
    polynomials = polynomials(x_polynomials).unsqueeze(0)

    left_polynomials_value = (
        torch.tensor([1, -1]).repeat(number_of_polynomials // 2 + 1)[:number_of_polynomials].unsqueeze(0).unsqueeze(-1).to(device)
    )

    boundaries_BXT = boundaries_BTX.transpose(1, 2)
    # careful: dont confuse BXT and BTX
    solution_DG = F.interpolate(boundaries_BXT.unsqueeze(1), size=(points_per_cell * nx, nt), mode="bilinear").squeeze(1)

    # just specify a Riemann initial condition
    # batch_size = 1
    # solution_DG = torch.zeros((batch_size, points_per_cell*cells, max_t)).squeeze(1)
    # idx_mid_x = solution_DG.shape[1] // 2
    # solution_DG[:, :idx_mid_x, 0] = 0.2
    # solution_DG[:, idx_mid_x:, 0] = 0.9

    # print(solution_DG.shape, idx_mid_x)
    # solution_DG[:, :idx_mid_x, 0] = 1.
    # solution_DG[:, idx_mid_x:, 0] = 0.

    # solution_DG = torch.nn.functional.interpolate(torch.FloatTensor(boundaries_BTX.T).unsqueeze(0).unsqueeze(0), size=(points_per_cell * cells, max_t), mode="bilinear").squeeze(0)
    batch_size = boundaries_BTX.shape[0]

    weights_dg = torch.empty(batch_size, number_of_polynomials, cells, device=device)
    weights_dg_speed = torch.empty(batch_size, number_of_polynomials, cells, device=device)

    ##### Initial weights Discontinuous Galerkin
    for cell in range(cells):
        indices = slice(cell * points_per_cell, (cell + 1) * points_per_cell)
        weights_dg[:, :, cell] = (
            mass_matrix_inv.squeeze(-1) * (weights_leggauss * (polynomials * solution_DG[:, indices, 0].unsqueeze(1))).sum(dim=-1) * half_cell_size
        )

    weights_dg_speed = weights_dg.clone()

    solution_DG[:, :, 0] = torch.einsum("ijk,ijl->ikl", polynomials, weights_dg).permute(0, 2, 1).reshape(batch_size, -1)

    left_boundary_indexes = torch.arange(-1, cells * points_per_cell, points_per_cell)
    right_boundary_indexes = torch.arange(0, cells * points_per_cell + 1, points_per_cell)
    left_boundary_indexes[0] = 0
    right_boundary_indexes[-1] = cells * points_per_cell - 1

    ##### Discontinuous Galerkin
    DG_update(
        max_t,
        solution_DG,
        boundaries_BTX,
        weights_dg,
        left_boundary_indexes,
        right_boundary_indexes,
        godunov_flux,
        flow,
        mass_matrix_inv,
        half_cell_size,
        batch_size,
        cells,
        points_per_cell,
        weights_leggauss,
        polynomials,
        polynomials_prime,
        left_polynomials_value,
        dt,
    )

    solution_DG = (solution_DG.transpose(1, 2) * weights_leggauss.repeat(1, 1, cells)).view(
        (solution_DG.shape[0], solution_DG.shape[2], solution_DG.shape[1] // points_per_cell, points_per_cell)
    ).sum(dim=3).transpose(1, 2) / 2

    solution_DG = torch.clamp(solution_DG, 0.0, flow.kmax)

    return solution_DG.transpose(1, 2)
