from typing import Callable, Optional, Sequence, Tuple, Union
from torchdiffeq import odeint
from tqdm import tqdm
from flow_matching.solver.riemannian_ode_solver import (
    RiemannianODESolver, interp, _euler_step, _midpoint_step, _rk4_step
)
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils.manifolds import Manifold, Euclidean
from flow_matching.utils import gradient, ModelWrapper
import torch
from torch import Tensor
import math
from sbmfi.core.linalg import LinAlg
from torch import nn
from sbmfi.inference.manifolds import BallManifold, ConvexPolytopeManifold
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Distribution
from scipy.special import gammaln


def plot_losses_vs_steps(losses, axmin=None, axmax=None):
    steps = np.arange(len(losses))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(steps, losses, c='blue', alpha=0.9, edgecolors='w', s=10)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Machine Learning Losses vs Training Steps')
    ax.grid(True)

    if axmin is None:
        axmin = min(losses) * 0.85
    if axmax is None:
        axmax = max(losses) * (1.0 / 0.85)

    ax.set_ylim(axmin, axmax)
    return fig


class UniformBall(Distribution):
    def __init__(self, K: int, dtype=np.float32, device='cpu'):
        self._K = K
        self._la = LinAlg(backend='torch', device=device, dtype=dtype)
        self._log_ball_vol = (K / 2) * np.log(np.pi) - gammaln(K / 2 + 1)

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape, self._K)
        else:
            shape = (*shape, self._K)
        return self._la.sample_unit_hyper_sphere_ball(shape, ball=True)

    def log_prob(self, values):
        return torch.full(values.shape[:-1], -self._log_ball_vol, dtype=values.dtype, device=values.device)


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield: nn.Module, manifold: Manifold):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.manifold.projx(x)
        v = self.vecfield(x, t)
        v = self.manifold.proju(x, v)
        return v


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x=x, t=t)


# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(self, input_dim, time_dim: int = 1, hidden_dim: int = 128, hidden_layers=6):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), Swish()]

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            *layers,
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1).float()
        h = torch.cat([x, t], dim=1).float()
        output = self.main(h)

        return output.reshape(*sz)

def sample_and_div(
        ode_solver: ODESolver,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        return_div: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
) -> Union[Tensor, Sequence[Tensor], Tuple[Tensor, Tensor]]:
    """Sample forward and optionally compute log probabilities.

    Args:
        return_div: If True, compute and return log probabilities. Requires log_p0.
        log_p0: Log probability function of initial distribution, required if return_log_p=True
    """
    if return_div:
        if not exact_divergence:
            z = (torch.randn_like(x_init).to(x_init.device) < 0) * 2.0 - 1.0

    def ode_func(t, x):
        return ode_solver.velocity_model(x=x, t=t, **model_extras)

    def dynamics_func(t, states):
        xt = states[0]
        with torch.set_grad_enabled(True):
            xt.requires_grad_()
            ut = ode_func(t, xt)

            if exact_divergence:
                # Compute exact divergence
                div = 0
                for i in range(ut.flatten(1).shape[1]):
                    div += gradient(ut[:, i], xt, create_graph=True)[:, i]
            else:
                # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                ut_dot_z = torch.einsum(
                    "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                )
                grad_ut_dot_z = gradient(ut_dot_z, xt)
                div = torch.einsum(
                    "ij,ij->i",
                    grad_ut_dot_z.flatten(start_dim=1),
                    z.flatten(start_dim=1),
                )

        return ut.detach(), div.detach()

    ode_opts = {"step_size": step_size} if step_size is not None else {}

    if return_div:
        y_init = (x_init, torch.zeros(x_init.shape[0], device=x_init.device))
        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )
        if return_intermediates:
            return sol, log_det[-1]
        else:
            return sol[-1], log_det[-1, None]

    sol = odeint(
        ode_func,
        x_init,
        time_grid,
        method=method,
        options=ode_opts,
        atol=atol,
        rtol=rtol,
    )
    if return_intermediates:
        return sol
    else:
        return sol[-1]


def riem_sample_and_div(
    riem_solver: RiemannianODESolver,
    x_init: Tensor,
    step_size: float,
    xt_batch_size=None,
    return_div: bool = True,
    projx: bool = True,
    proju: bool = True,
    method: str = "euler",
    time_grid: Tensor = torch.tensor([0.0, 1.0]),
    return_intermediates: bool = False,
    verbose: bool = False,
    enable_grad: bool = False,
    **model_extras,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Solve the Riemannian ODE and, in addition, compute the integrated divergence.

    This method is similar to `sample`, but during the integration it computes the divergence
    (i.e. the trace of the Jacobian of the projected velocity field) at each time step and
    accumulates dt * divergence to approximate the log determinant of the transformation.

    Args:
        x_init (Tensor): initial conditions.
        step_size (float): step size.
        return_div (bool): If True, return a tuple (final_sample, log_det); otherwise, return sample only.
        projx (bool): Whether to project the state onto the manifold at each step.
        proju (bool): Whether to project the velocity onto the tangent plane at each step.
        method (str): One of ["euler", "midpoint", "rk4"].
        time_grid (Tensor): Time grid (will be sorted). If step_size is not None, a uniform discretization is used.
        return_intermediates (bool): If True, return samples at the times specified in time_grid.
        verbose (bool): Whether to display a progress bar.
        enable_grad (bool): Whether to compute gradients during sampling.
        **model_extras: Additional keyword arguments for the velocity model.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: If return_div is True, returns a tuple containing the final sample (or sequence if return_intermediates is True)
        and the final log determinant; otherwise, returns the final sample only.
    """
    # Select the integration step function.
    step_fns = {
        "euler": _euler_step,
        "midpoint": _midpoint_step,
        "rk4": _rk4_step,
    }

    if not any([isinstance(riem_solver.manifold, x) for x in (BallManifold, ConvexPolytopeManifold, Euclidean)]):
        raise NotImplementedError('currently not yet implemented for manifolds with a non-Euclidean metric')

    assert method in step_fns.keys(), f"Unknown method {method}"
    step_fn = step_fns[method]

    # Define a velocity function that includes any necessary projections.
    def velocity_func(x, t):
        return riem_solver.velocity_model(x=x, t=t, **model_extras)

    # Prepare the time discretization.
    time_grid = torch.sort(time_grid.to(device=x_init.device)).values
    if step_size is None:
        t_discretization = time_grid
    else:
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        assert (t_final - t_init) > step_size, (
            f"Time interval [{t_init}, {t_final}] must be larger than step_size {step_size}."
        )
        n_steps = math.ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [step_size * i for i in range(n_steps)] + [t_final],
            device=x_init.device,
        )
    t0s = t_discretization[:-1]
    if verbose:
        t0s = tqdm(t0s)

    # Initialize divergence accumulator.
    log_det = torch.zeros(x_init.shape[0], device=x_init.device)

    # Optionally collect intermediate states.
    if return_intermediates:
        xts = []
        i_ret = 0

    with torch.set_grad_enabled(True):
        xt = x_init

        if xt_batch_size is None:
            xt_batch_size = xt.shape[0]

        for t0, t1 in zip(t0s, t_discretization[1:]):
            dt = t1 - t0

            # --- Compute divergence of the velocity field at the current state.
            # We need gradients here even if enable_grad is False,
            # so we use a context that always allows grad.

            N = xt.shape[0]
            divergence_chunks = []
            for start in range(0, N, xt_batch_size):
                end = min(start + xt_batch_size, N)
                sub_xt = xt[start:end].clone()
                sub_xt.requires_grad_(True)
                sub_vt = velocity_func(sub_xt, t0)  # shape: [sub_batch, d]
                sub_div = 0.0
                d = sub_xt.shape[1]
                for i in range(d):
                    grad_i = torch.autograd.grad(
                        sub_vt[:, i].sum(), sub_xt, retain_graph=True, create_graph=True
                    )[0][:, i]
                    sub_div = sub_div + grad_i
                # Detach the sub-batch divergence.
                divergence_chunks.append(sub_div.detach())
            divergence = torch.cat(divergence_chunks, dim=0)
            log_det = log_det + dt * divergence

            # xt.requires_grad_(True)
            # --- Take the integration step.
            xt_next = step_fn(
                velocity_func,
                xt,
                t0,
                dt,
                manifold=riem_solver.manifold,
                projx=projx,
                proju=proju,
            )


            # --- If returning intermediates, interpolate between xt and xt_next.
            if return_intermediates:
                while (
                    i_ret < len(time_grid)
                    and t0 <= time_grid[i_ret]
                    and time_grid[i_ret] <= t1
                ):
                    xts.append(
                        interp(riem_solver.manifold, xt, xt_next, t0, t1, time_grid[i_ret])
                    )
                    i_ret += 1
            xt = xt_next.detach()  # detach to prevent unnecessary graph buildup

    if return_intermediates:
        final_sample = torch.stack(xts, dim=0)
    else:
        final_sample = xt

    if return_div:
        return final_sample, log_det
    else:
        return final_sample



if __name__ == "__main__":
    device='cpu'
    from flow_matching.path import GeodesicProbPath
    # import numpy as np
    # from manifolds import BallManifold
    # from scipy.special import gammaln
    #
    # torch_linalg = LinAlg(backend='torch', device=device, dtype=np.float32)
    # def sample_ball(shape):
    #     return torch_linalg.sample_unit_hyper_sphere_ball(shape, ball=True)
    #
    # K = 4
    # log_ball_vol = (K / 2) * np.log(np.pi) - gammaln(K / 2 + 1)
    # def log_p_ball(values):
    #     return torch.full(values.shape[:-1], -log_ball_vol, dtype=values.dtype, device=values.device)

    # aff_model = MLP(input_dim=4, time_dim=1, hidden_dim=512).to(device)
    # aff_model.load_state_dict(torch.load(r"C:\python_projects\sbmfi\arxiv_polytope\aff_model.pt"))
    # wrapped_vf = WrappedModel(aff_model)

    # solver = ODESolver(velocity_model=wrapped_vf)
    # time_grid = torch.Tensor([0.0, 1.0]).to(dtype=torch.float, device=device)
    # x_init = sample_ball((50,4))
    # dang = solver.sample(x_init, step_size=0.05, method='midpoint')
    # ding = sample_and_div(
    #         ode_solver=solver,
    #         x_init=x_init,
    #         time_grid=time_grid,
    #         step_size=0.05,
    #         method= "midpoint",
    #         return_intermediates = False,
    #         return_div= True,
    #         exact_divergence=True,
    # )

    # manifold = BallManifold(dim=4)
    # ball_model = ProjectToTangent(  # Ensures we can just use Euclidean divergence.
    #     MLP(  # Vector field in the ambient space.
    #         input_dim=4,
    #         hidden_dim=512,
    #     ),
    #     manifold=manifold,
    # ).to(device)
    #
    # ball_model.load_state_dict(torch.load(r"C:\python_projects\sbmfi\arxiv_polytope\ball_model.pt"))
    #
    # step_size = torch.Tensor([0.05]).to(dtype=torch.float, device=device)
    #
    # x_init = sample_ball((20000, 4))
    # wrapped_vf = WrappedModel(ball_model)
    #
    # solver = RiemannianODESolver(velocity_model=wrapped_vf, manifold=manifold)  # create an ODESolver class
    #
    # # sol = solver.sample(x_init, 0.05)
    # sol2, logq = riem_sample_and_div(solver, x_init, xt_batch_size=1000, step_size=0.05, return_log_p=True, exact_divergence=True)


