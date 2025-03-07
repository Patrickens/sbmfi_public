import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from flow_matching.utils.manifolds import Manifold
import math



class MaxEntManifold(Manifold):
    def __init__(self, vertices: torch.Tensor, tol: float = 1e-8, max_iter: int = 100):
        """
        Args:
            vertices (Tensor): a tensor of shape (n, d) containing the vertices (e.g. for a hypercube).
            tol (float): tolerance for the root solver.
            max_iter (int): maximum number of iterations for LBFGS.
        """
        super().__init__()
        # Save vertices as a buffer so that they are moved to the proper device automatically.
        self.register_buffer("vertices", vertices)
        self.tol = tol
        self.max_iter = max_iter

    def max_entropy_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the maximum entropy (barycentric) coordinates for point(s) x.

        Args:
            x (Tensor): target point(s) on the manifold, shape (B,d)

        Returns:
            Tensor: maximum entropy coordinates (lambdas), shape (B, n)
        """
        _, lambdas = self._solve_mu(x)
        return lambdas

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct the point from its maximum entropy coordinates.

        Args:
            x (Tensor): target point(s) on the manifold, shape (B,d)

        Returns:
            Tensor: reconstructed point(s), shape (B,d)
        """
        lambdas = self.max_entropy_coordinates(x)
        return torch.matmul(lambdas, self.vertices)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""Computes the exponential map on the MaxEntManifold.

        The idea is to compute the dual coordinate μ corresponding to x,
        add the tangent vector u, and then map back to the manifold.

        Args:
            x (Tensor): base point(s) on the manifold, shape (B,d)
            u (Tensor): tangent vector(s) at x, shape (B,d)

        Returns:
            Tensor: the point(s) expₓ(u) on the manifold, shape (B,d)
        """
        mu_x, _ = self._solve_mu(x)  # Get the dual coordinate for x.
        mu_new = mu_x + u  # Move in the dual space.
        dot = torch.matmul(mu_new, self.vertices.T)
        exp_dot = torch.exp(dot)
        Z = torch.sum(exp_dot, dim=1, keepdim=True)
        lambdas = exp_dot / Z
        x_new = torch.matmul(lambdas, self.vertices)
        return x_new

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computes the logarithmic map on the MaxEntManifold.

        We define the log map as the difference of the dual coordinates:
            logₓ(y) = μ_y - μₓ.

        Args:
            x (Tensor): base point(s) on the manifold, shape (B,d)
            y (Tensor): target point(s) on the manifold, shape (B,d)

        Returns:
            Tensor: tangent vector(s) at x, shape (B,d)
        """
        mu_x, _ = self._solve_mu(x)
        mu_y, _ = self._solve_mu(y)
        return mu_y - mu_x

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        r"""Projects an arbitrary point x to the manifold.

        In this simple implementation we assume that x is already in the interior
        of the convex hull defined by the vertices.

        Args:
            x (Tensor): point(s) to be projected, shape (B,d)

        Returns:
            Tensor: projected point(s), shape (B,d)
        """
        # For a general polytope one might solve a quadratic program.
        # Here we assume that x is already valid.
        return x

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""Projects an arbitrary tangent vector u at x onto the tangent space.

        For points in the interior of the convex hull, the tangent space is all of ℝᵈ.

        Args:
            x (Tensor): base point(s) on the manifold, shape (B,d)
            u (Tensor): vector(s) to be projected, shape (B,d)

        Returns:
            Tensor: projected tangent vector(s), shape (B,d)
        """
        # In the interior, the tangent space is all of ℝᵈ.
        return u

    def _solve_mu(self, x: torch.Tensor):
        xdim = x.dim()
        if xdim == 1:
            x = x[None, :]
        B, d = x.shape
        mu = torch.zeros(B, d, device=x.device, dtype=x.dtype, requires_grad=True)

        def loss_fn(mu):
            dot = torch.matmul(mu, self.vertices.T)
            exp_dot = torch.exp(dot)
            Z = torch.sum(exp_dot, dim=1, keepdim=True)
            lambdas = exp_dot / Z
            rec = torch.matmul(lambdas, self.vertices)
            return 0.5 * torch.sum((rec - x) ** 2)  # Sum over batch to return a scalar

        grad_fn = torch.func.grad_and_value(loss_fn)

        optimizer = torch.optim.LBFGS(
            [mu],
            max_iter=self.max_iter,
            tolerance_grad=self.tol,
            tolerance_change=self.tol,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            grad, loss = grad_fn(mu)
            mu.grad = grad.view_as(mu)  # Ensure correct shape
            return loss.sum()  # Loss is already a scalar

        optimizer.step(closure)

        dot = torch.matmul(mu, self.vertices.T)
        exp_dot = torch.exp(dot)
        Z = torch.sum(exp_dot, dim=1, keepdim=True)
        lambdas = exp_dot / Z

        return mu, lambdas


class BallManifold(Manifold):
    """
    A Euclidean ball manifold representing the open ball in ℝ^n:
        { x in ℝ^n : ||x|| < radius }.

    The exponential map is defined as:
        expmap(x, u) = projx(x + u),
    and the logarithmic map as:
        logmap(x, y) = y - x.

    The projection operation ensures that any point that would fall outside
    the open ball is scaled back to lie inside.
    """

    def __init__(self, dim: int, radius: float = 1.0, eps: float = 1e-9):
        """
        Args:
            dim (int): Dimension of the ambient space ℝ^n.
            radius (float): Radius of the ball.
            eps (float): A small constant to ensure points remain strictly inside the ball.
        """
        super(BallManifold, self).__init__()
        self.dim = dim
        self.radius = radius
        self.eps = eps

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """
        For the Euclidean open ball, the exponential map is given by addition,
        followed by a projection to enforce the ball constraint.
        """
        y = x + u
        return self.projx(y)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        The logarithmic map in the Euclidean setting is simply subtraction.
        """
        return y - x

    def projx(self, x: Tensor) -> Tensor:
        """
        Projects points onto the open ball by scaling any point that lies outside
        (or too close to the boundary) back to have norm (radius - eps).

        Args:
            x (Tensor): Tensor of shape (..., n) representing points in ℝ^n.

        Returns:
            Tensor: The projected points, each with norm at most (radius - eps).
        """
        # Compute the Euclidean norm along the last dimension.
        norm = x.norm(p=2, dim=-1, keepdim=True)
        # For points with norm greater than (radius - eps), compute the scaling factor.
        factor = torch.where(norm > (self.radius - self.eps),
                             (self.radius - self.eps) / (norm + 1e-12),
                             torch.ones_like(norm))
        return factor * x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """
        In Euclidean space, the tangent space at any point is ℝ^n, so the tangent
        vector u is already valid.
        """
        return u


class PoincareBallManifold(Manifold):
    """
    Implements operations on the n-dimensional Poincaré ball model
    of hyperbolic space with negative curvature -c.

    Attributes:
        c (float): positive curvature parameter (default: 1.0).
        eps (float): small epsilon for numerical stability.
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        super().__init__()
        # Curvature c > 0 => negative curvature -c in the Poincaré model
        self.c = c
        self.eps = eps

    ########################################################
    #  Utility Functions
    ########################################################
    def _norm(self, x: Tensor, keepdim: bool = True) -> Tensor:
        """Compute Euclidean norm of x along last dimension."""
        return torch.norm(x, p=2, dim=-1, keepdim=keepdim).clamp_min(self.eps)

    def _lambda_x(self, x: Tensor) -> Tensor:
        r"""
        Compute the conformal factor:
            λ_x^c = 2 / (1 - c * ||x||^2).
        Used in exponential/logarithmic maps.
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True).clamp_min(self.eps)
        return 2.0 / (1.0 - self.c * x2).clamp_min(self.eps)

    def _mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Möbius addition in the Poincaré ball (with curvature c):

        x ⊕_c y =
          ( (1 + 2c <x,y> + c||y||^2) x + (1 - c||x||^2) y )
          ------------------------------------------------
                       (1 + 2c <x,y> + c^2 ||x||^2 ||y||^2)
        """
        xy = torch.sum(x * y, dim=-1, keepdim=True)  # <x,y>
        x2 = torch.sum(x * x, dim=-1, keepdim=True)  # ||x||^2
        y2 = torch.sum(y * y, dim=-1, keepdim=True)  # ||y||^2

        num = (1.0 + 2.0 * self.c * xy + self.c * y2) * x + (1.0 - self.c * x2) * y
        denom = 1.0 + 2.0 * self.c * xy + (self.c ** 2) * x2 * y2
        return num / denom.clamp_min(self.eps)

    def _mobius_neg(self, x: Tensor) -> Tensor:
        """Möbius 'negative' just the usual -x in R^n, used in logmap."""
        return -x

    ########################################################
    #  Required Methods
    ########################################################
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        r"""
        Exponential map at point x applied to tangent vector u.

        Formula (c > 0):
            exp_x^c(u) = x ⊕_c(
                tanh( sqrt(c)/2 * λ_x^c * ||u|| ) * (u / ( sqrt(c)*||u|| ))
            )

        where λ_x^c = 2 / (1 - c ||x||^2).

        Args:
            x (Tensor): BxN manifold points (on Poincaré ball).
            u (Tensor): BxN tangent vectors at x.

        Returns:
            Tensor: BxN points in the Poincaré ball.
        """
        norm_u = self._norm(u)  # ||u||
        lambda_x = self._lambda_x(x)  # λ_x^c

        # scale = tanh( sqrt(c)/2 * λ_x^c * ||u|| )
        scaled_factor = torch.tanh(
            torch.sqrt(torch.tensor(self.c, device=x.device)) * lambda_x * norm_u / 2.0
        )

        # direction = u / (||u|| * sqrt(c))
        direction = u / norm_u / torch.sqrt(torch.tensor(self.c, device=x.device))

        gamma = scaled_factor * direction  # same shape as x, y
        return self._mobius_add(x, gamma)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Logarithmic map at x of point y on the ball.

        Formula (c > 0):
            log_x^c(y) =
                2 / ( sqrt(c) * λ_x^c ) * artanh( sqrt(c) * ||-x ⊕_c y|| )
                * ( (-x ⊕_c y) / ||-x ⊕_c y|| )

        Args:
            x (Tensor): BxN manifold points (on Poincaré ball).
            y (Tensor): BxN manifold points (on Poincaré ball).

        Returns:
            Tensor: BxN tangent vectors at x.
        """
        # Möbius subtract is mobius_add(x, -y) or the other way around.
        # But for log, we compute z = (-x) ⊕_c y
        z = self._mobius_add(self._mobius_neg(x), y)
        norm_z = self._norm(z)

        lambda_x = self._lambda_x(x)  # λ_x^c

        # factor = 2/( sqrt(c)*λ_x^c ) * artanh( sqrt(c)*||z|| )
        # direction = z / ||z||
        scale = (
                        2.0
                        / (
                                torch.sqrt(torch.tensor(self.c, device=x.device)) * lambda_x
                        ).clamp_min(self.eps)
                ) * torch.atanh(
            torch.sqrt(torch.tensor(self.c, device=x.device)) * norm_z.clamp_max(1 - self.eps)
        ).clamp_min(self.eps)

        direction = z / norm_z
        return scale * direction

    def projx(self, x: Tensor) -> Tensor:
        """
        Project points x onto the open Poincaré ball of radius 1/sqrt(c).
        By default, we ensure the norm < 1/sqrt(c).
        """
        maxnorm = 1.0 / math.sqrt(self.c)
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        # cond has shape [B, 1]. We want [B] for row-wise indexing in x.
        cond = (norm_x > maxnorm).squeeze(-1)  # shape [B]

        safe_x = x.clone()
        # safe_x[cond] has shape [num_cond_true, N]
        # norm_x[cond] has shape [num_cond_true, 1]
        # which will broadcast to [num_cond_true, N]
        safe_x[cond] = (
                               safe_x[cond] / norm_x[cond]
                       ) * (maxnorm - self.eps)
        return safe_x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Project a tangent vector u onto the tangent space at x.
        For the Poincaré disk, the tangent space at x is isomorphic to R^n,
        so we usually do not need a special projection (identity).
        """
        # Typically identity for Poincaré ball, but
        # you could also multiply by the conformal factor if needed for
        # certain gradient adjustments. For now, do identity.
        return u


class ConvexPolytopeManifold(Manifold):
    r"""
    A manifold defined by the convex polytope

        P = { x in ℝⁿ : A x ≤ b }.

    The following operations are defined using Euclidean projections:
      - Exponential map: expₓ(u) = projₓ(x + u)
      - Logarithmic map: logₓ(y) = y - x
      - Tangent projection: proju(x, u) projects u onto the tangent cone at x

    The projections are computed by solving a dual quadratic program via
    iterative gradient descent updates that continue until convergence.
    """

    def __init__(self, A: torch.Tensor, b: torch.Tensor,
                 proj_max_iters: int = 50, proj_step: float = 1e-2,
                 proju_max_iters: int = 10, proju_step: float = 1e-2,
                 tol: float = 1e-5):
        """
        Args:
            A (Tensor): Constraint matrix of shape [m, n].
            b (Tensor): Constraint right-hand side, shape [m].
            proj_max_iters (int): Maximum iterations for point projection.
            proj_step (float): Step size for point projection dual updates.
            proju_max_iters (int): Maximum iterations for tangent projection.
            proju_step (float): Step size for tangent projection dual updates.
            tol (float): Convergence tolerance for the dual updates.
        """
        super().__init__()
        self.register_buffer("A", A)  # [m, n]
        self.register_buffer("b", b)  # [m]
        self.proj_max_iters = proj_max_iters
        self.proj_step = proj_step
        self.proju_max_iters = proju_max_iters
        self.proju_step = proju_step
        self.tol = tol
        # Precompute Q = A Aᵀ (used in both dual problems).
        Q = A @ A.t()  # [m, m]
        self.register_buffer("Q", Q)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Projects a batch of points x (shape [B, n]) onto the convex polytope

            P = { x in ℝⁿ : A x ≤ b }.

        This is achieved by solving the dual quadratic program

            min_{λ ≥ 0} ½ λᵀ Q λ - λᵀ (A x - b)

        via iterative gradient descent on the dual variable λ. The iteration stops
        when the maximum change in λ falls below tol or when proj_max_iters is reached.
        The projection is then recovered by

            projₓ(x) = x - Aᵀ λ.
        """
        B, n = x.shape
        m = self.A.shape[0]
        # Compute c = A x - b for each sample.
        c = x @ self.A.t() - self.b.unsqueeze(0)  # [B, m]
        lam = torch.zeros(B, m, device=x.device, dtype=x.dtype)
        iteration = 0
        while True:
            grad = lam @ self.Q - c  # [B, m]
            lam_new = torch.clamp(lam - self.proj_step * grad, min=0.0)
            # Check convergence: if maximum change is below tol.
            if torch.max(torch.abs(lam_new - lam)) < self.tol:
                lam = lam_new
                break
            lam = lam_new
            iteration += 1
            if iteration >= self.proj_max_iters:
                break
        x_proj = x - lam @ self.A
        return x_proj

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Projects a tangent vector u at x onto the tangent cone of P at x.

        For x on the boundary (i.e. where A x is nearly b), define the active
        constraints as those for which

            A_i x ≥ b_i - tol.

        For each sample, we solve the dual problem

            min_{λ ≥ 0} ½ λᵀ Q λ - λᵀ ( (u @ Aᵀ) ⊙ active )

        and then recover the projected tangent vector by

            proju(x, u) = u - Aᵀ λ.
        The iterative updates stop when the maximum change in λ is below tol or when
        proju_max_iters is reached.
        """
        B, n = x.shape
        m = self.A.shape[0]
        Ax = x @ self.A.t()  # [B, m]
        active = (Ax >= (self.b.unsqueeze(0) - self.tol)).to(u.dtype)  # [B, m]
        masked = (u @ self.A.t()) * active  # [B, m]
        lam = torch.zeros(B, m, device=x.device, dtype=u.dtype)
        iteration = 0
        while True:
            grad = lam @ self.Q - masked  # [B, m]
            lam_new = torch.clamp(lam - self.proju_step * grad, min=0.0)
            lam_new = lam_new * active  # ensure λ=0 for inactive constraints
            if torch.max(torch.abs(lam_new - lam)) < self.tol:
                lam = lam_new
                break
            lam = lam_new
            iteration += 1
            if iteration >= self.proju_max_iters:
                break
        u_proj = u - lam @ self.A
        return u_proj

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Exponential map defined by:

            expₓ(u) = projₓ(x + u)
        """
        return self.projx(x + u)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Logarithmic map defined by:

            logₓ(y) = y - x.
        """
        return y - x


def helmert_matrix(n: int) -> torch.Tensor:
    """
    Computes a (n-1) x n Helmert submatrix whose rows form an orthonormal basis
    for the subspace of R^n orthogonal to the vector of ones.
    """
    H = torch.zeros(n - 1, n)
    for i in range(1, n):
        H[i - 1, :i] = 1.0 / i
        H[i - 1, i] = -1.0
    # Normalize each row so that they become unit vectors.
    H = H / H.norm(dim=1, keepdim=True)
    return H


class Simplex(Manifold):
    r"""
    A manifold representing the interior of the probability simplex endowed with Aitchison geometry.

    In Aitchison geometry, points are compositions (strictly positive vectors summing to one)
    and the natural geometry is induced via the isometric log-ratio (ilr) transform.

    Given an orthonormal basis \(V\) (of shape \((n-1)\times n\))—here constructed via the Helmert
    submatrix—the ilr transform and its inverse are given by:

    .. math::

        \operatorname{ilr}(x) &= \log(x)\,V^T, \\
        \operatorname{ilr}^{-1}(z) &= \operatorname{C}\Big(\exp\big(V\,z\big)\Big),

    where \(\operatorname{C}(\cdot)\) denotes closure (normalization to sum 1).

    The tangent space at \(x\) is isometrically identified with \(\mathbb{R}^{n-1}\) via the differential
    of the ilr transform:

    .. math::

        d\,\operatorname{ilr}_x(u) = V^T\Big(\frac{u}{x}\Big).

    Its inverse is given by:

    .. math::

        u = x\odot \Big(V\,v\Big), \quad\text{if } v \text{ is the ilr-coordinate of } u.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim (int): Number of parts of the composition (ambient dimension of the simplex).
        """
        super().__init__()
        self.dim = dim
        # Compute and register the Helmert submatrix as the orthonormal basis V.
        # V has shape (dim-1, dim).
        self.register_buffer('V', helmert_matrix(dim))

    def closure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the closure operator to ensure the vector is a composition:
        all entries are strictly positive and sum to 1.
        """
        x = torch.clamp(x, min=1e-6)  # avoid zeros
        return x / x.sum(dim=-1, keepdim=True)

    def ilr(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the isometric log-ratio transform of a composition.

        Args:
            x (Tensor): Composition (strictly positive, summing to one) of shape (..., dim)
        Returns:
            Tensor: ilr coordinates of shape (..., dim-1)
        """
        return torch.matmul(torch.log(x), self.V.t())

    def ilr_inv(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse isometric log-ratio transform.

        Args:
            z (Tensor): ilr coordinates of shape (..., dim-1)
        Returns:
            Tensor: Composition in the simplex (shape (..., dim))
        """
        x = torch.exp(torch.matmul(z, self.V))
        return self.closure(x)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Projects an arbitrary vector onto the simplex.

        Here we use the closure operator to map any (nonnegative) vector
        to a valid composition.

        Args:
            x (Tensor): input tensor of shape (..., dim)
        Returns:
            Tensor: composition (strictly positive entries summing to 1)
        """
        return self.closure(x)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Projects an arbitrary ambient vector u onto the tangent space at x in Aitchison geometry.

        The projection is performed via the differential of the ilr transform. That is,
        we first compute the ilr representation of u:

        .. math::

            v = V^T\Big(\frac{u}{x}\Big),

        and then map it back to obtain the tangent vector in the original space:

        .. math::

            u_{\text{proj}} = x \odot \Big(V\,v\Big).

        Args:
            x (Tensor): Base point in the simplex (composition), shape (..., dim)
            u (Tensor): Ambient vector, shape (..., dim)
        Returns:
            Tensor: Tangent vector at x (in the original space), shape (..., dim)
        """
        # Convert u into ilr coordinates.
        v = torch.matmul(u / x, self.V.t())
        # Map back into the tangent space at x.
        u_proj = x * torch.matmul(v, self.V)
        return u_proj

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Exponential map in Aitchison geometry.

        Given a base point x and a tangent vector u (both in the original space),
        we compute their ilr representations, perform Euclidean addition in the ilr space,
        and then map back via the inverse ilr transform.

        In formulas:

        .. math::

            z &= \operatorname{ilr}(x) = \log(x)\,V^T, \\
            v &= V^T\Big(\frac{u}{x}\Big), \\
            z_{\text{new}} &= z + v, \\
            x_{\text{new}} &= \operatorname{ilr}^{-1}(z_{\text{new}}) = \operatorname{C}\Big(\exp\big(V\,z_{\text{new}}\big)\Big).

        Args:
            x (Tensor): Base point in the simplex, shape (..., dim)
            u (Tensor): Tangent vector at x, shape (..., dim)
        Returns:
            Tensor: New point on the simplex.
        """
        z = self.ilr(x)
        v = torch.matmul(u / x, self.V.t())
        z_new = z + v
        return self.ilr_inv(z_new)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Logarithmic map in Aitchison geometry.

        Computes the tangent vector at x that “points toward” y. In ilr coordinates,
        this corresponds to the difference:

        .. math::

            v = \operatorname{ilr}(y) - \operatorname{ilr}(x),

        which is then mapped back to the original space via the inverse differential:

        .. math::

            u = x \odot \Big(V\,v\Big).

        Args:
            x (Tensor): Base point in the simplex, shape (..., dim)
            y (Tensor): Target point in the simplex, shape (..., dim)
        Returns:
            Tensor: Tangent vector at x, shape (..., dim)
        """
        z_x = self.ilr(x)
        z_y = self.ilr(y)
        v = z_y - z_x
        u = x * torch.matmul(v, self.V)
        return u


if __name__ == "__main__":
    from flow_matching.path import GeodesicProbPath,  CondOTProbPath
    from flow_matching.solver import ODESolver, RiemannianODESolver