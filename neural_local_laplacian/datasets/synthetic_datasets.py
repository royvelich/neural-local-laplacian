# Standard library
from typing import Tuple, List, Optional, Union, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod

# torch
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data

# numpy
import numpy as np

# scipy
from scipy.spatial import Delaunay

# neural signatures
from neural_local_laplacian.datasets.base_datasets import CoeffGenerationMethod
from neural_local_laplacian.utils.pose_transformers import PoseTransformer
from neural_local_laplacian.utils.utils import build_patches_from_vertices


# =============================================
# Generic helpers
# =============================================

def _coerce_scalar_or_range(value, name: str, cast):
    """Coerce a scalar / 1-elt / 2-elt sequence to a ``(lo, hi)`` tuple of ``cast``.

    Accepts plain Python scalars, list / tuple, and OmegaConf ``ListConfig``
    (anything iterable that yields ``cast``-convertible elements).  Validates
    ``lo <= hi``.  A scalar input returns ``(v, v)`` so callers can always
    treat the result as a uniform-random range without branching.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = cast(value)
        return (v, v)
    try:
        items = [cast(x) for x in value]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{name} must be a scalar or 1/2-element sequence, "
            f"got {value!r}: {e}")
    if len(items) == 1:
        return (items[0], items[0])
    if len(items) == 2:
        if items[0] > items[1]:
            raise ValueError(f"{name}: lo {items[0]} > hi {items[1]}")
        return (items[0], items[1])
    raise ValueError(
        f"{name} must be a scalar or 1/2-element sequence, got "
        f"{len(items)} elements: {items}")


# =============================================
# Polynomial evaluation (reusable)
# =============================================

def get_polynomial_pairs(order: int) -> List[Tuple[int, int]]:
    """Get list of (i, j) exponent pairs for polynomial of given order."""
    return [(i, j) for i in range(order + 1) for j in range(order + 1) if 0 < i + j <= order]


def evaluate_polynomial(x: torch.Tensor, y: torch.Tensor,
                        coefficients: torch.Tensor,
                        pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """Evaluate a 2D polynomial z = Σ c_mn x^m y^n.

    Args:
        x, y: Coordinate tensors (any shape, must be broadcastable).
        coefficients: Tensor of coefficients matching ``pairs``.
        pairs: List of (m, n) exponent pairs.

    Returns:
        Polynomial values with the same shape as x.
    """
    z = torch.zeros_like(x)
    for c, (i, j) in zip(coefficients, pairs):
        z = z + c * (x ** i) * (y ** j)
    return z


def sample_polynomial_monge_surface(
    rng: np.random.Generator,
    order_range: Tuple[int, int],
    coefficient_scale_range: Tuple[float, ...],
    coeff_generation_method,
    polynomial_offset_range: Tuple[float, ...] = (0.0, 0.0),
):
    """Sample a random polynomial Monge surface ``z = p(x + ox, y + oy)``.

    Used by both the patch-level ``PolynomialSurfaceDataset`` and the
    surface-level ``MongeSurfaceVariationalDataset`` so the polynomial
    sampling logic lives in exactly one place.

    Args:
        rng: numpy random generator.
        order_range: ``(min_order, max_order)`` polynomial order range.
        coefficient_scale_range: scalar or 2-tuple range for the coefficient
            magnitudes (used as ``coefficient_scale`` below).
        coeff_generation_method: a :class:`CoeffGenerationMethod` value
            (``UNIFORM`` or ``NORMAL``).
        polynomial_offset_range: range for the random ``(ox, oy)`` offset
            applied before evaluation, sliding the surface under the grid.

    Returns:
        surface_func: callable ``(x, y) -> z`` that closes over the sampled
            coefficients and offset.
        surface_params: dict ``{'coefficients', 'pairs', 'order', 'offset'}``.
    """
    def _sample_range(r):
        if len(r) == 2:
            return float(rng.uniform(low=r[0], high=r[1]))
        return float(r[0])

    order = int(rng.integers(low=order_range[0], high=order_range[1] + 1))
    pairs = get_polynomial_pairs(order)
    num_coeffs = len(pairs)
    scale = _sample_range(coefficient_scale_range)

    if coeff_generation_method == CoeffGenerationMethod.UNIFORM:
        coefficients = torch.tensor(
            2 * (rng.uniform(size=num_coeffs) - 0.5) * scale)
    elif coeff_generation_method == CoeffGenerationMethod.NORMAL:
        coefficients = torch.tensor(rng.normal(size=num_coeffs) * scale)
    else:
        raise ValueError(
            f"Invalid coefficient generation method: {coeff_generation_method}")

    offset_x = _sample_range(polynomial_offset_range)
    offset_y = _sample_range(polynomial_offset_range)

    def surface_func(x, y, _c=coefficients, _p=pairs,
                     _ox=offset_x, _oy=offset_y):
        return evaluate_polynomial(x + _ox, y + _oy, _c, _p)

    surface_params = {
        'coefficients': coefficients,
        'pairs': pairs,
        'order': order,
        'offset': (offset_x, offset_y),
    }
    return surface_func, surface_params


# =============================================
# Monge surface samplers (polynomial / random Fourier)
# =============================================

class BaseMongeSurfaceSampler(ABC):
    """Abstract base for Monge-patch surface samplers.

    A surface sampler decides which family of analytic surfaces is used
    to drive synthetic-patch training and visualisation.  It produces
    a closure ``surface_func(x, y) -> z`` plus a serialisable parameters
    dict that callers can store / log.

    Subclasses
        :class:`PolynomialMongeSurfaceSampler`     — random polynomials
        :class:`RandomFourierMongeSurfaceSampler`  — GP-like, random
            Fourier features with a power-law spectral envelope
            (LBO conversation §22).
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator):
        """Draw a fresh surface.

        Returns:
            surface_func: callable ``(x, y) -> z`` (autograd-compatible).
            surface_params: dict of plain Python / torch primitives that
                describes the sampled surface.  Must round-trip through
                logging (no callables in here besides the optional
                ``_surface_func`` key handled by the dataset).
        """


class PolynomialMongeSurfaceSampler(BaseMongeSurfaceSampler):
    """Random polynomial Monge surface ``z = p(x + ox, y + oy)``.

    Thin wrapper around :func:`sample_polynomial_monge_surface` so the
    same logic that has driven training for months is now reachable via
    a Hydra-instantiable object.  All args have the same meanings.
    """

    def __init__(
        self,
        order_range: Tuple[int, int],
        coefficient_scale_range: Tuple[float, ...],
        coeff_generation_method: CoeffGenerationMethod,
        polynomial_offset_range: Tuple[float, ...] = (0.0, 0.0),
    ):
        self._order_range = order_range
        self._coefficient_scale_range = coefficient_scale_range
        self._coeff_generation_method = coeff_generation_method
        self._polynomial_offset_range = polynomial_offset_range

    def sample(self, rng):
        return sample_polynomial_monge_surface(
            rng=rng,
            order_range=self._order_range,
            coefficient_scale_range=self._coefficient_scale_range,
            coeff_generation_method=self._coeff_generation_method,
            polynomial_offset_range=self._polynomial_offset_range,
        )


class RandomFourierMongeSurfaceSampler(BaseMongeSurfaceSampler):
    r"""GP-like Monge surface via random Fourier features.

    Samples a surface ``z = h(u, v)`` with closed-form derivatives:

        h(u, v) = Σ_{|k1|+|k2| ≤ K}  σ_{k1,k2} · [a_{k1,k2} · cos(π k1 u + π k2 v)
                                                  + b_{k1,k2} · sin(π k1 u + π k2 v)]

    where ``a, b ~ N(0, 1)`` are drawn per-surface and the spectral
    envelope is

        σ_{k1,k2} = magnitude_scale / (1 + k1² + k2²)^(β / 2)

    Higher ``K`` admits higher-frequency modes; higher ``β`` makes the
    surface smoother.  ``magnitude_scale`` controls the overall amplitude.

    The polynomial sampler's per-surface offset is unnecessary here:
    the random phases already give a uniformly-translated ensemble.

    All three of ``K``, ``beta``, ``magnitude_scale`` accept either a
    scalar (fixed across surfaces) or a 1- or 2-element sequence (a
    range from which a fresh value is drawn per surface — uniform
    integer for ``K``, uniform float for the other two).  This mirrors
    the ``order_range`` / ``coefficient_scale_range`` style used by
    the polynomial sampler.

    Args:
        K: Maximum frequency along each axis (modes with ``|k1| + |k2| ≤ K``
           are kept).  Scalar or ``(K_lo, K_hi)``.
        beta: Spectral decay exponent.  ``β = 2`` is a smoothish default.
            Scalar or ``(β_lo, β_hi)``.  Higher β → smoother surfaces.
        magnitude_scale: Overall amplitude (multiplies σ).  Scalar or
            ``(scale_lo, scale_hi)``.
        l1_truncate: If True, truncate by ``|k1| + |k2| ≤ K`` (cone);
            otherwise by ``|k1|, |k2| ≤ K`` (square).  Default True
            matches the LBO conversation recipe.
    """

    def __init__(
        self,
        K=10,
        beta=2.0,
        magnitude_scale=1.0,
        l1_truncate: bool = True,
    ):
        self._K_range = _coerce_scalar_or_range(K, name='K', cast=int)
        self._beta_range = _coerce_scalar_or_range(beta, name='beta', cast=float)
        self._magnitude_scale_range = _coerce_scalar_or_range(
            magnitude_scale, name='magnitude_scale', cast=float)
        if self._K_range[0] < 0:
            raise ValueError(f"K must be >= 0, got range {self._K_range}")
        if self._beta_range[0] < 0:
            raise ValueError(f"beta must be >= 0, got range {self._beta_range}")
        self._l1_truncate = bool(l1_truncate)

        # (k1, k2) grids only depend on K (and l1_truncate, which is fixed
        # per instance), so cache by K — typically a small handful of
        # distinct values across an epoch.  σ depends on (β, scale) and is
        # cheap to recompute per sample.
        self._grid_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_grid(self, K_int: int):
        """Return cached (k1, k2) frequency grids for this ``K``."""
        cached = self._grid_cache.get(K_int)
        if cached is not None:
            return cached
        ks = np.arange(-K_int, K_int + 1)
        K1, K2 = np.meshgrid(ks, ks, indexing='ij')
        K1 = K1.flatten()
        K2 = K2.flatten()
        if self._l1_truncate:
            keep = (np.abs(K1) + np.abs(K2)) <= K_int
            K1 = K1[keep]
            K2 = K2[keep]
        k1 = torch.from_numpy(K1.astype(np.float32))
        k2 = torch.from_numpy(K2.astype(np.float32))
        self._grid_cache[K_int] = (k1, k2)
        return k1, k2

    def sample(self, rng: np.random.Generator):
        # Per-surface draws of K / β / magnitude_scale from their ranges.
        K_lo, K_hi = self._K_range
        K_int = int(rng.integers(low=K_lo, high=K_hi + 1)) if K_lo != K_hi else K_lo

        beta_lo, beta_hi = self._beta_range
        beta = (float(rng.uniform(low=beta_lo, high=beta_hi))
                if beta_lo != beta_hi else beta_lo)

        scale_lo, scale_hi = self._magnitude_scale_range
        magnitude_scale = (float(rng.uniform(low=scale_lo, high=scale_hi))
                           if scale_lo != scale_hi else scale_lo)

        k1, k2 = self._get_grid(K_int)
        # σ envelope at the sampled (β, scale): magnitude_scale / (1 + k1² + k2²)^(β/2)
        sigma = magnitude_scale / (1.0 + k1 ** 2 + k2 ** 2) ** (beta / 2.0)

        M = sigma.shape[0]
        sigma_np = sigma.numpy()
        a = torch.from_numpy(
            (rng.standard_normal(size=M).astype(np.float32) * sigma_np))
        b = torch.from_numpy(
            (rng.standard_normal(size=M).astype(np.float32) * sigma_np))

        def surface_func(x, y, _a=a, _b=b, _k1=k1, _k2=k2):
            # x, y can be any shape; broadcast against (M,) basis.
            x_e = x.unsqueeze(-1)                     # (..., 1)
            y_e = y.unsqueeze(-1)
            phase = torch.pi * (_k1 * x_e + _k2 * y_e)
            return (_a * torch.cos(phase) + _b * torch.sin(phase)).sum(dim=-1)

        params = {
            'family': 'random_fourier',
            'K': K_int,
            'beta': beta,
            'magnitude_scale': magnitude_scale,
            'l1_truncate': self._l1_truncate,
            # Coefficients kept for reproducibility / debugging; small.
            'a': a.detach().cpu().clone(),
            'b': b.detach().cpu().clone(),
        }
        return surface_func, params


# =============================================
# Laplace-Beltrami via autograd
# =============================================

def compute_laplace_beltrami(surface_func, h_func, x, y):
    """Compute Δ_LB(h) on surface z = surface_func(x, y) using autograd.

    Both ``surface_func`` and ``h_func`` must be autograd-compatible
    callables ``(x, y) -> scalar_tensor``.

    Args:
        surface_func: Callable defining the surface z = f(x, y).
        h_func: Callable defining the test function h(x, y).
        x, y: (N, 1) tensors with ``requires_grad=True``.

    Returns:
        (N,) tensor of Δ_LB(h) values.
    """
    # Surface derivatives → metric
    z = surface_func(x, y)
    f_x = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
    f_y = torch.autograd.grad(z.sum(), y, create_graph=True)[0]

    det_g = 1.0 + f_x ** 2 + f_y ** 2
    sqrt_det_g = det_g.sqrt()

    # Inverse metric
    g11 = (1.0 + f_y ** 2) / det_g
    g12 = -(f_x * f_y) / det_g
    g22 = (1.0 + f_x ** 2) / det_g

    # Test function derivatives
    h = h_func(x, y)
    h_x = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
    h_y = torch.autograd.grad(h.sum(), y, create_graph=True)[0]

    # Flux: F_i = √det(g) · g^{ij} · h_j
    F_x = sqrt_det_g * (g11 * h_x + g12 * h_y)
    F_y = sqrt_det_g * (g12 * h_x + g22 * h_y)

    # Divergence
    dFx_dx = torch.autograd.grad(F_x.sum(), x, create_graph=True)[0]
    dFy_dy = torch.autograd.grad(F_y.sum(), y, create_graph=True)[0]

    # Δ_LB(h) = (1/√det(g)) · (∂F_x/∂x + ∂F_y/∂y)
    lb_h = (dFx_dx + dFy_dy) / sqrt_det_g

    return lb_h.squeeze(-1)


def compute_surface_gradient(surface_func, h_func, x, y):
    """Compute the surface gradient ∇_S(h) in ℝ³ on surface z = f(x,y).

    The surface gradient is the Jacobian J times the contravariant
    gradient components: ∇_S(h) = J · g^{ij} h_j.

    Args:
        surface_func, h_func: Autograd-compatible callables.
        x, y: (N, 1) tensors with requires_grad=True.

    Returns:
        (N, 3) surface gradient vectors in ℝ³.
    """
    z = surface_func(x, y)
    f_x = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
    f_y = torch.autograd.grad(z.sum(), y, create_graph=True)[0]

    det_g = 1.0 + f_x ** 2 + f_y ** 2

    g11 = (1.0 + f_y ** 2) / det_g
    g12 = -(f_x * f_y) / det_g
    g22 = (1.0 + f_x ** 2) / det_g

    h = h_func(x, y)
    h_x = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
    h_y = torch.autograd.grad(h.sum(), y, create_graph=True)[0]

    # Contravariant components
    c1 = g11 * h_x + g12 * h_y
    c2 = g12 * h_x + g22 * h_y

    # Jacobian columns: ∂(x,y,z)/∂u = (1,0,f_x), ∂(x,y,z)/∂v = (0,1,f_y)
    grad_3d = torch.cat([c1, c2, c1 * f_x + c2 * f_y], dim=-1)

    return grad_3d


def compute_lb_coefficients(surface_func, x, y):
    """Precompute LB operator coefficients from surface metric.

    On a parametric surface z = f(x,y), the Laplace-Beltrami operator has
    the form:

        Δ_LB(h) = A·h_xx + 2B·h_xy + C·h_yy + D·h_x + E·h_y

    where A, B, C, D, E depend only on the surface (not on h).

    Args:
        surface_func: Callable ``(x, y) -> z``.
        x, y: (N, 1) tensors with requires_grad=True.

    Returns:
        Dict with detached tensors (each shape (N, 1) or squeezed):
        A, B, C (inverse metric), D, E (Christoffel-like), f_x, f_y.
    """
    z = surface_func(x, y)
    f_x = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
    f_y = torch.autograd.grad(z.sum(), y, create_graph=True)[0]

    # Second derivatives of surface
    f_xx, f_xy = torch.autograd.grad(f_x.sum(), [x, y], create_graph=True)
    _, f_yy = torch.autograd.grad(f_y.sum(), [x, y], create_graph=True)

    # Metric quantities
    det_g = 1.0 + f_x ** 2 + f_y ** 2
    sqrt_det_g = det_g.sqrt()

    # Inverse metric = LB second-order coefficients
    A = (1.0 + f_y ** 2) / det_g     # g^11
    B = -(f_x * f_y) / det_g          # g^12
    C = (1.0 + f_x ** 2) / det_g     # g^22

    # First-order coefficients: D, E from derivatives of (√det · g^ij)
    # α₁ = √det · g^11, α₂ = √det · g^12, α₃ = √det · g^22
    # D = (∂α₁/∂x + ∂α₂/∂y) / √det,  E = (∂α₂/∂x + ∂α₃/∂y) / √det
    a1 = (1.0 + f_y ** 2) / sqrt_det_g
    a2 = -(f_x * f_y) / sqrt_det_g
    a3 = (1.0 + f_x ** 2) / sqrt_det_g

    da1_dx = torch.autograd.grad(a1.sum(), x, create_graph=True)[0]
    da2_dx = torch.autograd.grad(a2.sum(), x, create_graph=True)[0]
    da2_dy = torch.autograd.grad(a2.sum(), y, create_graph=True)[0]
    da3_dy = torch.autograd.grad(a3.sum(), y, create_graph=True)[0]

    D = (da1_dx + da2_dy) / sqrt_det_g
    E = (da2_dx + da3_dy) / sqrt_det_g

    return {
        'A': A.detach(), 'B': B.detach(), 'C': C.detach(),
        'D': D.detach(), 'E': E.detach(),
        'f_x': f_x.detach(), 'f_y': f_y.detach(),
    }


def compute_h_derivatives(h_func, x, y):
    """Compute h and its first/second derivatives on a fresh autograd graph.

    Handles test functions that may not depend on one (or both) of x, y
    (e.g., the coordinate test functions ``h(x,y) = x``, ``h(x,y) = y``)
    by using ``allow_unused=True`` and substituting zero tensors for any
    gradients PyTorch reports as missing from the graph.

    Args:
        h_func: Callable ``(x, y) -> scalar tensor``.
        x, y: (N, 1) tensors (will be cloned with requires_grad).

    Returns:
        Dict with detached tensors: h, h_x, h_y, h_xx, h_xy, h_yy.
    """
    x_h = x.clone().detach().requires_grad_(True)
    y_h = y.clone().detach().requires_grad_(True)

    h = h_func(x_h, y_h)
    if h.dim() == 1:
        h = h.unsqueeze(-1)

    # If h itself is detached from the graph (e.g. h is literally a
    # constant tensor like torch.full_like(...)), all derivatives are
    # zero and we can short-circuit before any grad() call.
    if not h.requires_grad:
        zero_x = torch.zeros_like(x_h)
        zero_y = torch.zeros_like(y_h)
        return {
            'h': h.detach(),
            'h_x': zero_x, 'h_y': zero_y,
            'h_xx': zero_x.clone(), 'h_xy': zero_y.clone(),
            'h_yy': zero_y.clone(),
        }

    # First derivatives. ``allow_unused=True`` returns None instead of
    # raising when h doesn't depend on the wrt-tensor. We track which
    # gradients came from the real autograd path vs. our zero fallback,
    # because only the real ones are valid inputs to subsequent grad()
    # calls — a fresh torch.zeros_like() tensor is not in any graph.
    h_x_real = torch.autograd.grad(
        h.sum(), x_h, create_graph=True, allow_unused=True)[0]
    h_y_real = torch.autograd.grad(
        h.sum(), y_h, create_graph=True, allow_unused=True)[0]

    h_x = h_x_real if h_x_real is not None else torch.zeros_like(x_h)
    h_y = h_y_real if h_y_real is not None else torch.zeros_like(y_h)

    # Second derivatives. Two reasons we may need to skip the autograd call:
    #   (a) h_*_real is None — h didn't depend on the variable at all.
    #   (b) h_*_real is a constant tensor with no grad_fn — happens when
    #       h is linear in that variable (e.g., h(x,y)=x → h_x is constant
    #       1, with requires_grad=False even under create_graph=True).
    # In both cases the second derivative is mathematically zero and we
    # avoid calling grad() on a tensor that isn't in the graph.
    h_x_in_graph = (h_x_real is not None) and h_x_real.requires_grad
    h_y_in_graph = (h_y_real is not None) and h_y_real.requires_grad

    if not h_x_in_graph:
        h_xx = torch.zeros_like(x_h)
        h_xy = torch.zeros_like(y_h)
    else:
        h_xx_raw, h_xy_raw = torch.autograd.grad(
            h_x_real.sum(), [x_h, y_h],
            retain_graph=True, allow_unused=True)
        h_xx = h_xx_raw if h_xx_raw is not None else torch.zeros_like(x_h)
        h_xy = h_xy_raw if h_xy_raw is not None else torch.zeros_like(y_h)

    if not h_y_in_graph:
        h_yy = torch.zeros_like(y_h)
    else:
        _, h_yy_raw = torch.autograd.grad(
            h_y_real.sum(), [x_h, y_h],
            create_graph=False, allow_unused=True)
        h_yy = h_yy_raw if h_yy_raw is not None else torch.zeros_like(y_h)

    return {
        'h': h.detach(), 'h_x': h_x.detach(), 'h_y': h_y.detach(),
        'h_xx': h_xx.detach(), 'h_xy': h_xy.detach(), 'h_yy': h_yy.detach(),
    }


def apply_lb_from_coefficients(coeffs, h_derivs):
    """Compute Δ_LB(h) = A·h_xx + 2B·h_xy + C·h_yy + D·h_x + E·h_y.

    Args:
        coeffs: Dict from compute_lb_coefficients (A, B, C, D, E, f_x, f_y).
        h_derivs: Dict from compute_h_derivatives (h_x, h_y, h_xx, h_xy, h_yy).

    Returns:
        lb: (N,) Laplace-Beltrami values.
        grad_3d: (N, 3) surface gradient vectors.
    """
    lb = (coeffs['A'] * h_derivs['h_xx']
          + 2.0 * coeffs['B'] * h_derivs['h_xy']
          + coeffs['C'] * h_derivs['h_yy']
          + coeffs['D'] * h_derivs['h_x']
          + coeffs['E'] * h_derivs['h_y']).squeeze(-1)

    # Surface gradient: (c1, c2, c1*f_x + c2*f_y)
    c1 = coeffs['A'] * h_derivs['h_x'] + coeffs['B'] * h_derivs['h_y']
    c2 = coeffs['B'] * h_derivs['h_x'] + coeffs['C'] * h_derivs['h_y']
    grad_3d = torch.cat([c1, c2, c1 * coeffs['f_x'] + c2 * coeffs['f_y']], dim=-1)

    return lb, grad_3d


def compute_lb_and_gradient_batch(surface_func, h_funcs, x0, y0, verbose=False):
    """Compute Δ_LB(h) and ∇_S(h) for multiple test functions.

    Precomputes LB operator coefficients from the surface once, then evaluates
    each test function on a fresh autograd graph.  No graph accumulation.

    Args:
        surface_func: Callable ``(x, y) -> z`` defining the surface.
        h_funcs: List of callables ``(x, y) -> scalar``.
        x0, y0: (N, 1) tensors with requires_grad=True (evaluation points).
        verbose: If True, print per-phase timing.

    Returns:
        laplacians: (P,) or (N, P) tensor of Δ_LB(h_p) values (detached).
        gradients:  (P, 3) or (N, P, 3) tensor of ∇_S(h_p) vectors (detached).
    """
    import time as _time
    P = len(h_funcs)
    N = x0.shape[0]

    # Precompute LB coefficients from surface (once, then detached)
    t0 = _time.perf_counter()
    coeffs = compute_lb_coefficients(surface_func, x0, y0)
    t_coeffs = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    if N == 1:
        # Single point (origin) — return (P,) and (P, 3)
        laplacians = torch.zeros(P, dtype=torch.float32)
        gradients = torch.zeros(P, 3, dtype=torch.float32)
        for p, h in enumerate(h_funcs):
            h_derivs = compute_h_derivatives(h, x0, y0)
            lb, grad_3d = apply_lb_from_coefficients(coeffs, h_derivs)
            laplacians[p] = lb.squeeze()
            gradients[p] = grad_3d.squeeze(0)
    else:
        # Multiple points — return (N, P) and (N, P, 3)
        laplacians = torch.zeros(N, P, dtype=torch.float32)
        gradients = torch.zeros(N, P, 3, dtype=torch.float32)
        for p, h in enumerate(h_funcs):
            h_derivs = compute_h_derivatives(h, x0, y0)
            lb, grad_3d = apply_lb_from_coefficients(coeffs, h_derivs)
            laplacians[:, p] = lb
            gradients[:, p] = grad_3d
    t_funcs = _time.perf_counter() - t0

    if verbose:
        print(f"      [lb_batch] N={N}, P={P}: coeffs={t_coeffs*1e3:.0f}ms, "
              f"h_derivs_loop={t_funcs*1e3:.0f}ms ({t_funcs/P*1e3:.1f}ms/func)")

    return laplacians, gradients


# =============================================
# Continuous Dirichlet form on a Monge patch
# =============================================

def gauss_legendre_2d_grid(
    U_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    n: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes and weights on a 2D rectangle.

    Args:
        U_bounds: ``((u_lo, u_hi), (v_lo, v_hi))`` parameter-domain rectangle.
        n: nodes per axis (so total Q = n²).
        dtype: output dtype.

    Returns:
        u_grid: ``(n, n)`` u-coordinates of quadrature points.
        v_grid: ``(n, n)`` v-coordinates of quadrature points.
        weights: ``(n, n)`` quadrature weights (incorporate the change-of-variables
            Jacobian for a non-unit rectangle).
    """
    (u_lo, u_hi), (v_lo, v_hi) = U_bounds

    # Reference nodes/weights on [-1, 1]
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(n)

    # Affine map [-1, 1] → [lo, hi] with Jacobian (hi - lo)/2
    u_nodes = 0.5 * (u_hi - u_lo) * nodes_1d + 0.5 * (u_hi + u_lo)
    v_nodes = 0.5 * (v_hi - v_lo) * nodes_1d + 0.5 * (v_hi + v_lo)
    u_weights = 0.5 * (u_hi - u_lo) * weights_1d
    v_weights = 0.5 * (v_hi - v_lo) * weights_1d

    u_grid_np, v_grid_np = np.meshgrid(u_nodes, v_nodes, indexing='ij')
    weights_np = u_weights[:, None] * v_weights[None, :]

    return (
        torch.from_numpy(u_grid_np).to(dtype),
        torch.from_numpy(v_grid_np).to(dtype),
        torch.from_numpy(weights_np).to(dtype),
    )


def compute_dirichlet_bilinear_form_continuous(
    surface_func,
    h_funcs: List,
    U_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    quadrature_n: int = 30,
) -> torch.Tensor:
    """Continuous Dirichlet bilinear form on a Monge patch via Gauss-Legendre.

    Computes the symmetric P×P matrix

        B[ℓ, p] = ∫_U  ⟨∇_g f̃_ℓ, ∇_g f̃_p⟩_g  √det g  du dv ,

    where each ``f̃_ℓ(u, v) = h_funcs[ℓ](u, v)`` is a function on the parameter
    domain (intrinsic, *not* an ambient ℝ³ function pulled back), and the
    metric is the Monge metric induced by ``z = surface_func(u, v)``:

        g = (( 1+h_u² , h_u h_v ), ( h_u h_v , 1+h_v² )),    √det g = √(1+h_u²+h_v²).

    Closed-form expansion:

        ⟨∇_g f̃_ℓ, ∇_g f̃_p⟩_g  =  g^{11} f̃_u^{(ℓ)} f̃_u^{(p)}
                                + g^{12} ( f̃_u^{(ℓ)} f̃_v^{(p)} + f̃_v^{(ℓ)} f̃_u^{(p)} )
                                + g^{22} f̃_v^{(ℓ)} f̃_v^{(p)} .

    The diagonal ``B[ℓ, ℓ] = 2 · E(f_ℓ)``.

    Args:
        surface_func: callable ``(u, v) -> z`` (autograd-compatible).
        h_funcs: list of P callables ``(u, v) -> scalar`` defined on the chart.
        U_bounds: parameter-domain rectangle.
        quadrature_n: Gauss-Legendre nodes per axis. n=30 gives ~1e-12 for
            smooth integrands.

    Returns:
        ``(P, P)`` symmetric tensor (detached — analytic ground truth).
    """
    P = len(h_funcs)
    if P == 0:
        return torch.zeros(0, 0)

    u_grid, v_grid, w_grid = gauss_legendre_2d_grid(U_bounds, quadrature_n)

    # Flatten into Q = n² quadrature points
    u_flat = u_grid.flatten()    # (Q,)
    v_flat = v_grid.flatten()
    w_flat = w_grid.flatten()
    Q = u_flat.shape[0]

    # Surface derivatives at quadrature points (autograd through surface_func).
    # We give the surface (Q, 1)-shaped inputs to match the convention used
    # by compute_lb_coefficients elsewhere in this module.
    u_s = u_flat.clone().unsqueeze(-1).requires_grad_(True)
    v_s = v_flat.clone().unsqueeze(-1).requires_grad_(True)
    z = surface_func(u_s, v_s)
    if z.dim() > 2:
        z = z.squeeze(-1)
    # Compute both partials in one grad() call so the saved-tensor graph
    # is freed exactly once (a sequence of two grad() calls would trigger
    # ``Trying to backward through the graph a second time``).
    h_u, h_v = torch.autograd.grad(z.sum(), [u_s, v_s], create_graph=False)
    h_u = h_u.squeeze(-1)   # (Q,)
    h_v = h_v.squeeze(-1)

    det_g = 1.0 + h_u ** 2 + h_v ** 2
    sqrt_det_g = det_g.sqrt()
    g11 = (1.0 + h_v ** 2) / det_g
    g12 = -(h_u * h_v) / det_g
    g22 = (1.0 + h_u ** 2) / det_g

    # Test-function derivatives at the same quadrature points.
    f_tilde_u = torch.zeros(Q, P)
    f_tilde_v = torch.zeros(Q, P)
    for p_idx, h_func in enumerate(h_funcs):
        u_h = u_flat.clone().unsqueeze(-1).requires_grad_(True)
        v_h = v_flat.clone().unsqueeze(-1).requires_grad_(True)
        f_tilde = h_func(u_h, v_h)
        if f_tilde.dim() > 2:
            f_tilde = f_tilde.squeeze(-1)

        if not f_tilde.requires_grad:
            # h is a constant tensor → derivatives are zero (already zero-init).
            continue

        df_du, df_dv = torch.autograd.grad(
            f_tilde.sum(), [u_h, v_h],
            create_graph=False, allow_unused=True)
        if df_du is not None:
            f_tilde_u[:, p_idx] = df_du.squeeze(-1)
        if df_dv is not None:
            f_tilde_v[:, p_idx] = df_dv.squeeze(-1)

    # Pointwise integrand B(ℓ, p) at every quadrature point, then quadrature sum.
    # Outer products: (Q, P, P)
    fu_outer    = f_tilde_u.unsqueeze(-1) * f_tilde_u.unsqueeze(-2)
    fv_outer    = f_tilde_v.unsqueeze(-1) * f_tilde_v.unsqueeze(-2)
    cross_outer = (f_tilde_u.unsqueeze(-1) * f_tilde_v.unsqueeze(-2)
                   + f_tilde_v.unsqueeze(-1) * f_tilde_u.unsqueeze(-2))

    # Broadcast g-coefficients along (P, P) — they're (Q,) functions of (u, v).
    g11_b = g11.unsqueeze(-1).unsqueeze(-1)
    g12_b = g12.unsqueeze(-1).unsqueeze(-1)
    g22_b = g22.unsqueeze(-1).unsqueeze(-1)
    sqrt_det_g_b = sqrt_det_g.unsqueeze(-1).unsqueeze(-1)

    integrand = (g11_b * fu_outer + g12_b * cross_outer + g22_b * fv_outer) * sqrt_det_g_b
    bilinear = (w_flat.unsqueeze(-1).unsqueeze(-1) * integrand).sum(dim=0)   # (P, P)

    # Symmetrise (numerically) and detach — this is GT.
    bilinear = 0.5 * (bilinear + bilinear.T)
    return bilinear.detach()


def compute_dirichlet_energy_continuous(
    surface_func,
    h_funcs: List,
    U_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    quadrature_n: int = 30,
) -> torch.Tensor:
    """Continuous Dirichlet energy per probe: ``E(f_ℓ) = ½ ∫_U ‖∇_g f̃_ℓ‖² √det g``.

    Equals ``½ · diag(compute_dirichlet_bilinear_form_continuous(...))``.

    Args:
        surface_func: callable ``(u, v) -> z``.
        h_funcs: list of P callables.
        U_bounds: parameter-domain rectangle.
        quadrature_n: Gauss-Legendre nodes per axis.

    Returns:
        ``(P,)`` energies (detached).
    """
    bilinear = compute_dirichlet_bilinear_form_continuous(
        surface_func, h_funcs, U_bounds, quadrature_n=quadrature_n)
    return 0.5 * torch.diagonal(bilinear)


# =============================================
# Test function sampling
# =============================================

class BaseTestFunctionSampler(ABC):
    """Abstract base for test-function samplers.

    Subclasses produce a set of ``P`` scalar test functions
    ``h_p(x, y)`` defined on the Monge chart of a synthetic surface.
    The base class then computes, given the surface ``surface_func``:

        test_func_deltas      (K, P)  h_p(grid) - h_p(origin)
        test_func_values      (K, P)  h_p(grid)
        test_func_laplacians  (P,)    Δ_g h_p at the origin
        test_func_gradients   (P, 3)  ∇_S h_p at the origin (in ℝ³)

    plus the optional per-grid-point ``test_func_lb_all_points`` and
    ``test_func_gradients_all_points`` when the corresponding flags are
    set.

    Subclasses must implement:
        ``_enumerate(rng) -> List[spec]``
            Decide which P test functions to use.  Specs are opaque to
            the base class (param dicts / exponent tuples / …).

        ``_specs_to_callables(specs, surface_func) -> List[Callable]``
            Convert specs to ``h_p(x, y)`` callables for the autograd
            path and grid evaluation.  Ambient-coordinate samplers can
            close ``surface_func`` into the callable (e.g. ``z = f(x, y)``).

    Subclasses may optionally implement:
        ``_specs_to_analytic_data(specs, x_grid, y_grid)``
            Return ``(deltas, values, h_d)`` without using autograd, where
            ``h_d`` is a dict of partial derivatives of each test function
            at the origin (keys ``h, h_x, h_y, h_xx, h_xy, h_yy``, each
            shape ``(P,)``).  Returning ``None`` (the default) forces the
            autograd path.

    Args:
        normalize_target: Target normalization mode.
            ``'none'``           — raw Δ_LB values.
            ``'unit_magnitude'`` — divide Δ_LB and deltas by |Δ_LB| per probe.
            ``'unit_variance'``  — divide deltas / Δ_LB so deltas have unit var.
        derivative_mode: ``'analytic'`` (try the closed-form metric path
            via subclass-supplied origin derivatives) or ``'autograd'``
            (always go through ``compute_lb_and_gradient_batch``).  Falls
            back to autograd automatically when the subclass returns
            ``None`` from ``_specs_to_analytic_data``.
        compute_lb_all_points: If True, also compute Δ_LB at every grid
            point (always via autograd).
        compute_gradients_all_points: If True, also compute ∇_S at every
            grid point (always via autograd).
        compute_continuous_energy: If True, ``sample()`` additionally returns
            ``test_func_continuous_energy`` of shape ``(P,)``  the continuous
            Dirichlet energy ``E(f_ℓ)`` per probe, computed by Gauss-Legendre
            quadrature on the parameter domain. Used as the GT target for
            ``DirichletEnergyTestLoss`` / variational training.
        compute_continuous_bilinear: If True, ``sample()`` additionally returns
            ``test_func_continuous_bilinear`` of shape ``(P, P)``  the
            continuous bilinear form ``E(f_ℓ, f_p)`` per probe pair. Used as
            the GT target for ``BilinearFormTestLoss``.
        quadrature_n: Gauss-Legendre nodes per axis for the continuous
            quadrature. Total quadrature points = quadrature_n². Default 30
            gives ~1e-12 accuracy for smooth integrands.
        verbose: Print per-call diagnostics.
    """

    def __init__(
        self,
        normalize_target: str = 'none',
        derivative_mode: str = 'analytic',
        compute_lb_all_points: bool = False,
        compute_gradients_all_points: bool = False,
        compute_continuous_energy: bool = False,
        compute_continuous_bilinear: bool = False,
        quadrature_n: int = 30,
        verbose: bool = False,
    ):
        if derivative_mode not in ('analytic', 'autograd'):
            raise ValueError(
                f"derivative_mode must be 'analytic' or 'autograd', got '{derivative_mode}'")
        if normalize_target not in ('none', 'unit_magnitude', 'unit_variance'):
            raise ValueError(
                f"normalize_target must be 'none' | 'unit_magnitude' | 'unit_variance', "
                f"got '{normalize_target}'")
        if quadrature_n < 1:
            raise ValueError(f"quadrature_n must be >= 1, got {quadrature_n}")
        self.normalize_target = normalize_target
        self.derivative_mode = derivative_mode
        self.compute_lb_all_points = compute_lb_all_points
        self.compute_gradients_all_points = compute_gradients_all_points
        self.compute_continuous_energy = compute_continuous_energy
        self.compute_continuous_bilinear = compute_continuous_bilinear
        self.quadrature_n = int(quadrature_n)
        self.verbose = verbose

    # ----- subclass API --------------------------------------------------

    @abstractmethod
    def _enumerate(self, rng: np.random.Generator) -> List[Any]:
        ...

    @abstractmethod
    def _specs_to_callables(self, specs: List[Any], surface_func) -> List[Any]:
        ...

    def _specs_to_analytic_data(self, specs: List[Any],
                                x_grid: torch.Tensor,
                                y_grid: torch.Tensor):
        return None

    # ----- main API ------------------------------------------------------

    def sample(
        self,
        surface_func,
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
        rng: np.random.Generator,
        U_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute test-function probes on a single Monge-patch surface.

        Args:
            surface_func: callable ``(u, v) -> z``.
            x_grid, y_grid: ``(K,)`` parameter-domain coordinates of the K
                vertices on this patch.  Probes are evaluated at these
                positions and at the origin.
            rng: numpy random generator.
            U_bounds: ``((u_lo, u_hi), (v_lo, v_hi))`` parameter-domain
                rectangle for the continuous quadrature (only used when
                ``compute_continuous_energy`` or ``compute_continuous_bilinear``
                is set).  If ``None``, derived from the extent of the supplied
                grid as ``((x.min(), x.max()), (y.min(), y.max()))``.
        """
        import time as _time
        _t = {} if self.verbose else None

        if _t is not None: _t0 = _time.perf_counter()
        specs = self._enumerate(rng)
        P = len(specs)
        K = len(x_grid)
        if _t is not None: _t['enumerate'] = _time.perf_counter() - _t0

        # ---- Try the analytic path first (subclass may decline) -------
        analytic_result = None
        if self.derivative_mode == 'analytic':
            if _t is not None: _t0 = _time.perf_counter()
            analytic_result = self._specs_to_analytic_data(specs, x_grid, y_grid)
            if _t is not None: _t['analytic_specs'] = _time.perf_counter() - _t0

        if analytic_result is not None:
            deltas, values, h_d = analytic_result
            if _t is not None: _t0 = _time.perf_counter()
            x0 = torch.tensor([[0.0]], requires_grad=True)
            y0 = torch.tensor([[0.0]], requires_grad=True)
            coeffs = compute_lb_coefficients(surface_func, x0, y0)
            # apply_lb_from_coefficients expects (N, 1)-shaped derivatives;
            # h_d is (P,)-shaped, so unsqueeze.  coeffs are (1, 1) and
            # broadcast into (P, 1).
            h_derivs = {k: h_d[k].unsqueeze(-1)
                        for k in ('h_x', 'h_y', 'h_xx', 'h_xy', 'h_yy')}
            laplacians, gradients = apply_lb_from_coefficients(coeffs, h_derivs)
            laplacians = laplacians.detach()
            gradients = gradients.detach()
            if _t is not None: _t['lb_origin'] = _time.perf_counter() - _t0
        else:
            if _t is not None: _t0 = _time.perf_counter()
            h_funcs = self._specs_to_callables(specs, surface_func)
            if _t is not None: _t['build_callables'] = _time.perf_counter() - _t0

            if _t is not None: _t0 = _time.perf_counter()
            with torch.no_grad():
                deltas = torch.zeros(K, P, dtype=torch.float32)
                values = torch.zeros(K, P, dtype=torch.float32)
                for p, h in enumerate(h_funcs):
                    h_grid = h(x_grid, y_grid)
                    h_center = h(torch.tensor(0.0), torch.tensor(0.0))
                    values[:, p] = h_grid
                    deltas[:, p] = h_grid - h_center
            if _t is not None: _t['deltas'] = _time.perf_counter() - _t0

            if _t is not None: _t0 = _time.perf_counter()
            x0 = torch.tensor([[0.0]], requires_grad=True)
            y0 = torch.tensor([[0.0]], requires_grad=True)
            laplacians, gradients = compute_lb_and_gradient_batch(
                surface_func, h_funcs, x0, y0, verbose=self.verbose)
            if _t is not None: _t['lb_origin'] = _time.perf_counter() - _t0

        # ---- Optional per-grid-point quantities (always autograd) -----
        lb_all_points = None
        gradients_all_points = None
        if self.compute_lb_all_points or self.compute_gradients_all_points:
            if _t is not None: _t0 = _time.perf_counter()
            h_funcs_all = self._specs_to_callables(specs, surface_func)
            x_all = x_grid.clone().unsqueeze(-1).requires_grad_(True)
            y_all = y_grid.clone().unsqueeze(-1).requires_grad_(True)
            lb_all, grad_all = compute_lb_and_gradient_batch(
                surface_func, h_funcs_all, x_all, y_all, verbose=self.verbose)
            if self.compute_lb_all_points:
                lb_all_points = lb_all
            if self.compute_gradients_all_points:
                gradients_all_points = grad_all
            if _t is not None: _t['lb_all_pts'] = _time.perf_counter() - _t0

        # ---- Verbose preview ------------------------------------------
        if self.verbose:
            mode_str = 'analytic' if analytic_result is not None else 'autograd'
            print(f"\n  [{type(self).__name__}] K={K}, P={P}, "
                  f"normalize={self.normalize_target}, mode={mode_str}")
            print(f"    RAW Δ_LB at origin: min={laplacians.min():.4f}  "
                  f"max={laplacians.max():.4f}  mean={laplacians.mean():.4f}  "
                  f"absmin={laplacians.abs().min():.6f}")
            print(f"    RAW deltas: min={deltas.min():.4f}  max={deltas.max():.4f}  "
                  f"mean_abs={deltas.abs().mean():.4f}")
            if lb_all_points is not None:
                print(f"    RAW Δ_LB all pts: min={lb_all_points.min():.4f}  "
                      f"max={lb_all_points.max():.4f}")
            for p in list(range(min(5, P))) + ([P - 1] if P > 5 else []):
                lb_str = f"Δ_LB(origin)={laplacians[p]:+.4f}"
                d_str = f"δh=[{deltas[:, p].min():.4f}, {deltas[:, p].max():.4f}]"
                lb_all_str = ""
                if lb_all_points is not None:
                    lb_all_str = (f"  Δ_LB_all=[{lb_all_points[:, p].min():.4f}, "
                                  f"{lb_all_points[:, p].max():.4f}]")
                prefix = "..." if p == P - 1 and P > 5 else ""
                print(f"    {prefix}h_{p}: {lb_str}  {d_str}{lb_all_str}")

        # ---- Normalisation --------------------------------------------
        if self.normalize_target == 'unit_magnitude':
            mag = laplacians.abs().clamp(min=1e-8)
            laplacians = laplacians / mag
            deltas = deltas / mag.unsqueeze(0)
            values = values / mag.unsqueeze(0)
            gradients = gradients / mag.unsqueeze(-1)
            if lb_all_points is not None:
                lb_all_points = lb_all_points / mag.unsqueeze(0)
            if gradients_all_points is not None:
                gradients_all_points = gradients_all_points / mag.view(1, -1, 1)
        elif self.normalize_target == 'unit_variance':
            var = deltas.var(dim=0).clamp(min=1e-8)
            std = var.sqrt()
            deltas = deltas / std.unsqueeze(0)
            values = values / std.unsqueeze(0)
            laplacians = laplacians / std
            gradients = gradients / std.unsqueeze(-1)
            if lb_all_points is not None:
                lb_all_points = lb_all_points / std.unsqueeze(0)
            if gradients_all_points is not None:
                gradients_all_points = gradients_all_points / std.view(1, -1, 1)

        if self.verbose and self.normalize_target != 'none':
            print(f"    POST-NORM Δ_LB at origin: min={laplacians.min():.4f}  "
                  f"max={laplacians.max():.4f}")
            print(f"    POST-NORM deltas: min={deltas.min():.4f}  "
                  f"max={deltas.max():.4f}")
            if lb_all_points is not None:
                print(f"    POST-NORM Δ_LB all pts: min={lb_all_points.min():.4f}  "
                      f"max={lb_all_points.max():.4f}")

        # ---- Optional continuous Dirichlet form (variational training) ----
        continuous_bilinear = None
        continuous_energy = None
        if self.compute_continuous_bilinear or self.compute_continuous_energy:
            if _t is not None: _t0 = _time.perf_counter()
            if U_bounds is None:
                # Fall back to the extent of the supplied vertex grid.
                u_lo = float(x_grid.min().item()); u_hi = float(x_grid.max().item())
                v_lo = float(y_grid.min().item()); v_hi = float(y_grid.max().item())
                U_bounds_eff = ((u_lo, u_hi), (v_lo, v_hi))
            else:
                U_bounds_eff = U_bounds

            h_funcs_for_quad = self._specs_to_callables(specs, surface_func)
            # The bilinear form contains the energy on its diagonal; compute
            # it once when either flag is set, derive the energy if needed.
            continuous_bilinear = compute_dirichlet_bilinear_form_continuous(
                surface_func, h_funcs_for_quad, U_bounds_eff,
                quadrature_n=self.quadrature_n)
            if self.compute_continuous_energy:
                continuous_energy = 0.5 * torch.diagonal(continuous_bilinear)
            if _t is not None: _t['continuous_form'] = _time.perf_counter() - _t0

        if _t is not None:
            total = sum(_t.values())
            parts = [f"{k}={v * 1e3:.0f}ms" for k, v in _t.items()]
            print(f"    [timing] total={total * 1e3:.0f}ms  {', '.join(parts)}")

        result = {
            'test_func_deltas': deltas,
            'test_func_values': values,
            'test_func_laplacians': laplacians,
            'test_func_gradients': gradients,
        }
        if lb_all_points is not None:
            result['test_func_lb_all_points'] = lb_all_points
        if gradients_all_points is not None:
            result['test_func_gradients_all_points'] = gradients_all_points
        if continuous_bilinear is not None and self.compute_continuous_bilinear:
            result['test_func_continuous_bilinear'] = continuous_bilinear     # (P, P)
        if continuous_energy is not None:
            result['test_func_continuous_energy'] = continuous_energy         # (P,)
        return result


class RandomFamilyTestFunctionSampler(BaseTestFunctionSampler):
    """Samples random test functions from configured families.

    Supports three families:
      - ``poly``: Random polynomials h(x,y) = Σ c_mn x^m y^n.
      - ``trig``: Trigonometric h(x,y) = sin(ωx·x + φx) · cos(ωy·y + φy).
      - ``exp``:  Gaussian h(x,y) = exp(-((x-cx)² + (y-cy)²) / (2σ²)).

    Optionally appends coordinate functions x, y, z as extra test functions.

    Args:
        num_test_funcs: Number of random test functions per patch.
        include_coordinates: If True, append x, y, z (3 extra functions).
        families: Dict of family configs. Keys: 'poly', 'trig', 'exp'.
            Each value is a dict with 'weight' (sampling probability) plus
            family-specific parameters.
        normalize_target: Target normalization mode.
            'none': raw Δ_LB values.
            'unit_magnitude': divide Δ_LB by |Δ_LB| per function.
            'unit_variance': divide deltas and Δ_LB so deltas have unit var.
    """

    def __init__(
            self,
            num_test_funcs: int = 10,
            include_coordinates: bool = False,
            families: Optional[Dict[str, Any]] = None,
            normalize_target: str = 'none',
            compute_lb_all_points: bool = False,
            compute_gradients_all_points: bool = False,
            compute_continuous_energy: bool = False,
            compute_continuous_bilinear: bool = False,
            quadrature_n: int = 30,
            verbose: bool = False,
            derivative_mode: str = 'analytic',
    ):
        super().__init__(
            normalize_target=normalize_target,
            derivative_mode=derivative_mode,
            compute_lb_all_points=compute_lb_all_points,
            compute_gradients_all_points=compute_gradients_all_points,
            compute_continuous_energy=compute_continuous_energy,
            compute_continuous_bilinear=compute_continuous_bilinear,
            quadrature_n=quadrature_n,
            verbose=verbose,
        )
        self.num_test_funcs = num_test_funcs
        self.include_coordinates = include_coordinates

        if families is None:
            families = {
                'poly': {'weight': 0.4, 'order_range': [1, 4],
                         'coeff_scale': 1.0, 'coeff_method': 'uniform'},
                'trig': {'weight': 0.4, 'frequency_range': [0.5, 4.0],
                         'phase_range': [0.0, 6.283]},
                'exp':  {'weight': 0.2, 'sigma_range': [0.3, 2.0],
                         'center_range': [-1.0, 1.0]},
            }

        self._families = {}
        self._family_weights = []
        self._family_names = []
        for name, cfg in families.items():
            w = cfg.get('weight', 1.0)
            self._families[name] = cfg
            self._family_names.append(name)
            self._family_weights.append(w)

        total_w = sum(self._family_weights)
        self._family_probs = [w / total_w for w in self._family_weights]

    # ----- family samplers ------------------------------------------------

    def _sample_poly_func(self, rng: np.random.Generator):
        """Return a polynomial test function callable."""
        cfg = self._families['poly']
        order_lo, order_hi = cfg.get('order_range', [1, 4])
        scale = cfg.get('coeff_scale', 1.0)
        method = cfg.get('coeff_method', 'uniform')

        order = int(rng.integers(order_lo, order_hi + 1))
        pairs = get_polynomial_pairs(order)
        if method == 'normal':
            coeffs = torch.tensor(rng.normal(size=len(pairs)) * scale,
                                  dtype=torch.float32)
        else:
            coeffs = torch.tensor(
                (2 * rng.uniform(size=len(pairs)) - 1) * scale,
                dtype=torch.float32)

        def h(x, y):
            return evaluate_polynomial(x, y, coeffs, pairs)
        return h

    def _sample_trig_func(self, rng: np.random.Generator):
        """Return a trigonometric test function callable."""
        cfg = self._families['trig']
        freq_lo, freq_hi = cfg.get('frequency_range', [0.5, 4.0])
        phase_lo, phase_hi = cfg.get('phase_range', [0.0, 6.283])

        wx = float(rng.uniform(freq_lo, freq_hi))
        wy = float(rng.uniform(freq_lo, freq_hi))
        px = float(rng.uniform(phase_lo, phase_hi))
        py = float(rng.uniform(phase_lo, phase_hi))

        def h(x, y):
            return torch.sin(wx * x + px) * torch.cos(wy * y + py)
        return h

    def _sample_exp_func(self, rng: np.random.Generator):
        """Return a Gaussian test function callable."""
        cfg = self._families['exp']
        sigma_lo, sigma_hi = cfg.get('sigma_range', [0.3, 2.0])
        ctr_lo, ctr_hi = cfg.get('center_range', [-1.0, 1.0])

        sigma = float(rng.uniform(sigma_lo, sigma_hi))
        cx = float(rng.uniform(ctr_lo, ctr_hi))
        cy = float(rng.uniform(ctr_lo, ctr_hi))

        def h(x, y):
            return torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        return h

    def _sample_func(self, rng: np.random.Generator):
        """Sample one random test function from the configured families."""
        family = rng.choice(self._family_names, p=self._family_probs)
        if family == 'poly':
            return self._sample_poly_func(rng)
        elif family == 'trig':
            return self._sample_trig_func(rng)
        elif family == 'exp':
            return self._sample_exp_func(rng)
        else:
            raise ValueError(f"Unknown test function family: {family}")

    # ----- param-based sampling (for analytic mode) -----------------------

    def _sample_func_params(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Sample one test function, returning a parameter dict."""
        family = rng.choice(self._family_names, p=self._family_probs)
        if family == 'poly':
            cfg = self._families['poly']
            order_lo, order_hi = cfg.get('order_range', [1, 4])
            scale = cfg.get('coeff_scale', 1.0)
            method = cfg.get('coeff_method', 'uniform')
            order = int(rng.integers(order_lo, order_hi + 1))
            pairs = get_polynomial_pairs(order)
            if method == 'normal':
                coeffs = torch.tensor(rng.normal(size=len(pairs)) * scale, dtype=torch.float32)
            else:
                coeffs = torch.tensor((2 * rng.uniform(size=len(pairs)) - 1) * scale, dtype=torch.float32)
            return {'family': 'poly', 'coeffs': coeffs, 'pairs': pairs, 'order': order}
        elif family == 'trig':
            cfg = self._families['trig']
            flo, fhi = cfg.get('frequency_range', [0.5, 4.0])
            plo, phi = cfg.get('phase_range', [0.0, 6.283])
            return {'family': 'trig',
                    'wx': float(rng.uniform(flo, fhi)), 'wy': float(rng.uniform(flo, fhi)),
                    'px': float(rng.uniform(plo, phi)), 'py': float(rng.uniform(plo, phi))}
        elif family == 'exp':
            cfg = self._families['exp']
            slo, shi = cfg.get('sigma_range', [0.3, 2.0])
            clo, chi = cfg.get('center_range', [-1.0, 1.0])
            return {'family': 'exp',
                    'cx': float(rng.uniform(clo, chi)), 'cy': float(rng.uniform(clo, chi)),
                    'sigma': float(rng.uniform(slo, shi))}
        else:
            raise ValueError(f"Unknown family: {family}")

    @staticmethod
    def _params_to_callable(params: Dict[str, Any], surface_func=None):
        """Convert a parameter dict to a callable h(x, y).

        ``surface_func`` is only required for the 'coord' family with
        ``kind='z'`` (i.e. h = f(x, y)).
        """
        fam = params['family']
        if fam == 'poly':
            coeffs, pairs = params['coeffs'], params['pairs']
            return lambda x, y: evaluate_polynomial(x, y, coeffs, pairs)
        if fam == 'trig':
            wx, wy, px, py = params['wx'], params['wy'], params['px'], params['py']
            return lambda x, y: torch.sin(wx * x + px) * torch.cos(wy * y + py)
        if fam == 'exp':
            cx, cy, sigma = params['cx'], params['cy'], params['sigma']
            return lambda x, y: torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        if fam == 'coord':
            kind = params['kind']
            if kind == 'x':
                return lambda x, y: x.squeeze(-1) if x.dim() > 1 else x
            if kind == 'y':
                return lambda x, y: y.squeeze(-1) if y.dim() > 1 else y
            if kind == 'z':
                if surface_func is None:
                    raise ValueError("surface_func is required for coord/z")
                def _z(x, y, sf=surface_func):
                    z = sf(x, y)
                    return z.squeeze(-1) if z.dim() > 1 else z
                return _z
        raise ValueError(f"Unknown family: {fam}")

    # ----- BaseTestFunctionSampler API -----------------------------------

    def _enumerate(self, rng: np.random.Generator) -> List[Dict[str, Any]]:
        specs = [self._sample_func_params(rng) for _ in range(self.num_test_funcs)]
        if self.include_coordinates:
            specs.append({'family': 'coord', 'kind': 'x'})
            specs.append({'family': 'coord', 'kind': 'y'})
            specs.append({'family': 'coord', 'kind': 'z'})
        return specs

    def _specs_to_callables(self, specs, surface_func):
        return [self._params_to_callable(p, surface_func) for p in specs]

    def _specs_to_analytic_data(self, specs, x_grid, y_grid):
        # Coordinate functions need surface second-derivatives at the
        # origin and are not yet wired into the analytic path; fall back
        # to autograd whenever they are present.
        if any(p['family'] == 'coord' for p in specs):
            return None

        P = len(specs)
        K = len(x_grid)
        deltas = torch.zeros(K, P, dtype=torch.float32)
        values = torch.zeros(K, P, dtype=torch.float32)
        h_d = {k: torch.zeros(P, dtype=torch.float32)
               for k in ['h', 'h_x', 'h_y', 'h_xx', 'h_xy', 'h_yy']}

        # Group by family for batched evaluation
        groups: Dict[str, Tuple[List[int], List[Dict[str, Any]]]] = {}
        for i, s in enumerate(specs):
            groups.setdefault(s['family'], ([], []))
            groups[s['family']][0].append(i)
            groups[s['family']][1].append(s)

        eval_fns = {'poly': self._eval_poly_batch,
                    'trig': self._eval_trig_batch,
                    'exp': self._eval_exp_batch}
        deriv_fns = {'poly': self._derivs_poly_at_origin,
                     'trig': self._derivs_trig_at_origin,
                     'exp': self._derivs_exp_at_origin}

        for fam, (indices, params) in groups.items():
            idx = torch.tensor(indices, dtype=torch.long)
            vals = eval_fns[fam](params, x_grid, y_grid)  # (N_fam, K)
            fd = deriv_fns[fam](params)
            values[:, idx] = vals.T
            deltas[:, idx] = (vals - fd['h'][:, None]).T
            for k in h_d:
                h_d[k][idx] = fd[k]

        return deltas, values, h_d

    # ----- vectorized batch evaluation ------------------------------------

    @staticmethod
    def _eval_poly_batch(params_list, x, y):
        """Evaluate all polynomials at grid points. Returns (N, K)."""
        if not params_list:
            return torch.zeros(0, len(x))
        max_order = max(p['order'] for p in params_list)
        all_pairs = get_polynomial_pairs(max_order)
        pair_to_idx = {pair: i for i, pair in enumerate(all_pairs)}
        N, T, K = len(params_list), len(all_pairs), len(x)
        coeff_mat = torch.zeros(N, T)
        for i, p in enumerate(params_list):
            for c, pair in zip(p['coeffs'], p['pairs']):
                coeff_mat[i, pair_to_idx[pair]] = c
        monomials = torch.stack([x ** m * y ** n for m, n in all_pairs])  # (T, K)
        return coeff_mat @ monomials  # (N, K)

    @staticmethod
    def _eval_trig_batch(params_list, x, y):
        """Evaluate all trig functions at grid points. Returns (N, K)."""
        if not params_list:
            return torch.zeros(0, len(x))
        wx = torch.tensor([p['wx'] for p in params_list])
        wy = torch.tensor([p['wy'] for p in params_list])
        px = torch.tensor([p['px'] for p in params_list])
        py = torch.tensor([p['py'] for p in params_list])
        return (torch.sin(wx[:, None] * x[None, :] + px[:, None]) *
                torch.cos(wy[:, None] * y[None, :] + py[:, None]))

    @staticmethod
    def _eval_exp_batch(params_list, x, y):
        """Evaluate all Gaussian functions at grid points. Returns (N, K)."""
        if not params_list:
            return torch.zeros(0, len(x))
        cx = torch.tensor([p['cx'] for p in params_list])
        cy = torch.tensor([p['cy'] for p in params_list])
        sigma = torch.tensor([p['sigma'] for p in params_list])
        dx = x[None, :] - cx[:, None]
        dy = y[None, :] - cy[:, None]
        return torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma[:, None] ** 2))

    # ----- vectorized analytic derivatives at origin ----------------------

    @staticmethod
    def _derivs_poly_at_origin(params_list):
        """Analytic derivatives of all polynomials at (0, 0). Returns dict of (N,)."""
        if not params_list:
            return {k: torch.zeros(0) for k in ['h', 'h_x', 'h_y', 'h_xx', 'h_xy', 'h_yy']}
        max_order = max(p['order'] for p in params_list)
        all_pairs = get_polynomial_pairs(max_order)
        pair_to_idx = {pair: i for i, pair in enumerate(all_pairs)}
        N, T = len(params_list), len(all_pairs)
        cm = torch.zeros(N, T)
        for i, p in enumerate(params_list):
            for c, pair in zip(p['coeffs'], p['pairs']):
                cm[i, pair_to_idx[pair]] = c

        def _col(m, n):
            idx = pair_to_idx.get((m, n))
            return cm[:, idx] if idx is not None else torch.zeros(N)

        return {
            'h': torch.zeros(N),  # no constant term
            'h_x': _col(1, 0), 'h_y': _col(0, 1),
            'h_xx': 2.0 * _col(2, 0), 'h_xy': _col(1, 1), 'h_yy': 2.0 * _col(0, 2),
        }

    @staticmethod
    def _derivs_trig_at_origin(params_list):
        """Analytic derivatives of all trig functions at (0, 0). Returns dict of (N,)."""
        if not params_list:
            return {k: torch.zeros(0) for k in ['h', 'h_x', 'h_y', 'h_xx', 'h_xy', 'h_yy']}
        wx = torch.tensor([p['wx'] for p in params_list])
        wy = torch.tensor([p['wy'] for p in params_list])
        px = torch.tensor([p['px'] for p in params_list])
        py = torch.tensor([p['py'] for p in params_list])
        sp, cp = torch.sin(px), torch.cos(px)
        sq, cq = torch.sin(py), torch.cos(py)
        return {
            'h': sp * cq,
            'h_x': wx * cp * cq, 'h_y': -wy * sp * sq,
            'h_xx': -(wx ** 2) * sp * cq, 'h_xy': -wx * wy * cp * sq,
            'h_yy': -(wy ** 2) * sp * cq,
        }

    @staticmethod
    def _derivs_exp_at_origin(params_list):
        """Analytic derivatives of all Gaussian functions at (0, 0). Returns dict of (N,)."""
        if not params_list:
            return {k: torch.zeros(0) for k in ['h', 'h_x', 'h_y', 'h_xx', 'h_xy', 'h_yy']}
        cx = torch.tensor([p['cx'] for p in params_list])
        cy = torch.tensor([p['cy'] for p in params_list])
        sigma = torch.tensor([p['sigma'] for p in params_list])
        s2, s4 = sigma ** 2, sigma ** 4
        h0 = torch.exp(-(cx ** 2 + cy ** 2) / (2 * s2))
        return {
            'h': h0,
            'h_x': h0 * cx / s2, 'h_y': h0 * cy / s2,
            'h_xx': h0 * (cx ** 2 / s4 - 1 / s2),
            'h_xy': h0 * cx * cy / s4,
            'h_yy': h0 * (cy ** 2 / s4 - 1 / s2),
        }


class MonomialBasisTestFunctionSampler(BaseTestFunctionSampler):
    """Deterministic test-function basis: all monomials up to degree d.

    Two variants controlled by ``variables``:

      ``chart`` (default)
          Bivariate monomials  h(x, y) = x^m · y^n
          with 1 ≤ m + n ≤ max_degree, evaluated on the parameter
          domain.  P = (d+1)(d+2)/2 - 1 probes (the constant is
          omitted because the operator kills constants by construction).

          Note: at the origin, every chart monomial of degree m + n ≥ 3
          has zero first AND second partials, so its analytic Δ_g target
          at the origin is identically zero.  The grid deltas are still
          informative, so the loss still constrains the predicted
          stiffness action to be small for those probes — useful as
          high-frequency null-space training signal.  For genuine
          higher-order surface-aware information use ``ambient``.

      ``ambient``
          Trivariate monomials  h(x, y, z) = x^a · y^b · z^c
          with 1 ≤ a + b + c ≤ max_degree, evaluated on Σ as
          ``x^a · y^b · f(x, y)^c``.  P = (d+1)(d+2)(d+3)/6 - 1.
          Captures genuine z-dependence (extrinsic geometry).  Forces
          the autograd path because the closed-form Monge formula
          would need surface second-derivatives composed through
          f(x, y)^c.

    Args:
        max_degree: Maximum total degree d.
        variables: ``'chart'`` or ``'ambient'``.
        normalize_target / derivative_mode / compute_lb_all_points /
        compute_gradients_all_points / verbose: see
        :class:`BaseTestFunctionSampler`.  ``derivative_mode='analytic'``
        is honoured for ``variables='chart'``; the ambient variant
        always uses autograd.
    """

    def __init__(
        self,
        max_degree: int,
        variables: str = 'chart',
        normalize_target: str = 'none',
        derivative_mode: str = 'analytic',
        compute_lb_all_points: bool = False,
        compute_gradients_all_points: bool = False,
        compute_continuous_energy: bool = False,
        compute_continuous_bilinear: bool = False,
        quadrature_n: int = 30,
        verbose: bool = False,
    ):
        super().__init__(
            normalize_target=normalize_target,
            derivative_mode=derivative_mode,
            compute_lb_all_points=compute_lb_all_points,
            compute_gradients_all_points=compute_gradients_all_points,
            compute_continuous_energy=compute_continuous_energy,
            compute_continuous_bilinear=compute_continuous_bilinear,
            quadrature_n=quadrature_n,
            verbose=verbose,
        )
        if variables not in ('chart', 'ambient'):
            raise ValueError(
                f"variables must be 'chart' or 'ambient', got '{variables}'")
        if int(max_degree) < 1:
            raise ValueError(f"max_degree must be ≥ 1, got {max_degree}")
        self.max_degree = int(max_degree)
        self.variables = variables

    # ----- BaseTestFunctionSampler API -----------------------------------

    def _enumerate(self, rng: np.random.Generator) -> List[Dict[str, Any]]:
        d = self.max_degree
        if self.variables == 'chart':
            return [{'family': 'mono_chart', 'm': m, 'n': n}
                    for m in range(d + 1)
                    for n in range(d + 1)
                    if 1 <= m + n <= d]
        return [{'family': 'mono_ambient', 'a': a, 'b': b, 'c': c}
                for a in range(d + 1)
                for b in range(d + 1)
                for c in range(d + 1)
                if 1 <= a + b + c <= d]

    def _specs_to_callables(self, specs, surface_func):
        funcs = []
        for s in specs:
            if s['family'] == 'mono_chart':
                m, n = s['m'], s['n']
                funcs.append(lambda x, y, m=m, n=n: (x ** m) * (y ** n))
            elif s['family'] == 'mono_ambient':
                a, b, c = s['a'], s['b'], s['c']
                def _f(x, y, a=a, b=b, c=c, sf=surface_func):
                    z = sf(x, y)
                    if z.dim() > x.dim():
                        z = z.squeeze(-1)
                    return (x ** a) * (y ** b) * (z ** c)
                funcs.append(_f)
            else:
                raise ValueError(f"Unknown spec family: {s['family']}")
        return funcs

    def _specs_to_analytic_data(self, specs, x_grid, y_grid):
        # Ambient monomials require composition with the surface, so
        # the closed-form Monge formula at origin is not used here.
        if any(s['family'] == 'mono_ambient' for s in specs):
            return None

        P = len(specs)
        K = len(x_grid)
        deltas = torch.zeros(K, P, dtype=torch.float32)
        values = torch.zeros(K, P, dtype=torch.float32)
        h_d = {k: torch.zeros(P, dtype=torch.float32)
               for k in ['h', 'h_x', 'h_y', 'h_xx', 'h_xy', 'h_yy']}

        for p, s in enumerate(specs):
            m, n = s['m'], s['n']
            vals = (x_grid ** m) * (y_grid ** n)  # h on the grid
            values[:, p] = vals
            # h(0, 0) = 0 because m + n ≥ 1 → δh = vals
            deltas[:, p] = vals
            # Analytic derivatives at origin of h = x^m y^n.  Only
            # five (m, n) pairs give a non-zero entry; everything else
            # vanishes at the origin because each remaining derivative
            # carries a positive power of x or y.
            if (m, n) == (1, 0):
                h_d['h_x'][p] = 1.0
            elif (m, n) == (0, 1):
                h_d['h_y'][p] = 1.0
            elif (m, n) == (2, 0):
                h_d['h_xx'][p] = 2.0
            elif (m, n) == (1, 1):
                h_d['h_xy'][p] = 1.0
            elif (m, n) == (0, 2):
                h_d['h_yy'][p] = 2.0
            # else: all entries remain zero (default-initialised)

        return deltas, values, h_d


# Back-compat alias: the original class name still resolves to the
# random-family sampler so out-of-tree configs and scripts keep working.
TestFunctionSampler = RandomFamilyTestFunctionSampler


# =============================================
# Grid Sampler Classes
# =============================================

class GridSampler(ABC):
    """Abstract base class for grid sampling strategies."""

    @abstractmethod
    def sample(self, grid_range: Tuple[float, float], rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x, y coordinates within the given grid range.

        Args:
            grid_range: Tuple of (min_val, max_val) defining the sampling range
            rng: Random number generator for reproducible sampling

        Returns:
            Tuple of (x, y) tensors containing sampled coordinates
        """
        pass


class RegularGridSampler(GridSampler):
    """Samples points on a regular rectangular grid."""

    def __init__(self, num_points: int):
        """
        Initialize the regular grid sampler.

        Args:
            num_points: Total number of points to sample (will be adjusted to nearest perfect square)
        """
        self._num_points = num_points
        # Adjust to nearest perfect square for regular grid
        self._grid_size = int(np.sqrt(num_points))
        self._actual_points = self._grid_size ** 2

        if self._actual_points != num_points:
            print(f"Warning: Adjusted grid points from {num_points} to {self._actual_points} (nearest perfect square)")

    def sample(self, grid_range: Tuple[float, float], rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points on a regular grid.

        Args:
            grid_range: Tuple of (min_val, max_val) defining the sampling range
            rng: Random number generator (not used for regular grid, but kept for interface consistency)

        Returns:
            Tuple of (x, y) tensors with regularly spaced coordinates
        """
        x_linspace = torch.linspace(start=grid_range[0], end=grid_range[1], steps=self._grid_size)
        y_linspace = torch.linspace(start=grid_range[0], end=grid_range[1], steps=self._grid_size)
        x, y = torch.meshgrid(x_linspace, y_linspace, indexing='ij')
        return x.flatten(), y.flatten()

    @property
    def num_points(self) -> int:
        """Get the actual number of points that will be sampled."""
        return self._actual_points


class RandomGridSampler(GridSampler):
    """Samples points uniformly at random within the grid range."""

    def __init__(self, num_points_range: Union[int, Tuple[int, int]]):
        """
        Initialize the random grid sampler.

        Args:
            num_points_range: Either a single integer for fixed number of points,
                            or a tuple (min_points, max_points) for variable sampling
        """
        # Handle OmegaConf objects (common with Hydra)
        try:
            from omegaconf import ListConfig, DictConfig
            if isinstance(num_points_range, (ListConfig, DictConfig)):
                # Convert OmegaConf to regular Python types
                num_points_range = list(num_points_range) if isinstance(num_points_range, ListConfig) else num_points_range
        except ImportError:
            # OmegaConf not available, continue with regular handling
            pass

        # Convert different input types and provide better error messages
        if isinstance(num_points_range, int):
            self._num_points_range = (num_points_range, num_points_range)
        elif isinstance(num_points_range, (tuple, list)):
            if len(num_points_range) == 2:
                # Convert to integers if they're not already
                try:
                    self._num_points_range = (int(num_points_range[0]), int(num_points_range[1]))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"num_points_range elements must be convertible to integers, got {num_points_range}: {e}")

                if self._num_points_range[0] > self._num_points_range[1]:
                    raise ValueError(f"Min points {self._num_points_range[0]} > max points {self._num_points_range[1]}")
            else:
                raise ValueError(f"num_points_range tuple/list must have exactly 2 elements, got {len(num_points_range)} elements: {num_points_range}")
        else:
            raise ValueError(f"num_points_range must be an integer or a tuple/list of two integers, got {type(num_points_range)}: {num_points_range}")

        # Validate minimum points
        if self._num_points_range[0] < 3:
            raise ValueError(f"Minimum number of points must be >= 3, got {self._num_points_range[0]}")

        # Validate that points are positive integers
        if not all(isinstance(x, int) and x > 0 for x in self._num_points_range):
            raise ValueError(f"All point counts must be positive integers, got {self._num_points_range}")

    def sample(self, grid_range: Tuple[float, float], rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points uniformly at random within the grid range.

        Args:
            grid_range: Tuple of (min_val, max_val) defining the sampling range
            rng: Random number generator for reproducible sampling

        Returns:
            Tuple of (x, y) tensors with randomly sampled coordinates
        """
        # Sample number of points if range is provided
        if self._num_points_range[0] == self._num_points_range[1]:
            num_points = self._num_points_range[0]
        else:
            num_points = int(rng.integers(
                low=self._num_points_range[0],
                high=self._num_points_range[1] + 1
            ))

        # Generate random points
        points = rng.uniform(
            low=grid_range[0],
            high=grid_range[1],
            size=(num_points, 2)
        )

        return torch.tensor(points[:, 0]), torch.tensor(points[:, 1])

    @property
    def num_points_range(self) -> Tuple[int, int]:
        """Get the range of number of points that can be sampled."""
        return self._num_points_range


# =============================================
# Differential Geometry and Dataset Classes
# =============================================

class DifferentialGeometryComponent(Enum):
    MEAN_CURVATURE = 'mean_curvature'
    GAUSSIAN_CURVATURE = 'gaussian_curvature'
    PRINCIPAL_CURVATURES = 'principal_curvatures'
    PRINCIPAL_DIRECTIONS_2D = 'principal_directions_2d'
    PRINCIPAL_DIRECTIONS_3D = 'principal_directions_3d'
    PRINCIPAL_DIRECTIONS = 'principal_directions'  # Legacy alias for both 2D and 3D
    CURVATURE_GRADIENTS_2D = 'curvature_gradients_2d'
    CURVATURE_GRADIENTS_3D = 'curvature_gradients_3d'
    SIGNATURE = 'signature'


# Mapping from enum components to output dictionary keys
DIFF_GEOM_COMPONENT_KEYS = {
    DifferentialGeometryComponent.MEAN_CURVATURE: ['H'],
    DifferentialGeometryComponent.GAUSSIAN_CURVATURE: ['K'],
    DifferentialGeometryComponent.PRINCIPAL_CURVATURES: ['k1', 'k2'],
    DifferentialGeometryComponent.PRINCIPAL_DIRECTIONS_2D: ['v1_2d', 'v2_2d'],
    DifferentialGeometryComponent.PRINCIPAL_DIRECTIONS_3D: ['v1_3d', 'v2_3d'],
    DifferentialGeometryComponent.PRINCIPAL_DIRECTIONS: ['v1_2d', 'v2_2d', 'v1_3d', 'v2_3d'],  # Both
    DifferentialGeometryComponent.CURVATURE_GRADIENTS_2D: ['grad_H_2d', 'grad_K_2d'],
    DifferentialGeometryComponent.CURVATURE_GRADIENTS_3D: ['grad_H_3d', 'grad_K_3d'],
    DifferentialGeometryComponent.SIGNATURE: ['signature'],
}


class SyntheticSurfaceDataset(ABC, Dataset):
    """Base class for synthetic surface datasets."""

    def __init__(
            self,
            epoch_size: int,
            pose_transformers: Optional[List[PoseTransformer]] = None,
            seed: int = 0,
            conv_k_nearest: Optional[int] = None,
            normalize_to_unit_sphere: bool = False,
            position_noise_std: Union[float, List[float], Tuple[float, ...]] = 0.0,
    ):
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._epoch_size = epoch_size
        self._pose_transformers = pose_transformers if pose_transformers is not None else []
        self._conv_k_nearest = conv_k_nearest
        self._normalize_to_unit_sphere = normalize_to_unit_sphere
        # Normalize position_noise_std to a validated (lo, hi) tuple. Accepts:
        #   - float (e.g. 0.01)                   → (0.01, 0.01)  — fixed std (back-compat)
        #   - 1-element sequence [0.01]           → (0.01, 0.01)  — fixed std
        #   - 2-element sequence [0.0, 0.05]      → (0.0, 0.05)   — uniform range
        # Convention matches grid_radius_range. At each get(idx) call, one std
        # is drawn from this range (via self._rng) and shared across all
        # surfaces produced by that call — just like grid_radius is shared.
        self._position_noise_std_range = self._normalize_noise_std(position_noise_std)

    @staticmethod
    def _normalize_noise_std(value: Union[float, List[float], Tuple[float, ...]]) -> Tuple[float, float]:
        """Coerce a position_noise_std input to a validated (lo, hi) tuple.

        Accepts a scalar, or a 1/2-element sequence (including OmegaConf
        ListConfig, which supports len() and integer indexing). Returns a
        tuple of plain Python floats.

        Raises ValueError for malformed input (negative values, lo > hi,
        wrong sequence length, or non-numeric elements).
        """
        # Scalar case — the historical API.
        if isinstance(value, (int, float)):
            v = float(value)
            if v < 0:
                raise ValueError(f"position_noise_std must be non-negative, got {v}")
            return (v, v)
        # Sequence case (list / tuple / OmegaConf ListConfig).
        try:
            items = [float(x) for x in value]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"position_noise_std must be a non-negative float or a 1/2-element "
                f"sequence of non-negative floats, got {value!r}: {e}"
            )
        if len(items) == 1:
            lo = hi = items[0]
        elif len(items) == 2:
            lo, hi = items
        else:
            raise ValueError(
                f"position_noise_std sequence must have 1 or 2 elements, got "
                f"{len(items)}: {items}"
            )
        if lo < 0 or hi < 0:
            raise ValueError(
                f"position_noise_std values must be non-negative, got ({lo}, {hi})"
            )
        if lo > hi:
            raise ValueError(f"position_noise_std min {lo} > max {hi}")
        return (lo, hi)

    def reset_rng(self) -> None:
        """Reset random number generator to initial seed."""
        self._rng = np.random.default_rng(seed=self._seed)

    def len(self) -> int:
        return self._epoch_size

    def _generate_surfaces(self) -> List[Data]:
        """
        Generate multiple surfaces and add features to them.

        This is the main orchestration method that:
        1. Calls the derived class implementation to generate raw surface data
        2. Optionally normalizes each surface to the unit sphere
        3. Optionally adds positional noise to vertex positions
        4. Adds features to each surface using the feature extractor

        Returns:
            List of Data objects with positions, normals, differential geometry, and features
        """
        # Generate raw surfaces using derived class implementation
        surfaces = self._generate_raw_surfaces()

        # Optionally normalize to unit sphere (before adding features)
        if self._normalize_to_unit_sphere:
            for surface in surfaces:
                self._normalize_surface_to_unit_sphere(surface)

        # Optionally add positional noise (after normalization, before features)
        # GT targets (normals, curvatures, test_func_laplacians) remain clean.
        # Only data.pos is perturbed — the model must be robust to noisy input.
        # When position_noise_std is a range [lo, hi], we draw one std value
        # from self._rng per get(idx) call and apply it to all surfaces
        # generated by that call (matching how grid_radius is shared).
        # The per-point Gaussian noise is also drawn from self._rng so the
        # entire data generation is deterministic given idx (self._rng is
        # reseeded as default_rng(self._seed + idx) at the top of get()).
        lo, hi = self._position_noise_std_range
        if hi > 0:
            noise_std = float(self._rng.uniform(lo, hi)) if hi > lo else hi
            if noise_std > 0:
                for surface in surfaces:
                    noise_np = self._rng.standard_normal(
                        size=tuple(surface.pos.shape)) * noise_std
                    noise = torch.from_numpy(noise_np).to(
                        dtype=surface.pos.dtype, device=surface.pos.device)
                    surface.pos = surface.pos + noise

        # Add features to each surface
        for surface in surfaces:
            self._add_surface_features(surface)

        return surfaces

    @staticmethod
    def _normalize_surface_to_unit_sphere(data: Data) -> None:
        """
        Normalize a surface patch so positions lie within the unit sphere.

        Divides positions by d = max(||p||) and rescales all differential
        geometry quantities to remain consistent under uniform scaling by 1/d:

            pos             -> pos / d
            H               -> H * d            (mean curvature ~ 1/length)
            K               -> K * d^2           (Gaussian curvature ~ 1/length^2)
            k1, k2          -> k1 * d, k2 * d
            gt_vertex_areas -> areas / d^2
            normals         -> unchanged         (unit vectors)
            v1, v2          -> unchanged         (direction only)
            grad_H          -> grad_H * d^2      (curvature gradient ~ 1/length^2)
            grad_K          -> grad_K * d^3      (curvature gradient ~ 1/length^3)

        Stores d as data.patch_scale_factor for area rescaling during inference.
        Operates in-place.
        """
        d = torch.norm(data.pos, dim=1).max().clamp(min=1e-8)

        # Rescale positions
        data.pos = data.pos / d

        # Store scale factor for area rescaling during inference
        data['patch_scale_factor'] = d.unsqueeze(0)

        # Rescale origin position (if present)
        if 'origin_pos' in data:
            data['origin_pos'] = data['origin_pos'] / d

        # --- Differential geometry rescaling ---
        # Mean curvature: H ~ 1/length
        if 'H' in data:
            data['H'] = data['H'] * d

        # Gaussian curvature: K ~ 1/length^2
        if 'K' in data:
            data['K'] = data['K'] * (d ** 2)

        # Principal curvatures: k ~ 1/length
        if 'k1' in data:
            data['k1'] = data['k1'] * d
        if 'k2' in data:
            data['k2'] = data['k2'] * d

        # GT vertex areas: area ~ length^2
        if 'gt_vertex_areas' in data:
            data['gt_vertex_areas'] = data['gt_vertex_areas'] / (d ** 2)

        # Curvature gradients in 3D: grad(H) ~ 1/length^2, grad(K) ~ 1/length^3
        if 'grad_H_3d' in data:
            data['grad_H_3d'] = data['grad_H_3d'] * (d ** 2)
        if 'grad_K_3d' in data:
            data['grad_K_3d'] = data['grad_K_3d'] * (d ** 3)

        # Curvature gradients in 2D parameter space (same scaling)
        if 'grad_H_2d' in data:
            data['grad_H_2d'] = data['grad_H_2d'] * (d ** 2)
        if 'grad_K_2d' in data:
            data['grad_K_2d'] = data['grad_K_2d'] * (d ** 3)

        # Normals, principal directions (v1, v2): unchanged (unit/direction vectors)

        # Test function targets: Δ_LB ~ 1/length², surface gradient ~ 1/length
        # Deltas (function value differences) are unchanged.
        if 'test_func_laplacians' in data:
            data['test_func_laplacians'] = data['test_func_laplacians'] * (d ** 2)
        if 'test_func_lb_all_points' in data:
            data['test_func_lb_all_points'] = data['test_func_lb_all_points'] * (d ** 2)
        if 'test_func_gradients' in data:
            data['test_func_gradients'] = data['test_func_gradients'] * d
        if 'test_func_gradients_all_points' in data:
            data['test_func_gradients_all_points'] = data['test_func_gradients_all_points'] * d

    @abstractmethod
    def _generate_raw_surfaces(self) -> List[Data]:
        """
        Generate multiple raw surfaces without features.

        This method should be implemented by derived classes to generate
        surface data with positions, normals, and differential geometry,
        but WITHOUT features (the 'x' attribute).

        Returns:
            List of Data objects with surface geometry but no features
        """
        pass

    def _repose_surface_and_quantities(self, data: Data, normals: Optional[torch.Tensor] = None) -> Data:
        """Apply pose transformations sequentially to surface and transform differential quantities accordingly."""
        if not self._pose_transformers:
            return data

        # Use passed normals if provided, otherwise get from data object
        if normals is not None:
            # Use the explicitly passed normal (e.g., origin normal)
            if normals.dim() == 2 and normals.shape[0] == 1:
                # Shape: (1, 3) -> extract to (3,)
                normal = normals[0]
            elif normals.dim() == 1:
                # Shape: (3,) -> use as is
                normal = normals
            else:
                # Handle unexpected shapes - take first normal
                normal = normals.flatten()[:3]
        else:
            # Fallback: get normal from data object (original behavior)
            normal = data['normal'][0] if 'normal' in data else torch.tensor([0., 0., 1.])

        # Apply each transformer sequentially
        for pose_transformer in self._pose_transformers:
            # Get translation and rotation from this transformer
            translation, rotation_matrix = pose_transformer.transform(data.pos, normal)

            # Apply translation and rotation to positions
            data.pos = data.pos + translation
            data.pos = torch.matmul(data.pos, rotation_matrix.T)

            # Transform origin position if present
            if 'origin_pos' in data:
                data['origin_pos'] = data['origin_pos'] + translation
                data['origin_pos'] = torch.matmul(data['origin_pos'], rotation_matrix.T)

            # Transform normals using rotation matrix
            if 'normal' in data:
                data['normal'] = torch.matmul(data['normal'], rotation_matrix.T)

            # Transform differential geometry quantities
            vector_3d_keys = ['v1_3d', 'v2_3d', 'grad_H_3d', 'grad_K_3d',
                              'test_func_gradients', 'test_func_gradients_all_points']
            for key in vector_3d_keys:
                if key in data:
                    data[key] = torch.matmul(data[key], rotation_matrix.T)

            # Transform the origin normal for the next transformer
            normal = torch.matmul(normal.unsqueeze(0), rotation_matrix.T).squeeze(0)

        return data

    def _add_surface_features(self, data: Data) -> None:
        """Store raw positions as features — feature extraction is handled by the model."""
        data['x'] = data.pos

    def get(self, idx: int) -> List[Data]:
        """Generate multiple samplings of the same surface.

        Each idx produces a unique deterministic surface, ensuring different
        DDP ranks (which receive different indices from DistributedSampler)
        generate different training data.
        """
        # [DDP-DIAG] Verify each rank receives different indices. Logs the
        # first few idx values seen per (rank, worker) pair, then goes silent.
        if not hasattr(self, '_ddp_diag_count'):
            self._ddp_diag_count = 0
        if self._ddp_diag_count < 8:
            import os
            rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '?'))
            world_size = os.environ.get('WORLD_SIZE', '?')
            try:
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info is not None else 'main'
            except Exception:
                worker_id = '?'
            print(
                f"[DDP-DIAG] rank={rank}/{world_size} worker={worker_id} "
                f"got idx={idx} (call #{self._ddp_diag_count})",
                flush=True,
            )
            self._ddp_diag_count += 1

        self._rng = np.random.default_rng(self._seed + idx)
        surfaces = self._generate_surfaces()

        # [DET-DIAG] Hash of the surface output, per idx, for the first few
        # calls. If two runs produce different hashes for the same idx, the
        # dataset path itself is non-deterministic. If hashes match across
        # runs but training still diverges, the bug is downstream of data.
        if not hasattr(self, '_det_diag_count'):
            self._det_diag_count = 0
        if self._det_diag_count < 4:
            import os
            import hashlib
            rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '?'))
            try:
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info is not None else 'main'
            except Exception:
                worker_id = '?'
            # Hash the position tensors (the most non-determinism-prone field).
            h = hashlib.md5()
            for s in surfaces:
                if hasattr(s, 'pos') and s.pos is not None:
                    # bit-exact hash of the float bytes
                    arr = s.pos.detach().cpu().numpy()
                    h.update(arr.tobytes())
                    h.update(str(arr.shape).encode())
            digest = h.hexdigest()[:16]
            n_surfaces = len(surfaces)
            n_pts = sum(s.pos.shape[0] if hasattr(s, 'pos') and s.pos is not None
                        else 0 for s in surfaces)
            print(
                f"[DET-DIAG] rank={rank} worker={worker_id} idx={idx} "
                f"n_surfaces={n_surfaces} total_pts={n_pts} "
                f"pos_hash={digest}",
                flush=True,
            )
            self._det_diag_count += 1

        return surfaces


class ParametricSurfaceDataset(SyntheticSurfaceDataset):
    """Dataset for parametric surfaces with differential geometry computation."""

    def __init__(
            self,
            grid_samplers: List[GridSampler],
            grid_radius_range: Tuple[float, float],
            points_scale_range: Tuple[float, float],
            diff_geom_components: Optional[List[DifferentialGeometryComponent]] = None,
            diff_geom_at_origin_only: bool = False,
            flip_normal_if_negative_curvature: bool = False,
            include_origin_in_grid: bool = False,
            surface_sampler: Optional[BaseMongeSurfaceSampler] = None,
            test_func_sampler: Optional[BaseTestFunctionSampler] = None,
            test_func_cfg: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._grid_samplers = grid_samplers
        self._grid_radius_range = self._validate_range(param_range=grid_radius_range, name="grid_radius_range")
        self._points_scale_range = self._validate_range(param_range=points_scale_range, name="points_scale_range")
        self._diff_geom_at_origin_only = diff_geom_at_origin_only
        self._flip_normal_if_negative_curvature = flip_normal_if_negative_curvature
        self._include_origin_in_grid = include_origin_in_grid

        # Optional injected surface sampler (Hydra-instantiated).  When
        # set, ``_generate_surface_parameters`` and
        # ``_evaluate_surface_with_parameters`` route through it; the
        # closed-over surface_func is stashed in the params dict under
        # key '_surface_func' so the existing single-pass flow needs no
        # changes.  Subclasses (e.g. PolynomialSurfaceDataset) may keep
        # their bespoke parameter-sampling and ignore this.
        self._surface_sampler = surface_sampler

        # Test function sampler — prefer the directly-instantiated object
        # (Hydra-style ``_target_`` config); fall back to the legacy
        # ``test_func_cfg`` flat dict, which builds a
        # RandomFamilyTestFunctionSampler.
        if test_func_sampler is not None and test_func_cfg is not None:
            raise ValueError(
                "Specify either 'test_func_sampler' (preferred, instantiated "
                "BaseTestFunctionSampler) or 'test_func_cfg' (legacy flat "
                "dict), not both.")
        if test_func_sampler is not None:
            self._test_func_sampler = test_func_sampler
        elif test_func_cfg is not None:
            try:
                from omegaconf import DictConfig
                if isinstance(test_func_cfg, DictConfig):
                    from omegaconf import OmegaConf
                    test_func_cfg = OmegaConf.to_container(test_func_cfg, resolve=True)
            except ImportError:
                pass
            self._test_func_sampler = RandomFamilyTestFunctionSampler(**test_func_cfg)
        else:
            self._test_func_sampler = None

        # Available differential geometry components
        available_components = list(DifferentialGeometryComponent)
        if diff_geom_components is None:
            self._diff_geom_components = available_components
        else:
            invalid = set(diff_geom_components) - set(available_components)
            if invalid:
                raise ValueError(f"Invalid diff_geom_components: {invalid}. Available: {available_components}")
            self._diff_geom_components = diff_geom_components

    def _validate_range(self, param_range: Tuple[float, float], name: str) -> Tuple[float, float]:
        """Validate parameter range."""
        if len(param_range) not in [1, 2]:
            raise ValueError(f"{name} must have 1 or 2 elements, got {len(param_range)}")
        if len(param_range) == 2 and param_range[0] > param_range[1]:
            raise ValueError(f"{name} min value {param_range[0]} > max value {param_range[1]}")
        return param_range

    def _sample_parameter(self, param_range: Tuple[float, ...]) -> float:
        """Sample parameter from range."""
        if len(param_range) == 2:
            return float(self._rng.uniform(low=param_range[0], high=param_range[1]))
        return param_range[0]

    def _compute_origin_data(self, surface_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute all data at origin (0,0) in a single pass.

        Returns dict with: x, y, z, dz_dx, dz_dy, normal, H (mean curvature)
        """
        x_origin = torch.tensor([0.0], requires_grad=True)
        y_origin = torch.tensor([0.0], requires_grad=True)
        z_origin = self._evaluate_surface_with_parameters(x=x_origin, y=y_origin, surface_params=surface_params)

        dz_dx = torch.autograd.grad(outputs=z_origin, inputs=x_origin, create_graph=True, retain_graph=True)[0]
        dz_dy = torch.autograd.grad(outputs=z_origin, inputs=y_origin, create_graph=True, retain_graph=True)[0]

        # Compute normal (without flipping - flipping uses H which we compute next)
        normal = self._compute_normal_from_derivatives(dz_dx, dz_dy)

        # Compute H for potential normal flipping
        H, _, _, _, _, _ = self._compute_curvature_quantities(dz_dx=dz_dx, dz_dy=dz_dy, x=x_origin, y=y_origin)

        return {
            'x': x_origin,
            'y': y_origin,
            'z': z_origin,
            'dz_dx': dz_dx,
            'dz_dy': dz_dy,
            'normal': normal,
            'H': H
        }

    def _compute_normal_from_derivatives(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> torch.Tensor:
        """Compute normalized surface normal from partial derivatives."""
        normal = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=1)
        return F.normalize(normal, p=2, dim=1)

    def _compute_surface_normals(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor,
                                 H_at_origin: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute surface normals from derivatives.

        Args:
            dz_dx, dz_dy: Precomputed derivatives
            H_at_origin: Mean curvature at origin (for flipping, if enabled)
        """
        normal = self._compute_normal_from_derivatives(dz_dx, dz_dy)

        # Apply normal flipping logic if flag is set
        if self._flip_normal_if_negative_curvature and H_at_origin is not None:
            normal = self._apply_normal_flipping(normal, H_at_origin)

        return normal

    def _apply_normal_flipping(self, normal: torch.Tensor, H_at_origin: torch.Tensor) -> torch.Tensor:
        """Flip normal if mean curvature at origin is negative."""
        if H_at_origin.item() < 0:
            return -normal
        return normal

    def _create_surface_mesh(self, pos: torch.Tensor) -> torch.Tensor:
        """Create triangular mesh from 2D positions."""
        try:
            pos_2d = pos[:, :2].detach().numpy()
            if len(pos_2d) < 3:
                raise ValueError("Need at least 3 points for triangulation")
            return torch.from_numpy(Delaunay(points=pos_2d).simplices).T
        except Exception as e:
            raise RuntimeError(f"Failed to create surface mesh: {e}")

    @staticmethod
    def _compute_barycentric_vertex_areas(
        pos: torch.Tensor, face: torch.Tensor,
    ) -> torch.Tensor:
        """Compute barycentric (1/3) vertex areas from a triangle mesh.

        Each vertex gets 1/3 of the area of each adjacent face.

        Args:
            pos: (N, 3) vertex positions.
            face: (3, F) face indices (PyG convention).

        Returns:
            areas: (N,) per-vertex areas.
        """
        N = pos.shape[0]
        # face is (3, F) — columns are triangles
        face = face.long()  # scatter_add_ requires int64
        v0 = pos[face[0]]  # (F, 3)
        v1 = pos[face[1]]
        v2 = pos[face[2]]
        face_areas = 0.5 * torch.linalg.norm(
            torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)  # (F,)
        third_areas = face_areas / 3.0
        areas = torch.zeros(N, dtype=pos.dtype)
        areas.scatter_add_(0, face[0], third_areas)
        areas.scatter_add_(0, face[1], third_areas)
        areas.scatter_add_(0, face[2], third_areas)
        return areas

    def _compute_first_derivatives(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute first partial derivatives ÃƒÂ¢Ã‹â€ Ã¢â‚¬Å¡z/ÃƒÂ¢Ã‹â€ Ã¢â‚¬Å¡x and ÃƒÂ¢Ã‹â€ Ã¢â‚¬Å¡z/ÃƒÂ¢Ã‹â€ Ã¢â‚¬Å¡y."""
        dz_dx = torch.autograd.grad(
            outputs=z.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        dz_dy = torch.autograd.grad(
            outputs=z.sum(),
            inputs=y,
            create_graph=True,
            retain_graph=True
        )[0]
        return dz_dx, dz_dy

    def _compute_second_derivatives(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute second partial derivatives from first derivatives."""
        d2z_dx2, d2z_dxdy = torch.autograd.grad(outputs=dz_dx.sum(), inputs=[x, y], create_graph=True)
        _, d2z_dy2 = torch.autograd.grad(outputs=dz_dy.sum(), inputs=[x, y], create_graph=True)
        return d2z_dx2, d2z_dxdy, d2z_dy2

    def _compute_shape_operator(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor, d2z_dx2: torch.Tensor, d2z_dxdy: torch.Tensor, d2z_dy2: torch.Tensor) -> torch.Tensor:
        """Compute shape operator matrix."""
        E = 1 + dz_dx ** 2
        F = dz_dx * dz_dy
        G = 1 + dz_dy ** 2
        L = d2z_dx2 / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        M = d2z_dxdy / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        N = d2z_dy2 / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        det = E * G - F ** 2
        shape_operator = torch.stack([
            torch.stack([G * L - F * M, G * M - F * N], dim=-1),
            torch.stack([E * M - F * L, E * N - F * M], dim=-1)
        ], dim=-2) / det.unsqueeze(dim=-1).unsqueeze(dim=-2)
        return shape_operator

    def _compute_principal_curvatures(self, shape_operator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract principal curvatures and directions from shape operator."""
        eigenvalues, eigenvectors = torch.linalg.eig(input=shape_operator)
        k1, k2 = eigenvalues.real[..., 0], eigenvalues.real[..., 1]
        v1, v2 = eigenvectors.real[..., 0], eigenvectors.real[..., 1]
        return k1, k2, v1, v2

    def _compute_curvatures(self, k1: torch.Tensor, k2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and Gaussian curvatures."""
        H = (k1 + k2) / 2
        K = k1 * k2
        return H, K

    def _compute_curvature_gradients(self, H: torch.Tensor, K: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Compute gradients of curvatures in parameter space."""
        grad_H = torch.autograd.grad(outputs=H.sum(), inputs=[x, y], create_graph=True)
        grad_K = torch.autograd.grad(outputs=K.sum(), inputs=[x, y], create_graph=True)
        return grad_H, grad_K

    def _compute_jacobian(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of surface parameterization."""
        return torch.stack([
            torch.stack([torch.ones_like(dz_dx), torch.zeros_like(dz_dx)], dim=-1),
            torch.stack([torch.zeros_like(dz_dy), torch.ones_like(dz_dy)], dim=-1),
            torch.stack([dz_dx, dz_dy], dim=-1)
        ], dim=-2)

    def _map_to_3d(self, jacobian: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, grad_H: torch.Tensor, grad_K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map 2D vector fields to 3D using Jacobian."""
        v1_3d = torch.einsum('ijk,ik->ij', jacobian, v1)
        v2_3d = torch.einsum('ijk,ik->ij', jacobian, v2)
        grad_H_3d = torch.einsum('ijk,ik->ij', jacobian, grad_H)
        grad_K_3d = torch.einsum('ijk,ik->ij', jacobian, grad_K)
        return v1_3d, v2_3d, grad_H_3d, grad_K_3d

    def _compute_3d_euclidean_signatures(self, H: torch.Tensor, K: torch.Tensor, grad_H_3d: torch.Tensor, grad_K_3d: torch.Tensor, v1_3d: torch.Tensor, v2_3d: torch.Tensor) -> torch.Tensor:
        """Compute invariant 3D Euclidean signatures."""
        # Normalize principal directions
        v1_3d_norm = v1_3d / torch.norm(v1_3d, dim=1, keepdim=True)
        v2_3d_norm = v2_3d / torch.norm(v2_3d, dim=1, keepdim=True)

        # Compute directional derivatives
        H_1 = torch.sum(grad_H_3d * v1_3d_norm, dim=1)
        H_2 = torch.sum(grad_H_3d * v2_3d_norm, dim=1)
        K_1 = torch.sum(grad_K_3d * v1_3d_norm, dim=1)
        K_2 = torch.sum(grad_K_3d * v2_3d_norm, dim=1)

        # Stack signature components
        signature = torch.stack([H, K, H_1, H_2, K_1, K_2], dim=1)
        return signature

    def _compute_curvature_quantities(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all curvature-related quantities."""
        # Second derivatives
        d2z_dx2, d2z_dxdy, d2z_dy2 = self._compute_second_derivatives(dz_dx=dz_dx, dz_dy=dz_dy, x=x, y=y)

        # Shape operator and principal curvatures
        shape_operator = self._compute_shape_operator(
            dz_dx=dz_dx, dz_dy=dz_dy,
            d2z_dx2=d2z_dx2, d2z_dxdy=d2z_dxdy, d2z_dy2=d2z_dy2
        )
        k1, k2, v1_2d, v2_2d = self._compute_principal_curvatures(shape_operator=shape_operator)
        H, K = self._compute_curvatures(k1=k1, k2=k2)

        return H, K, k1, k2, v1_2d, v2_2d

    def _compute_gradients_2d(self, H: torch.Tensor, K: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute curvature gradients in 2D parameter space."""
        grad_H_2d, grad_K_2d = self._compute_curvature_gradients(H=H, K=K, x=x, y=y)
        grad_H_2d = torch.stack(grad_H_2d, dim=-1)
        grad_K_2d = torch.stack(grad_K_2d, dim=-1)
        return grad_H_2d, grad_K_2d

    def _compute_3d_quantities(self, jacobian: torch.Tensor, v1_2d: torch.Tensor, v2_2d: torch.Tensor,
                               grad_H_2d: torch.Tensor, grad_K_2d: torch.Tensor,
                               H: torch.Tensor, K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all 3D quantities (principal directions, gradients, signature)."""
        v1_3d, v2_3d, grad_H_3d, grad_K_3d = self._map_to_3d(
            jacobian=jacobian, v1=v1_2d, v2=v2_2d,
            grad_H=grad_H_2d, grad_K=grad_K_2d
        )
        signature = self._compute_3d_euclidean_signatures(
            H=H, K=K, grad_H_3d=grad_H_3d, grad_K_3d=grad_K_3d,
            v1_3d=v1_3d, v2_3d=v2_3d
        )
        return {
            'v1_3d': v1_3d,
            'v2_3d': v2_3d,
            'grad_H_3d': grad_H_3d,
            'grad_K_3d': grad_K_3d,
            'signature': signature
        }

    def _compute_all_differential_geometry(self, x: torch.Tensor, y: torch.Tensor,
                                           dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute ALL differential geometry quantities unconditionally."""
        # Curvature quantities
        H, K, k1, k2, v1_2d, v2_2d = self._compute_curvature_quantities(
            dz_dx=dz_dx, dz_dy=dz_dy, x=x, y=y
        )

        # Gradients in 2D
        grad_H_2d, grad_K_2d = self._compute_gradients_2d(H=H, K=K, x=x, y=y)

        # Jacobian and 3D quantities
        jacobian = self._compute_jacobian(dz_dx=dz_dx, dz_dy=dz_dy)
        quantities_3d = self._compute_3d_quantities(
            jacobian=jacobian, v1_2d=v1_2d, v2_2d=v2_2d,
            grad_H_2d=grad_H_2d, grad_K_2d=grad_K_2d, H=H, K=K
        )

        # Combine all quantities
        return {
            'H': H,
            'K': K,
            'k1': k1,
            'k2': k2,
            'v1_2d': v1_2d,
            'v2_2d': v2_2d,
            'grad_H_2d': grad_H_2d,
            'grad_K_2d': grad_K_2d,
            **quantities_3d
        }

    def _filter_differential_geometry(self, all_diff_geom: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Filter to only keep quantities specified in _diff_geom_components."""
        result = {}
        for component in self._diff_geom_components:
            for key in DIFF_GEOM_COMPONENT_KEYS.get(component, []):
                if key in all_diff_geom:
                    result[key] = all_diff_geom[key]
        return result

    def _compute_differential_geometry(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                       dz_dx: torch.Tensor, dz_dy: torch.Tensor,
                                       origin_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute requested differential geometry quantities using precomputed first derivatives."""
        if not self._diff_geom_components:
            return {}

        if self._diff_geom_at_origin_only:
            if origin_data is None:
                raise ValueError("origin_data required when diff_geom_at_origin_only=True")
            x_eval = origin_data['x']
            y_eval = origin_data['y']
            dz_dx_eval = origin_data['dz_dx']
            dz_dy_eval = origin_data['dz_dy']
        else:
            x_eval, y_eval, dz_dx_eval, dz_dy_eval = x, y, dz_dx, dz_dy

        all_diff_geom = self._compute_all_differential_geometry(x=x_eval, y=y_eval, dz_dx=dz_dx_eval, dz_dy=dz_dy_eval)
        return self._filter_differential_geometry(all_diff_geom)

    def _create_raw_surface_data(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                 dz_dx: torch.Tensor, dz_dy: torch.Tensor, points_scale: float,
                                 surface_params: Optional[Dict[str, Any]] = None) -> Tuple[Data, torch.Tensor]:
        """
        Create raw surface data object with positions, mesh, normals, and differential geometry.
        Does NOT add features - those are added later by the base class.
        Does NOT apply pose transformation - that is deferred to _generate_raw_surfaces
        so that test-function fields (and any other ℝ³ quantities attached after this
        returns) are included in the single, centralized rotation pass.
        Modified to handle origin-only computation and center the surface at origin.

        Returns:
            data: The Data object with positions, normals, diff-geom (un-rotated).
            origin_normal: The canonical origin normal in the un-rotated frame,
                           used as the reference normal for the pose transformer chain.
        """
        # Compute all origin data once (derivatives, normal, H for flipping)
        origin_data = self._compute_origin_data(surface_params)

        # Create positions and translate so that (0,0,z_center) becomes (0,0,0)
        center_point = torch.stack([origin_data['x'], origin_data['y'], origin_data['z']], dim=1) * points_scale  # (1, 3)
        positions = torch.stack([x, y, z], dim=1) * points_scale  # (N, 3)
        positions = positions - center_point  # Translate so center is at origin

        # Create base surface data with translated positions
        data = Data()
        data['pos'] = positions.detach()
        data['face'] = self._create_surface_mesh(pos=data.pos).detach()

        # Compute GT barycentric vertex areas from 3D mesh
        all_vertex_areas = self._compute_barycentric_vertex_areas(
            data['pos'], data['face'])
        if self._diff_geom_at_origin_only:
            # Store only origin vertex's area (same batching as H, normals)
            origin_idx = (positions.detach() ** 2).sum(dim=1).argmin().item()
            data['gt_vertex_areas'] = all_vertex_areas[origin_idx:origin_idx+1].detach()
        else:
            data['gt_vertex_areas'] = all_vertex_areas.detach()

        # Store the origin position (0,0,0) after centering - will be transformed with pose
        data['origin_pos'] = torch.zeros(1, 3)

        # Compute origin normal (with potential flipping based on H)
        origin_normal = self._compute_surface_normals(
            dz_dx=origin_data['dz_dx'],
            dz_dy=origin_data['dz_dy'],
            H_at_origin=origin_data['H']
        )

        # Compute normals: origin-only or all points
        if self._diff_geom_at_origin_only:
            data['normal'] = origin_normal.detach()
        else:
            data['normal'] = self._compute_surface_normals(
                dz_dx=dz_dx,
                dz_dy=dz_dy,
                H_at_origin=origin_data['H']
            ).detach()

        # Compute differential geometry (unified method handles origin-only vs all points)
        diff_geom = self._compute_differential_geometry(
            x=x, y=y, z=z, dz_dx=dz_dx, dz_dy=dz_dy, origin_data=origin_data
        )
        for key, value in diff_geom.items():
            data[key] = value.detach()

        # Apply pose transformation to positions and differential quantities
        # ALWAYS pass the origin normal, regardless of the diff_geom_at_origin_only flag
        # NOTE: Pose transformation is intentionally NOT applied here. It is deferred
        # to _generate_raw_surfaces so that test-function fields (and any other ℝ³
        # quantities attached after this returns) are included in the single,
        # centralized rotation pass. See _repose_surface_and_quantities for the
        # authoritative list of rotated keys (vector_3d_keys).

        # NOTE: Features are NOT added here - they're added by the base class
        # after calling _generate_raw_surfaces()

        return data, origin_normal

    def _ensure_origin_in_grid(self, x: torch.Tensor, y: torch.Tensor,
                               grid_range: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """
        Ensure the origin (0,0) is included in the grid points.

        Args:
            x, y: Grid point coordinates
            grid_range: The (min, max) range of the grid

        Returns:
            x, y: Updated coordinates (with origin added if needed)
            origin_idx: Index of the origin point in the grid, or None if origin is outside grid range
        """
        # Check if origin is within the grid range
        if not (grid_range[0] <= 0.0 <= grid_range[1]):
            return x, y, None

        # Check if origin already exists in the grid (within tolerance)
        tolerance = 1e-6
        distances_sq = x ** 2 + y ** 2
        min_dist_sq = distances_sq.min().item()

        if min_dist_sq < tolerance ** 2:
            # Origin already exists, find its index
            origin_idx = distances_sq.argmin().item()
            return x, y, origin_idx

        # Add origin to the grid
        x = torch.cat([x, torch.tensor([0.0])])
        y = torch.cat([y, torch.tensor([0.0])])
        origin_idx = len(x) - 1

        return x, y, origin_idx

    def _generate_raw_surfaces(self) -> List[Data]:
        """Generate multiple samplings of the same parametric surface using grid samplers."""
        # Generate surface parameters once (shared across all samplings)
        surface_params = self._generate_surface_parameters()

        # Sample other parameters once
        grid_radius = self._sample_parameter(param_range=self._grid_radius_range)
        points_scale = self._sample_parameter(param_range=self._points_scale_range)
        grid_range = (-grid_radius, grid_radius)  # Always centered at (0, 0)

        # Snapshot the rng state BEFORE the per-grid-sampler loop so that every
        # grid sampling of this surface sees the SAME set of test-function probes.
        # Without this, calling self._test_func_sampler.sample(rng=self._rng)
        # inside the loop advances self._rng each iteration, and TestFunctionSampler
        # draws a different random probe set per surface (via _sample_func_params
        # / _sample_func). That makes "probe index k" refer to a different function
        # on surface 0 vs surface 1 — visible in hybrid viz as inconsistent probe
        # coloring and gradient arrows between the mesh and point cloud. Training
        # is unaffected because it uses a single grid sampler.
        tf_rng_state = (self._rng.bit_generator.state
                        if self._test_func_sampler is not None else None)

        surfaces = []
        for grid_sampler in self._grid_samplers:
            # Generate grid for this sampling using the grid sampler
            x, y = grid_sampler.sample(grid_range=grid_range, rng=self._rng)

            # Optionally ensure origin is in the grid
            origin_idx = None
            if self._include_origin_in_grid:
                x, y, origin_idx = self._ensure_origin_in_grid(x, y, grid_range)

            with torch.enable_grad():
                # Convert to float32 and enable gradients
                x = x.to(dtype=torch.float32).requires_grad_(True)
                y = y.to(dtype=torch.float32).requires_grad_(True)

                # Evaluate surface using shared parameters
                z = self._evaluate_surface_with_parameters(x=x, y=y, surface_params=surface_params).to(dtype=torch.float32)

                # Compute first derivatives BEFORE creating surface data
                dz_dx, dz_dy = self._compute_first_derivatives(x=x, y=y, z=z)

                # Create surface data object WITHOUT features and WITHOUT pose
                # transform (deferred — see comment below).
                data, origin_normal = self._create_raw_surface_data(
                    x=x, y=y, z=z, dz_dx=dz_dx, dz_dy=dz_dy,
                    points_scale=points_scale, surface_params=surface_params
                )

                # Compute test function data (if sampler is configured).
                # IMPORTANT: test functions are computed in the ORIGINAL parameter
                # frame (using the un-rotated x, y grid and surface_func). Their
                # ℝ³ gradients are therefore expressed in that frame and MUST be
                # rotated together with positions/normals/principal directions
                # below. We attach them BEFORE _repose_surface_and_quantities so
                # the centralized vector_3d_keys loop rotates them as well.
                if self._test_func_sampler is not None:
                    # Build a surface callable from the current surface_params
                    def surface_func(xv, yv, _sp=surface_params):
                        return self._evaluate_surface_with_parameters(xv, yv, _sp)

                    # Use a fresh rng seeded to the pre-loop snapshot so every
                    # surface samples the SAME probe functions. The probes are
                    # then evaluated on this surface's (x, y) grid. See the
                    # tf_rng_state comment before the loop for context.
                    tf_rng = np.random.default_rng()
                    tf_rng.bit_generator.state = tf_rng_state
                    tf_data = self._test_func_sampler.sample(
                        surface_func=surface_func,
                        x_grid=x.detach(),
                        y_grid=y.detach(),
                        rng=tf_rng,
                    )
                    data['test_func_deltas'] = tf_data['test_func_deltas']
                    data['test_func_values'] = tf_data['test_func_values']
                    data['test_func_laplacians'] = tf_data['test_func_laplacians'].unsqueeze(0)  # (1, P) for batching
                    data['test_func_gradients'] = tf_data['test_func_gradients'].unsqueeze(0)    # (1, P, 3) for batching
                    if 'test_func_lb_all_points' in tf_data:
                        data['test_func_lb_all_points'] = tf_data['test_func_lb_all_points']    # (K, P) per-node
                    if 'test_func_gradients_all_points' in tf_data:
                        data['test_func_gradients_all_points'] = tf_data['test_func_gradients_all_points']  # (K, P, 3) per-node

            # Apply pose transformation ONCE to positions and ALL ℝ³ quantities
            # (normals, principal directions, curvature gradients, AND test-function
            # gradients). This is the single source of truth for rotation — adding
            # a new ℝ³ field only requires updating vector_3d_keys in
            # _repose_surface_and_quantities.
            data = self._repose_surface_and_quantities(data=data, normals=origin_normal)

            # Store origin index if we added/found one
            if origin_idx is not None:
                data['origin_idx'] = torch.tensor([origin_idx])

            surfaces.append(data)

        return surfaces

    # Default sampler-driven implementation.  Subclasses may override
    # for bespoke families (e.g. PolynomialSurfaceDataset's pre-sampler
    # backward-compat path).

    def _generate_surface_parameters(self) -> Dict[str, Any]:
        """Sample fresh parameters via ``self._surface_sampler``.

        The closed-over ``surface_func`` is stashed in the returned dict
        under ``'_surface_func'`` so the existing flow (which threads
        ``surface_params`` from this method to
        ``_evaluate_surface_with_parameters``) needs no change.
        """
        if self._surface_sampler is None:
            raise NotImplementedError(
                f"{type(self).__name__} requires either a configured "
                f"surface_sampler or an override of "
                f"_generate_surface_parameters")
        surface_func, params = self._surface_sampler.sample(self._rng)
        params['_surface_func'] = surface_func
        return params

    def _evaluate_surface_with_parameters(
            self, x: torch.Tensor, y: torch.Tensor,
            surface_params: Dict[str, Any]) -> torch.Tensor:
        """Evaluate via the closed-over surface_func stashed in the params."""
        surface_func = surface_params.get('_surface_func')
        if surface_func is None:
            raise NotImplementedError(
                f"{type(self).__name__} expected '_surface_func' in "
                f"surface_params (set by sampler-driven "
                f"_generate_surface_parameters); got keys "
                f"{list(surface_params.keys())}")
        return surface_func(x, y)


class PolynomialSurfaceDataset(ParametricSurfaceDataset):
    """Dataset for polynomial surfaces with optional coordinate offset."""

    def __init__(
            self,
            order_range: Optional[Tuple[int, int]] = None,
            coefficient_scale_range: Optional[Tuple[float, float]] = None,
            coeff_generation_method: Optional[CoeffGenerationMethod] = None,
            polynomial_offset_range: Tuple[float, float] = (0.0, 0.0),
            surface_sampler: Optional[BaseMongeSurfaceSampler] = None,
            **kwargs
    ):
        """
        Initialize PolynomialSurfaceDataset.

        Two configuration styles are supported:

        1. **Legacy (back-compat)**: pass the polynomial-specific args
           (``order_range``, ``coefficient_scale_range``, ``coeff_generation_method``,
           ``polynomial_offset_range``).  A :class:`PolynomialMongeSurfaceSampler`
           is constructed internally and forwarded to the base class.  This
           keeps every existing yaml working unchanged.

        2. **Sampler injection**: pass ``surface_sampler=...``  (any
           :class:`BaseMongeSurfaceSampler`) and omit the polynomial args.
           Lets new yamls swap in :class:`RandomFourierMongeSurfaceSampler`
           or any other family without subclassing.

        Mixing the two styles in one call raises ``ValueError``.

        Args:
            order_range: Range of polynomial orders (min, max).
            coefficient_scale_range: Range for scaling coefficients.
            coeff_generation_method: ``CoeffGenerationMethod.UNIFORM`` or ``NORMAL``.
            polynomial_offset_range: Range for random ``(ox, oy)`` offset.
            surface_sampler: Pre-instantiated surface sampler.  Mutually
                exclusive with the polynomial args.
            **kwargs: Forwarded to :class:`ParametricSurfaceDataset`.
        """
        polynomial_args_given = any(arg is not None for arg in (
            order_range, coefficient_scale_range, coeff_generation_method))
        if surface_sampler is not None and polynomial_args_given:
            raise ValueError(
                "PolynomialSurfaceDataset accepts either an injected "
                "'surface_sampler' OR the legacy polynomial args "
                "(order_range / coefficient_scale_range / "
                "coeff_generation_method), not both.")

        if surface_sampler is None:
            # Legacy path: synthesize a PolynomialMongeSurfaceSampler.
            if not polynomial_args_given:
                raise ValueError(
                    "PolynomialSurfaceDataset requires either a "
                    "'surface_sampler' or the legacy polynomial args.")
            order_range = self._validate_order_range(order_range=order_range)
            coefficient_scale_range = self._validate_range(
                param_range=coefficient_scale_range, name="coefficient_scale_range")
            polynomial_offset_range = self._validate_range(
                param_range=polynomial_offset_range, name="polynomial_offset_range")
            surface_sampler = PolynomialMongeSurfaceSampler(
                order_range=order_range,
                coefficient_scale_range=coefficient_scale_range,
                coeff_generation_method=coeff_generation_method,
                polynomial_offset_range=polynomial_offset_range,
            )
            # Keep the historical attributes for any external introspection.
            self._order_range = order_range
            self._coefficient_scale_range = coefficient_scale_range
            self._polynomial_offset_range = polynomial_offset_range
            self._coeff_generation_method = coeff_generation_method

        super().__init__(surface_sampler=surface_sampler, **kwargs)

    def _validate_order_range(self, order_range: Tuple[int, int]) -> Tuple[int, int]:
        """Validate polynomial order range."""
        if len(order_range) != 2:
            raise ValueError(f"Order range must have 2 elements, got {len(order_range)}")
        if order_range[0] < 1 or order_range[1] < 1:
            raise ValueError("Polynomial order must be >= 1")
        if order_range[0] > order_range[1]:
            raise ValueError(f"Order range min {order_range[0]} > max {order_range[1]}")
        return order_range

    @staticmethod
    def _get_polynomial_pairs(order: int) -> List[Tuple[int, int]]:
        """Get list of (i, j) exponent pairs for polynomial of given order."""
        return get_polynomial_pairs(order)


# =============================================
# Surface-level dataset for variational training
# =============================================

class _VariationalSurfaceData(Data):
    """PyG Data subclass for surface-level variational batches.

    The surface is laid out as one PyG graph element with ``n*k`` rows of
    per-patch features (``x``, ``pos``, ``patch_idx``) plus surface-level
    fields (``vertex_pos``, ``vertex_normals``, ``vertex_areas``, ``knn``,
    ``test_func_*``).  The custom ``__cat_dim__`` / ``__inc__`` keep
    ``patch_idx`` from being incremented when PyG batches multiple
    surfaces.  The variational training pipeline currently runs with
    ``batch_size=1`` (one surface per step), so cross-surface incrementing
    of ``knn`` is not yet supported.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'patch_idx':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'patch_idx':
            return 0
        if key == 'knn':
            # batch_size=1 only for now: knn indices are not offset.
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class MongeSurfaceVariationalDataset(Dataset):
    """Surface-level synthetic dataset for variational / Dirichlet-energy training.

    Each ``__getitem__`` returns one Monge-patch surface with ``n`` vertices,
    their k-NN patches, per-vertex normals/areas, and probe-function data
    including the analytic ``(P, P)`` continuous Dirichlet bilinear-form GT.

    The model's ``forward()`` processes all ``n`` patches in a single batched
    call (the patch fields ``x`` / ``pos`` / ``patch_idx`` already encode
    them as a multi-patch batch).  The variational losses then read the
    surface-level fields from the same ``Data`` object to compare discrete
    and continuous Dirichlet forms.

    Args:
        surface_sampler:  Pre-instantiated :class:`BaseMongeSurfaceSampler`
            (e.g. :class:`PolynomialMongeSurfaceSampler` or
            :class:`RandomFourierMongeSurfaceSampler`).  Drives surface
            generation; the rest of the pipeline (quadrature, k-NN,
            probe evaluation) is family-agnostic.
        num_vertices_range:  Per-surface vertex count.  Either an int
            (fixed) or a 2-tuple (uniform random per surface).
        grid_radius:  Half-side of the parameter-domain square U =
            [-r, r]^2 used both for vertex sampling and for the continuous
            quadrature.
        position_noise_std:  Gaussian noise std added to 3D vertex
            positions *after* analytic normals / areas are computed from
            the clean surface.  Either a scalar or a 2-tuple range.
        k:  k-NN neighbour count per patch.  Either an int (fixed across
            surfaces) or a 2-element ``[lo, hi]`` range (uniform integer
            random draw per surface).  ``num_vertices_range[0]`` must be
            strictly greater than ``max(k)``.
        test_func_sampler:  Configured :class:`BaseTestFunctionSampler`
            instance.  Must have ``compute_continuous_bilinear=True`` (or
            ``compute_continuous_energy=True``) so the GT targets are
            attached to each item.
        epoch_size:  Number of surfaces drawn per training epoch.
        seed:  RNG seed.

    Yields :class:`_VariationalSurfaceData` items with fields:
        x                              (n*k, 3)  per-patch features
        pos                            (n*k, 3)  per-patch positions (centred)
        patch_idx                      (n*k,)    patch assignment 0..n-1
        vertex_indices                 (n*k,)    global vertex idx per row
        vertex_pos                     (n, 3)    vertex positions
        vertex_normals                 (n, 3)    analytic surface normals
        vertex_areas                   (n,)      uniform A_total/n
        knn                            (n, k)    global neighbour indices
        test_func_values               (n, P)    probe values at vertices
        test_func_continuous_bilinear  (P, P)    GT bilinear form (if enabled)
        test_func_continuous_energy    (P,)      GT energies         (if enabled)
    """

    def __init__(
        self,
        surface_sampler: BaseMongeSurfaceSampler,
        num_vertices_range,
        test_func_sampler,
        grid_radius: float = 1.0,
        position_noise_std=0.0,
        k=15,
        epoch_size: int = 100,
        seed: int = 0,
        quadrature_n_for_total_area: int = 20,
    ):
        super().__init__()
        self._surface_sampler = surface_sampler
        self._grid_radius = float(grid_radius)
        self._epoch_size = int(epoch_size)
        self._test_func_sampler = test_func_sampler
        self._quadrature_n_area = int(quadrature_n_for_total_area)

        # k may be a scalar or a 2-element [lo, hi] range; a fresh value
        # is drawn per surface in get().  Each surface internally has a
        # fixed k (since build_patches_from_vertices takes a single k);
        # variation across surfaces exposes the model to multiple
        # neighbour-count regimes.
        self._k_range = _coerce_scalar_or_range(k, name='k', cast=int)
        if self._k_range[0] < 1:
            raise ValueError(f"k must be >= 1, got range {self._k_range}")

        if isinstance(num_vertices_range, int):
            self._num_vertices_range = (num_vertices_range, num_vertices_range)
        else:
            self._num_vertices_range = (
                int(num_vertices_range[0]), int(num_vertices_range[1]))
        # Lower bound on n must beat the *largest* k draw so every patch
        # has enough neighbours.
        if self._num_vertices_range[0] <= self._k_range[1]:
            raise ValueError(
                f"num_vertices_range[0]={self._num_vertices_range[0]} must be > "
                f"max k = {self._k_range[1]}")

        self._position_noise_std = SyntheticSurfaceDataset._normalize_noise_std(position_noise_std)
        self._rng = np.random.default_rng(seed=seed)

    # PyG Dataset hooks ----------------------------------------------------

    def len(self) -> int:
        return self._epoch_size

    def get(self, idx) -> _VariationalSurfaceData:
        # 1. Sample a Monge surface from the configured family.
        surface_func, _surface_params = self._surface_sampler.sample(self._rng)

        U_bounds = ((-self._grid_radius, self._grid_radius),
                    (-self._grid_radius, self._grid_radius))

        # 2. Sample vertex parameter coords on U.
        n = int(self._rng.integers(low=self._num_vertices_range[0],
                                    high=self._num_vertices_range[1] + 1))
        u = torch.from_numpy(self._rng.uniform(
            low=-self._grid_radius, high=self._grid_radius, size=n)).float()
        v = torch.from_numpy(self._rng.uniform(
            low=-self._grid_radius, high=self._grid_radius, size=n)).float()

        # 3. Lift vertices to ℝ³ and compute analytic normals via autograd.
        # Compute both partials in a single grad() call so the graph is freed
        # exactly once (PyTorch frees saved tensors after the first call
        # unless retain_graph=True; computing both at once avoids that).
        with torch.enable_grad():
            u_g = u.clone().requires_grad_(True)
            v_g = v.clone().requires_grad_(True)
            z_g = surface_func(u_g, v_g)
            h_u, h_v = torch.autograd.grad(
                z_g.sum(), [u_g, v_g], create_graph=False)
        z = z_g.detach()
        h_u = h_u.detach()
        h_v = h_v.detach()

        clean_pos = torch.stack([u, v, z], dim=-1)                     # (n, 3)
        normal_unnorm = torch.stack(
            [-h_u, -h_v, torch.ones_like(h_u)], dim=-1)
        vertex_normals = F.normalize(normal_unnorm, p=2, dim=-1)        # (n, 3)

        # 4. Total surface area via Gauss-Legendre on √det g; uniform per-vertex.
        u_q, v_q, w_q = gauss_legendre_2d_grid(U_bounds, self._quadrature_n_area)
        u_q_flat = u_q.flatten()
        v_q_flat = v_q.flatten()
        w_q_flat = w_q.flatten()
        with torch.enable_grad():
            u_qg = u_q_flat.clone().unsqueeze(-1).requires_grad_(True)
            v_qg = v_q_flat.clone().unsqueeze(-1).requires_grad_(True)
            z_q = surface_func(u_qg, v_qg)
            if z_q.dim() > 1:
                z_q = z_q.squeeze(-1)
            h_u_q, h_v_q = torch.autograd.grad(
                z_q.sum(), [u_qg, v_qg], create_graph=False)
            h_u_q = h_u_q.squeeze(-1)
            h_v_q = h_v_q.squeeze(-1)
        sqrt_det_g_q = (1.0 + h_u_q.detach() ** 2 + h_v_q.detach() ** 2).sqrt()
        total_area = float((w_q_flat * sqrt_det_g_q).sum().item())
        vertex_areas = torch.full((n,), total_area / n, dtype=torch.float32)

        # 5. Optional position noise (applied AFTER analytic quantities are computed).
        lo, hi = self._position_noise_std
        noise_std = float(self._rng.uniform(low=lo, high=hi)) if hi > lo else lo
        if noise_std > 0:
            noise = torch.from_numpy(
                self._rng.normal(scale=noise_std, size=(n, 3))).float()
            noisy_pos = clean_pos + noise
        else:
            noisy_pos = clean_pos

        # 6. Build k-NN patches.  k is drawn fresh per surface from its
        # configured range (forward() handles any per-surface k internally).
        k_lo, k_hi = self._k_range
        k_this = int(self._rng.integers(low=k_lo, high=k_hi + 1)) if k_lo != k_hi else k_lo
        patch = build_patches_from_vertices(noisy_pos, k=k_this, device=None)
        # patch: pos (n*k, 3), x (n*k, 3), patch_idx (n*k,),
        #        vertex_indices (n*k,), neighbor_index_matrix (n, k)

        # 7. Probe values at vertices and continuous bilinear / energy GT.
        tf = self._test_func_sampler.sample(
            surface_func=surface_func,
            x_grid=u, y_grid=v,
            rng=self._rng,
            U_bounds=U_bounds,
        )

        # 8. Bundle into a single PyG Data object.
        data = _VariationalSurfaceData(
            x=patch.x,
            pos=patch.pos,
            patch_idx=patch.patch_idx,
            vertex_indices=patch.vertex_indices,
            knn=patch.neighbor_index_matrix,
            vertex_pos=noisy_pos,
            vertex_normals=vertex_normals,
            vertex_areas=vertex_areas,
            test_func_values=tf['test_func_values'],
        )
        if 'test_func_continuous_bilinear' in tf:
            data.test_func_continuous_bilinear = tf['test_func_continuous_bilinear']
        if 'test_func_continuous_energy' in tf:
            data.test_func_continuous_energy = tf['test_func_continuous_energy']
        if 'test_func_gradients_all_points' in tf:
            # In the variational layout, the sampler's "all points" axis
            # is the n vertices; expose under the more semantic name that
            # GradientVectorTestLoss reads.
            data.test_func_gradients_at_vertices = tf['test_func_gradients_all_points']
        return data