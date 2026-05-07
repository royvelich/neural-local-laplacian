"""Discrete Dirichlet-energy utilities for the variational training pipeline.

Given the network's predicted gradient coefficients ``grad_coeffs[i, j] = g_ij``
(per patch i, per neighbour j) and probe-function values at the vertices,
these helpers compute:

    ĝ_i^{(ℓ)}  =  Σ_α  g_iα · f_ℓ(v_α)
                =  Σ_j  g_ij · (f_ℓ(v_j) − f_ℓ(v_i))                    (using g_ii = −Σ_j g_ij)

i.e. the discrete surface gradient at each vertex applied to each test
function, and from there the discrete Dirichlet energy and bilinear form

    Ê(f_ℓ)        = ½ Σ_i  A_i · ‖ĝ_i^{(ℓ)}‖²
    Ê(f_ℓ, f_p)  =   Σ_i  A_i · ⟨ĝ_i^{(ℓ)}, ĝ_i^{(p)}⟩

These are the discrete-side counterparts of the continuous Dirichlet
form computed in :mod:`neural_local_laplacian.datasets.synthetic_datasets`
(``compute_dirichlet_bilinear_form_continuous`` and friends). All ops
are autograd-compatible and device-agnostic.
"""

import torch


def predicted_gradient_per_vertex(
    grad_coeffs: torch.Tensor,
    knn: torch.Tensor,
    values_at_vertices: torch.Tensor,
) -> torch.Tensor:
    """Compute the discrete surface gradient ĝ_i^{(ℓ)} = Σ_j g_ij · δf_ℓ_j.

    Equivalent to ``Σ_α g_iα f_ℓ(v_α)`` with the consistency self-coefficient
    ``g_ii = −Σ_j g_ij`` substituted in, so the operator annihilates constants
    by construction.

    Args:
        grad_coeffs: ``(n, k, 3)`` predicted gradient coefficients per
            (vertex i, neighbour j).
        knn: ``(n, k)`` long tensor of global neighbour indices.
        values_at_vertices: ``(n, P)`` probe-function values at every vertex.

    Returns:
        ``(n, P, 3)`` predicted surface gradient at each vertex for each probe.
    """
    n, k, _ = grad_coeffs.shape
    if knn.shape != (n, k):
        raise ValueError(
            f"knn shape {tuple(knn.shape)} does not match grad_coeffs (n={n}, k={k})")
    if values_at_vertices.shape[0] != n:
        raise ValueError(
            f"values_at_vertices has n={values_at_vertices.shape[0]}, "
            f"expected {n}")

    # Gather neighbour values: (n, k, P)
    neighbour_values = values_at_vertices[knn]
    # Center values broadcast: (n, 1, P)
    center_values = values_at_vertices.unsqueeze(1)
    # Per-edge function differences: (n, k, P)
    deltas = neighbour_values - center_values
    # ĝ_i^{(ℓ)}[d] = Σ_j grad_coeffs[i, j, d] · deltas[i, j, ℓ]
    return torch.einsum('nkd,nkp->npd', grad_coeffs, deltas)


def discrete_dirichlet_energy(
    pred_grads: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    """Compute Ê(f_ℓ) = ½ Σ_i A_i ‖ĝ_i^{(ℓ)}‖² for each probe.

    Args:
        pred_grads: ``(n, P, 3)`` predicted surface gradients (output of
            :func:`predicted_gradient_per_vertex`).
        areas: ``(n,)`` per-vertex areas.

    Returns:
        ``(P,)`` discrete Dirichlet energy per probe.
    """
    if areas.shape[0] != pred_grads.shape[0]:
        raise ValueError(
            f"areas has n={areas.shape[0]}, expected {pred_grads.shape[0]}")
    sq_norms = (pred_grads ** 2).sum(dim=-1)              # (n, P)
    return 0.5 * (areas.unsqueeze(-1) * sq_norms).sum(dim=0)  # (P,)


def discrete_dirichlet_bilinear(
    pred_grads: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    """Compute Ê(f_ℓ, f_p) = Σ_i A_i ⟨ĝ_i^{(ℓ)}, ĝ_i^{(p)}⟩ for every probe pair.

    The diagonal equals ``2 · discrete_dirichlet_energy(pred_grads, areas)``.

    Args:
        pred_grads: ``(n, P, 3)`` predicted surface gradients.
        areas: ``(n,)`` per-vertex areas.

    Returns:
        ``(P, P)`` discrete bilinear form. Symmetric.
    """
    if areas.shape[0] != pred_grads.shape[0]:
        raise ValueError(
            f"areas has n={areas.shape[0]}, expected {pred_grads.shape[0]}")
    weighted = areas[:, None, None] * pred_grads          # (n, P, 3)
    return torch.einsum('nld,npd->lp', weighted, pred_grads)
