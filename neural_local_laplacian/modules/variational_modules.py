"""Surface-level variational training module.

This module subclasses :class:`LaplacianTransformerModule` (and friends),
inheriting all of the architecture, the area / gradient heads, the
``forward()`` method, every patch-level validation step, and the optimiser
configuration.  The only override is :meth:`training_step`, which:

  1. Accepts a PyG Batch composed of one or more
     :class:`_VariationalSurfaceData` items (each = one Monge surface
     with ``n`` vertices and their k-NN patches stacked along the patch
     axis).
  2. Runs ``forward()`` once per surface to produce
     ``grad_coeffs (n, k, 3)``, ``areas (n,)``, etc.
  3. Builds a :class:`LossContext` with the surface-level fields
     (``knn``, ``vertex_normals``, ``test_func_values``,
     ``test_func_continuous_bilinear``, ``test_func_continuous_energy``)
     so the variational losses can read them.
  4. Iterates ``self._loss_configs`` exactly like the parent module, with
     the same NaN diagnostics, gradient-clipping, and W&B logging.

Validation (mesh / fmap / patch-MCV) is unchanged — those steps are
inherited and dispatched by ``LaplacianLocalModule.validation_step``.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from neural_local_laplacian.modules.laplacian_modules import (
    LaplacianTransformerModule,
    LaplacianDeepSetModule,
)
from neural_local_laplacian.modules.losses import LossContext


def _build_per_surface_batch(surface_data) -> Batch:
    """Wrap one ``_VariationalSurfaceData`` item into a PyG ``Batch`` of one
    graph element so :meth:`forward` can consume it with its existing
    ``batch.x / batch.pos / batch.patch_idx`` contract."""
    return Batch.from_data_list([surface_data])


class _VariationalTrainingMixin:
    """Mixin providing :meth:`training_step` for variational training.

    Kept separate from the architecture-specific subclasses below so the
    same training loop is shared between the transformer and DeepSet
    variants.
    """

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        # The data module wraps each batch in a list (mirroring the existing
        # patch-level convention). Unwrap if needed.
        if isinstance(batch, list):
            if len(batch) == 0:
                raise ValueError("Empty training batch")
            batch = batch[0]

        # PyG Batch of N surfaces → process them sequentially. Each surface
        # internally bundles its n*k per-patch rows (with patch_idx) plus
        # surface-level fields (knn, vertex_normals, test_func_*).
        if isinstance(batch, Batch):
            surface_items = batch.to_data_list()
        else:
            surface_items = [batch]

        per_surface_losses: List[torch.Tensor] = []
        per_surface_components: List[Dict[str, float]] = []

        for surface in surface_items:
            forward_batch = _build_per_surface_batch(surface)
            forward_result = self.forward(forward_batch)

            grad_coeffs = forward_result['grad_coeffs']        # (n, k, 3)
            areas = forward_result['areas']                    # (n,)
            attention_mask = forward_result['attention_mask']  # (n, k)
            stiffness = forward_result['stiffness_weights']    # (n, k)

            # Surface-level GT fields (attached by MongeSurfaceVariationalDataset).
            knn = getattr(surface, 'knn', None)
            vertex_pos = getattr(surface, 'vertex_pos', None)
            vertex_normals = getattr(surface, 'vertex_normals', None)
            vertex_mcv = getattr(surface, 'vertex_mcv', None)
            test_func_values = getattr(surface, 'test_func_values', None)
            tf_continuous_bilinear = getattr(
                surface, 'test_func_continuous_bilinear', None)
            tf_continuous_energy = getattr(
                surface, 'test_func_continuous_energy', None)
            tf_gradients_at_vertices = getattr(
                surface, 'test_func_gradients_at_vertices', None)
            tf_lap_at_vertices = getattr(
                surface, 'test_func_laplacians_at_vertices', None)
            geodesic_sources = getattr(surface, 'geodesic_sources', None)
            geodesic_distances = getattr(surface, 'geodesic_distances', None)
            heat_kernel_times = getattr(surface, 'heat_kernel_times', None)
            heat_kernel_gt = getattr(surface, 'heat_kernel_gt', None)
            heat_kernel_diagonal_only = getattr(
                surface, 'heat_kernel_diagonal_only', None)

            # The variational losses operate on whole-surface tensors; no
            # MCV target is needed.  We still set predicted_mcv/target_mcv
            # to dummy zeros so the LossContext dataclass (which requires
            # them) is well-formed; nothing reads them in this pipeline.
            n = grad_coeffs.shape[0]
            dummy_mcv = grad_coeffs.new_zeros((n, 3))

            ctx = LossContext(
                predicted_mcv=dummy_mcv,
                target_mcv=dummy_mcv,
                grad_coeffs=grad_coeffs,
                attention_mask=attention_mask,
                areas=areas,
                stiffness_weights=stiffness,
                knn=knn,
                vertex_pos=vertex_pos,
                vertex_normals=vertex_normals,
                vertex_mcv=vertex_mcv,
                test_func_values=test_func_values,
                test_func_continuous_bilinear=tf_continuous_bilinear,
                test_func_continuous_energy=tf_continuous_energy,
                test_func_gradients_at_vertices=tf_gradients_at_vertices,
                test_func_laplacians_at_vertices=tf_lap_at_vertices,
                geodesic_sources=geodesic_sources,
                geodesic_distances=geodesic_distances,
                heat_kernel_times=heat_kernel_times,
                heat_kernel_gt=heat_kernel_gt,
                heat_kernel_diagonal_only=heat_kernel_diagonal_only,
                gt_vertex_areas=getattr(surface, 'vertex_areas', None),
            )

            total_loss: torch.Tensor = grad_coeffs.new_zeros(())
            components: Dict[str, float] = {}
            any_weighted = False
            for loss_config in self._loss_configs:
                unweighted = loss_config.loss_module(ctx)
                name = loss_config.loss_module.__class__.__name__

                if (self._enable_nan_diagnostics
                        and torch.is_tensor(unweighted)
                        and not torch.isfinite(unweighted).all()):
                    raise RuntimeError(
                        f"Variational loss {name} returned a non-finite value: "
                        f"{unweighted.detach()}")

                components[f'train/loss/{name}'] = (
                    unweighted.item() if torch.is_tensor(unweighted) else float(unweighted))
                if loss_config.weight is not None:
                    total_loss = total_loss + loss_config.weight * unweighted
                    any_weighted = True

            if not any_weighted:
                raise ValueError(
                    "Variational training requires at least one loss config "
                    "with a non-None weight.")

            per_surface_losses.append(total_loss)
            per_surface_components.append(components)

        # Average across the surfaces in this batch.
        total_loss = torch.stack(per_surface_losses).mean()

        # Aggregate per-loss-component logs across surfaces.
        if per_surface_components:
            agg: Dict[str, float] = {}
            for k in per_surface_components[0]:
                vals = [d[k] for d in per_surface_components if k in d]
                agg[k] = sum(vals) / len(vals) if vals else 0.0

            batch_size = len(surface_items)
            for name, val in agg.items():
                self.log(name, val, on_step=False, on_epoch=True, logger=True,
                         batch_size=batch_size, sync_dist=True)

        self.log('train/loss', total_loss.item(), on_step=False, on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=len(surface_items), sync_dist=True)

        return {"loss": total_loss}


class VariationalLaplacianTransformerModule(_VariationalTrainingMixin,
                                             LaplacianTransformerModule):
    """Transformer-based variational module.

    Same hyperparameters as :class:`LaplacianTransformerModule`; only
    :meth:`training_step` is overridden via the mixin.
    """


class VariationalLaplacianDeepSetModule(_VariationalTrainingMixin,
                                         LaplacianDeepSetModule):
    """DeepSet-based variational module."""
