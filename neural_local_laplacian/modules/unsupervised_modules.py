# Standard library imports
import importlib
from typing import List, Union, Type, Dict, Callable, Optional
import inspect
from abc import ABC, abstractmethod

import numpy as np
# Third-party library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Dropout, Linear
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from omegaconf import DictConfig, OmegaConf
import wandb
from lightly.models.modules import SimSiamProjectionHead, SimSiamPredictionHead
from lightly.loss import NegativeCosineSimilarity
from deltaconv.models import DeltaNetBase
from deltaconv.nn import MLP
import open3d as o3d

# Local imports
from neural_local_laplacian.modules.architectures import SignaturePredictionModuleBase
from neural_local_laplacian.utils import utils


class UnsupervisedSignaturePredictionModule(SignaturePredictionModuleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._val_predictions = {}

    @abstractmethod
    def training_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        pass

    def validation_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.

        This method is common to all derived classes and implements the
        validation logic. It processes two batches and stores the predictions
        and ground truths for later use in matching rate calculation.

        Args:
            batch (List[Batch]): The input batches of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the validation loss.
        """
        # Process both batches
        pred_signatures1 = self(batch=batch[0])
        pred_signatures2 = self(batch=batch[1])

        # Store predictions and ground truths
        self._val_predictions.append((pred_signatures1, pred_signatures2))

        return {
            'pred_signatures1': pred_signatures1,
            'pred_signatures2': pred_signatures2
        }

    def on_validation_epoch_end(self) -> None:
        for key in self._val_predictions.keys():

            val_predictions = self._val_predictions[key]

            pred1_list = [p[0] for p in val_predictions]
            pred2_list = [p[1] for p in val_predictions]
            group1_list = [p[2] for p in val_predictions]
            group2_list = [p[3] for p in val_predictions]

            # Concatenate local tensors
            local_pred1 = torch.cat(pred1_list)
            local_pred2 = torch.cat(pred2_list)
            local_group1 = torch.cat(group1_list)
            local_group2 = torch.cat(group2_list)

            # Gather from all processes
            gathered_pred1 = self.all_gather(local_pred1)
            gathered_pred2 = self.all_gather(local_pred2)
            gathered_group1 = self.all_gather(local_group1)
            gathered_group2 = self.all_gather(local_group2)

            # Reshape gathered tensors
            all_pred1 = gathered_pred1.reshape(-1, gathered_pred1.shape[-1])
            all_pred2 = gathered_pred2.reshape(-1, gathered_pred2.shape[-1])
            all_group1 = gathered_group1.reshape(-1)
            all_group2 = gathered_group2.reshape(-1)

            # Get unique groups
            unique_groups = torch.unique(all_group1)

            # Initialize lists to store rates for each group
            top1_rates = []
            top5_rates = []
            top10_rates = []

            # For each group, compute top-k metrics
            for group_id in unique_groups:
                # Get predictions for current group
                group_mask1 = (all_group1 == group_id)
                group_mask2 = (all_group2 == group_id)

                group_pred1 = all_pred1[group_mask1]
                group_pred2 = all_pred2[group_mask2]

                if len(group_pred1) == 0 or len(group_pred2) == 0:
                    continue

                # Compute distances only for this group
                group_distances = torch.cdist(group_pred1, group_pred2)

                # Get top-k matches for this group
                k = min(10, len(group_distances[0]))
                topk_matches = torch.topk(group_distances, k=k, dim=1, largest=False)
                topk_indices = topk_matches.indices

                # Create ground truth indices for this group
                gt_indices = torch.arange(len(group_pred1), device=topk_indices.device)

                # Calculate top-k match rates for this group
                top1_correct = (topk_indices[:, 0] == gt_indices) if k >= 1 else torch.zeros_like(gt_indices)
                top5_correct = (topk_indices[:, :min(5, k)] == gt_indices.unsqueeze(1)).any(dim=1) if k >= 5 else torch.zeros_like(gt_indices)
                top10_correct = (topk_indices[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)

                # Store rates for this group
                top1_rates.append(top1_correct.float().mean().item())
                top5_rates.append(top5_correct.float().mean().item())
                top10_rates.append(top10_correct.float().mean().item())

            # Calculate mean rates across all groups
            mean_top1_rate = sum(top1_rates) / len(top1_rates) if top1_rates else 0.0
            mean_top5_rate = sum(top5_rates) / len(top5_rates) if top5_rates else 0.0
            mean_top10_rate = sum(top10_rates) / len(top10_rates) if top10_rates else 0.0

            # Log mean metrics
            self.log(f"val_top1_rate_dataloader{key}", mean_top1_rate, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"val_top5_rate_dataloader{key}", mean_top5_rate, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"val_top10_rate_dataloader{key}", mean_top10_rate, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Clear stored predictions
        self._val_predictions = {}


class SimSiamSignaturePredictionModule(UnsupervisedSignaturePredictionModule):
    def __init__(self, criterion: torch.nn.Module, output_channels: int, **kwargs):
        super().__init__(**kwargs)
        self._output_channels = output_channels
        self._projection_head = SimSiamProjectionHead(input_dim=output_channels, hidden_dim=2048, output_dim=2048)
        self._prediction_head = SimSiamPredictionHead(input_dim=2048, hidden_dim=512, output_dim=2048)
        self._criterion = criterion

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch1, batch2 = batch[0][0], batch[0][1]

        # Get predictions
        pred_signature1 = self(batch=batch1)
        pred_signature2 = self(batch=batch2)

        # SimSiam components
        z1 = self._projection_head(pred_signature1)
        z2 = self._projection_head(pred_signature2)
        p1 = self._prediction_head(z1)
        p2 = self._prediction_head(z2)

        # SimSiam loss
        loss = 0.5 * (self._criterion(p1, z2.detach()) + self._criterion(p2, z1.detach()))

        self.log("simsiam loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))

        return {
            'loss': loss
        }

    def validation_step(self, batch: List[Batch], batch_idx: int, dataloader_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.

        This method is common to all derived classes and implements the
        validation logic. It processes two batches and stores the predictions
        and ground truths for later use in matching rate calculation.

        Args:
            batch (List[Batch]): The input batches of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the validation loss.
        """
        # Process both batches
        pred_signatures1 = self(batch=batch[0][0])
        pred_signatures2 = self(batch=batch[0][1])

        if dataloader_idx not in self._val_predictions:
            self._val_predictions[dataloader_idx] = []

        # Store predictions and ground truths
        self._val_predictions[dataloader_idx].append((pred_signatures1, pred_signatures2, batch[0][0].mesh_id, batch[0][1].mesh_id))

        return {
            'pred_signatures1': pred_signatures1,
            'pred_signatures2': pred_signatures2,
            "dataloader_idx": dataloader_idx
        }


class SoftCorrespondenceModule(SignaturePredictionModuleBase):
    def __init__(self, sinkhorn_temp: float, sinkhorn_n_iter: int, sinkhorn_slack: bool, **kwargs):
        super().__init__(**kwargs)
        self._sinkhorn_temp = sinkhorn_temp
        self._sinkhorn_n_iter = sinkhorn_n_iter
        self._sinkhorn_slack = sinkhorn_slack

    def _pre_preprocess_mlp(self, batch: Batch) -> torch.Tensor:
        x = batch.x.reshape(-1, batch.x.shape[-1])
        return x

    def _pre_gnn(self, batch: Batch) -> torch.Tensor:
        x = batch.post_mlp_x.reshape(batch.x.shape[0], batch.x.shape[1], -1)
        x, _ = torch.max(x, dim=1)
        return x

    @staticmethod
    def _create_labels(
            points: np.ndarray,
            indices: Union[List[int], np.ndarray],
            offset: List[float] = [0.1, 0.1, 0.1],
            scale: float = 1.0
    ) -> List[o3d.visualization.gui.Label3D]:
        """
        Create 3D text labels for specified point indices.

        Args:
            points: Nx3 array of points
            indices: Indices of points to label
            offset: 3D offset for label positioning
            scale: Scale of the text labels

        Returns:
            List of Label3D objects
        """
        labels = []
        for idx in indices:
            pos = points[idx] + np.array(offset)
            # Convert position to float32 as required by Label3D
            pos_float32 = pos.astype(np.float32)
            label = o3d.visualization.gui.Label3D(str(idx), pos_float32)
            label.scale = scale
            label.color = o3d.visualization.gui.Color(1.0, 1.0, 1.0)  # White text
            labels.append(label)
        return labels

    @staticmethod
    def _render_dual_point_clouds(
            points1: np.ndarray,
            highlight_indices1: Union[List[int], np.ndarray],
            points2: np.ndarray,
            highlight_indices2: Union[List[int], np.ndarray],
            base_color: List[float] = [0.5, 0.5, 0.5],
            base_size: float = 2.0,
            highlight_size: float = 5.0
    ) -> None:
        """
        Render two point clouds side by side, with corresponding highlighted points sharing random colors
        and having larger size.

        Args:
            points1: Nx3 array of point coordinates for first cloud
            highlight_indices1: List or array of indices to highlight in first cloud
            points2: Nx3 array of point coordinates for second cloud
            highlight_indices2: List or array of indices to highlight in second cloud
            base_color: RGB color for non-highlighted points (default: gray)
            base_size: Size for non-highlighted points
            highlight_size: Size for highlighted points
        """

        # Validate input dimensions and lengths
        def validate_points_and_indices(
                points: np.ndarray,
                indices: Union[List[int], np.ndarray],
                cloud_name: str
        ) -> None:
            if points.shape[1] != 3:
                raise ValueError(f"{cloud_name} points must be Nx3 array, got shape {points.shape}")
            if max(indices) >= len(points):
                raise ValueError(f"{cloud_name} highlight indices out of bounds")

        validate_points_and_indices(points1, highlight_indices1, "First cloud")
        validate_points_and_indices(points2, highlight_indices2, "Second cloud")

        if len(highlight_indices1) != len(highlight_indices2):
            raise ValueError("Highlight indices must have the same length")

        def create_colored_point_cloud(
                points: np.ndarray,
                highlight_indices: Union[List[int], np.ndarray],
                highlight_colors: np.ndarray,
                base_color: List[float]
        ) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
            """
            Create two point clouds: one for base points and one for highlighted points.
            Returns a tuple of (base_cloud, highlight_cloud).
            """
            # Create a mask for non-highlighted points
            mask = np.ones(len(points), dtype=bool)
            mask[highlight_indices] = False

            # Create point cloud for non-highlighted points
            base_pcd = o3d.geometry.PointCloud()
            base_pcd.points = o3d.utility.Vector3dVector(points[mask])
            base_pcd.colors = o3d.utility.Vector3dVector(np.tile(base_color, (np.sum(mask), 1)))

            # Create point cloud for highlighted points
            highlight_pcd = o3d.geometry.PointCloud()
            highlight_pcd.points = o3d.utility.Vector3dVector(points[highlight_indices])
            highlight_pcd.colors = o3d.utility.Vector3dVector(highlight_colors)

            return base_pcd, highlight_pcd

        # Generate random colors for highlighted points
        num_highlights = len(highlight_indices1)
        random_colors = np.random.rand(num_highlights, 3)  # Random RGB values between 0 and 1

        # Create point cloud objects
        base_pcd1, highlight_pcd1 = create_colored_point_cloud(points1, highlight_indices1, random_colors, base_color)
        base_pcd2, highlight_pcd2 = create_colored_point_cloud(points2, highlight_indices2, random_colors, base_color)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Dual Point Cloud Viewer', width=1600, height=800)

        # Position point clouds side by side
        def position_point_cloud(
                pcd: o3d.geometry.PointCloud,
                offset_x: float
        ) -> None:
            center = pcd.get_center()
            pcd.translate((-center[0] + offset_x, -center[1], -center[2]))

        # Position all point clouds
        # position_point_cloud(base_pcd1, -2)
        # position_point_cloud(highlight_pcd1, -2)
        # position_point_cloud(base_pcd2, 2)
        # position_point_cloud(highlight_pcd2, 2)

        # Add point clouds to visualizer
        vis.add_geometry(base_pcd1)
        vis.add_geometry(highlight_pcd1)
        vis.add_geometry(base_pcd2)
        vis.add_geometry(highlight_pcd2)

        # Configure rendering options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # Black background

        # Set point sizes for each render
        def set_point_size_for_cloud(vis: o3d.visualization.Visualizer, cloud: o3d.geometry.PointCloud, size: float) -> None:
            vis.update_geometry(cloud)
            vis.get_render_option().point_size = size
            vis.poll_events()
            vis.update_renderer()

        # Set sizes for base and highlighted points
        set_point_size_for_cloud(vis, base_pcd1, base_size)
        set_point_size_for_cloud(vis, base_pcd2, base_size)
        set_point_size_for_cloud(vis, highlight_pcd1, highlight_size)
        set_point_size_for_cloud(vis, highlight_pcd2, highlight_size)

        # Set camera view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_lookat([0, 0, 0])

        # Run visualizer
        vis.run()
        vis.destroy_window()

    @staticmethod
    def _correspondence_to_permutation_matrix(
            correspondence_tensor: torch.Tensor,
            num_vertices_A: Optional[int] = None,
            num_vertices_B: Optional[int] = None
    ) -> torch.Tensor:
        """
        Convert a correspondence tensor to a permutation/partial permutation matrix.

        Args:
            correspondence_tensor (torch.Tensor): 1D tensor where correspondence_tensor[i] = j means
                                                vertex i in mesh A corresponds to vertex j in mesh B.
                                                Should be a 1D integer tensor.
            num_vertices_A (Optional[int]): Number of vertices in mesh A.
                                          If None, will be inferred from correspondence tensor.
            num_vertices_B (Optional[int]): Number of vertices in mesh B.
                                          If None, will be max(correspondence_tensor) + 1.

        Returns:
            torch.Tensor: Sparse permutation matrix P where P[i,j] = 1 if vertex i in A
                         corresponds to vertex j in B, 0 otherwise. Shape will be
                         (num_vertices_A, num_vertices_B).

        Raises:
            ValueError: If correspondence_tensor is not 1D or contains invalid indices.

        Example:
            >>> correspondence = torch.tensor([1, 0, 3, 2])  # vertex 0->1, 1->0, 2->3, 3->2
            >>> P = correspondence_to_permutation_matrix(correspondence)
            >>> print(P)
            tensor([[0., 1., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 1., 0.]])
        """
        # Input validation
        if correspondence_tensor.dim() != 1:
            raise ValueError("correspondence_tensor must be a 1D tensor")

        # Determine matrix dimensions
        if num_vertices_A is None:
            num_vertices_A = len(correspondence_tensor)
        if num_vertices_B is None:
            num_vertices_B = int(correspondence_tensor.max().item()) + 1

        # # Validate indices
        # if torch.any(correspondence_tensor < 0) or torch.any(correspondence_tensor >= num_vertices_B):
        #     raise ValueError(f"Invalid vertex indices in correspondence_tensor. "
        #                      f"All indices must be in range [0, {num_vertices_B})")

        # Create indices for sparse matrix construction
        rows: torch.Tensor = torch.arange(len(correspondence_tensor), device=correspondence_tensor.device)
        cols: torch.Tensor = correspondence_tensor

        # Create values (all ones for binary permutation matrix)
        values: torch.Tensor = torch.ones_like(rows, dtype=torch.float)

        # Create sparse tensor
        indices: torch.Tensor = torch.stack([rows, cols])
        P_sparse: torch.Tensor = torch.sparse_coo_tensor(
            indices,
            values,
            (num_vertices_A, num_vertices_B)
        )

        # Convert to dense tensor
        P_dense: torch.Tensor = P_sparse.to_dense()

        return P_dense

    def _shared_step(self, batch: List[Batch], batch_idx: int, train: bool, dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        data_batch = batch[0]
        gt_correspondences_list = [data['gt_correspondences'] for data in batch[1][0].to_data_list()]
        batch1, batch2 = data_batch[0], data_batch[1]
        batch1.x = self(batch=batch1)
        batch2.x = self(batch=batch2)

        data1_list = batch1.to_data_list()
        data2_list = batch2.to_data_list()

        # SoftCorrespondenceModule._render_dual_point_clouds(
        #     points1=data1_list[0].pos.detach().cpu().numpy(),
        #     highlight_indices1=data1_list[0].anchor_indices.detach().cpu().numpy(),
        #     points2=data2_list[0].pos.detach().cpu().numpy(),
        #     highlight_indices2=data2_list[0].anchor_indices.detach().cpu().numpy(),
        # )

        anchors1_list = []
        anchors2_list = []
        for data1, data2 in zip(data1_list, data2_list):
            anchors1 = data1.x[data1.anchor_indices]
            anchors2 = data2.x[data2.anchor_indices]
            anchors1_list.append(anchors1)
            anchors2_list.append(anchors2)

        # anchors1_stacked = torch.stack(anchors1_list, dim=0)
        # anchors2_stacked = torch.stack(anchors2_list, dim=0)

        # print(anchors1_list[0][0])
        # print(anchors2_list[0][0])

        similarities = [torch.mm(a, b.transpose(0, 1)) for a, b in zip(anchors1_list, anchors2_list)]

        sinks = [torch.nn.functional.softmax(similarity, dim=1) for similarity in similarities]

        # similarity = torch.bmm(anchors1_stacked, anchors2_stacked.transpose(1, 2))

        # sinks = [utils.gumbel_sinkhorn(log_alpha=similarity, temp=self._sinkhorn_temp, n_iter=self._sinkhorn_n_iter, slack=self._sinkhorn_slack) for similarity in similarities]
        # identity = torch.eye(sink.shape[1], device=sink.device).expand(sink.shape[0], -1, -1)

        selected_values_list = [sink[torch.arange(sink.shape[0]), gt_correspondence] for sink, gt_correspondence in zip(sinks, gt_correspondences_list)]

        # permutations = [SoftCorrespondenceModule._correspondence_to_permutation_matrix(correspondence_tensor=gt_correspondences_list[i], num_vertices_B=anchors2_list[i].shape[0]) for i in range(len(gt_correspondences_list))]

        # 3. Cross entropy loss (commonly used for permutation learning)
        eps = 1e-10  # small epsilon to prevent log(0)
        # ce_loss = [-torch.sum(permutation * torch.log(sink + eps), dim=1) for sink, permutation in zip(sinks, permutations)]

        ce_loss = [-torch.log(selected_values + eps) for selected_values in selected_values_list]

        # diffs = [(sink - permutation) for sink, permutation in zip(sinks, permutations)]
        # norms = [torch.norm(diff, p='fro') for diff in diffs]

        # loss = torch.mean(torch.stack(ce_loss))
        loss = torch.mean(torch.concat(ce_loss))


        # diff = sink - identity
        # loss = torch.norm(diff, p='fro', dim=(1, 2)).mean()

        # name = 'train_loss' if train else f'val_loss_{dataloader_idx}'
        # self.log(name=name, value=loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch[0]))

        name = 'train_loss' if train else f'val_loss'
        self.log(name=name, value=loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch[0]))

        return {
            'loss': loss,
            'sinks': sinks
        }

    def training_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._shared_step(batch=batch, batch_idx=batch_idx, train=True)

    def validation_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._shared_step(batch=batch, batch_idx=batch_idx, train=False)

    # def validation_step(self, batch: List[Batch], batch_idx: int, dataloader_idx: int) -> Dict[str, torch.Tensor]:
    #     return self._shared_step(batch=batch, batch_idx=batch_idx, train=False, dataloader_idx=dataloader_idx)

    # def on_validation_epoch_end(self, outputs) -> None:
    #     pass