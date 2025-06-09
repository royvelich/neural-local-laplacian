# # Standard library imports
# import importlib
# from typing import List, Union, Type, Dict, Callable
# import inspect
# from abc import ABC, abstractmethod
#
# # Third-party library imports
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Sequential, Dropout, Linear
# import pytorch_lightning as pl
# from torch_geometric.data import Batch
# from torch_geometric.nn import global_mean_pool, global_max_pool
# from omegaconf import DictConfig, OmegaConf
# import wandb
# from lightly.models.modules import SimSiamProjectionHead, SimSiamPredictionHead
# from lightly.loss import NegativeCosineSimilarity
#
# # Local imports
# from deltaconv.models import DeltaNetBase
# from deltaconv.nn import MLP
#
#
# class SignaturePredictionModuleBase(ABC, pl.LightningModule):
#     """
#     An abstract base class for signature prediction modules using PyTorch Lightning.
#
#     This class defines the common structure and functionality for signature prediction
#     modules. It implements shared methods for training, validation, and optimization,
#     while leaving the specific forward pass implementation to derived classes.
#
#     Derived classes must implement the `forward` method to define the specific
#     architecture for signature prediction.
#
#     Attributes:
#         _grid_size (int): Size of the grid used in signature prediction.
#         _optimizer_cfg (DictConfig): Configuration for the optimizer.
#         _output_dim (int): Dimension of the output signature.
#         _val_predictions (List): List to store validation predictions.
#         _val_ground_truths (List): List to store validation ground truths.
#     """
#
#     def __init__(self, grid_size: int, optimizer_cfg: DictConfig):
#         """
#         Initialize the base Signature Prediction Module.
#
#         Args:
#             grid_size (int): Size of the grid used in signature prediction.
#             optimizer_cfg (DictConfig): Configuration for the optimizer.
#         """
#         super().__init__()
#         self.save_hyperparameters()
#         self._grid_size = grid_size
#         self._optimizer_cfg = optimizer_cfg
#         # self._output_dim = (grid_size ** 2) * 6
#         self._val_predictions = []
#         self._val_ground_truths = []
#
#         # SimSiam components using lightly
#         self.projection_head = SimSiamProjectionHead(input_dim=2048, hidden_dim=1024, output_dim=1024)
#         self.prediction_head = SimSiamPredictionHead(input_dim=1024, hidden_dim=512, output_dim=1024)
#
#         # SimSiam loss
#         self.criterion = NegativeCosineSimilarity()
#
#     def setup(self, stage):
#         def exclude_fn(path: str):
#             if 'lightning_logs' in path:
#                 return True
#             if 'outputs' in path:
#                 return True
#             if 'wandb' in path:
#                 return True
#             if '.git' in path:
#                 return True
#
#             return False
#
#         def include_fn(path: str):
#             return True if path.endswith('.py') or path.endswith('.yml') or path.endswith('.yaml') else False
#
#         if self.trainer.global_rank == 0 and wandb.run is not None:
#             self.logger.experiment.log_code(root=".", exclude_fn=exclude_fn, include_fn=include_fn)
#             dict_cfg = OmegaConf.to_container(self.trainer.cfg, resolve=True)
#             self.logger.experiment.config.update(dict_cfg)
#
#     @abstractmethod
#     def forward(self, batch: Batch) -> torch.Tensor:
#         """
#         Abstract method for the forward pass of the Signature Prediction Module.
#
#         This method must be implemented by all derived classes to define the
#         specific architecture for signature prediction.
#
#         Args:
#             batch (Batch): Input batch of data.
#
#         Returns:
#             torch.Tensor: Predicted signature.
#         """
#         pass
#
#     def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
#         batch1, batch2 = batch[0], batch[1]
#
#         # Get predictions
#         pred_signature1 = self(batch=batch1)
#         pred_signature2 = self(batch=batch2)
#
#         # Ground truth loss
#         # gt_signature = batch1['signature'].reshape(len(batch1), -1)
#         # loss_gt1 = F.mse_loss(input=pred_signature1, target=gt_signature)
#         # loss_gt2 = F.mse_loss(input=pred_signature2, target=gt_signature)
#
#         # SimSiam components
#         z1 = self.projection_head(pred_signature1)
#         z2 = self.projection_head(pred_signature2)
#         p1 = self.prediction_head(z1)
#         p2 = self.prediction_head(z2)
#
#         # SimSiam loss
#         loss_simsiam = 0.5 * (self.criterion(p1, z2.detach()) + self.criterion(p2, z1.detach()))
#
#         # Combined loss
#         loss = loss_simsiam
#
#         # self.log("pred1 vs. gt loss", loss_gt1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
#         # self.log("pred2 vs. gt loss", loss_gt2, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
#         self.log("simsiam loss", loss_simsiam, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
#
#         return loss
#
#     def validation_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
#         """
#         Perform a single validation step.
#
#         This method is common to all derived classes and implements the
#         validation logic. It processes two batches and stores the predictions
#         and ground truths for later use in matching rate calculation.
#
#         Args:
#             batch (List[Batch]): The input batches of data.
#             batch_idx (int): The index of the current batch.
#
#         Returns:
#             Dict[str, torch.Tensor]: A dictionary containing the validation loss.
#         """
#         # Process both batches
#         pred_signatures1 = self(batch=batch[0])
#         pred_signatures2 = self(batch=batch[1])
#
#         # Store predictions and ground truths
#         self._val_predictions.append((pred_signatures1, pred_signatures2))
#         # self._val_ground_truths.append((batch[0]['signature'], batch[1]['signature']))
#
#         # Compute the MSE loss for both batches
#         # gt_signatures1 = batch[0]['signature'].reshape(len(batch[0]), -1)
#         # gt_signatures2 = batch[1]['signature'].reshape(len(batch[1]), -1)
#
#         # loss1 = F.mse_loss(input=pred_signatures1, target=gt_signatures1)
#         # loss2 = F.mse_loss(input=pred_signatures2, target=gt_signatures2)
#         # avg_loss = (loss1 + loss2) / 2
#
#         # Log the loss
#         # self.log(name="val_loss", value=avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
#         #
#         # return {"val_loss": avg_loss}
#
#     def on_validation_epoch_end(self) -> None:
#         """
#         Called at the end of the validation epoch.
#
#         This method is common to all derived classes. It calculates the matching
#         rate across all validation samples and logs metrics.
#         """
#         all_pred1 = torch.cat([p[0] for p in self._val_predictions])
#         all_pred2 = torch.cat([p[1] for p in self._val_predictions])
#
#         # Compute pairwise distances between all signatures
#         distances = torch.cdist(all_pred1, all_pred2)
#
#         # Find the closest match for each patch in the first batch
#         closest_matches = torch.argmin(distances, dim=1)
#
#         # Calculate the rate of correct matches
#         correct_matches = (closest_matches == torch.arange(len(closest_matches), device=closest_matches.device))
#         match_rate = correct_matches.float().mean()
#
#         # Log the match rate
#         self.log(name="val_match_rate", value=match_rate, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
#
#         # Clear stored predictions and ground truths
#         self._val_predictions.clear()
#         # self._val_ground_truths.clear()
#
#     def configure_optimizers(self) -> torch.optim.Optimizer:
#         """
#         Configure the optimizer for the model.
#
#         This method is common to all derived classes and sets up the optimizer
#         based on the provided configuration.
#
#         Returns:
#             torch.optim.Optimizer: The configured optimizer.
#         """
#         return self._optimizer_cfg(params=self.parameters())
