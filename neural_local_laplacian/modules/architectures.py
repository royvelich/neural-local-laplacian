# Standard library imports
import importlib
from typing import List, Type, Callable, Optional, Dict
from abc import ABC, abstractmethod
import inspect

# Neural signatures
import neural_local_laplacian.utils.utils as utils

# Third-party library imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import wandb
from torch_geometric.data import Batch
from torch_geometric.utils import to_torch_csr_tensor
from torch_cluster import fps
import torch_geometric
from torch_geometric.nn import knn_interpolate, knn_graph
from hydra.utils import instantiate
from deltaconv.models import DeltaNetBase
from deltaconv.nn import MLP


class ConfigurableGNNBase(nn.Module):
    # def __init__(self, conv_class_name: str, channels: List[int], use_batch_norm: bool, activation_name: str, k: Optional[int] = None):
    def __init__(self, conv_layers: List[torch_geometric.nn.conv.MessagePassing], mlp_layers: Optional[List[torch.nn.Module]], k: int, concat_residual: bool, recompute_knn: bool):
        super().__init__()
        # self._conv_layers = torch.nn.ModuleList([instantiate(conv_layer) for conv_layer in conv_layers])
        self._conv_layers = torch.nn.ModuleList(conv_layers)
        if mlp_layers is not None:
            self._mlp_layers = torch.nn.ModuleList(mlp_layers)
        else:
            self._mlp_layers = [None for _ in range(len(self._conv_layers))]
        self._k = k
        self._recompute_knn = recompute_knn
        self._concat_residual = concat_residual
        # self._conv_class_name = conv_class_name
        # self._channels = channels
        # self._use_batch_norm = use_batch_norm
        # self._activation_name = activation_name
        # self._activation = utils.import_object(full_type_name=activation_name)()
        # self._conv_class = utils.import_object(full_type_name=conv_class_name)
        self._conv_input_args = utils.get_input_args(forward_method=self._conv_layers[0].forward)
        # self._k = k


class ConfigurableGNN(ConfigurableGNNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._layers, self._batch_norms = utils.create_layers(channels=self._channels, conv_class=self._conv_class, use_batch_norm=self._use_batch_norm, backward=False, concat_residual=False)

    def forward(self, batch: Batch) -> torch.Tensor:
        if not self._recompute_knn:
            batch['edge_index'] = knn_graph(x=batch['pos'], k=self._k, batch=batch['batch'])

        x_list = [batch['x']]
        for i, (conv_layer, mlp_layer) in enumerate(zip(self._conv_layers, self._mlp_layers)):
            # Prepare input arguments for the layer
            layer_inputs = {}
            for key in self._conv_input_args:
                if key == 'x':
                    layer_inputs[key] = x_list[-1]
                elif key == 'edge_index' and self._recompute_knn:
                    layer_inputs[key] = knn_graph(x=x_list[-1], k=self._k, batch=batch['batch'])
                else:
                    layer_inputs[key] = batch[key]
                # if key == 'x':
                #     if i == 0:
                #         layer_inputs[key] = batch[key]
                #     else:
                #         layer_inputs[key] = x
                # else:
                #     layer_inputs[key] = batch[key]

            # Apply the layer
            x = conv_layer(**layer_inputs)
            x = mlp_layer(x)
            x_list.append(x)

            # # Apply batch norm and activation if it's not the last layer
            # if i < len(self._layers) - 1:
            #     if self._use_batch_norm:
            #         x = self._batch_norms[i](x)
            #     x = self._activation(x)

        if self._concat_residual:
            return torch.concat(x_list[1:], dim=-1)

        return x_list[-1]


# https://discuss.pytorch.org/t/what-is-the-output-of-a-topkpooling-layer/125564/2
class ConfigurableUNetGNN(ConfigurableGNNBase):
    def __init__(self, pool_ratio: float, concat_residual: bool, k: int, **kwargs):
        super().__init__(**kwargs)
        self._pool_ratio = pool_ratio
        self._concat_residual = concat_residual
        self._k = k
        self._encoder_layers, self._encoder_batch_norms = utils.create_layers(channels=self._channels + [self._channels[-1]], conv_class=self._conv_class, use_batch_norm=self._use_batch_norm, backward=False, concat_residual=False)
        self._decoder_layers, self._decoder_batch_norms = utils.create_layers(channels=self._channels, conv_class=self._conv_class, use_batch_norm=self._use_batch_norm, backward=True, concat_residual=concat_residual)
        pass

    def forward(self, batch: Batch) -> torch.Tensor:
        residual_data_list = []
        current_data = {}
        all_args = self._conv_input_args + ['batch']
        for key in all_args:
            if key != 'edge_index':
                current_data[key] = batch[key]

        for i, layer in enumerate(self._encoder_layers):
            edge_index = knn_graph(x=current_data['pos'], k=self._k, batch=current_data['batch'])
            current_data['edge_index'] = edge_index

            layer_inputs = {}
            for key in self._conv_input_args:
                layer_inputs[key] = current_data[key]

            # Apply point transformer block
            x = layer(**layer_inputs)
            x = self._activation(x)
            current_data['x'] = x

            residual_data_list = [current_data.copy()] + residual_data_list

            if i < len(self._encoder_layers) - 1:
                # FPS sampling
                idx = fps(src=current_data['pos'], batch=current_data['batch'], ratio=self._pool_ratio)

                # Update points and features
                for key in all_args:
                    if key != 'edge_index':
                        current_data[key] = current_data[key][idx]

        decoder_inputs = residual_data_list[0]

        # Decoder path
        for i, (decoder_layer, residual_data) in enumerate(zip(self._decoder_layers, residual_data_list[1:])):
            # Interpolate features
            x = knn_interpolate(x=decoder_inputs['x'], pos_x=decoder_inputs['pos'], pos_y=residual_data['pos'], batch_x=decoder_inputs['batch'], batch_y=residual_data['batch'], k=3)

            # Skip connection
            decoder_inputs = residual_data
            decoder_inputs['x'] = residual_data['x'] + x if not self._concat_residual else torch.cat([residual_data['x'], x], dim=-1)

            layer_inputs = {}
            for key in self._conv_input_args:
                layer_inputs[key] = decoder_inputs[key]

            # Apply point transformer block
            x = decoder_layer(**layer_inputs)
            # if i < len(self._decoder_layers) - 1:
            x = self._activation(x)
            decoder_inputs['x'] = x

        return decoder_inputs['x']


class ConfigurableMLP(nn.Module):
    def __init__(self, channels: List[int], use_batch_norm: bool, activation: Optional[str]):
        super().__init__()
        self._layers = nn.ModuleList()
        self._batch_norms = nn.ModuleList()
        self._channels = channels
        self._use_batch_norm = use_batch_norm
        if activation is not None:
            self._activations = [utils.import_object(full_type_name=activation)() for _ in range(len(channels) - 1)]
        else:
            self._activations = None
        for i in range(len(channels) - 1):
            self._layers.append(nn.Linear(in_features=channels[i], out_features=channels[i+1]))
            if self._use_batch_norm and i < len(channels) - 2:
                self._batch_norms.append(nn.BatchNorm1d(num_features=channels[i+1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self._layers[:-1]):
            x = layer(x)
            if self._use_batch_norm:
                x = self._batch_norms[i](x)
            if self._activations is not None:
                x = self._activations[i](x)
        x = self._layers[-1](x)
        return x



    # def __init__(self, pregnn_mlp_cfg: DictConfig, **kwargs):
    #     kwargs['gnn_cfg'].keywords['channels'] = [pregnn_mlp_cfg.keywords['channels'][-1]] + kwargs['gnn_cfg'].keywords['channels']
    #     super().__init__(**kwargs)
    #     self._pregnn_mlp = pregnn_mlp_cfg()
    #
    # def _preprocess_features(self, batch: Batch) -> torch.Tensor:
    #     shape1 = batch.x.shape
    #     x = self._pregnn_mlp(batch.x.reshape(-1, batch.x.shape[-1]))
    #     x = x.reshape(shape1[0], shape1[1], -1)
    #     x, _ = torch.max(x, dim=1)
    #     return x



class SignaturePredictionModuleBase(ABC, pl.LightningModule):
    def __init__(self, gnn: ConfigurableGNNBase, preprocess_mlp: ConfigurableMLP, postprocess_mlp: ConfigurableMLP, postgnn_mlp: ConfigurableMLP, pooling_layers: Optional[List[str]], optimizer_cfg: DictConfig):
        super().__init__()
        self._gnn = gnn
        self._preprocess_mlp = preprocess_mlp
        self._postprocess_mlp = postprocess_mlp
        self._postgnn_mlp = postgnn_mlp
        self._pooling_layers = pooling_layers
        self._optimizer_cfg = optimizer_cfg

    def setup(self, stage):
        def exclude_fn(path: str):
            if 'lightning_logs' in path:
                return True
            if 'outputs' in path:
                return True
            if 'wandb' in path:
                return True
            if '.git' in path:
                return True

            return False

        def include_fn(path: str):
            return True if path.endswith('.py') or path.endswith('.yml') or path.endswith('.yaml') else False

        if self.trainer.global_rank == 0 and wandb.run is not None:
            self.logger.experiment.log_code(root=".", exclude_fn=exclude_fn, include_fn=include_fn)
            dict_cfg = OmegaConf.to_container(self.trainer.cfg, resolve=True)
            self.logger.experiment.config.update(dict_cfg)

    def _pre_preprocess_mlp(self, batch: Batch) -> torch.Tensor:
        x = batch.x
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])
        return x

    def _pre_gnn(self, batch: Batch) -> torch.Tensor:
        if len(batch.x.shape) > 2:
            x = batch.preprocess_mlp_out.reshape(batch.x.shape[0], batch.x.shape[1], -1)
            x, _ = torch.max(x, dim=1)
        else:
            x = batch.preprocess_mlp_out
        return x

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass of the Signature Prediction DeltaConv Module.

        Args:
            batch (Batch): Input batch of data.

        Returns:
            torch.Tensor: Predicted signature.
        """

        batch.preprocess_mlp_in = self._pre_preprocess_mlp(batch=batch)

        if self._preprocess_mlp is not None:
            batch.preprocess_mlp_out = self._preprocess_mlp(x=batch.preprocess_mlp_in)
        else:
            batch.preprocess_mlp_out = batch.preprocess_mlp_in

        batch.x = self._pre_gnn(batch=batch)
        batch.x = self._gnn(batch)

        if not isinstance(batch.x, list):
            x = [batch.x]
        else:
            x = batch.x
        x = torch.cat(x, dim=1)

        if self._postgnn_mlp is not None:
            x = self._postgnn_mlp(x)

        pooling_features_list = []
        for pooling_layer in self._pooling_layers:
            pooling_fn = utils.import_object(full_type_name=pooling_layer)
            pooling_out = pooling_fn(x=x, batch=batch.batch)
            pooling_features_list.append(pooling_out)

        pooling_cat = torch.cat(pooling_features_list, dim=1)

        if self._postprocess_mlp is not None:
            out = self._postprocess_mlp(pooling_cat)
        else:
            out = pooling_cat

        # if self._prepooling_mlp is not None:
        #     out = self._prepooling_mlp(out)
        #
        # pooling_features_list = []
        # for pooling_layer in self._pooling_layers:
        #     pooling_fn = import_object(full_type_name=pooling_layer)
        #     x = pooling_fn(x=out, batch=batch.batch)
        #     pooling_features_list.append(x)
        #
        # out = torch.cat(pooling_features_list, dim=1)
        # if self._postpooling_mlp is not None:
        #     out = self._postpooling_mlp(out)

        return out

    @abstractmethod
    def training_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def validation_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._optimizer_cfg(params=self.parameters())
