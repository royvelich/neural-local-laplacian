# standard library
from typing import List, Type, Callable, Optional, Dict

# neural laplacian
from neural_local_laplacian.utils import utils

# torch
import torch
import torch.nn as nn

# torch_geometric
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.nn import knn_interpolate, knn_graph

# torch_cluster
from torch_cluster import fps


class ConfigurableGNNBase(nn.Module):
    def __init__(self, conv_layers: List[torch_geometric.nn.conv.MessagePassing], mlp_layers: Optional[List[torch.nn.Module]], k: int, concat_residual: bool, recompute_knn: bool):
        super().__init__()
        self._conv_layers = torch.nn.ModuleList(conv_layers)
        if mlp_layers is not None:
            self._mlp_layers = torch.nn.ModuleList(mlp_layers)
        else:
            self._mlp_layers = [None for _ in range(len(self._conv_layers))]
        self._k = k
        self._recompute_knn = recompute_knn
        self._concat_residual = concat_residual
        self._conv_input_args = utils.get_input_args(forward_method=self._conv_layers[0].forward)


class ConfigurableGNN(ConfigurableGNNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, batch: Batch) -> torch.Tensor:
        # if not self._recompute_knn:
        #     batch['edge_index'] = knn_graph(x=batch['pos'], k=self._k, batch=batch['batch'])

        x_list = [batch['x']]
        for i, (conv_layer, mlp_layer) in enumerate(zip(self._conv_layers, self._mlp_layers)):
            # Prepare input arguments for the layer
            layer_inputs = {}
            for key in self._conv_input_args:
                if key == 'x':
                    layer_inputs[key] = x_list[-1]
                # elif key == 'edge_index' and self._recompute_knn:
                #     layer_inputs[key] = knn_graph(x=x_list[-1], k=self._k, batch=batch['batch'])
                else:
                    layer_inputs[key] = batch[key]

            # Apply the layer
            x = conv_layer(**layer_inputs)
            if mlp_layer is not None:
                x = mlp_layer(x)
            x_list.append(x)

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
    def __init__(self, channels: List[int], use_batch_norm: bool = False, use_layer_norm: bool = False, activation: Optional[str] = None):
        super().__init__()
        self._layers = nn.ModuleList()
        self._normalizations = nn.ModuleList()
        self._channels = channels
        self._use_batch_norm = use_batch_norm
        self._use_layer_norm = use_layer_norm

        if activation is not None:
            self._activations = [utils.import_object(full_type_name=activation)() for _ in range(len(channels) - 1)]
        else:
            self._activations = None

        for i in range(len(channels) - 1):
            self._layers.append(nn.Linear(in_features=channels[i], out_features=channels[i + 1]))

            # Add normalization layer if needed (except for the last layer)
            if i < len(channels) - 2:
                if self._use_batch_norm:
                    self._normalizations.append(nn.BatchNorm1d(num_features=channels[i + 1]))
                elif self._use_layer_norm:
                    self._normalizations.append(nn.LayerNorm(normalized_shape=channels[i + 1]))
                else:
                    self._normalizations.append(None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self._layers[:-1]):
            x = layer(x)

            # Apply normalization if available
            if (self._use_batch_norm or self._use_layer_norm) and i < len(self._normalizations):
                if self._normalizations[i] is not None:
                    x = self._normalizations[i](x)

            # Apply activation if available
            if self._activations is not None:
                x = self._activations[i](x)

        x = self._layers[-1](x)
        return x


class ConfigurablePooling(torch.nn.Module):
    def __init__(self, pooling_layers: List[str]):
        super().__init__()
        self._pooling_layers = pooling_layers

    def forward(self, batch: Batch, x: torch.Tensor) -> torch.Tensor:
        pooling_features_list = []
        for pooling_layer in self._pooling_layers:
            pooling_fn = utils.import_object(full_type_name=pooling_layer)
            pooling_out = pooling_fn(x=x, batch=batch.batch)
            pooling_features_list.append(pooling_out)

        pooling_cat = torch.cat(pooling_features_list, dim=1)

        return pooling_cat