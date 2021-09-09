# Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch import nn
from torch.nn import functional as F
from galileo.platform.export import export


@export('galileo.pytorch')
class BaseAggregator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 use_concat_in_aggregator: bool = True,
                 bias: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_concat_in_aggregator = use_concat_in_aggregator
        self.bias = bias
        input_dim = 2 * input_dim if use_concat_in_aggregator else input_dim
        self.fc_layer = nn.Linear(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r'''
        initialized using Glorot uniform initialization
        '''
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_layer.weight, gain=gain)

    def aggregate(self, inputs):
        raise NotImplementedError

    def concat(self, self_feat, neigh_feat):
        if self.use_concat_in_aggregator:
            return torch.cat([self_feat, neigh_feat], dim=-1)
        return torch.add(self_feat, neigh_feat)

    def forward(self, inputs):
        r'''
        inputs is list of tensor
        case 1:
            self feat shape (*, feat dim)
            neighbor feat shape (*, nbr dim, feat dim)
        case 2:
            self_feat shape (*, self dim, feat dim)
            neighbor feat shape (*, nbr dim, feat dim)
        output shape:
            case 1: (*, output_dim)
            case 2: (*, self dim, output_dim)
        '''
        self_feat, neigh_feat = inputs
        if self_feat.dim() == neigh_feat.dim():
            # reshape to case 1
            self_shape = list(self_feat.shape)
            neigh_shape = self_shape[:-1] + [-1, self_shape[-1]]
            neigh_feat = neigh_feat.view(neigh_shape)
        agg_feat = self.aggregate(neigh_feat)
        output = self.concat(self_feat, agg_feat)
        output = self.fc_layer(output)
        return output


@export('galileo.pytorch')
class MeanAggregator(BaseAggregator):
    def aggregate(self, inputs):
        return torch.mean(inputs, dim=-2)


@export('galileo.pytorch')
class PoolAggregator(BaseAggregator):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 use_concat_in_aggregator: bool = True,
                 bias: bool = False):
        super().__init__(input_dim, output_dim, use_concat_in_aggregator, bias)
        self.neigh_fc_layer = nn.Linear(input_dim, input_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        r'''
        initialized using Glorot uniform initialization
        '''
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.neigh_fc_layer.weight, gain=gain)

    def aggregate(self, inputs):
        output = F.relu(self.neigh_fc_layer(inputs))
        return torch.mean(output, dim=-2)


MeanPoolAggregator = PoolAggregator
export('galileo.pytorch').var('MeanPoolAggregator', MeanPoolAggregator)


@export('galileo.pytorch')
class MaxPoolAggregator(PoolAggregator):
    def aggregate(self, inputs):
        output = F.relu(self.neigh_fc_layer(inputs))
        return torch.max(output, dim=-2)[0]


@export('galileo.pytorch')
class GCNAggregator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 use_concat_in_aggregator: bool = True,
                 bias: bool = False):
        super().__init__()
        self.fc_layer = nn.Linear(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r'''
        initialized using Glorot uniform initialization
        '''
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_layer.weight, gain=gain)

    def forward(self, inputs):
        r'''
        inputs is list of tensor
        case 1:
            self feat shape (*, feat dim)
            neighbor feat shape (*, nbr dim, feat dim)
        case 2:
            self_feat shape (*, self dim, feat dim)
            neighbor feat shape (*, nbr dim, feat dim)
        output shape:
            case 1: (*, output_dim)
            case 2: (*, self dim, output_dim)
        '''
        self_feat, neigh_feat = inputs
        if self_feat.dim() == neigh_feat.dim():
            # reshape to case 1
            self_shape = list(self_feat.shape)
            neigh_shape = self_shape[:-1] + [-1, self_shape[-1]]
            neigh_feat = neigh_feat.view(neigh_shape)
        if self_feat.dim() + 1 == neigh_feat.dim():
            self_feat = torch.unsqueeze(self_feat, dim=-2)
        concated_feat = torch.cat([self_feat, neigh_feat], dim=-2)
        mean_feat = torch.mean(concated_feat, dim=-2)
        output = self.fc_layer(mean_feat)
        return output


aggregators = {
    'mean': MeanAggregator,
    'meanpool': MeanPoolAggregator,
    'maxpool': MaxPoolAggregator,
    'gcn': GCNAggregator,
}


@export('galileo.pytorch')
def get_aggregator(name):
    return aggregators.get(name)
