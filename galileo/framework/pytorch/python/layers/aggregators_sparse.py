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
from .dense import Dense


@export('galileo.pytorch')
class BaseAggregatorSparse(nn.Module):
    '''
    base aggregator aggregates target and neigbor feature

    args:
        input_dim: input dimensionality
        output_dim: dimensionality of the output space
        use_concat_in_aggregator: concat target and neigbor feature
        bias: bias
    '''
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 use_concat_in_aggregator: bool = True,
                 bias: bool = False,
                 **kwargs):
        super().__init__()
        try:
            from torch_scatter import scatter
        except ImportError:
            raise ImportError('torch-scatter is required by *AggregatorSparse')
        self.scatter = scatter
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_concat_in_aggregator = use_concat_in_aggregator
        self.bias = bias
        self.build_kernels()

    def build_kernels(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError

    def reset_parameters(self, *layers):
        r'''
        initialized using Glorot uniform initialization
        '''
        gain = nn.init.calculate_gain('relu')
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)


@export('galileo.pytorch')
class MeanAggregatorSparse(BaseAggregatorSparse):
    r'''
    \brief mean or mean-1k

    \f$ h_v=\sigma(W(h_v^{self}|mean(h_{N(v)}))) $\f
    '''
    def build_kernels(self):
        input_dim = 2 * self.input_dim if self.use_concat_in_aggregator \
            else self.input_dim
        self.fc_layer = nn.Linear(input_dim, self.output_dim, bias=self.bias)
        self.reset_parameters(self.fc_layer)

    def forward(self, inputs):
        r'''
        inputs: (self_feat, nbr_feat, relation_src_indices)
        '''
        self_feat = inputs[0]
        agg_feat = self.aggregate(inputs)
        output = self.concat(self_feat, agg_feat)
        return self.fc_layer(output)

    def concat(self, self_feat, neigh_feat):
        if self.use_concat_in_aggregator:
            return torch.cat([self_feat, neigh_feat], dim=-1)
        return torch.add(self_feat, neigh_feat)

    def aggregate(self, inputs):
        self_feat, nbr_feat, relation_src_indices = inputs
        num_nodes = self_feat.shape[0]
        return self.scatter(nbr_feat,
                            relation_src_indices,
                            dim=0,
                            dim_size=num_nodes,
                            reduce='mean')


@export('galileo.pytorch')
class Mean2kAggregatorSparse(MeanAggregatorSparse):
    r'''
    \brief mean-2k

    \f$ h_v=\sigma(W_1h_v^{self}|W_2mean(h_{N(v)}))) $\f
    '''
    def build_kernels(self):
        if self.use_concat_in_aggregator:
            self_units = self.output_dim // 2
            nbr_units = self.output_dim - self_units
        else:
            self_units = nbr_units = self.output_dim
        self.self_fc_layer = nn.Linear(self.input_dim,
                                       self_units,
                                       bias=self.bias)
        self.nbr_fc_layer = nn.Linear(self.input_dim,
                                      nbr_units,
                                      bias=self.bias)
        self.reset_parameters(self.self_fc_layer, self.nbr_fc_layer)

    def forward(self, inputs):
        r'''
        inputs: (self_feat, nbr_feat, relation_src_indices)
        '''
        self_feat, nbr_feat, relation_src_indices = inputs
        self_out = self.self_fc_layer(self_feat)
        agg_feat = super().aggregate(inputs)
        nbr_out = self.nbr_fc_layer(agg_feat)
        output = self.concat(self_out, nbr_out)
        return output


@export('galileo.pytorch')
class PoolAggregatorSparse(MeanAggregatorSparse):
    r'''
    \brief mean pool

    \f$ h_v=\sigma(W(h_v^{self}|mean(\sigma(W_ph_{N(v)})))) $\f
    '''
    def build_kernels(self):
        super().build_kernels()
        self.neigh_fc_layer = nn.Linear(self.input_dim,
                                        self.input_dim,
                                        bias=self.bias)
        self.reset_parameters(self.neigh_fc_layer)

    def aggregate(self, inputs):
        output = F.relu(self.neigh_fc_layer(inputs[1]))
        return super().aggregate((inputs[0], output, inputs[2]))


MeanPoolAggregatorSparse = PoolAggregatorSparse
export('galileo.pytorch').var('MeanPoolAggregatorSparse',
                              MeanPoolAggregatorSparse)


@export('galileo.pytorch')
class MaxPoolAggregatorSparse(PoolAggregatorSparse):
    def aggregate(self, inputs):
        self_feat, nbr_feat, relation_src_indices = inputs
        num_nodes = self_feat.shape[0]
        nbr_feat = F.relu(self.neigh_fc_layer(nbr_feat))
        return self.scatter(nbr_feat,
                            relation_src_indices,
                            dim=0,
                            dim_size=num_nodes,
                            reduce='max')


@export('galileo.pytorch')
class GCNAggregatorSparse(BaseAggregatorSparse):
    def build_kernels(self):
        self.fc_layer = nn.Linear(self.input_dim,
                                  self.output_dim,
                                  bias=self.bias)
        self.reset_parameters(self.fc_layer)

    def forward(self, inputs):
        self_feat, nbr_feat, relation_src_indices = inputs
        num_nodes = self_feat.shape[0]
        nbr_sum = self.scatter(nbr_feat,
                               relation_src_indices,
                               dim=0,
                               dim_size=num_nodes,
                               reduce='sum')
        degs = self.scatter(torch.ones_like(nbr_feat),
                            relation_src_indices,
                            dim=0,
                            dim_size=num_nodes,
                            reduce='sum')
        output = (self_feat + nbr_sum) / (degs + 1)
        output = self.fc_layer(output)
        return output


aggregators = {
    'mean': MeanAggregatorSparse,
    'mean-1k': MeanAggregatorSparse,
    'mean-2k': Mean2kAggregatorSparse,
    'meanpool': MeanPoolAggregatorSparse,
    'maxpool': MaxPoolAggregatorSparse,
    'gcn': GCNAggregatorSparse,
}


@export('galileo.pytorch')
def get_aggregator_sparse(name):
    return aggregators.get(name)
