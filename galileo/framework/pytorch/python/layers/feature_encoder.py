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
import torch.nn as nn
import torch.nn as init
from galileo.platform.export import export


@export('galileo.pytorch')
class DenseFeatureEncoder(nn.Module):
    '''
    dense feature encoder

    args:
        dense_feature_dims: int or list[int], dense feature dimension
        output_dim: int, output dim after encode
    '''
    def __init__(self, dense_feature_dims, output_dim):
        super().__init__()
        if isinstance(dense_feature_dims, int):
            dense_feature_dims = [dense_feature_dims]
        self.dense_feature_dims = dense_feature_dims
        self.output_dim = output_dim
        self.fc = nn.Linear(sum(dense_feature_dims), output_dim, bias=False)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            dense_feats = torch.cat(inputs, dim=-1)
        else:
            dense_feats = inputs
        dense_feature = self.fc(dense_feats)
        return dense_feature


@export('galileo.pytorch')
class SparseFeatureEncoder(nn.Module):
    '''
    sparse feature encoder

    args:
        sparse_feature_maxs: int or list[int], the max value in saprse feature set,
            used to set embedding size
        sparse_feature_embedding_dims: int or list[int], embedding dim
    '''
    def __init__(self, sparse_feature_maxs, sparse_feature_embedding_dims):
        super().__init__()
        if isinstance(sparse_feature_maxs, int):
            sparse_feature_maxs = [sparse_feature_maxs]
        if isinstance(sparse_feature_embedding_dims, int):
            sparse_feature_embedding_dims = [sparse_feature_embedding_dims
                                             ] * len(sparse_feature_maxs)
        self.sparse_embeddings = nn.ModuleList()
        for max_value, dim in zip(sparse_feature_maxs,
                                  sparse_feature_embedding_dims):
            self.sparse_embeddings.append(nn.Embedding(max_value + 1, dim))

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = torch.split(inputs, 1, dim=-1)
        assert isinstance(
            inputs,
            (list, tuple)), 'invalid inputs type for SparseFeatureEncoder'
        assert len(inputs) == len(self.sparse_embeddings)
        embeddings = [
            torch.squeeze(sparse_embedding(sparse_feature), dim=-2)
            for sparse_embedding, sparse_feature in zip(
                self.sparse_embeddings, inputs)
        ]
        return embeddings
