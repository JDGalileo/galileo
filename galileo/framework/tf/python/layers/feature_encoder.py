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

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from galileo.platform.export import export


@export('galileo.tf')
class DenseFeatureEncoder(Layer):
    '''
    dense feature encoder

    args:
        dense_feature_dims: int or list[int], dense feature dimension
            for compat with pytorch
        output_dim: int, output dim after encode
    '''
    def __init__(self, dense_feature_dims, output_dim, **kwargs):
        super().__init__(**kwargs)
        if isinstance(dense_feature_dims, int):
            dense_feature_dims = [dense_feature_dims]
        self.dense_feature_dims = dense_feature_dims
        self.output_dim = output_dim
        self.fc = Dense(output_dim, use_bias=False)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            dense_feats = tf.concat(inputs, axis=-1)
        else:
            dense_feats = inputs
        dense_feature = self.fc(dense_feats)
        return dense_feature

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(dense_feature_dims=self.dense_feature_dims,
                 output_dim=self.output_dim))
        return config


@export('galileo.tf')
class SparseFeatureEncoder(Layer):
    r'''
    sparse feature encoder

    args:
        sparse_feature_maxs: list, the max value in saprse feature set,
            used to set embedding size
        sparse_feature_embedding_dims: int or list[int], embedding dim
    '''
    def __init__(self, sparse_feature_maxs, sparse_feature_embedding_dims,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(sparse_feature_maxs, int):
            sparse_feature_maxs = [sparse_feature_maxs]
        if isinstance(sparse_feature_embedding_dims, int):
            sparse_feature_embedding_dims = [sparse_feature_embedding_dims
                                             ] * len(sparse_feature_maxs)
        self.sparse_feature_maxs = sparse_feature_maxs
        self.sparse_feature_embedding_dims = sparse_feature_embedding_dims
        self.sparse_embeddings = [
            Embedding(max_value + 1, dim) for max_value, dim in zip(
                sparse_feature_maxs, sparse_feature_embedding_dims)
        ]

    def call(self, inputs):
        if tf.is_tensor(inputs):
            inputs = tf.split(inputs, len(self.sparse_embeddings), axis=-1)
        assert isinstance(
            inputs,
            (list, tuple)), 'invalid inputs type for SparseFeatureEncoder'
        assert len(inputs) == len(self.sparse_embeddings)
        # squeeze may throw exception when axis=-2 is not 1
        embeddings = [
            tf.squeeze(sparse_embedding(sparse_feature), axis=-2)
            for sparse_embedding, sparse_feature in zip(
                self.sparse_embeddings, inputs)
        ]
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(sparse_feature_maxs=self.sparse_feature_maxs,
                 sparse_feature_embedding_dims=self.
                 sparse_feature_embedding_dims))
        return config
