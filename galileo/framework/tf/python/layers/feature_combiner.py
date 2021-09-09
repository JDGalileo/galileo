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
from tensorflow.keras.layers import Layer, Dense
from galileo.framework.tf.python.layers.feature_encoder import (
    DenseFeatureEncoder,
    SparseFeatureEncoder,
)
from galileo.platform.export import export


@export('galileo.tf')
class FeatureCombiner(Layer):
    r'''
    FeatureCombiner combine dense and sparse feature

    args:
        dense_feature_dims: int or list[int], dense feature dims
        sparse_feature_maxs: int or list[int], the max value in sparse
            feature set, used to set embedding size
        sparse_feature_embedding_dims: int or list[int], for sub layer embedding
        hidden_dim: must be specified when add feature
        feature_combiner: str in ['concat', 'sum'], combine strategy,
            how to combine feature when use multi feature
            concat: concat all dense and sparse features
                optional encode to hidden_dim, when hidden_dim is given
            add: encode dense and sparse features to hidden_dim and add
    '''
    def __init__(self,
                 dense_feature_dims=None,
                 sparse_feature_maxs=None,
                 sparse_feature_embedding_dims=None,
                 hidden_dim: int = None,
                 feature_combiner: str = 'concat',
                 **kwargs):
        super().__init__(**kwargs)
        if dense_feature_dims is None and sparse_feature_maxs is None:
            raise ValueError('one of dense or sparse feature '
                             'must be specified')
        if feature_combiner not in ['add', 'concat']:
            raise ValueError('feature_combiner is either "add" or "concat".')
        if feature_combiner == 'add' and hidden_dim is None:
            raise ValueError('hidden_dim must be specified when add feature.')
        dense_feature_dims = dense_feature_dims or []
        sparse_feature_maxs = sparse_feature_maxs or []
        sparse_feature_embedding_dims = sparse_feature_embedding_dims or []
        self.feature_combiner = feature_combiner
        self.hidden_dim = hidden_dim

        if isinstance(dense_feature_dims, int):
            dense_feature_dims = [dense_feature_dims]
        if feature_combiner == 'add':
            # add combiner use a same hidden_dim
            sparse_feature_embedding_dims = hidden_dim
        if isinstance(sparse_feature_maxs, int):
            sparse_feature_maxs = [sparse_feature_maxs]
        if isinstance(sparse_feature_embedding_dims, int):
            sparse_feature_embedding_dims = [sparse_feature_embedding_dims
                                             ] * len(sparse_feature_maxs)
        assert len(sparse_feature_maxs) == len(sparse_feature_embedding_dims)

        self.dense_feature_encoder = None
        if dense_feature_dims and feature_combiner == 'add':
            self.dense_feature_encoder = DenseFeatureEncoder(
                dense_feature_dims, hidden_dim)

        self.sparse_feature_encoder = None
        if sparse_feature_maxs and sparse_feature_embedding_dims:
            self.sparse_feature_encoder = SparseFeatureEncoder(
                sparse_feature_maxs, sparse_feature_embedding_dims)

        self.fc = None
        if feature_combiner == 'concat' and hidden_dim:
            self.fc = Dense(hidden_dim, use_bias=False)

        self.dense_feature_dims = dense_feature_dims
        self.sparse_feature_maxs = sparse_feature_maxs
        self.sparse_feature_embedding_dims = sparse_feature_embedding_dims

    def call(self, inputs):
        r'''
        \param inputs list/dict
        \return tensor
        '''
        if isinstance(inputs, (list, tuple)):
            dense_feature, sparse_feature = inputs[:2]
        elif isinstance(inputs, dict):
            dense_feature = inputs.get('dense')
            sparse_feature = inputs.get('sparse')
        else:
            dense_feature, sparse_feature = inputs, None
        features = []
        if dense_feature is not None:
            if isinstance(dense_feature, (list, tuple)):
                dense_feature = tf.concat(dense_feature, axis=-1)
            if self.dense_feature_encoder:
                dense_feature = self.dense_feature_encoder(dense_feature)
            features.append(dense_feature)

        if sparse_feature is not None and self.sparse_feature_encoder:
            sparse_embeddings = self.sparse_feature_encoder(sparse_feature)
            features.extend(sparse_embeddings)

        if self.feature_combiner == 'add':
            feature = tf.add_n(features)
        else:
            feature = tf.concat(features, axis=-1)
            if self.fc is not None:
                feature = self.fc(feature)
        return feature

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                dense_feature_dims=self.dense_feature_dims,
                sparse_feature_maxs=self.sparse_feature_maxs,
                sparse_feature_embedding_dims=self.
                sparse_feature_embedding_dims,
                hidden_dim=self.hidden_dim,
                feature_combiner=self.feature_combiner,
            ))
        return config
