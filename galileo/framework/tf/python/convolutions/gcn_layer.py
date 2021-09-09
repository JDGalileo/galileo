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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from galileo.platform.export import export
from galileo.framework.python.base_message_passing import BaseMessagePassing


@export('galileo.tf')
class GCNLayer(Layer, BaseMessagePassing):
    r'''
    \brief GCN convolution tf layer, sparse version

    `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`
    '''
    def __init__(self,
                 output_dim: int,
                 bias: bool = True,
                 dropout_rate: float = 0.0,
                 activation=None,
                 normalization=None,
                 **kwargs):
        r'''
        \param output_dim output dim of layer
        \param bias bias of layer
        \param dropout_rate feature dropout rate
        \param activation callable, apply activation to
            the updated vertices features
        \param normalization callable, apply normalization to
            the updated vertices features
        '''
        # tensorflow replace the base class to base_layer_v1.Layer
        # when use estimator, so can't call Layer.__init__()
        self.__class__.__bases__[0].__init__(self, **kwargs)
        BaseMessagePassing.__init__(self)

        self.output_dim = output_dim
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.normalization = normalization

        self.feature_dropout = Dropout(dropout_rate)

    def build(self, input_shape):
        input_dim = input_shape['features'][-1]
        self.kernels = self.add_weight(name='kernels',
                                       shape=(input_dim, self.output_dim),
                                       initializer='glorot_normal')
        if self.bias:
            self.biases = self.add_weight(name='biases',
                                          shape=(self.output_dim, ),
                                          initializer='zeros')

    def call(self, inputs, training=None):
        h = self.message_and_aggregate(inputs, training=training)
        return self.update(h)

    def message_and_aggregate(self, inputs, training=None):
        edge_srcs = inputs['edge_srcs']
        edge_dsts = inputs['edge_dsts']
        features = inputs['features']

        num_nodes = tf.shape(features)[0]
        features = self.feature_dropout(features, training=training)
        features = tf.matmul(features, self.kernels)

        edge_weights = inputs.get('edge_weights')
        if edge_weights is None:
            edge_weights = tf.ones_like(edge_dsts, dtype=tf.float32)

        degrees = tf.ones_like(edge_srcs, dtype=tf.float32)
        degrees = tf.math.unsorted_segment_sum(degrees,
                                               edge_srcs,
                                               num_segments=num_nodes)
        degrees = tf.clip_by_value(degrees,
                                   clip_value_min=1,
                                   clip_value_max=np.inf)
        norm = tf.pow(degrees, -0.5)
        edge_weights = tf.gather(norm, edge_srcs) * edge_weights * tf.gather(
            norm, edge_dsts)
        # edge_weights can be computed before all layers

        nbr_features = tf.gather(features, edge_dsts)
        nbr_features = nbr_features * tf.expand_dims(edge_weights, 1)
        reduced = tf.math.unsorted_segment_sum(nbr_features,
                                               edge_srcs,
                                               num_segments=num_nodes)
        if self.bias:
            reduced += self.biases
        # shape [features.shape[0], output_dim]
        return reduced

    def update(self, inputs):
        if callable(self.activation):
            inputs = self.activation(inputs)
        if callable(self.normalization):
            inputs = self.normalization(inputs)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                output_dim=self.output_dim,
                bias=self.bias,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                normalization=self.normalization,
            ))
        return config
