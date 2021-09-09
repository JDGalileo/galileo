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
from tensorflow.keras.layers import Layer, Dropout, Dense
from galileo.platform.export import export
from galileo.framework.python.base_message_passing import BaseMessagePassing
from galileo.framework.tf.python.layers.aggregators_sparse import (
    get_aggregator_sparse)


@export('galileo.tf')
class SAGESparseLayer(Layer, BaseMessagePassing):
    r'''
    \brief graphSAGE convolution tf layer, sparse version

    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`
    '''
    def __init__(self,
                 output_dim: int,
                 aggregator_name: str = 'mean',
                 use_concat_in_aggregator: bool = True,
                 bias: bool = True,
                 dropout_rate: float = 0.0,
                 activation=None,
                 normalization=None,
                 **kwargs):
        r'''
        \param output_dim output dim of layer
        \param aggregator_name aggregator name, one of
            "mean, mean-1k, mean-2k, gcn, meanpool, maxpool"
        \param use_concat_in_aggregator concat if True else sum when aggregate
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
        self.aggregator_name = aggregator_name
        self.use_concat_in_aggregator = use_concat_in_aggregator
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.normalization = normalization

        aggregator_class = get_aggregator_sparse(aggregator_name)
        self.aggregator = aggregator_class(output_dim,
                                           use_concat_in_aggregator, bias)
        self.feature_dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        r'''
        \param inputs relation graph
            dict(
                relation_indices=tensor,
                feature=tensor,
                relation_weight=tensor,
            )
        '''
        h = self.message_and_aggregate(inputs, training=training)
        return self.update(h)

    def message_and_aggregate(self, inputs, training=None):
        relation_indices = inputs['relation_indices']
        feature = inputs['feature']
        relation_weight = inputs.get('relation_weight')
        feature_n = tf.gather(feature, relation_indices[1])
        if relation_weight is not None:
            feature_n = feature_n * relation_weight
        feature_n = self.feature_dropout(feature_n, training=training)
        return self.aggregator((feature, feature_n, relation_indices[0]))

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
                aggregator_name=self.aggregator_name,
                use_concat_in_aggregator=self.use_concat_in_aggregator,
                bias=self.bias,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                normalization=self.normalization,
            ))
        return config
