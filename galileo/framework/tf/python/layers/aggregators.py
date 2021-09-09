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
from tensorflow.keras.layers import (
    Layer,
    Dense,
)
from galileo.platform.export import export


@export('galileo.tf')
class BaseAggregator(Layer):
    '''
    base aggregator aggregates target and neigbor feature

    args:
        units: dimensionality of the output space
        use_concat_in_aggregator: concat target and neigbor feature
    '''
    def __init__(self,
                 units: int,
                 use_concat_in_aggregator: bool = True,
                 bias: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_concat_in_aggregator = use_concat_in_aggregator
        self.bias = bias
        self.fc_layer = Dense(units,
                              activation=None,
                              use_bias=bias,
                              kernel_initializer='glorot_uniform')

    def call(self, inputs):
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
        self_feat, nbr_feat = inputs
        if self_feat.shape.rank == nbr_feat.shape.rank:
            # reshape to case 1
            self_shape = tf.shape(self_feat)
            ss = tf.strided_slice(self_shape, [0], [-1])
            neigh_shape = tf.concat([ss, [-1], [self_feat.shape[-1]]], axis=0)
            nbr_feat = tf.reshape(nbr_feat, neigh_shape)
        agg_feat = self.aggregate(nbr_feat)
        if self.use_concat_in_aggregator:
            output = tf.concat([self_feat, agg_feat], -1)
        else:
            output = tf.add(self_feat, agg_feat)
        return self.fc_layer(output)

    def aggregate(self, inputs):
        raise NotImplementedError()

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(units=self.units,
                 bias=self.bias,
                 use_concat_in_aggregator=self.use_concat_in_aggregator))
        return config


@export('galileo.tf')
class MeanAggregator(BaseAggregator):
    def aggregate(self, inputs):
        return tf.reduce_mean(inputs, axis=-2)


@export('galileo.tf')
class PoolAggregator(BaseAggregator):
    def __init__(self,
                 units: int,
                 use_concat_in_aggregator: bool = True,
                 bias: bool = False,
                 **kwargs):
        super().__init__(units, use_concat_in_aggregator, bias, **kwargs)

    def build(self, input_shape):
        self_shape, _ = input_shape
        self_shape = tf.TensorShape(self_shape)
        last_dim = tf.compat.dimension_at_index(self_shape, -1)
        assert last_dim
        self.neigh_fc_layer = Dense(last_dim,
                                    activation='relu',
                                    use_bias=self.bias,
                                    kernel_initializer='glorot_uniform')

    def aggregate(self, inputs):
        output = self.neigh_fc_layer(inputs)
        return tf.reduce_mean(output, axis=-2)


MeanPoolAggregator = PoolAggregator
export('galileo.tf').var('MeanPoolAggregator', MeanPoolAggregator)


@export('galileo.tf')
class MaxPoolAggregator(PoolAggregator):
    def aggregate(self, inputs):
        output = self.neigh_fc_layer(inputs)
        return tf.reduce_max(output, axis=-2)


@export('galileo.tf')
class GCNAggregator(Layer):
    def __init__(self,
                 units: int,
                 use_concat_in_aggregator: bool = True,
                 bias: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_concat_in_aggregator = use_concat_in_aggregator
        self.bias = bias
        self.fc_layer = Dense(units,
                              activation=None,
                              use_bias=bias,
                              kernel_initializer='glorot_uniform')

    def call(self, inputs):
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
        self_feat, nbr_feat = inputs
        if self_feat.shape.rank == nbr_feat.shape.rank:
            self_shape = tf.shape(self_feat)
            ss = tf.strided_slice(self_shape, [0], [-1])
            neigh_shape = tf.concat([ss, [-1], [self_feat.shape[-1]]], axis=0)
            nbr_feat = tf.reshape(nbr_feat, neigh_shape)
        if self_feat.shape.rank + 1 == nbr_feat.shape.rank:
            self_feat = tf.expand_dims(self_feat, axis=-2)
        concated_feat = tf.concat([self_feat, nbr_feat], axis=-2)
        mean_feat = tf.reduce_mean(concated_feat, axis=-2)
        output = self.fc_layer(mean_feat)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(units=self.units,
                 bias=self.bias,
                 use_concat_in_aggregator=self.use_concat_in_aggregator))
        return config


aggregators = {
    'mean': MeanAggregator,
    'meanpool': MeanPoolAggregator,
    'maxpool': MaxPoolAggregator,
    'gcn': GCNAggregator,
}


@export('galileo.tf')
def get_aggregator(name):
    return aggregators.get(name)
