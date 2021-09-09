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
class BaseAggregatorSparse(Layer):
    '''
    base aggregator aggregates target and neigbor feature

    args:
        output_dim: dimensionality of the output space
        use_concat_in_aggregator: concat target and neigbor feature
        bias: bias
    '''
    def __init__(self,
                 output_dim: int,
                 use_concat_in_aggregator: bool = True,
                 bias: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.use_concat_in_aggregator = use_concat_in_aggregator
        self.bias = bias

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(output_dim=self.output_dim,
                 bias=self.bias,
                 use_concat_in_aggregator=self.use_concat_in_aggregator))
        return config


@export('galileo.tf')
class MeanAggregatorSparse(BaseAggregatorSparse):
    r'''
    \brief mean or mean-1k

    \f$ h_v=\sigma(W(h_v^{self}|mean(h_{N(v)}))) $\f
    '''
    def build(self, input_shape):
        self.fc_layer = Dense(self.output_dim,
                              activation=None,
                              use_bias=self.bias,
                              kernel_initializer='glorot_uniform')

    def call(self, inputs):
        r'''
        inputs: (self_feat, nbr_feat, relation_src_indices)
        '''
        self_feat = inputs[0]
        agg_feat = self.aggregate(inputs)
        if self.use_concat_in_aggregator:
            output = tf.concat([self_feat, agg_feat], -1)
        else:
            output = tf.add(self_feat, agg_feat)
        return self.fc_layer(output)

    def aggregate(self, inputs):
        self_feat, nbr_feat, relation_src_indices = inputs
        num_nodes = tf.shape(self_feat)[0]
        return tf.math.unsorted_segment_mean(nbr_feat,
                                             relation_src_indices,
                                             num_segments=num_nodes)


@export('galileo.tf')
class Mean2kAggregatorSparse(MeanAggregatorSparse):
    r'''
    \brief mean-2k

    \f$ h_v=\sigma(W_1h_v^{self}|W_2mean(h_{N(v)}))) $\f
    '''
    def build(self, input_shape):
        if self.use_concat_in_aggregator:
            self_units = self.output_dim // 2
            nbr_units = self.output_dim - self_units
        else:
            self_units = nbr_units = self.output_dim
        self.self_fc_layer = Dense(self_units,
                                   activation=None,
                                   use_bias=self.bias,
                                   kernel_initializer='glorot_uniform')
        self.nbr_fc_layer = Dense(nbr_units,
                                  activation=None,
                                  use_bias=self.bias,
                                  kernel_initializer='glorot_uniform')

    def call(self, inputs):
        r'''
        inputs: (self_feat, nbr_feat, relation_src_indices)
        '''
        self_feat, nbr_feat, relation_src_indices = inputs
        self_out = self.self_fc_layer(self_feat)
        agg_feat = super().aggregate(inputs)
        nbr_out = self.nbr_fc_layer(agg_feat)
        if self.use_concat_in_aggregator:
            output = tf.concat([self_out, nbr_out], -1)
        else:
            output = tf.add(self_out, nbr_out)
        return output


@export('galileo.tf')
class PoolAggregatorSparse(MeanAggregatorSparse):
    r'''
    \brief mean pool

    \f$ h_v=\sigma(W(h_v^{self}|mean(\sigma(W_ph_{N(v)})))) $\f
    '''
    def build(self, input_shape):
        super().build(input_shape)
        nbr_shape = tf.TensorShape(input_shape[1])
        last_dim = tf.compat.dimension_at_index(nbr_shape, -1)
        assert last_dim
        self.neigh_fc_layer = Dense(last_dim,
                                    activation='relu',
                                    use_bias=self.bias,
                                    kernel_initializer='glorot_uniform')

    def aggregate(self, inputs):
        output = self.neigh_fc_layer(inputs[1])
        return super().aggregate((inputs[0], output, inputs[2]))


MeanPoolAggregatorSparse = PoolAggregatorSparse
export('galileo.tf').var('MeanPoolAggregatorSparse', MeanPoolAggregatorSparse)


@export('galileo.tf')
class MaxPoolAggregatorSparse(PoolAggregatorSparse):
    def aggregate(self, inputs):
        self_feat, nbr_feat, relation_src_indices = inputs
        nbr_feat = self.neigh_fc_layer(nbr_feat)
        num_nodes = tf.shape(self_feat)[0]
        return tf.math.unsorted_segment_max(nbr_feat,
                                            relation_src_indices,
                                            num_segments=num_nodes)


@export('galileo.tf')
class GCNAggregatorSparse(BaseAggregatorSparse):
    def build(self, input_shape):
        self.fc_layer = Dense(self.output_dim,
                              activation=None,
                              use_bias=self.bias,
                              kernel_initializer='glorot_uniform')

    def call(self, inputs):
        self_feat, nbr_feat, relation_src_indices = inputs
        num_nodes = tf.shape(self_feat)[0]
        nbr_sum = tf.math.unsorted_segment_sum(nbr_feat, relation_src_indices,
                                               num_nodes)
        degs = tf.math.unsorted_segment_sum(tf.ones_like(nbr_feat),
                                            relation_src_indices, num_nodes)
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


@export('galileo.tf')
def get_aggregator_sparse(name):
    return aggregators.get(name)
