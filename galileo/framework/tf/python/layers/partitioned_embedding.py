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
from tensorflow.keras.layers import Embedding
from galileo.platform.export import export


@export('galileo.tf')
class PartitionedEmbedding(Embedding):
    '''
    partition embedding for ParameterServerStrategy in estimator

    args:
        units: dimensionality of the output space
        use_concat_in_aggregator: concat target and neigbor feature
    '''
    def __init__(self, input_dim: int, output_dim: int, num_of_ps: int,
                 **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.num_of_ps = num_of_ps
        self.partitioner = tf.compat.v1.fixed_size_partitioner(
            num_shards=num_of_ps, axis=0)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            partitioner=self.partitioner,
            getter=tf.compat.v1.get_variable)
        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update(dict(num_of_ps=self.num_of_ps))
        return config
