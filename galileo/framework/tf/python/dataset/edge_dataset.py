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

import random
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from galileo.framework.tf.python.ops import get_entity_dataset
from galileo.platform.export import export


@export('galileo.tf')
class EdgeDataset(dataset_ops.DatasetSource):
    r'''
    edge dataset

    args:
        edge_types:
        batch_size:

    output:
        list of tensor, (src, dst, type) shape:[1, batch_size]
    '''
    def __init__(self, edge_types: list, batch_size: int, **kwargs):
        del kwargs
        if len(edge_types) > 1:
            edge_types = [random.choice(edge_types)]
        self.batch_size = batch_size
        entity_dataset = get_entity_dataset()
        super().__init__(
            entity_dataset(tf.convert_to_tensor(edge_types, dtype=tf.uint8),
                           count=self.batch_size,
                           category='edge',
                           **self._flat_structure))

    @property
    def element_spec(self):
        return (tf.TensorSpec([1, self.batch_size], dtype=tf.int64),
                tf.TensorSpec([1, self.batch_size], dtype=tf.int64),
                tf.TensorSpec([1, self.batch_size], dtype=tf.uint8))
