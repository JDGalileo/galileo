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
class VertexDataset(dataset_ops.DatasetSource):
    r'''
    vertex dataset

    args:
        vertex_type:
        batch_size:

    output:
        [1, batch_size] tensor
    '''
    def __init__(self, vertex_type: list, batch_size: int, **kwargs):
        del kwargs
        if len(vertex_type) > 1:
            vertex_type = [random.choice(vertex_type)]
        self.batch_size = batch_size
        entity_dataset = get_entity_dataset()
        super().__init__(
            entity_dataset(tf.convert_to_tensor(vertex_type, dtype=tf.uint8),
                           count=self.batch_size,
                           category='vertex',
                           **self._flat_structure))

    @property
    def element_spec(self):
        return tf.TensorSpec([1, self.batch_size], dtype=tf.int64)
