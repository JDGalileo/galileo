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

import pytest
import tensorflow as tf
import galileo.tf as gt
from galileo.tests import expected_data
from galileo.tests.utils import numpy_equal_unique, numpy_equal_unique_axis


def test_vertex_dataset(prepare_tf_env):
    vertex_dataset = gt.VertexDataset(vertex_type=[0], batch_size=64)
    for ele in vertex_dataset.take(5):
        assert numpy_equal_unique(expected_data.all_vertex_types[0], ele)


def test_edge_dataset(prepare_tf_env):
    edge_dataset = gt.EdgeDataset(edge_types=[1], batch_size=64)
    for ele in edge_dataset.take(5):
        srcs = tf.concat(tf.split(ele[0], 1, axis=0), axis=1)
        dsts = tf.concat(tf.split(ele[1], 1, axis=0), axis=1)
        transpose_edges = tf.transpose(tf.concat([srcs, dsts], axis=0))
        assert numpy_equal_unique_axis(expected_data.all_edges_types[1],
                                       transpose_edges.numpy())
