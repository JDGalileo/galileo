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

import os
import pytest
import tensorflow as tf
from galileo.tests import expected_data
from galileo.tests.utils import (
    numpy_equal_unique,
    numpy_equal_unique_axis,
    numpy_equal,
)
from galileo.tf import ops

valid_ids = ('valid_zero', 'valid_one', 'valid_all', 'global')

invalid_ids = ('invalid', 'partial_invalid')


@pytest.mark.parametrize('types,expected', (
    ([0], expected_data.all_vertex_types[0]),
    ([1], expected_data.all_vertex_types[1]),
    ([0, 1], expected_data.all_vertex),
    ([], expected_data.all_vertex),
),
                         ids=valid_ids)
def test_collect_vertex(prepare_tf_env, types, expected):
    count = 100
    vertex = ops.sample_vertices(types, count)[0]
    len_types = len(types) or 1
    assert numpy_equal([len_types, count], tf.shape(vertex).numpy())
    assert numpy_equal_unique(expected, vertex.numpy())


@pytest.mark.parametrize('types', (
    pytest.param([-1, 8]),
    pytest.param([0, 4, 5, 1]),
),
                         ids=invalid_ids)
def test_collect_invalid_vertex(prepare_tf_env, types):
    count = 100
    with pytest.raises(Exception):
        ops.sample_vertices(types, count)


@pytest.mark.parametrize('types,expected', (
    ([0], expected_data.all_edges_types[0]),
    ([1], expected_data.all_edges_types[1]),
    ([0, 1], expected_data.all_edges),
    ([], expected_data.all_edges),
),
                         ids=valid_ids)
def test_collect_edge(prepare_tf_env, types, expected):
    types_tensor = tf.constant(types, dtype=tf.uint8)
    count = 100
    edges = ops.sample_edges(types_tensor, count)
    assert len(edges) == 3
    len_types = len(types) or 1
    assert numpy_equal([len_types, count], tf.shape(edges[0]).numpy())
    assert numpy_equal([len_types, count], tf.shape(edges[1]).numpy())
    srcs = tf.concat(tf.split(edges[0], len_types, axis=0), axis=1)
    dsts = tf.concat(tf.split(edges[1], len_types, axis=0), axis=1)
    transpose_edges = tf.transpose(tf.concat([srcs, dsts], axis=0))
    assert numpy_equal_unique_axis(expected, transpose_edges.numpy())


@pytest.mark.parametrize('types', (
    pytest.param([-1, 8]),
    pytest.param([0, 4, 5, 1]),
),
                         ids=invalid_ids)
def test_collect_invalid_edge(prepare_tf_env, types):
    types_tensor = tf.constant(types, dtype=tf.uint8)
    count = 100
    with pytest.raises(Exception):
        ops.sample_edges(types_tensor, count)
