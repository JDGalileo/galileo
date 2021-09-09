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
from galileo.tests.utils import numpy_equal_unique, numpy_equal
from galileo.tf import ops

valid_ids = ('global', 'valid_zero', 'valid_one', 'valid_multi')
valid_topk_ids = ('global', 'valid_one', 'valid_multi')
invalid_ids = ('no_neighbor', 'invalid_types')

valid_params = (
    ([1001], [], [1000, 1003, 1004, 1005], [3.5, 5.5]),
    ([1001], [0], [1003, 1004, 1005], [3.5]),
    ([1001], [1], [1000], [5.5]),
    ([1000, 1002, 1006, 1009], [0, 1], [1001, 1001, 1000, 1000], [3.5, 5.5]),
)

valid_topk_params = (
    ([1001], [], [1000], [5.5]),
    ([1001], [1], [1000], [5.5]),
    ([1000, 1002, 1006, 1009], [0, 1], [1001, 1001, 1000,
                                        1000], [3.5, 3.5, 5.5, 5.5]),
)

invalid_params = (
    pytest.param([1003, 1004, 1005], []),
    pytest.param([1001], [-1, 5]),
)


@pytest.mark.parametrize('vertex,types,expected_nbr,expected_weight',
                         valid_params,
                         ids=valid_ids)
def test_valid_sample_neighbor_with_weight(prepare_tf_env, vertex, types,
                                           expected_nbr, expected_weight):
    count = 100
    is_weight = True
    neighbors, weights = ops.sample_neighbors(vertex, types, count, is_weight)
    assert numpy_equal([len(vertex), count], tf.shape(neighbors).numpy())
    assert numpy_equal([len(vertex), count], tf.shape(weights).numpy())
    assert numpy_equal_unique(expected_nbr, neighbors.numpy())
    assert numpy_equal_unique(expected_weight, weights.numpy())


@pytest.mark.parametrize('vertex,types', invalid_params, ids=invalid_ids)
def test_invalid_sample_neighbor_with_weight(prepare_tf_env, vertex, types):
    count = 100
    is_weight = True
    with pytest.raises(Exception):
        ops.sample_neighbors(vertex, types, count, is_weight)


@pytest.mark.parametrize('vertex,types,expected_nbr,expected_weight',
                         valid_params,
                         ids=valid_ids)
def test_valid_sample_neighbor_without_weight(prepare_tf_env, vertex, types,
                                              expected_nbr, expected_weight):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    types_tensor = tf.constant(types, dtype=tf.uint8)
    count = 100
    is_weight = False
    neighbors = ops.sample_neighbors(vertex_tensor, types_tensor, count,
                                     is_weight)[0]
    assert numpy_equal([len(vertex), count], tf.shape(neighbors).numpy())
    assert numpy_equal_unique(expected_nbr, neighbors.numpy())


@pytest.mark.parametrize('vertex,types', invalid_params, ids=invalid_ids)
def test_invalid_sample_neighbor_without_weight(prepare_tf_env, vertex, types):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    types_tensor = tf.constant(types, dtype=tf.uint8)
    count = 100
    is_weight = False
    with pytest.raises(Exception):
        ops.sample_neighbors(vertex_tensor, types_tensor, count, is_weight)[0]


@pytest.mark.parametrize('vertex,types,expected_nbr,expected_weight',
                         valid_topk_params,
                         ids=valid_topk_ids)
def test_valid_topk_neighbor_with_weight(prepare_tf_env, vertex, types,
                                         expected_nbr, expected_weight):
    count = 1
    is_weight = True
    neighbors = ops.get_topk_neighbors(vertex, types, count, is_weight)
    assert numpy_equal([len(vertex), 1], tf.shape(neighbors[0]).numpy())
    assert numpy_equal([len(vertex), 1], tf.shape(neighbors[1]).numpy())
    assert numpy_equal(expected_nbr, tf.reshape(neighbors[0], [-1]).numpy())
    assert numpy_equal(expected_weight, tf.reshape(neighbors[1], [-1]).numpy())


@pytest.mark.parametrize('vertex,types', invalid_params, ids=invalid_ids)
def test_invalid_topk_neighbor_with_weight(prepare_tf_env, vertex, types):
    count = 2
    is_weight = True
    with pytest.raises(Exception):
        ops.get_topk_neighbors(vertex, types, count, is_weight)


@pytest.mark.parametrize('vertex,types,expected_nbr,expected_weight',
                         valid_topk_params,
                         ids=valid_topk_ids)
def test_valid_topk_neighbor_without_weight(prepare_tf_env, vertex, types,
                                            expected_nbr, expected_weight):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    types_tensor = tf.constant(types, dtype=tf.uint8)
    count = 1
    is_weight = False
    neighbors = ops.get_topk_neighbors(vertex_tensor, types_tensor, count,
                                       is_weight)
    assert numpy_equal([len(vertex), 1], tf.shape(neighbors[0]).numpy())
    assert numpy_equal(expected_nbr, tf.reshape(neighbors[0], [-1]).numpy())


@pytest.mark.parametrize('vertex,types', invalid_params, ids=invalid_ids)
def test_invalid_topk_neighbor_without_weight(prepare_tf_env, vertex, types):
    count = 2
    is_weight = False
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    types_tensor = tf.constant(types, dtype=tf.uint8)
    with pytest.raises(Exception):
        ops.get_topk_neighbors(vertex_tensor, types_tensor, count, is_weight)


@pytest.mark.parametrize('vertex,types,expected_nbr,expected_weight',
                         valid_params,
                         ids=valid_ids)
def test_valid_full_neighbor_with_weight(prepare_tf_env, vertex, types,
                                         expected_nbr, expected_weight):
    is_weight = True
    neighbors = ops.get_full_neighbors(vertex, types, is_weight)
    assert 3 == len(neighbors)
    assert numpy_equal_unique(expected_nbr, neighbors[0].numpy())
    assert numpy_equal_unique(expected_weight, neighbors[1].numpy())


@pytest.mark.parametrize('vertex,types', invalid_params, ids=invalid_ids)
def test_invalid_full_neighbor_with_weight(prepare_tf_env, vertex, types):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    types_tensor = tf.constant(types, dtype=tf.uint8)
    is_weight = True
    with pytest.raises(Exception):
        ops.get_full_neighbors(vertex_tensor, types_tensor, is_weight)


@pytest.mark.parametrize('vertex,types,expected_nbr,expected_weight',
                         valid_params,
                         ids=valid_ids)
def test_valid_full_neighbor_without_weight(prepare_tf_env, vertex, types,
                                            expected_nbr, expected_weight):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    types_tensor = tf.constant(types, dtype=tf.uint8)
    is_weight = False
    neighbors = ops.get_full_neighbors(vertex_tensor, types_tensor, is_weight)
    assert 2 == len(neighbors)
    assert numpy_equal_unique(expected_nbr, neighbors[0].numpy())


@pytest.mark.parametrize('vertex,types', invalid_params, ids=invalid_ids)
def test_invalid_full_neighbor_without_weight(prepare_tf_env, vertex, types):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    types_tensor = tf.constant(types, dtype=tf.uint8)
    is_weight = False
    with pytest.raises(Exception):
        ops.get_full_neighbors(vertex_tensor, types_tensor, is_weight)


def test_empty_inputs(prepare_tf_env):
    vertex_tensor = tf.constant(1, dtype=tf.int64, shape=(0, ))
    types_tensor = tf.constant([0], dtype=tf.uint8)

    res = ops.sample_neighbors(vertex_tensor, types_tensor, 5, False)
    assert 1 == len(res)
    assert res[0].dtype == tf.int64
    assert res[0].shape[0] == 0
    assert res[0].shape[1] == 5

    res = ops.get_topk_neighbors(vertex_tensor, types_tensor, 2, False)
    assert 1 == len(res)
    assert res[0].dtype == tf.int64
    assert res[0].shape[0] == 0
    assert res[0].shape[1] == 2

    res = ops.get_full_neighbors(vertex_tensor, types_tensor, False)
    assert 2 == len(res)
    assert res[0].dtype == tf.int64
    assert res[0].shape[0] == 0
    assert res[1].dtype == tf.int32
    assert res[1].shape[0] == 0
    assert res[1].shape[1] == 2

    res = ops.sample_neighbors(vertex_tensor, types_tensor, 5, True)
    assert 2 == len(res)
    assert res[0].dtype == tf.int64
    assert res[0].shape[0] == 0
    assert res[0].shape[1] == 5
    assert res[1].dtype == tf.float32
    assert res[1].shape[0] == 0
    assert res[1].shape[1] == 5

    res = ops.get_topk_neighbors(vertex_tensor, types_tensor, 2, True)
    assert 2 == len(res)
    assert res[0].dtype == tf.int64
    assert res[0].shape[0] == 0
    assert res[0].shape[1] == 2
    assert res[1].dtype == tf.float32
    assert res[1].shape[0] == 0
    assert res[1].shape[1] == 2

    res = ops.get_full_neighbors(vertex_tensor, types_tensor, True)
    assert 3 == len(res)
    assert res[0].dtype == tf.int64
    assert res[0].shape[0] == 0
    assert res[1].dtype == tf.float32
    assert res[1].shape[0] == 0
    assert res[2].dtype == tf.int32
    assert res[2].shape[0] == 0
    assert res[2].shape[1] == 2
