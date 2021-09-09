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
from galileo.tests.utils import numpy_equal
from galileo.tf import ops

# vertex feature
valid_vertex_param_names = 'vertex,features,dims,expected'
invalid_vertex_param_names = 'vertex,features,dims'

valid_vertex_0_params = (pytest.param(
    [1002, 1003, 1004], ['cid', 'price'], [1, 1],
    ([[35], [35], [35]], [[3.5], [3.5], [3.5]])), )

invalid_vertex_0_params = (pytest.param([1002, 1003, 1004],
                                        ['invalid', 'price'], [1, 1]),
                           pytest.param([1102, 1003, 1104], ['cid', 'price'],
                                        [1, 1]))

valid_vertex_1_params = (pytest.param(
    [1006, 1007, 1009], ['age', 'test'], [2, 2],
    ([[55], [55], [55]], [[2001, 2002], [3001, 3002], [5001, 5002]])), )

invalid_vertex_1_params = (pytest.param([1006, 1007, 1009], ['age', 'invalid'],
                                        [2, 2]),
                           pytest.param([1106, 1007, 1109], ['age', 'test'],
                                        [2, 2]))

valid_ids = ('valid', )
invalid_ids = ('feature_invalid', 'vertex_invalid')


@pytest.mark.parametrize(valid_vertex_param_names,
                         valid_vertex_0_params,
                         ids=valid_ids)
def test_valid_collect_vertex_pod_feature_0(prepare_tf_env, vertex, features,
                                            dims, expected):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    res_features = ops.get_pod_feature([vertex_tensor], features, dims,
                                       [tf.int16, tf.float32])
    assert 2 == len(res_features)
    assert numpy_equal([3, 1], tf.shape(res_features[0]).numpy())
    assert numpy_equal([3, 1], tf.shape(res_features[1]).numpy())
    assert numpy_equal(expected[0], res_features[0].numpy())
    assert numpy_equal(expected[1], res_features[1].numpy())


@pytest.mark.parametrize(invalid_vertex_param_names,
                         invalid_vertex_0_params,
                         ids=invalid_ids)
def test_invalid_collect_vertex_pod_feature_0(prepare_tf_env, vertex, features,
                                              dims):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    with pytest.raises(Exception):
        ops.get_pod_feature([vertex_tensor], features, dims,
                            [tf.int16, tf.float32])


@pytest.mark.parametrize(valid_vertex_param_names,
                         valid_vertex_1_params,
                         ids=valid_ids)
def test_valid_collect_vertex_pod_feature_1(prepare_tf_env, vertex, features,
                                            dims, expected):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    res_features = ops.get_pod_feature([vertex_tensor], features, dims,
                                       [tf.int16, tf.int32])
    assert 2 == len(res_features)
    assert numpy_equal([3, 1], tf.shape(res_features[0]).numpy())
    assert numpy_equal([3, 2], tf.shape(res_features[1]).numpy())
    assert numpy_equal(expected[0], res_features[0].numpy())
    assert numpy_equal(expected[1], res_features[1].numpy())


@pytest.mark.parametrize(invalid_vertex_param_names,
                         invalid_vertex_1_params,
                         ids=invalid_ids)
def test_invalid_collect_vertex_pod_feature_1(prepare_tf_env, vertex, features,
                                              dims):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    with pytest.raises(Exception):
        ops.get_pod_feature([vertex_tensor], features, dims,
                            [tf.int16, tf.int32])


# edge feature
valid_edge_param_names = 'src,dst,types,features,dims,expected'
invalid_edge_param_names = 'src,dst,types,features,dims'

valid_edge_params = (pytest.param(
    [1001, 1007, 1009], [1000, 1000, 1000], [1, 1, 1],
    ['discounts', 'purchase_num', 'test'], [2, 2, 2],
    ([[3.5], [3.5], [3.5]], [[333], [333], [333]], [[1001, 1002], [3001, 3002],
                                                    [5001, 5002]])), )

invalid_edge_params = (pytest.param([1001, 1007, 1009], [1000, 1000, 1000],
                                    [1, 1, 1],
                                    ['invalid', 'purchase_num', 'test'],
                                    [2, 2, 2]),
                       pytest.param([1101, 1006, 1007], [1000, 1000, 1000],
                                    [1, 1, 1],
                                    ['discounts', 'purchase_num', 'test'],
                                    [2, 2, 2]))


@pytest.mark.parametrize(valid_edge_param_names,
                         valid_edge_params,
                         ids=valid_ids)
def test_valid_collect_edge_pod_feature(prepare_tf_env, src, dst, types,
                                        features, dims, expected):
    edge_features = ops.get_pod_feature([src, dst, types], features, dims,
                                        [tf.float32, tf.int32, tf.int32])
    assert numpy_equal([3, 1], tf.shape(edge_features[0]).numpy())
    assert numpy_equal([3, 1], tf.shape(edge_features[1]).numpy())
    assert numpy_equal([3, 2], tf.shape(edge_features[2]).numpy())
    assert numpy_equal(expected[0], edge_features[0].numpy())
    assert numpy_equal(expected[1], edge_features[1].numpy())
    assert numpy_equal(expected[2], edge_features[2].numpy())


@pytest.mark.parametrize(invalid_edge_param_names,
                         invalid_edge_params,
                         ids=invalid_ids)
def test_invalid_collect_edge_pod_feature(prepare_tf_env, src, dst, types,
                                          features, dims):
    with pytest.raises(Exception):
        ops.get_pod_feature([src, dst, types], features, dims,
                            [tf.float32, tf.int32, tf.int32])


def test_valid_vertex_type(prepare_tf_env):
    vertex_tensor = tf.constant([
        1000, 1002, 1003, 1004, 1005, 1001, 1006, 1007, 1008, 1009, 1010, 1005,
        1001, 1006, 1005, 1001, 1006, 1007, 1008, 1009, 1010, 1004, 1005, 1001
    ],
                                dtype=tf.int64)
    types = ops.get_pod_feature([vertex_tensor], ['vtype'], [1], [tf.uint8])
    expected_types = [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1],
                      [0], [1], [1], [0], [1], [1], [1], [1], [1], [1], [0],
                      [0], [1]]
    assert numpy_equal(expected_types, types[0].numpy())


def test_empty_inputs(prepare_tf_env):
    vertex = tf.constant(1, dtype=tf.int64, shape=(0, ))
    features = ['cid', 'price']
    dims = [1, 1]
    res = ops.get_pod_feature([vertex], features, dims, [tf.int16, tf.float32])
    assert 2 == len(res)
    assert res[0].dtype == tf.int16
    assert res[0].shape[0] == 0
    assert res[0].shape[1] == dims[0]
    assert res[1].dtype == tf.float32
    assert res[1].shape[0] == 0
    assert res[1].shape[1] == dims[1]

    types = tf.constant(1, dtype=tf.uint8, shape=(0, ))
    features = ['discounts', 'purchase_num', 'test']
    dims = [2, 2, 2]
    res = ops.get_pod_feature([vertex, vertex, types], features, dims,
                              [tf.float32, tf.int32, tf.int32])
    assert 3 == len(res)
    assert res[0].dtype == tf.float32
    assert res[0].shape[0] == 0
    assert res[0].shape[1] == dims[0]
    assert res[1].dtype == tf.int32
    assert res[1].shape[0] == 0
    assert res[1].shape[1] == dims[1]
    assert res[2].dtype == tf.int32
    assert res[2].shape[0] == 0
    assert res[2].shape[1] == dims[2]
