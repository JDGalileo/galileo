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
import torch
from galileo.tests import expected_data
from galileo.tests.utils import numpy_equal
from galileo.pytorch import ops

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
def test_valid_collect_vertex_pod_feature_0(prepare_pytorch_env, vertex,
                                            features, dims, expected):
    vertex_tensor = torch.tensor(vertex, dtype=torch.int64)
    res_features = ops.get_pod_feature([vertex_tensor], features, dims,
                                     [torch.int16, torch.float32])
    assert 2 == len(res_features)
    assert numpy_equal([3, 1], res_features[0].shape)
    assert numpy_equal([3, 1], res_features[1].shape)
    assert numpy_equal(expected[0], res_features[0].numpy())
    assert numpy_equal(expected[1], res_features[1].numpy())


@pytest.mark.parametrize(invalid_vertex_param_names,
                         invalid_vertex_0_params,
                         ids=invalid_ids)
def test_invalid_collect_vertex_pod_feature_0(prepare_pytorch_env, vertex,
                                              features, dims):
    vertex_tensor = torch.tensor(vertex, dtype=torch.int64)
    res_features = ops.get_pod_feature([vertex_tensor], features, dims,
                                     [torch.int16, torch.float32])
    assert 0 == len(res_features)


@pytest.mark.parametrize(valid_vertex_param_names,
                         valid_vertex_1_params,
                         ids=valid_ids)
def test_valid_collect_vertex_pod_feature_1(prepare_pytorch_env, vertex,
                                            features, dims, expected):
    vertex_tensor = torch.tensor(vertex, dtype=torch.int64)
    res_features = ops.get_pod_feature([vertex_tensor], features, dims,
                                     [torch.int16, torch.int32])
    assert 2 == len(res_features)
    assert numpy_equal([3, 1], res_features[0].shape)
    assert numpy_equal([3, 2], res_features[1].shape)
    assert numpy_equal(expected[0], res_features[0].numpy())
    assert numpy_equal(expected[1], res_features[1].numpy())


@pytest.mark.parametrize(invalid_vertex_param_names,
                         invalid_vertex_1_params,
                         ids=invalid_ids)
def test_invalid_collect_vertex_pod_feature_1(prepare_pytorch_env, vertex,
                                              features, dims):
    vertex_tensor = torch.tensor(vertex, dtype=torch.int64)
    res_features = ops.get_pod_feature([vertex_tensor], features, dims,
                                     [torch.int16, torch.int32])
    assert 0 == len(res_features)


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
def test_valid_collect_edge_pod_feature(prepare_pytorch_env, src, dst, types,
                                        features, dims, expected):
    edge_features = ops.get_pod_feature(
        [src, dst, types], features, dims,
        [torch.float32, torch.int32, torch.int32])
    assert numpy_equal([3, 1], edge_features[0].shape)
    assert numpy_equal([3, 1], edge_features[1].shape)
    assert numpy_equal([3, 2], edge_features[2].shape)
    assert numpy_equal(expected[0], edge_features[0].numpy())
    assert numpy_equal(expected[1], edge_features[1].numpy())
    assert numpy_equal(expected[2], edge_features[2].numpy())


@pytest.mark.parametrize(invalid_edge_param_names,
                         invalid_edge_params,
                         ids=invalid_ids)
def test_invalid_collect_edge_pod_feature(prepare_pytorch_env, src, dst, types,
                                          features, dims):
    edge_features = ops.get_pod_feature(
        [src, dst, types], features, dims,
        [torch.float32, torch.int32, torch.int32])
    assert 0 == len(edge_features)


def test_valid_vertex_type(prepare_pytorch_env):
    vertex_tensor = torch.tensor([
        1000, 1002, 1003, 1004, 1005, 1001, 1006, 1007, 1008, 1009, 1010, 1005,
        1001, 1006, 1005, 1001, 1006, 1007, 1008, 1009, 1010, 1004, 1005, 1001
    ],
                                 dtype=torch.int64)
    types = ops.get_pod_feature([vertex_tensor], ['vtype'], [1], [torch.uint8])
    expected_types = [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1],
                      [0], [1], [1], [0], [1], [1], [1], [1], [1], [1], [0],
                      [0], [1]]
    assert numpy_equal(expected_types, types[0].numpy())
