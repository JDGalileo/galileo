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
from galileo.tests.utils import (
    numpy_equal_unique,
    numpy_equal_unique_axis,
    numpy_equal,
)
from galileo.pytorch import ops

valid_ids = ('valid_zero', 'valid_one', 'valid_all', 'global')

invalid_ids = ('invalid', 'partial_invalid')


@pytest.mark.parametrize('types,expected', (
    ([0], expected_data.all_vertex_types[0]),
    ([1], expected_data.all_vertex_types[1]),
    ([0, 1], expected_data.all_vertex),
    ([], expected_data.all_vertex),
),
                         ids=valid_ids)
def test_collect_vertex(prepare_pytorch_env, types, expected):
    count = 100
    vertex = ops.sample_vertices(types, count)[0]
    len_types = len(types) or 1
    assert numpy_equal([len_types, count], vertex.shape)
    assert numpy_equal_unique(expected, vertex.numpy())


@pytest.mark.parametrize('types', (
    pytest.param([-1, 8]),
    pytest.param([0, 4, 5, 1]),
),
                         ids=invalid_ids)
def test_collect_invalid_vertex(prepare_pytorch_env, types):
    count = 100
    vertex = ops.sample_vertices(types, count)
    assert 0 == len(vertex)


@pytest.mark.parametrize('types,expected', (
    ([0], expected_data.all_edges_types[0]),
    ([1], expected_data.all_edges_types[1]),
    ([0, 1], expected_data.all_edges),
    ([], expected_data.all_edges),
),
                         ids=valid_ids)
def test_collect_edge(prepare_pytorch_env, types, expected):
    types_tensor = torch.tensor(types, dtype=torch.uint8)
    count = 100
    edges = ops.sample_edges(types_tensor, count)
    assert len(edges) == 3
    len_types = len(types) or 1
    assert numpy_equal([len_types, count], edges[0].shape)
    assert numpy_equal([len_types, count], edges[1].shape)
    srcs = torch.cat(torch.split(edges[0], [1] * len_types, dim=0), dim=1)
    dsts = torch.cat(torch.split(edges[1], [1] * len_types, dim=0), dim=1)
    transpose_edges = torch.transpose(torch.cat([srcs, dsts], dim=0), 0, 1)
    assert numpy_equal_unique_axis(expected, transpose_edges.numpy())


@pytest.mark.parametrize('types', (
    pytest.param([-1, 8]),
    pytest.param([0, 4, 5, 1]),
),
                         ids=invalid_ids)
def test_collect_invalid_edge(prepare_pytorch_env, types):
    types_tensor = torch.tensor(types, dtype=torch.uint8)
    count = 100
    edges = ops.sample_edges(types_tensor, count)
    assert 0 == len(edges)
