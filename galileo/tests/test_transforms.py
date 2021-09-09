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
import galileo.pytorch as gp
import tensorflow as tf
import galileo.tf as gt
from galileo.tests.utils import numpy_equal

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# batch size=5 num nodes=10
# indices [5, 9]
fanouts = [2, 3]
indices = [
    [2, 7, 4, 4, 2, 1, 3, 8, 1],
    [8, 9, 2, 3, 6, 4, 4, 4, 0],
    [4, 7, 1, 0, 1, 3, 0, 4, 1],
    [4, 2, 6, 1, 5, 1, 4, 0, 5],
    [3, 2, 2, 9, 8, 8, 0, 1, 7],
]
expect_no_sort = [
    [
        2, 8, 4, 4, 3, 2, 8, 4, 4, 3, 7, 9, 7, 2, 2, 7, 9, 7, 2, 2, 7, 9, 7, 2,
        2, 4, 2, 1, 6, 2, 4, 2, 1, 6, 2, 4, 2, 1, 6, 2
    ],
    [
        7, 9, 7, 2, 2, 4, 2, 1, 6, 2, 4, 3, 0, 1, 9, 2, 6, 1, 5, 8, 1, 4, 3, 1,
        8, 3, 4, 0, 4, 0, 8, 4, 4, 0, 1, 1, 0, 1, 5, 7
    ],
]
expect_sort = [
    [
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4,
        4, 4, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9, 9
    ],
    [
        0, 4, 1, 7, 4, 1, 9, 5, 8, 1, 8, 4, 0, 4, 1, 0, 7, 2, 2, 7, 2, 1, 6, 3,
        8, 1, 4, 0, 5, 4, 0, 2, 1, 1, 3, 9, 2, 3, 6, 4
    ],
]
expect_target = [2, 8, 4, 4, 3]


@pytest.mark.parametrize('sort_indices', (False, True))
def test_relation_transform_tf(sort_indices):
    rt = gt.RelationTransform(fanouts,
                              sort_indices=sort_indices,
                              sort_stable=True)
    res = rt.transform(
        dict(indices=tf.convert_to_tensor(indices),
             edge_weight=tf.random.normal((5, 9))))
    assert list(res.keys()) == [
        'relation_indices',
        'relation_weight',
        'target_indices',
    ]
    assert res['relation_indices'].shape == [2, 40]
    expect = expect_sort if sort_indices else expect_no_sort
    assert numpy_equal(res['relation_indices'].numpy(), expect)
    assert res['relation_weight'].shape == [40, 1]
    assert res['target_indices'].shape == [5]
    assert numpy_equal(res['target_indices'].numpy(), expect_target)


@pytest.mark.parametrize('sort_indices', (False, True))
def test_relation_transform_pytorch(sort_indices):
    rt = gp.RelationTransform(fanouts,
                              sort_indices=sort_indices,
                              sort_stable=True)
    res = rt.transform(
        dict(indices=torch.tensor(indices), edge_weight=torch.randn(5, 9)))
    assert list(res.keys()) == [
        'relation_indices',
        'relation_weight',
        'target_indices',
    ]
    assert numpy_equal(res['relation_indices'].shape, [2, 40])
    if sort_indices:
        # sort is not stable in pytorch
        assert numpy_equal(res['relation_indices'][0].numpy(), expect_sort[0])
    else:
        assert numpy_equal(res['relation_indices'].numpy(), expect_no_sort)
    assert numpy_equal(res['relation_weight'].shape, [40, 1])
    assert numpy_equal(res['target_indices'].shape, [5])
    assert numpy_equal(res['target_indices'].numpy(), expect_target)
