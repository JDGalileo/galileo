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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


@pytest.mark.parametrize('name', ('mean', 'meanpool', 'maxpool', 'gcn'))
@pytest.mark.parametrize('use_concat_in_aggregator', (True, False))
def test_aggregators_pt(name, use_concat_in_aggregator):
    agg = gp.get_aggregator(name)
    ag = agg(4, 5, use_concat_in_aggregator=use_concat_in_aggregator)
    # case 1: not same shape, neighbor size is 6
    src = torch.randn(4, 2, 4)
    neigh = torch.randn(4, 2, 6, 4)
    output = ag((src, neigh))
    assert output.shape == (4, 2, 5)
    # case 2: same shape, neighbor size is 3
    src = torch.randn(4, 2, 2, 4)
    neigh = torch.randn(4, 2, 6, 4)
    output = ag((src, neigh))
    assert output.shape == (4, 2, 2, 5)
    # case 3: same shape, neighbor size is 3
    src = torch.randn(4, 2, 1, 4)
    neigh = torch.randn(4, 2, 3, 4)
    output = ag((src, neigh))
    assert output.shape == (4, 2, 1, 5)


@pytest.mark.parametrize('name', ('mean', 'meanpool', 'maxpool', 'gcn'))
@pytest.mark.parametrize('use_concat_in_aggregator', (True, False))
def test_aggregators_tf(name, use_concat_in_aggregator):
    agg = gt.get_aggregator(name)
    ag = agg(5, use_concat_in_aggregator=use_concat_in_aggregator)
    # case 1: not same shape, neighbor size is 6
    src = tf.random.normal((4, 2, 4))
    neigh = tf.random.normal((4, 2, 6, 4))
    output = ag((src, neigh))
    assert output.shape == (4, 2, 5)
    # case 2: same shape, neighbor size is 3
    src = tf.random.normal((4, 2, 2, 4))
    neigh = tf.random.normal((4, 2, 6, 4))
    output = ag((src, neigh))
    assert output.shape == (4, 2, 2, 5)
    # case 3: same shape, neighbor size is 3
    src = tf.random.normal((4, 2, 1, 4))
    neigh = tf.random.normal((4, 2, 3, 4))
    output = ag((src, neigh))
    assert output.shape == (4, 2, 1, 5)


sparse_aggregators = (
    'mean',
    'mean-1k',
    'mean-2k',
    'meanpool',
    'maxpool',
    'gcn',
)
indices = [
    [
        2, 8, 4, 4, 3, 2, 8, 4, 4, 3, 7, 9, 7, 2, 2, 7, 9, 7, 2, 2, 7, 9, 7, 2,
        2, 4, 2, 1, 6, 2, 4, 2, 1, 6, 2, 4, 2, 1, 6, 2
    ],
    [
        7, 9, 7, 2, 2, 4, 2, 1, 6, 2, 4, 3, 0, 1, 9, 2, 6, 1, 5, 8, 1, 4, 3, 1,
        8, 3, 4, 0, 4, 0, 8, 4, 4, 0, 1, 1, 0, 1, 5, 7
    ],
]


@pytest.mark.parametrize('name', sparse_aggregators)
@pytest.mark.parametrize('use_concat_in_aggregator', (True, False))
def test_aggregators_sparse_pt(name, use_concat_in_aggregator):
    agg = gp.get_aggregator_sparse(name)
    ag = agg(6, 8, use_concat_in_aggregator=use_concat_in_aggregator)
    feature = torch.randn(10, 6)
    feature_n = feature.index_select(0, torch.tensor(indices[1]))
    output = ag((feature, feature_n, torch.tensor(indices[0])))
    assert output.shape == (10, 8)


@pytest.mark.parametrize('name', sparse_aggregators)
@pytest.mark.parametrize('use_concat_in_aggregator', (True, False))
def test_aggregators_sparse_tf(name, use_concat_in_aggregator):
    agg = gt.get_aggregator_sparse(name)
    ag = agg(8, use_concat_in_aggregator=use_concat_in_aggregator)
    feature = tf.random.normal((10, 6))
    feature_n = tf.gather(feature, indices[1])
    output = ag((feature, feature_n, indices[0]))
    assert output.shape == (10, 8)
