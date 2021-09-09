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

valid_params = (([1000, 1002], [[0], [0, 1]], [2, 3], [2, 9]),
                ([1000, 1002], [[0], []], [2, 3], [2, 9]))

invalid_params = (pytest.param([1200, 1002], [[0, -1, 10], [5]], [2, 3]), )

valid_ids = ['valid', 'global']
invalid_ids = [
    'invalid',
]


@pytest.mark.parametrize('vertex,metapath,hops,expected',
                         valid_params,
                         ids=valid_ids)
def test_valid_mulit_hop_with_weight(prepare_tf_env, vertex, metapath, hops,
                                     expected):
    counts = hops
    is_weight = True
    sequence_weight = ops.sample_seq_by_multi_hop(vertex, metapath, counts,
                                                  is_weight)
    assert 2 == len(sequence_weight)
    assert numpy_equal(expected, tf.shape(sequence_weight[0]).numpy())
    assert numpy_equal(expected, tf.shape(sequence_weight[1]).numpy())


@pytest.mark.parametrize('vertex,metapath,hops',
                         invalid_params,
                         ids=invalid_ids)
def test_invalid_mulit_hop_with_weight(prepare_tf_env, vertex, metapath, hops):
    counts = hops
    is_weight = True
    with pytest.raises(Exception):
        ops.sample_seq_by_multi_hop(vertex, metapath, counts, is_weight)


@pytest.mark.parametrize('vertex,metapath,hops,expected',
                         valid_params,
                         ids=valid_ids)
def test_valid_mulit_hop_without_weight(prepare_tf_env, vertex, metapath, hops,
                                        expected):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    counts = hops
    is_weight = False
    sequence = ops.sample_seq_by_multi_hop(vertex_tensor, metapath, counts,
                                           is_weight)
    assert 1 == len(sequence)
    assert numpy_equal(expected, tf.shape(sequence[0]).numpy())


@pytest.mark.parametrize('vertex,metapath,hops',
                         invalid_params,
                         ids=invalid_ids)
def test_invalid_mulit_hop_without_weight(prepare_tf_env, vertex, metapath,
                                          hops):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    counts = hops
    is_weight = False
    with pytest.raises(Exception):
        ops.sample_seq_by_multi_hop(vertex_tensor, metapath, counts, is_weight)


@pytest.mark.parametrize('vertex,metapath,hops,expected',
                         valid_params,
                         ids=valid_ids)
@pytest.mark.parametrize('p,q', ((1.0, 1.0), (0.5, 2.0)))
def test_valid_random_walk_seq(prepare_tf_env, vertex, metapath, hops,
                               expected, q, p):
    repetition = 2
    sequence = ops.sample_seq_by_random_walk(vertex, metapath, repetition, p,
                                             q)
    assert 4 == len(sequence)
    assert numpy_equal([len(vertex) * 2, len(metapath) + 1],
                       tf.shape(sequence).numpy())


@pytest.mark.parametrize('vertex,metapath,hops',
                         invalid_params,
                         ids=invalid_ids)
@pytest.mark.parametrize('p,q', ((1.0, 1.0), (0.5, 2.0)))
def test_invalid_random_walk_seq(prepare_tf_env, vertex, metapath, hops, p, q):
    repetition = 2
    with pytest.raises(Exception):
        ops.sample_seq_by_random_walk(vertex, metapath, repetition, p, q)


@pytest.mark.parametrize('vertex,metapath,hops,expected',
                         valid_params,
                         ids=valid_ids)
@pytest.mark.parametrize('p,q', ((1.0, 1.0), (0.5, 2.0)))
def test_valid_random_walk_pairs(prepare_tf_env, vertex, metapath, hops,
                                 expected, p, q):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    repetition = 2
    context_size = 2
    pairs = ops.sample_pairs_by_random_walk(vertex_tensor, metapath,
                                            repetition, context_size, p, q)
    assert 24 == len(pairs)
    assert numpy_equal([24, 2], tf.shape(pairs).numpy())


@pytest.mark.parametrize('vertex,metapath,hops',
                         invalid_params,
                         ids=invalid_ids)
@pytest.mark.parametrize('p,q', ((1.0, 1.0), (0.5, 2.0)))
def test_invalid_random_walk_pairs(prepare_tf_env, vertex, metapath, hops, p,
                                   q):
    vertex_tensor = tf.constant(vertex, dtype=tf.int64)
    repetition = 2
    context_size = 2
    with pytest.raises(Exception):
        ops.sample_pairs_by_random_walk(vertex_tensor, metapath, repetition,
                                        context_size, p, q)


def test_empty_inputs(prepare_tf_env):
    vertex = tf.constant(1, dtype=tf.int64, shape=(0, ))
    metapath = [[0], [0, 1]]
    counts = [2, 3]
    res = ops.sample_seq_by_multi_hop(vertex, metapath, counts, True)
    assert 2 == len(res)
    assert res[0].dtype == tf.int64
    assert res[0].shape[0] == 0
    assert res[0].shape[1] == 9
    assert res[1].dtype == tf.float32
    assert res[1].shape[0] == 0
    assert res[1].shape[1] == 9
    res = ops.sample_seq_by_multi_hop(vertex, metapath, counts, False)
    assert 1 == len(res)
    assert res[0].dtype == tf.int64
    assert res[0].shape[0] == 0
    assert res[0].shape[1] == 9
    res = ops.sample_seq_by_random_walk(vertex, metapath, 2, 1.0, 1.0)
    assert res.dtype == tf.int64
    assert res.shape[0] == 0
    assert res.shape[1] == 3
    res = ops.sample_seq_by_random_walk(vertex, metapath, 2, 2.0, 0.5)
    assert res.dtype == tf.int64
    assert res.shape[0] == 0
    assert res.shape[1] == 3
    res = ops.sample_pairs_by_random_walk(vertex, metapath, 2, 2, 1.0, 1.0)
    assert res.dtype == tf.int64
    assert res.shape[0] == 0
    assert res.shape[1] == 2
    res = ops.sample_pairs_by_random_walk(vertex, metapath, 2, 2, 2.0, 0.5)
    assert res.dtype == tf.int64
    assert res.shape[0] == 0
    assert res.shape[1] == 2


def test_graph_mode_metapath(prepare_tf_env):
    metapaths = [[[0], [0, 1]], [[0], [0, 1]]]
    counts = [2, 3]
    with tf.Graph().as_default():
        vertex = tf.constant([1000], dtype=tf.int64)
        for mp in metapaths:
            ops.sample_seq_by_multi_hop(vertex, mp, counts, True)

    def tr(v):
        v = tf.cast(v, dtype=tf.int64)
        for mp in metapaths:
            res = ops.sample_seq_by_multi_hop(v, mp, counts, True)
        return res

    ds = tf.data.Dataset.from_tensor_slices([1000, 1002]).map(tr)
    res = [d for d in ds]
    assert len(res) == 2
