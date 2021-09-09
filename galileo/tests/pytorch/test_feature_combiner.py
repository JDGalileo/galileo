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
import torch
from galileo.pytorch import FeatureCombiner
from galileo.tests.utils import numpy_equal


def test_feature_combiner_dense():
    fc = FeatureCombiner(dense_feature_dims=[16],
                         sparse_feature_maxs=None,
                         sparse_feature_embedding_dims=None,
                         hidden_dim=None,
                         feature_combiner='concat')
    dense = torch.rand(4, 2, 16)
    res = fc(dense)
    assert numpy_equal([4, 2, 16], res.shape)


def test_feature_combiner_sparse():
    fc = FeatureCombiner(dense_feature_dims=None,
                         sparse_feature_maxs=[32],
                         sparse_feature_embedding_dims=[16],
                         hidden_dim=None,
                         feature_combiner='concat')
    sparse = torch.randint(32, (4, 2, 1))
    res = fc((None, sparse))
    assert numpy_equal([4, 2, 16], res.shape)


def test_feature_combiner_all():
    fc = FeatureCombiner(dense_feature_dims=[32],
                         sparse_feature_maxs=[32],
                         sparse_feature_embedding_dims=[16],
                         hidden_dim=None,
                         feature_combiner='concat')
    dense = torch.rand(4, 2, 32)
    sparse = torch.randint(32, (4, 2, 1))
    res = fc((dense, sparse))
    assert numpy_equal([4, 2, 48], res.shape)
    res = fc([dense, sparse])
    assert numpy_equal([4, 2, 48], res.shape)
    res = fc(dict(dense=dense, sparse=sparse))
    assert numpy_equal([4, 2, 48], res.shape)


def test_feature_combiner_all_multi():
    fc = FeatureCombiner(dense_feature_dims=[32, 64],
                         sparse_feature_maxs=[32, 64, 16],
                         sparse_feature_embedding_dims=[8, 16, 32],
                         hidden_dim=None,
                         feature_combiner='concat')
    dense0 = torch.rand(4, 2, 32)
    dense1 = torch.rand(4, 2, 64)
    sparse = torch.randint(16, (4, 2, 3))
    res = fc(([dense0, dense1], sparse))
    assert numpy_equal([4, 2, 32 + 64 + 8 + 16 + 32], res.shape)


def test_feature_combiner_all_multi2():
    fc = FeatureCombiner(dense_feature_dims=[32, 64],
                         sparse_feature_maxs=[32, 64, 16],
                         sparse_feature_embedding_dims=[8, 16, 32],
                         hidden_dim=16,
                         feature_combiner='add')
    dense0 = torch.rand(4, 2, 32)
    dense1 = torch.rand(4, 2, 64)
    sparse = torch.randint(16, (4, 2, 3))
    res = fc(([dense0, dense1], sparse))
    assert numpy_equal([4, 2, 16], res.shape)
