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
import numpy as np
import galileo as g
from galileo.unify.models import Supervised, Unsupervised, Embedding

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class UnsupervisedTest(Unsupervised):
    def __init__(self, embedding_size, embedding_dim, **kwargs):
        self._target_encoder = Embedding(embedding_size, embedding_dim,
                                         **kwargs)
        self._context_encoder = self._target_encoder

    def target_encoder(self, inputs):
        return self._target_encoder(inputs)

    def context_encoder(self, inputs):
        return self._context_encoder(inputs)


@pytest.mark.parametrize('backend', ('tf', 'pytorch'))
def test_unsupervised(backend):
    ve = UnsupervisedTest(embedding_size=10, embedding_dim=16, backend=backend)
    res = ve(dict(target=[[1], [2], [3]]))
    assert 'ids' in res
    assert 'embeddings' in res
    assert res['ids'].shape == (3, 1)
    assert res['embeddings'].shape == (3, 1, 16)
    res = ve(
        dict(target=[[1], [2], [3]],
             context=[[3], [4], [5]],
             negative=[[5, 6], [7, 6], [7, 8]]))
    assert 'loss' in res
    assert 'mrr' in res


class SupervisedTest(Supervised):
    def __init__(self, *args, **kwargs):
        pass

    def encoder(self, inputs):
        return inputs


@pytest.mark.parametrize('backend', ('tf', 'pytorch'))
def test_supervised(backend):
    m = SupervisedTest(label_dim=3, embedding_dim=16, backend=backend)
    inputs = np.random.randn(1, 16)
    res = m(dict(features=inputs, labels=[[1, 1, 0]]))
    assert 'loss' in res
    assert 'f1_score' in res
    res = m(dict(features=inputs, target=[0, 1, 2]))
    assert 'ids' in res
    assert 'embeddings' in res
    assert res['ids'].shape == (3, )
    assert res['embeddings'].shape == (1, 16)
