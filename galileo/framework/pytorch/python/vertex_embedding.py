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

from galileo.platform.export import export
from galileo.framework.pytorch.python.unsupervised import Unsupervised
from galileo.framework.pytorch.python.layers.embedding import Embedding


@export('galileo.pytorch')
class VertexEmbedding(Unsupervised):
    r'''
    unsupervised model for vertex embedding

    args:
        embedding_size: Size of the vocabulary
        embedding_dim: Dimension of the dense embedding
        shared_embeddings: share target and context embedding
    '''
    def __init__(self,
                 embedding_size,
                 embedding_dim,
                 shared_embeddings=True,
                 **kwargs):
        super().__init__(**kwargs)
        assert embedding_size and embedding_size > 0
        assert embedding_dim and embedding_dim > 0

        self.target_embedding = Embedding(embedding_size, embedding_dim)
        if shared_embeddings:
            self.context_embedding = self.target_embedding
        else:
            self.context_embedding = Embedding(embedding_size, embedding_dim)

    def target_encoder(self, inputs):
        return self.target_embedding(inputs)

    def context_encoder(self, inputs):
        return self.context_embedding(inputs)
