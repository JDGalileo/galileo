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

from tensorflow.keras.layers import Embedding
from galileo.platform.export import export
from galileo.framework.tf.python.unsupervised import Unsupervised


@export('galileo.tf')
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
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.shared_embeddings = shared_embeddings

        self._target_encoder = Embedding(embedding_size, embedding_dim)
        if shared_embeddings:
            self._context_encoder = self._target_encoder
        else:
            self._context_encoder = Embedding(embedding_size, embedding_dim)

    def target_encoder(self, inputs):
        return self._target_encoder(inputs)

    def context_encoder(self, inputs):
        return self._context_encoder(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(embedding_size=self.embedding_size,
                 embedding_dim=self.embedding_dim,
                 shared_embeddings=self.shared_embeddings))
        return config
