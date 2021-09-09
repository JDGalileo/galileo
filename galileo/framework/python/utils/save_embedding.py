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
import numpy as np
from galileo.platform.log import log
from galileo.platform.export import export


@export()
def save_embedding(ids, embedding, save_embedding_dir, worker_id=0):
    r'''
    \brief save embedding for predict

    \param ids vertex id of embedding
    \param embedding embedding array
    \param save_embedding_dir dir for save embedding
    \param worker_id worker id
    '''
    os.makedirs(save_embedding_dir, exist_ok=True)
    ids = np.asarray(ids).flatten()
    ids_file = os.path.join(save_embedding_dir, f'ids_{worker_id}.npy')
    np.save(ids_file, ids)

    embedding = np.asarray(embedding).squeeze()
    embedding_file = os.path.join(save_embedding_dir,
                                  f'embedding_{worker_id}.npy')
    np.save(embedding_file, embedding)

    log.info(f'save embedding to {embedding_file}, ids to {ids_file}')
