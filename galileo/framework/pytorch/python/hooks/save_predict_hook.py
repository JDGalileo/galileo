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
from galileo.framework.pytorch.python.hooks.base import BaseHook
from galileo.framework.python.utils.save_embedding import save_embedding
from galileo.platform.log import log
from galileo.platform.export import export


@export('galileo.pytorch')
class SavePredictHook(BaseHook):
    r'''
    args:
        model_dir
        save_predict_fn: def save_predict_fn(ids, embeddings, dir, rank)
    '''
    def __init__(self, trainer):
        super().__init__()
        model_dir = trainer.run_config.get('model_dir')
        self.save_predict_dir = os.path.join(model_dir, 'predict_results')
        os.makedirs(self.save_predict_dir, exist_ok=True)
        self.save_predict_fn = trainer.run_config.get('save_predict_fn')
        self.global_rank = trainer.config['global_rank']

    def on_predict_end(self, outputs):
        ids = []
        embeddings = []
        for op in outputs:
            ids.append(op['ids'])
            embeddings.append(op['embeddings'])
        ids = np.concatenate(ids, axis=0)
        embeddings = np.concatenate(embeddings, axis=0)
        if not callable(self.save_predict_fn):
            # default save_embedding
            self.save_predict_fn = save_embedding
        self.save_predict_fn(ids, embeddings, self.save_predict_dir,
                             self.global_rank)
