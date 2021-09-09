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

import numpy as np
from collections import defaultdict
from galileo.platform.utils import get_gpu_status
from galileo.platform.export import export

import tensorflow as tf
from tensorflow.python.eager import context


@export('galileo.tf')
class GPUStatusCallback(tf.keras.callbacks.Callback):
    '''
    gpu usage status
    '''
    def __init__(self, num_gpu, summary_dir=None):
        super().__init__()
        self.num_gpu = num_gpu
        with context.eager_mode():
            self.summary_writer = tf.summary.create_file_writer(
                summary_dir) if summary_dir else None
        self.global_step = 0

    def on_train_begin(self, logs=None):
        self.stats = defaultdict(list)
        self.global_step = 0

    def on_batch_end(self, batch, logs=None):
        if self.num_gpu < 1:
            return
        self.global_step += 1
        for i in range(self.num_gpu):
            gpu_value = get_gpu_status(i).gpu
            self.stats[i].append(gpu_value)
            if self.summary_writer:
                with context.eager_mode():
                    with self.summary_writer.as_default():
                        tf.summary.scalar(f'gpu:{i}',
                                          gpu_value,
                                          step=self.global_step)
                    self.summary_writer.flush()

    def on_train_end(self, logs=None):
        if self.num_gpu < 1:
            return
        out = 'GPU Summary:'
        for k, v in self.stats.items():
            ts = np.array(v)
            a, b, c = ts.min(), ts.mean(), ts.max()
            out += f'\n\tmin/mean/max gpu:{k}: {a:.0f}/{b:.1f}/{c:.0f}'
        print(out, flush=True)
