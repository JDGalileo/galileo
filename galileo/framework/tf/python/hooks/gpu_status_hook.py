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

from collections import defaultdict
import numpy as np
import tensorflow as tf
from galileo.platform.utils import get_gpu_status
from galileo.platform.export import export


@export('galileo.tf')
class GpuStatusHook(tf.estimator.SessionRunHook):
    '''
    gpu usage status
    '''
    def __init__(self, num_gpus, logdir=None):
        super().__init__()
        self.num_gpus = num_gpus

    def begin(self):
        self.stats = defaultdict(list)

    def after_run(self, run_context, run_values):
        if self.num_gpus < 1:
            return
        for i in range(self.num_gpus):
            gpu_value = get_gpu_status(i).gpu
            self.stats[i].append(gpu_value)

    def end(self, session):
        if self.num_gpus < 1:
            return
        out = 'GPU Summary:'
        for k, v in self.stats.items():
            ts = np.array(v)
            a, b, c = ts.min(), ts.mean(), ts.max()
            out += f'\n\tmin/mean/max gpu:{k}: {a:.0f}/{b:.1f}/{c:.0f}'
        print(out, flush=True)
