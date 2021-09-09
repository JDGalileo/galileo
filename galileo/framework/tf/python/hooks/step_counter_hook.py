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

from timeit import default_timer
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from galileo.platform.utils import get_time_str
from galileo.platform.export import export


@export('galileo.tf')
class StepCounterHook(tf.estimator.StepCounterHook):
    '''
    summary global_step/sec
    '''
    def __init__(self,
                 every_n_steps=100,
                 every_n_secs=None,
                 output_dir=None,
                 summary_writer=None):
        super().__init__(every_n_steps, every_n_secs, output_dir,
                         summary_writer)

    def begin(self):
        super().begin()
        self.begin_time = default_timer()

    def end(self, session):
        et = default_timer() - self.begin_time
        out = 'Galileo summary:'
        out += f'\n\tTotal steps: {self._last_global_step}'
        step_per_sec = self._last_global_step / et
        out += f'\n\tAverage global_step/sec: {step_per_sec:.6f}'
        print(out, flush=True)
