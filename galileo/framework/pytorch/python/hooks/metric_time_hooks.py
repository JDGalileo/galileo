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

import torch
from timeit import default_timer
from galileo.framework.pytorch.python.hooks.base_log_hook import BaseLogHook
from galileo.framework.pytorch.python.hooks.statistics import Statistics
from galileo.platform.log import log
from galileo.platform.utils import get_time_str
from galileo.platform.export import export


@export('galileo.pytorch')
class MetricTimeHook(BaseLogHook):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.metrics = {}
        self.step_time = Statistics(':.3f')
        self.steps_per_sec = Statistics(':.3f')
        self.start_time = None
        self.start_epoch_time = None
        self.start_batch_time = None
        self.batch_time = None

    def on_train_begin(self):
        super().on_train_begin()
        self.start_time = default_timer()

    def on_epoch_begin(self, epoch, steps):
        super().on_epoch_begin(epoch, steps)
        self.start_epoch_time = default_timer()

    def on_batch_begin(self, step):
        super().on_batch_begin(step)
        now = default_timer()
        self.start_batch_time = now

    def on_batch_end(self, outputs):
        self.batch_time = default_timer() - self.start_batch_time
        if not self.metrics:
            for k, v in outputs.items():
                self.metrics[k] = Statistics(':.3f')
        for k, v in outputs.items():
            if torch.is_tensor(v):
                v = v.item()
            self.metrics[k].update(v)
        self.step_time.update(self.batch_time)
        if self._should_log():
            print(self._get_log_prefix(), end='')
            for k, v in self.metrics.items():
                print(f' {k}:{v.get_last_result()}', end='')
            print(f' step time:{self.step_time.get_last_result()}s', end='')
            print(flush=True)

    def on_epoch_end(self, outputs):
        sps = self.total_steps / (default_timer() - self.start_epoch_time)
        self.steps_per_sec.update(sps)
        print(f'Epoch:{self.cur_epoch + 1} steps/sec:'
              f'{self.steps_per_sec.get_last_result()}')

    def on_summary_end(self):
        print('Summary:')
        for k, v in self.metrics.items():
            print(f'\t{k} {v.get_result()}')
        print(f'\tstep time {self.step_time.get_result()}')
        print(f'\tsteps per sec: {self.steps_per_sec.get_result()}')
        print(flush=True)

    def on_train_end(self):
        self.on_summary_end()
        elaps = get_time_str(default_timer() - self.start_time)
        log.info(f'train done, Time {elaps}')

    def on_evaluate_begin(self):
        super().on_evaluate_begin()
        self.start_time = default_timer()

    def on_evaluate_end(self):
        self.on_summary_end()
        elaps = get_time_str(default_timer() - self.start_time)
        log.info(f'evaluate done, Time {elaps}')

    def on_predict_begin(self):
        self.start_time = default_timer()

    def on_predict_end(self, outputs):
        elaps = get_time_str(default_timer() - self.start_time)
        log.info(f'predict done, Time {elaps}')
