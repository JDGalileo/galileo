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

from galileo.framework.pytorch.python.hooks.base import BaseHook
from galileo.platform.default_values import DefaultValues
from galileo.platform.export import export


@export('galileo.pytorch')
class BaseLogHook(BaseHook):
    def __init__(self, trainer):
        log_steps = trainer.run_config.get('log_steps',
                                           DefaultValues.LOG_STEPS)
        if log_steps is None or log_steps <= 0:
            log_steps = 1
        self.log_steps = log_steps
        self.log_max_times_per_epoch = trainer.run_config.get(
            'log_max_times_per_epoch', DefaultValues.LOG_MAX_TIMES_PER_EPOCH)
        self.trainer = trainer
        self.start_epoch = 0
        self.total_epochs = 0
        self.total_steps = 0
        self.cur_step = 0
        self.cur_epoch = 0

    def on_train_begin(self):
        # checkpoint hook may change start_epoch
        self.start_epoch = self.trainer.run_config.get('start_epoch', 0)
        self.total_epochs = self.trainer.run_config.get('num_epochs', 0)

    def on_evaluate_begin(self):
        self.start_epoch = 0
        self.total_epochs = 1

    def on_epoch_begin(self, epoch, steps):
        self.total_steps = steps or 1
        self.cur_epoch = epoch
        # avoid too much batch log
        log_steps = min(self.log_steps, self.total_steps)
        if self.total_steps // log_steps > self.log_max_times_per_epoch:
            log_steps = self.total_steps // self.log_max_times_per_epoch
        self.log_steps = log_steps

    def on_batch_begin(self, step):
        self.cur_step = step

    def _should_log(self):
        return (self.cur_step % self.log_steps == 0
                or self.cur_step >= self.total_steps)

    def _get_log_prefix(self):
        def _get_fmtstr(total):
            num_digits = len(str(total // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return f'[{fmt}/{fmt.format(total)}]'

        epoch_fmt = _get_fmtstr(self.start_epoch + self.total_epochs)
        steps_fmt = _get_fmtstr(self.total_steps)
        fmtstr = 'Epoch:' + epoch_fmt + ' Steps:' + steps_fmt
        return fmtstr.format(self.cur_epoch + 1, self.cur_step)
