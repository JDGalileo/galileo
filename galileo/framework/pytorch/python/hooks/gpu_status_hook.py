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

from galileo.framework.pytorch.python.hooks.base_log_hook import BaseLogHook
from galileo.framework.pytorch.python.hooks.statistics import Statistics
from galileo.platform.utils import get_gpu_status
from galileo.platform.export import export


@export('galileo.pytorch')
class GpuStatusHook(BaseLogHook):
    def __init__(self, trainer):
        super().__init__(trainer)
        local_rank = trainer.config.get('local_rank')
        self.device = local_rank or 0
        self.gpu = Statistics(':.1f')

    def on_batch_end(self, outputs):
        name = 'gpu_{}'.format(self.device)
        gpu_value = get_gpu_status(self.device).gpu
        self.gpu.update(gpu_value)
        if self._should_log():
            print(
                f'{self._get_log_prefix()} {name}:{self.gpu.get_last_result()}',
                flush=True)

    def on_train_end(self):
        print(f'Summary:\n\t{name}:{self.gpu.get_result()}', flush=True)
