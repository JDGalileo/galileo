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
from galileo.framework.pytorch.python.hooks.base_log_hook import BaseLogHook
from galileo.framework.pytorch.python.hooks.hooklist import HookList
from galileo.framework.pytorch.python.hooks.checkpoint import CheckpointHook
from galileo.framework.pytorch.python.hooks.metric_time_hooks \
    import MetricTimeHook
from galileo.framework.pytorch.python.hooks.save_predict_hook \
    import SavePredictHook
from galileo.framework.pytorch.python.hooks.gpu_status_hook import GpuStatusHook
from galileo.platform.default_values import DefaultValues
from galileo.platform.export import export


@export('galileo.pytorch')
def get_hooks(trainer, optimizer):
    config = trainer.run_config
    custom_hooks = config.get('hooks', [])
    for hook in custom_hooks:
        assert isinstance(hook,
                          BaseHook), 'custom hook should inherit BaseHook'
    hookList = HookList()
    hookList.append(CheckpointHook(trainer, optimizer))
    hookList.append(SavePredictHook(trainer))
    hookList.append(MetricTimeHook(trainer))
    if (trainer.config['use_cuda']
            and config.get('gpu_status', DefaultValues.GPU_STATUS)):
        hookList.append(GpuStatusHook(trainer))
    # place custom_hooks to last
    return hookList.extend(custom_hooks)
