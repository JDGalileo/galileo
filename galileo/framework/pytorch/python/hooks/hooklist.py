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
from galileo.platform.export import export


@export('galileo.pytorch')
class HookList(BaseHook):
    def __init__(self, hooks=None):
        self.hook_list = []
        self.extend(hooks)
        self.step = 0

    def append(self, hook):
        if not isinstance(hook, BaseHook):
            log.warning(f'skip invalid hook {hook}, should '
                        'derive from galileo.BaseHook')
            return self
        self.hook_list.append(hook)
        return self

    def extend(self, hooks):
        hooks = hooks or []
        for hook in hooks:
            self.append(hook)
        return self

    def on_train_begin(self):
        self.step = 0
        for hook in self.hook_list:
            hook.on_train_begin()

    def on_train_end(self):
        for hook in self.hook_list:
            hook.on_train_end()

    def on_evaluate_begin(self):
        self.step = 0
        for hook in self.hook_list:
            hook.on_evaluate_begin()

    def on_evaluate_end(self):
        for hook in self.hook_list:
            hook.on_evaluate_end()

    def on_predict_begin(self):
        self.step = 0
        for hook in self.hook_list:
            hook.on_predict_begin()

    def on_predict_end(self, outputs):
        for hook in self.hook_list:
            hook.on_predict_end(outputs)

    def on_epoch_begin(self, epoch, steps):
        self.step = 0
        for hook in self.hook_list:
            hook.on_epoch_begin(epoch, steps)

    def on_epoch_end(self, outputs):
        for hook in self.hook_list:
            hook.on_epoch_end(outputs)

    def on_batch_begin(self):
        self.step += 1
        for hook in self.hook_list:
            hook.on_batch_begin(self.step)

    def on_batch_end(self, outputs):
        for hook in self.hook_list:
            hook.on_batch_end(outputs)
