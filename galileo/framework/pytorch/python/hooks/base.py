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

from abc import ABCMeta
from galileo.platform.export import export


@export('galileo.pytorch')
class BaseHook(metaclass=ABCMeta):
    r'''
    Abstract base class used to build new hook.

    Subclass this class and override any of the relevant hooks
    '''
    def on_train_begin(self):
        '''
        begin train hook
        '''

    def on_train_end(self):
        '''
        end train hook
        '''

    def on_evaluate_begin(self):
        '''
        begin evaluate hook
        '''

    def on_evaluate_end(self):
        '''
        end evaluate hook
        '''

    def on_predict_begin(self):
        '''
        begin predict hook
        '''

    def on_predict_end(self, outputs):
        '''
        end predict hook

        args:
            outputs: iterator
        '''

    def on_epoch_begin(self, epoch, steps):
        '''
        begin epoch hook

        args:
            epoch: current epoch
            steps: total steps
        '''

    def on_epoch_end(self, outputs):
        '''
        end epoch hook

        args:
            outputs: dict
        '''

    def on_batch_begin(self, step):
        '''
        begin batch hook

        args:
            steps: current step
        '''

    def on_batch_end(self, outputs):
        '''
        end batch hook

        args:
            outputs: dict
        '''
