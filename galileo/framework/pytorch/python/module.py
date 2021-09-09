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

from galileo.framework.python.base_module import BaseModule
from galileo.platform.export import export


@export('galileo.pytorch')
class Module(BaseModule):
    r'''
    pytorch module
    '''
    def __init__(self, config: dict = None, name: str = None):
        super().__init__(config, name)

    def train_step(self, inputs, model, optimizer):
        r'''
        train step, including forward and backward
        '''
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        return outputs

    def evaluate_step(self, inputs, model):
        r'''
        evaluate step
        '''
        return model(inputs)

    def predict_step(self, inputs, model):
        r'''
        predict step
        '''
        return model(inputs)
