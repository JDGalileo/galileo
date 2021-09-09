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

from abc import ABCMeta, abstractmethod
from .named_object import NamedObject
from galileo.platform.export import export


@export()
class BaseMessagePassing(NamedObject, metaclass=ABCMeta):
    r'''
    \brief Base message passing for GNN

    paper `"Neural Message Passing for Quantum Chemistry"
        <https://arxiv.org/abs/1704.01212>`
    '''
    def __init__(self, config: dict = None, name: str = None):
        r'''
        \param config dict, config
        \param name name of inputs
        '''
        super().__init__(name)
        self._config = config

    @property
    def config(self):
        r'''
        \brief get config
        '''
        return self._config

    def __call__(self, inputs, training=None):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        hs = []
        for ip in inputs:
            h = self.message(ip, training=training)
            h = self.aggregate(h)
            h = self.update(h)
            hs.append(h)
        outputs = inputs[1:]
        for i in range(1, len(hs)):
            outputs[i - 1]['src_feature'] = hs[i]
            outputs[i - 1]['dst_feature'] = hs[i - 1]
        if 1 == len(hs):
            outputs = [dict(src_feature=hs[0])]
        return outputs

    def message(self, inputs, training=None):
        r'''
        \brief message features on vertices and edges\n

        subclass should override this method

        \param inputs inputs for message passing
        \param training
        \return tensors
        '''
        return inputs

    def aggregate(self, inputs):
        r'''
        \brief aggregate messages from neighbors with target\n

        subclass should override this method

        \param inputs inputs for aggregate
        \return tensors
        '''
        return inputs

    def update(self, inputs):
        r'''
        \brief update target features\n

        subclass should override this method

        \param inputs inputs for update
        \return tensors
        '''
        return inputs

    def message_and_aggregate(self, inputs, training=None):
        r'''
        \brief message and aggregate features on vertices and edges\n

        subclass should override this method

        \param inputs inputs for message passing
        \param training
        \return tensors
        '''
        return inputs