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
class BaseModule(NamedObject, metaclass=ABCMeta):
    r'''
    \brief Base Module for tf and pytorch and more

    \param config config dict for module
    \param name module name
    '''
    def __init__(self, config: dict = None, name: str = None):
        super().__init__(name)
        self._config = config

    @property
    def config(self):
        r'''
        \brief get config
        '''
        return self._config

    @abstractmethod
    def train_step(self, inputs, model, optimizer):
        r'''
        \brief train step, including forward and backward
        '''

    @abstractmethod
    def evaluate_step(self, inputs, model):
        r'''
        \brief evaluate step
        '''

    @abstractmethod
    def predict_step(self, inputs, model):
        r'''
        \brief predict step
        '''
