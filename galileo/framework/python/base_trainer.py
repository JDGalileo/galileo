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
from .base_module import BaseModule
from .base_inputs import BaseInputs
from galileo.platform.export import export


@export()
class BaseTrainer(metaclass=ABCMeta):
    r'''
    \brief BaseTrainer for tf and pytorch and more

    \li setup distributed training
    \li train/evaluate/predict

    \attention API: galileo.BaseTrainer
    '''
    def __init__(self,
                 model,
                 inputs: BaseInputs = None,
                 module: BaseModule = None,
                 config: dict = None):
        r'''
        \param model Model for tf or pytorch
        \param inputs Inputs for model
        \param module Module for trainer, use default
        \param config dict, config
        '''
        self._config = config
        if inputs is not None and not isinstance(inputs, BaseInputs):
            raise ValueError(f'{inputs} should be subclass of BaseInputs')
        if module is not None and not isinstance(module, BaseModule):
            raise ValueError(f'{module} should be subclass of BaseModule')
        self.model = model
        self.inputs = inputs
        self.module = module

    @property
    def config(self):
        r'''
        \brief get config
        '''
        return self._config

    @abstractmethod
    def get_dataset(self, mode):
        r'''
        \brief get an dataset

        \param mode train/evaluate/predict
        '''

    @abstractmethod
    def get_optimizer(self):
        r'''
        \brief return an optimizer
        '''

    @abstractmethod
    def train(self, **kwargs):
        r'''
        \brief train

        \param kwargs config
        '''

    @abstractmethod
    def evaluate(self, **kwargs):
        r'''
        \brief evaluate

        \param kwargs config
        '''

    @abstractmethod
    def predict(self, **kwargs):
        r'''
        \brief predict

        \param kwargs config
        '''
