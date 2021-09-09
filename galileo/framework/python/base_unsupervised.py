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
from galileo.platform.export import export


@export()
class BaseUnsupervised(metaclass=ABCMeta):
    r'''
    \brief base unsupervised model for graph

    including target embedding and context embedding
    '''
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def target_encoder(self, inputs):
        r'''
        \brief unsupervised target encoder
        '''

    @abstractmethod
    def context_encoder(self, inputs):
        r'''
        \brief unsupervised context encoder
        '''

    @abstractmethod
    def compute_logits(self, target, context):
        r'''
        \brief compute logits
        '''

    @abstractmethod
    def loss_and_metrics(self, logits, negative_logits):
        r'''
        \return a dict of loss and metrics
        '''

    @abstractmethod
    def convert_ids_tensor(self, inputs):
        r'''
        \brief convert ids tensor
        '''

    @abstractmethod
    def convert_features_tensor(self, inputs):
        r'''
        \brief convert features tensor
        '''

    def unpack_sample(self, inputs):
        r'''
        \brief unpack sample

        \param inputs dict, keys of dict:\n
            case 1: (target,) for save embedding, target and target_ids are same\n
            case 2: (target, target_ids) for save embedding,
                target is features of target_ids\n
            case 3: (target, context, negative) for train and evaluate,
                all is features

        \return target, target_ids, None, only_embedding=True\n
                target, context, negative, only_embedding=False
        '''
        if not isinstance(inputs, dict):
            raise ValueError('inputs of unsupervised must be dict')
        if 'target' not in inputs:
            raise ValueError('inputs of unsupervised must contain key target')
        if 1 == len(inputs):
            return self.convert_ids_tensor(inputs['target']), None, None, True
        if 'target_ids' in inputs:
            return (self.convert_features_tensor(inputs['target']),
                    self.convert_ids_tensor(inputs['target_ids']), None, True)
        return (self.convert_features_tensor(inputs['target']),
                self.convert_features_tensor(inputs['context']),
                self.convert_features_tensor(inputs['negative']), False)

    def __call__(self, inputs, **kwargs):
        r'''
        \param inputs dict of tensors, \see unpack_sample
        \return
            a dict of ids and embeddings if only_embedding is True,
            otherwise return a dict of loss and metrics
        '''
        target, context, negative, only_embedding = self.unpack_sample(inputs)
        if only_embedding:
            ids = context if context is not None else target
            return dict(ids=ids, embeddings=self.target_encoder(target))
        target_embedding = self.target_encoder(target)
        context_embedding = self.context_encoder(context)
        negative_embedding = self.context_encoder(negative)

        logits = self.compute_logits(target_embedding, context_embedding)
        negative_logits = self.compute_logits(target_embedding,
                                              negative_embedding)
        return self.loss_and_metrics(logits, negative_logits)
