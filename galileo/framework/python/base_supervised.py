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
class BaseSupervised(metaclass=ABCMeta):
    r'''
    \brief base supervised model for graph

    \param label_dim label dim
    \param num_classes num of class
    '''
    def __init__(self, label_dim=None, num_classes=None, *args, **kwargs):
        if label_dim is not None:
            if label_dim == 1 and num_classes is None:
                raise ValueError('num_classes is required when label_dim is 1')
            if num_classes is None:
                num_classes = label_dim
            if label_dim > 1 and label_dim != num_classes:
                raise ValueError('label_dim must match with num_classes')

        self.label_dim = label_dim
        self.num_classes = num_classes

    @abstractmethod
    def encoder(self, inputs):
        r'''
        \brief supervised feature encoder
        '''

    def dense_encoder(self, inputs):
        r'''
        \brief a dense layer after encoder
        '''
        return inputs

    @abstractmethod
    def loss_and_metrics(self, labels, logits):
        r'''
        \brief compute loss and metrics
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

    @abstractmethod
    def convert_labels_tensor(self, inputs):
        r'''
        \brief convert labels tensor
        '''

    def unpack_sample(self, inputs):
        r'''
        \brief unpack sample

        \param inputs dict, keys of dict:\n
            case 1: (features, target) for save embedding\n
            case 2: (features, labels) for train and evaluate

        \return features, target_or_labels, only_embedding
        '''
        if not isinstance(inputs, dict):
            raise ValueError('inputs of supervised must be dict')
        if 'features' not in inputs:
            raise ValueError(
                'inputs of supervised must contain key `features`')
        if 'target' in inputs:
            return (self.convert_features_tensor(inputs['features']),
                    self.convert_ids_tensor(inputs['target']), True)
        if 'labels' in inputs:
            return (self.convert_features_tensor(inputs['features']),
                    self.convert_labels_tensor(inputs['labels']), False)
        raise ValueError(
            'inputs of supervised must contain key target or labels')

    def __call__(self, inputs, **kwargs):
        r'''
        \param inputs dict of tensors\n
            contains target, features and labels
        \see unpack_sample
        '''
        features, target_or_labels, only_embedding = self.unpack_sample(inputs)
        embedding = self.encoder(features)
        if only_embedding:
            return dict(ids=target_or_labels, embeddings=embedding)
        logits = self.dense_encoder(embedding)
        return self.loss_and_metrics(target_or_labels, logits)
