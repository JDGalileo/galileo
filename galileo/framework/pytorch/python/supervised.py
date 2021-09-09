# Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.nn import Module
from galileo.platform.export import export
from galileo.framework.python.base_supervised import BaseSupervised
from galileo.framework.pytorch.python.losses import get_loss
from galileo.framework.pytorch.python.metrics import get_metric
from galileo.framework.pytorch.python.layers.dense import Dense


@export('galileo.pytorch')
class Supervised(Module, BaseSupervised):
    r'''
    \brief supervised model

    compute the loss and metrics

    Methods that the subclass must implement:\n
        encoder

    \param loss_name: loss name
    \param metric_names: metric names, default is f1_score
    \param dense_input_dim: input dim for dense layer
    \param label_dim: label dim
    \param num_classes: num of class
    '''
    def __init__(self,
                 loss_name='multi_label_sm',
                 metric_names='f1_score',
                 dense_input_dim=None,
                 label_dim=None,
                 num_classes=None,
                 *args,
                 **kwargs):
        Module.__init__(self)
        BaseSupervised.__init__(self, label_dim, num_classes, *args, **kwargs)
        self.loss_name = loss_name
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        self.metric_names = metric_names
        self.dense_layer = None
        if dense_input_dim and self.num_classes:
            self.dense_layer = Dense(dense_input_dim,
                                     self.num_classes,
                                     use_bias=False)

    def encoder(self, inputs):
        raise NotImplementedError

    def dense_encoder(self, inputs):
        if callable(self.dense_layer):
            return self.dense_layer(inputs)
        return inputs

    def loss_and_metrics(self, labels, logits):
        r'''
        \return a dict of loss and metrics
        '''
        outputs = OrderedDict(loss=get_loss(self.loss_name)(labels, logits))
        sigmoid = torch.nn.Sigmoid()
        predictions = sigmoid(logits)
        predictions = torch.floor(predictions + 0.5).to(torch.long)
        for metric_name in self.metric_names:
            outputs[metric_name] = get_metric(metric_name)(labels, predictions)
            if 'f1_score' == metric_name:
                outputs[metric_name] = outputs[metric_name][0]
        return outputs

    def convert_ids_tensor(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return torch.tensor(inputs, dtype=torch.int64)
        if torch.is_tensor(inputs) and inputs.dtype != torch.int64:
            return inputs.to(dtype=torch.int64)
        return inputs

    def convert_features_tensor(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return torch.tensor(inputs, dtype=torch.float32)
        if torch.is_tensor(inputs) and inputs.dtype != torch.float32:
            return inputs.to(dtype=torch.float32)
        return inputs

    def convert_labels_tensor(self, inputs):
        labels = self.convert_ids_tensor(inputs)
        labels = torch.squeeze(labels)
        if self.label_dim == 1:
            labels = F.one_hot(labels, self.num_classes)
        return labels

    def forward(self, inputs):
        return BaseSupervised.__call__(self, inputs)
