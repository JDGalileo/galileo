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

from collections import OrderedDict
import torch
from torch.nn import Module
from galileo.framework.python.base_unsupervised import BaseUnsupervised
from galileo.platform.export import export
from galileo.framework.pytorch.python.metrics import get_metric
from galileo.framework.pytorch.python.losses import get_loss


@export('galileo.pytorch')
class Unsupervised(Module, BaseUnsupervised):
    r'''
    \brief unsupervised network embedding model

    compute the loss and metrics

    Methods that the subclass must implement:\n
        target_encoder,
        context_encoder,
    '''
    def __init__(self,
                 loss_name='neg_cross_entropy',
                 metric_names='mrr',
                 *args,
                 **kwargs):
        Module.__init__(self)
        BaseUnsupervised.__init__(self, *args, **kwargs)
        self.loss_name = loss_name
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        self.metric_names = metric_names

    def target_encoder(self, inputs):
        raise NotImplementedError('call abc method')

    def context_encoder(self, inputs):
        raise NotImplementedError('call abc method')

    def compute_logits(self, target, context):
        return torch.sum(target * context, dim=-1)

    def loss_and_metrics(self, logits, negative_logits):
        r'''
        \return a dict of loss and metrics
        '''
        outputs = OrderedDict(
            loss=get_loss(self.loss_name)(logits, negative_logits))
        for metric_name in self.metric_names:
            outputs[metric_name] = get_metric(metric_name)(logits,
                                                           negative_logits)
        return outputs

    def convert_ids_tensor(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return torch.tensor(inputs, dtype=torch.int64)
        if torch.is_tensor(inputs) and inputs.dtype != torch.int64:
            return inputs.to(dtype=torch.int64)
        return inputs

    def convert_features_tensor(self, inputs):
        return self.convert_ids_tensor(inputs)

    def forward(self, inputs):
        return BaseUnsupervised.__call__(self, inputs)
