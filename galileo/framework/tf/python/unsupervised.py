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

import tensorflow as tf
from collections import OrderedDict
from tensorflow.keras import Model

from galileo.platform.export import export
from galileo.framework.python.base_unsupervised import BaseUnsupervised
from galileo.framework.tf.python.metrics import get_metric
from galileo.framework.tf.python.losses import get_loss


@export('galileo.tf')
class Unsupervised(Model, BaseUnsupervised):
    r'''
    \brief unsupervised network embedding layer

    Methods that the subclass must implement:\n
        target_encoder,
        context_encoder,

    \param loss_name: loss name
    \param metric_names: metric names, default is mrr
    \param is_add_metrics: add loss and metrics layers for keras
    '''
    def __init__(self,
                 loss_name='neg_cross_entropy',
                 metric_names='mrr',
                 is_add_metrics=True,
                 *args,
                 **kwargs):
        Model.__init__(self, name=kwargs.get('name'))
        BaseUnsupervised.__init__(self, *args, **kwargs)
        self.loss_name = loss_name
        self.loss_obj = get_loss(self.loss_name)
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        self.metric_names = metric_names
        self.is_add_metrics = is_add_metrics
        self.metric_objs = {name: get_metric(name) for name in metric_names}

    def target_encoder(self, inputs):
        raise NotImplementedError('call abc method')

    def context_encoder(self, inputs):
        raise NotImplementedError('call abc method')

    def compute_logits(self, target, context):
        return tf.matmul(target, context, transpose_b=True)

    def loss_and_metrics(self, logits, negative_logits):
        r'''
        \return a dict of loss and metrics or y_true and y_pred
        '''
        outputs = OrderedDict(loss=self.loss_obj(logits, negative_logits))
        if self.is_add_metrics:
            self.add_loss(outputs['loss'])
        for name in self.metric_names:
            metric = self.metric_objs[name](logits, negative_logits)
            outputs[name] = metric
            if self.is_add_metrics:
                self.add_metric(metric, name=name)
        return outputs

    def convert_ids_tensor(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return tf.convert_to_tensor(inputs, dtype=tf.int64)
        if tf.is_tensor(inputs) and inputs.dtype != tf.int64:
            return tf.cast(inputs, dtype=tf.int64)
        return inputs

    def convert_features_tensor(self, inputs):
        return self.convert_ids_tensor(inputs)

    def call(self, inputs):
        return BaseUnsupervised.__call__(self, inputs)

    def get_config(self):
        config = Model.get_config(self)
        config.update(
            dict(
                loss_name=self.loss_name,
                metric_names=self.metric_names,
                is_add_metrics=self.is_add_metrics,
            ))
        return config
