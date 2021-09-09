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
import tensorflow as tf
from tensorflow.keras import Model

from galileo.platform.export import export
from galileo.framework.python.base_supervised import BaseSupervised
from galileo.framework.tf.python.losses import get_loss
from galileo.framework.tf.python.metrics import get_metric
from galileo.framework.tf.python.layers import Dense


@export('galileo.tf')
class Supervised(Model, BaseSupervised):
    r'''
    \brief supervised model

    Methods that the subclass must implement:
        encoder

    args:
    \param loss_name: loss name
    \param metric_names: metric names, default is categorical_accuracy
    \param dense_input_dim: input dim for dense layer
    \param label_dim: label dim
    \param num_classes: num of class
    \param is_add_metrics: add loss and metrics layers for keras
    '''
    __default_loss = {
        'class_name': 'BinaryCrossentropy',
        'config': {
            'from_logits': True,
            'reduction': tf.keras.losses.Reduction.NONE
        }
    }

    def __init__(self,
                 loss_name=__default_loss,
                 metric_names='categorical_accuracy',
                 dense_input_dim=None,
                 label_dim=None,
                 num_classes=None,
                 is_add_metrics=True,
                 *args,
                 **kwargs):
        Model.__init__(self, name=kwargs.get('name'))
        BaseSupervised.__init__(self, label_dim, num_classes, *args, **kwargs)
        self.loss_name = loss_name
        self.loss_obj = get_loss(self.loss_name)
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        self.metric_names = metric_names
        self.is_add_metrics = is_add_metrics
        self.metric_objs = {name: get_metric(name) for name in metric_names}
        self.dense_layer = None
        if dense_input_dim and self.num_classes:
            self.dense_layer = Dense(self.num_classes, bias=False)
        self.dense_input_dim = dense_input_dim

    def encoder(self, inputs):
        raise NotImplementedError

    def dense_encoder(self, inputs):
        if callable(self.dense_layer):
            inputs = tf.reshape(inputs, [-1, self.dense_input_dim])
            return self.dense_layer(inputs)
        return inputs

    def loss_and_metrics(self, labels, logits):
        r'''
        \return a dict of loss and metrics or y_true and y_pred
        '''
        outputs = OrderedDict(loss=self.loss_obj(labels, logits))
        if self.is_add_metrics:
            self.add_loss(outputs['loss'])
        for name in self.metric_names:
            metric = self.metric_objs[name](labels, logits)
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
        if isinstance(inputs, (list, tuple)):
            return tf.convert_to_tensor(inputs, dtype=tf.float32)
        if tf.is_tensor(inputs) and inputs.dtype != tf.float32:
            return tf.cast(inputs, dtype=tf.float32)
        return inputs

    def convert_labels_tensor(self, inputs):
        labels = self.convert_ids_tensor(inputs)
        if self.label_dim == 1:
            labels = tf.one_hot(labels, self.num_classes)
        return labels

    def call(self, inputs):
        return BaseSupervised.__call__(self, inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                loss_name=self.loss_name,
                metric_names=self.metric_names,
                dense_input_dim=self.dense_input_dim,
                label_dim=self.label_dim,
                num_classes=self.num_classes,
                is_add_metrics=self.is_add_metrics,
            ))
        return config
