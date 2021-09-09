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
from tensorflow.keras.metrics import Accuracy, deserialize
from galileo.platform.export import export


def mrr(logits, negative_logits):
    r'''
    Mean reciprocal rank score.
    '''
    with tf.name_scope('mrr'):
        scores = tf.concat([negative_logits, logits], axis=2)
        k = tf.shape(scores)[2]
        ranks_idx = tf.nn.top_k(scores, k=k)[1]
        ranks = tf.nn.top_k(-ranks_idx, k=k)[1]
        ranks = tf.cast(ranks[:, :, -1] + 1, tf.float32)
        return tf.reduce_mean(tf.math.reciprocal(ranks))


def f1_score(y_true, y_pred):
    r'''
    Calculate micro F1 score.

    NOTE: use this to evaluate for the whole epoch, not for one batch

    args:
        y_true: Tensor
        y_pred: Tensor
    return Tensor (dim == 1 and 0 <= val <= 1)
    '''
    with tf.name_scope('f1_score'):
        y_pred.shape.assert_is_compatible_with(y_true.shape)
        y_pred = tf.nn.sigmoid(y_pred)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        precision = tf.math.divide_no_nan(tp, tp + fp)
        recall = tf.math.divide_no_nan(tp, tp + fn)
        f1 = tf.math.divide_no_nan(2.0 * precision * recall,
                                   precision + recall)
        return f1


class MRR(Accuracy):
    r'''
    Mean reciprocal rank score.

    MeanMetricWrapper is not public api,
    so we inherit from Accuracy that is subclass of MeanMetricWrapper
    '''
    def __init__(self, name='mrr', dtype=None):
        super().__init__(name, dtype=dtype)
        self._fn = mrr


class F1Score(Accuracy):
    r'''
    Calculate micro F1 score.

    NOTE: use this to evaluate for the whole epoch, not for one batch

    MeanMetricWrapper is not public api,
    so we inherit from Accuracy that is subclass of MeanMetricWrapper
    '''
    def __init__(self, name='f1_score', dtype=None):
        super().__init__(name, dtype=dtype)
        self._fn = f1_score


@export('galileo.tf')
def get_metric(name):
    r'''
    get all custom metrics and keras buildin metrics

    custom metrics:
        mrr
        f1_score

    keras buildin metrics:
        https://www.tensorflow.org/api_docs/python/tf/keras/metrics

    examples:
        >>> get_metric('mrr')
        >>> get_metric('f1_score')
        >>> get_metric('categorical_accuracy')
        >>> get_metric('auc')
    '''
    return deserialize(name,
                       custom_objects=dict(
                           mrr=mrr,
                           f1_score=f1_score,
                           Mrr=MRR,
                           F1Score=F1Score,
                       ))
