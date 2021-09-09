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
from tensorflow.keras.losses import Loss, deserialize
from galileo.platform.export import export


def neg_cross_entropy(logits, negative_logits):
    with tf.name_scope('loss'):
        positive = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(logits), logits=logits)
        negative = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(negative_logits), logits=negative_logits)
        loss = (tf.reduce_sum(positive, axis=-1) +
                tf.reduce_sum(negative, axis=-1))
        return tf.reduce_mean(loss)


class NegCrossentropy(Loss):
    def call(self, y_true, y_pred):
        return neg_cross_entropy(y_true, y_pred)


@export('galileo.tf')
def get_loss(name):
    r'''
    get all custom losses and keras buildin losses

    custom losses:
        nce
        neg_cross_entropy
        NCE
        NegCrossentropy

    keras buildin losses:
        https://www.tensorflow.org/api_docs/python/tf/keras/losses

    examples:
        >>> get_loss('nce')
        >>> get_loss('neg_cross_entropy')
        <function neg_cross_entropy at >
        >>> get_loss('NegCrossentropy')
        <galileo.framework.tf.python.losses.NegCrossentropy object at >
        >>> get_loss('binary_crossentropy')
        <function binary_crossentropy at >
    '''
    return deserialize(name,
                       custom_objects=dict(
                           nce=neg_cross_entropy,
                           neg_cross_entropy=neg_cross_entropy,
                           NCE=NegCrossentropy,
                           NegCrossentropy=NegCrossentropy,
                       ))
