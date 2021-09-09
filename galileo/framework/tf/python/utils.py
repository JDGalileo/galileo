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
from galileo.platform.export import export


@export('galileo.tf')
def unique_pair(pair):
    r'''
    \brief unique a pair of tensor [N, 2]

    \return
        x shape [U]
        y shape [U]
        index shape [N]

    \examples
    >>>unique_pair([[2,3,2],[1,3,1]])
    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([2, 3])>
    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 3])>
    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([0, 1])>)
    '''
    if isinstance(pair, (list, tuple)):
        x, y = pair
        if not tf.is_tensor(x):
            x = tf.convert_to_tensor(x)
        if not tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
    elif tf.is_tensor(pair):
        # expected pair shape is [N, 2]
        x, y = tf.split(pair, [1, 1], axis=-1)
    else:
        raise ValueError('Not support type of pair', type(pair))
    x = tf.cast(tf.reshape(x, [-1]), tf.int64)
    y = tf.cast(tf.reshape(y, [-1]), tf.int64)
    # unique by Cantor pairing function
    pair_dup = ((x + y) * (x + y + 1) // 2 + y)
    pair_uniq = tf.unique(pair_dup)[0]
    pair_indices = tf.map_fn(
        lambda x: tf.argmax(tf.cast(tf.equal(pair_dup, x), tf.int64)),
        pair_uniq)
    x = tf.gather(x, pair_indices)
    y = tf.gather(y, pair_indices)
    return x, y, pair_indices
