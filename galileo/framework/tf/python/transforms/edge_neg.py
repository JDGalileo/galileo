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
from galileo.framework.python.base_transform import BaseTransform
from galileo.framework.tf.python.ops import TFOps as ops
from galileo.platform.export import export


@export('galileo.tf')
class EdgeNegTransform(BaseTransform):
    r'''
    \brief edge with negative sampling
    '''
    def __init__(self, vertex_type: list, negative_num: int, **kwargs):
        r'''
        \param vertex_type
        \param negative_num
        '''
        del kwargs
        config = dict(vertex_type=vertex_type, negative_num=negative_num)
        super().__init__(config=config)

    def transform(self, src, dst, types):
        r'''
        \param src src vertices of edges
        \param dst dst vertices of edges
        \param types edge vertices
        \return dict(target=tensor,context=tensor,negative=tensor)
        '''
        del types
        vertex_type = self.config['vertex_type']
        negative_num = self.config['negative_num']

        target = tf.reshape(src, [-1, 1])
        context = tf.reshape(dst, [-1, 1])
        size = tf.size(target)
        negative = ops.sample_vertices(types=vertex_type,
                                       count=size * negative_num)[0]
        negative = tf.reshape(negative, [size, negative_num])
        return dict(target=target, context=context, negative=negative)
