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
class RandomWalkNegTransform(BaseTransform):
    r'''
    \brief randomwalk with negative sampling

    \par example
    \code{.py}
    >>> from galileo.tf import RandomWalkNegTransform
    >>> transform = RandomWalkNegTransform([0],[0],3,2,1,
            walk_length=3).transform
    >>> res = transform([2,4,6])
    >>> res.keys()
    dict_keys(['target', 'context', 'negative'])
    >>> res['target'].shape
    TensorShape([30, 1])
    >>> res['context'].shape
    TensorShape([30, 1])
    >>> res['negative'].shape
    TensorShape([30, 3])
    \endcode
    '''
    def __init__(self,
                 vertex_type: list,
                 edge_types: list,
                 negative_num: int,
                 context_size: int,
                 repetition: int = 1,
                 walk_p: float = 1.,
                 walk_q: float = 1.,
                 walk_length=None,
                 metapath=None,
                 **kwargs):
        r'''
        \param vertex_type
        \param edge_types
        \param negative_num
        \param context_size
        \param repetition
        \param walk_p
        \param walk_q
        \param walk_length
        \param metapath
        '''
        if walk_length is None and metapath is None:
            raise ValueError(
                'one of walk_length and metapath must be specified')

        if metapath is None:
            metapath = [edge_types] * walk_length

        config = dict(
            vertex_type=vertex_type,
            edge_types=edge_types,
            negative_num=negative_num,
            context_size=context_size,
            repetition=repetition,
            walk_p=walk_p,
            walk_q=walk_q,
            walk_length=walk_length,
            metapath=metapath,
        )
        super().__init__(config=config)

    def transform(self, inputs):
        r'''
        \param inputs vertices
        \return dict(target=tensor,context=tensor,negative=tensor)
        '''
        vertex_type = self.config['vertex_type']
        context_size = self.config['context_size']
        negative_num = self.config['negative_num']
        repetition = self.config['repetition']
        walk_p = self.config['walk_p']
        walk_q = self.config['walk_q']
        metapath = self.config['metapath']

        if not tf.is_tensor(inputs):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.int64)
        vertices = tf.reshape(inputs, [-1])
        pair = ops.sample_pairs_by_random_walk(vertices=vertices,
                                               metapath=metapath,
                                               repetition=repetition,
                                               context_size=context_size,
                                               p=walk_p,
                                               q=walk_q)
        target, context = tf.split(pair, [1, 1], axis=-1)
        size = tf.size(target)
        negative = ops.sample_vertices(types=vertex_type,
                                       count=negative_num * size)[0]
        negative = tf.reshape(negative, (size, negative_num))
        return dict(target=target, context=context, negative=negative)
