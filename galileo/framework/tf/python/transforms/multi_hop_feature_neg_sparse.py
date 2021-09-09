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
from galileo.framework.tf.python.ops import TFOps as ops
from galileo.platform.export import export
from .multi_hop_feature_sparse import MultiHopFeatureSparseTransform


@export('galileo.tf')
class MultiHopFeatureNegSparseTransform(MultiHopFeatureSparseTransform):
    r'''
    \brief transform for multi hop features with negative sampling, sparse version

    This is inputs for Unsupervised graphSAGE

    \par Examples:
    \code{.py}
        >>> from galileo.tf import MultiHopFeatureNegSparseTransform
        >>> transform = MultiHopFeatureNegSparseTransform([[0],[0]],[2,3],[0],5,
                False,['feature'],5).transform
        >>> res = transform([2,4,6])
        >>> res.keys()
        dict_keys(['target', 'context', 'negative'])
        >>> res['target'].keys()
        dict_keys(['ids', 'indices', 'dense'])
        >>> res['target']['ids'].shape
        TensorShape([22])
        >>> res['target']['indices'].shape
        TensorShape([3, 1, 9])
        >>> res['context']['dense'].shape
        TensorShape([17, 5])
        >>> res['negative']['ids'].shape
        TensorShape([78])
        >>> res['negative']['dense'].shape
        TensorShape([78, 5])
    \endcode
    '''
    def __init__(
        self,
        metapath: list,
        fanouts: list,
        vertex_type: list,
        negative_num: int,
        edge_weight: bool = False,
        dense_feature_names: list = None,
        dense_feature_dims=None,
        sparse_feature_names: list = None,
        sparse_feature_dims=None,
        **kwargs,
    ):
        r'''
        \param metapath list of list, edge types of multi hop
        \param fanouts number of multi hop
        \param vertex_type vertex type
        \param negative_num number of negative
        \param edge_weight has weight or not
        \param dense_feature_names list of str
        \param dense_feature_dims int or list[int]
        \param sparse_feature_names list of str
        \param sparse_feature_dims int or list[int]
        '''
        super().__init__(metapath,
                         fanouts,
                         edge_weight,
                         vertex_type=vertex_type,
                         negative_num=negative_num,
                         dense_feature_names=dense_feature_names,
                         dense_feature_dims=dense_feature_dims,
                         sparse_feature_names=sparse_feature_names,
                         sparse_feature_dims=sparse_feature_dims,
                         **kwargs)

    def transform(self, inputs):
        r'''
        \param inputs vertices

        \return
        dict(target=dict,context=dict,negative=dict)\n
        inner dict:
            dict(ids=tensor, indices=tensor, dense=tensor,
            sparse=tensor, edge_weight=tensor)
        '''
        metapath = self.config['metapath']
        vertex_type = self.config['vertex_type']
        negative_num = self.config['negative_num']
        if not tf.is_tensor(inputs):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.int64)
        size = tf.size(inputs)
        vertices = tf.reshape(inputs, [size, 1])
        target_features = super().transform(vertices)

        vertices_f = tf.reshape(inputs, [size])
        context = ops.sample_neighbors(vertices_f,
                                       metapath[0],
                                       count=1,
                                       has_weight=False)
        context = tf.reshape(context[0], [size, 1])
        context_features = super().transform(context)

        negative = ops.sample_vertices(types=vertex_type,
                                       count=size * negative_num)[0]
        negative = tf.reshape(negative, [size, negative_num])
        negative_features = super().transform(negative)
        return dict(target=target_features,
                    context=context_features,
                    negative=negative_features)
