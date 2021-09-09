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
class MultiHopFeatureLabelSparseTransform(MultiHopFeatureSparseTransform):
    r'''
    \brief transform for multi hop features and label

    This is inputs for Supervised graphSAGE

    \par Examples:
    \code{.py}
        >>> from galileo.tf import MultiHopFeatureLabelSparseTransform
        >>> transform = MultiHopFeatureLabelSparseTransform([[0],[0]],[2,3],
                'label',7,False,['feature'],5).transform
        >>> res = transform([2,4])
        >>> res.keys()
        dict_keys(['features', 'labels'])
        >>> res['labels'].shape
        TensorShape([2, 7])
        >>> res['features'].keys()
        dict_keys(['ids', 'indices', 'dense'])
        >>> res['features']['ids'].shape
        TensorShape([13])
        >>> res['features']['indices'].shape
        TensorShape([2, 9])
        >>> res['features']['dense'].shape
        TensorShape([13, 5])
    \endcode
    '''
    def __init__(
        self,
        metapath: list,
        fanouts: list,
        label_name: str,
        label_dim: int,
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
        \param label_name label feature name
        \param label_dim label dim
        \param edge_weight has weight or not
        \param dense_feature_names list of str
        \param dense_feature_dims int or list[int]
        \param sparse_feature_names list of str
        \param sparse_feature_dims int or list[int]
        '''
        super().__init__(metapath,
                         fanouts,
                         edge_weight,
                         label_name=label_name,
                         label_dim=label_dim,
                         dense_feature_names=dense_feature_names,
                         dense_feature_dims=dense_feature_dims,
                         sparse_feature_names=sparse_feature_names,
                         sparse_feature_dims=sparse_feature_dims,
                         **kwargs)

    def transform(self, inputs):
        r'''
        \param inputs vertices
        \return
        dict(features=dict,labels=dict)\n
        inner dict:
            dict(ids=tensor, dense=tensor, sparse=tensor, edge_weight=tensor)
        '''
        label_name = self.config['label_name']
        label_dim = self.config['label_dim']
        if not tf.is_tensor(inputs):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.int64)
        vertices = tf.reshape(inputs, [-1])
        features = super().transform(vertices)
        outputs = dict(features=features)
        labels = ops.get_pod_feature([vertices], [label_name], [label_dim],
                                     [tf.float32])
        outputs['labels'] = labels[0]
        return outputs
