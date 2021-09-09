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

import torch
from galileo.framework.pytorch.python.ops import PTOps as ops
from galileo.platform.export import export
from galileo.framework.python.utils.utils import get_fanouts_dim
from .multi_hop import MultiHopNeighborTransform


@export('galileo.pytorch')
class MultiHopFeatureSparseTransform(MultiHopNeighborTransform):
    r'''
    \brief transform for multi hop features, sparse version

    \par Examples:
    \code{.py}
        >>> from galileo.pytorch import MultiHopFeatureSparseTransform
        >>> transform = MultiHopFeatureSparseTransform([[0],[0]],[2,3],
                False,['feature'],5).transform
        >>> res = transform([2,4])
        >>> res.keys()
        dict_keys(['ids', 'indices', 'dense'])
        >>> res['ids'].shape
        torch.Size([10])
        >>> res['indices'].shape
        torch.Size([2, 9])
        >>> res['dense'].shape
        torch.Size([10, 5])

        >>> transform = MultiHopFeatureSparseTransform([[0],[0]],[2,3],
                True,['feature'],5).transform
        >>> res = transform([[2],[4],[8]])
        >>> res.keys()
        dict_keys(['ids', 'indices', 'dense', 'edge_weight'])
        >>> res['ids'].shape
        torch.Size([17])
        >>> res['indices'].shape
        torch.Size([3, 1, 9])
        >>> res['dense'].shape
        torch.Size([17, 5])
        >>> res['edge_weight'].shape
        torch.Size([3, 9])
    \endcode
    '''
    def __init__(
        self,
        metapath: list,
        fanouts: list,
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
        \param edge_weight has weight or not
        \param dense_feature_names list of str
        \param dense_feature_dims int or list[int]
        \param sparse_feature_names list of str
        \param sparse_feature_dims int or list[int]
        '''
        if dense_feature_names is None and sparse_feature_names is None:
            raise ValueError('one of dense or sparse feature '
                             'names must be specified')
        if dense_feature_names is not None:
            assert dense_feature_dims, ('must set dense_feature_dims'
                                        '(int or list[int])')
            if isinstance(dense_feature_dims, int):
                dense_feature_dims = [dense_feature_dims
                                      ] * len(dense_feature_names)
            assert len(dense_feature_names) == len(dense_feature_dims)
        if sparse_feature_names is not None:
            if sparse_feature_dims is None:
                sparse_feature_dims = 1
            if isinstance(sparse_feature_dims, int):
                sparse_feature_dims = [sparse_feature_dims
                                       ] * len(sparse_feature_names)
            for dim in sparse_feature_dims:
                if dim != 1:
                    raise ValueError('Only support one dim sparse feature')
            assert len(sparse_feature_names) == len(sparse_feature_dims)
        self.fanouts_dim = get_fanouts_dim(fanouts)
        super().__init__(metapath,
                         fanouts,
                         edge_weight,
                         dense_feature_names=dense_feature_names,
                         dense_feature_dims=dense_feature_dims,
                         sparse_feature_names=sparse_feature_names,
                         sparse_feature_dims=sparse_feature_dims,
                         **kwargs)

    def transform(self, inputs):
        r'''
        \param inputs vertices
        \return
        dict(ids=tensor, indices=tensor, dense=tensor,
            sparse=tensor, edge_weight=tensor)\n
            \li ids shape [U]
            \li indices shape inputs.shape + fanouts_dim
            \li dense sparse shape [U, dim]
            \li edge_weight shape [N, fanouts_dim]
        '''
        edge_weight = self.config['edge_weight']
        dense_feature_names = self.config['dense_feature_names']
        dense_feature_dims = self.config['dense_feature_dims']
        sparse_feature_names = self.config['sparse_feature_names']
        sparse_feature_dims = self.config['sparse_feature_dims']

        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.int64)
        multi_hops = self.sample_multi_hop(inputs)
        ids, indices = multi_hops[0].flatten().unique(return_inverse=True)
        indices = indices.reshape(inputs.shape + (self.fanouts_dim, ))
        outputs = dict(ids=ids, indices=indices.contiguous())
        dense_features = self.get_feature(ids, dense_feature_names,
                                          dense_feature_dims, torch.float32)
        if dense_features is not None:
            outputs['dense'] = dense_features
        sparse_features = self.get_feature(ids, sparse_feature_names,
                                           sparse_feature_dims, torch.int64)
        if sparse_features is not None:
            outputs['sparse'] = sparse_features
        if edge_weight:
            outputs['edge_weight'] = multi_hops[1]
        return outputs

    def get_feature(
        self,
        vertices,
        feature_names,
        feature_dims,
        feature_type,
    ):
        if feature_names is None:
            return None
        features = ops.get_pod_feature([vertices], feature_names, feature_dims,
                                       [feature_type] * len(feature_names))
        if len(features) == 0:
            raise ValueError('Error get feature, see logs for details')
        features = torch.cat(features, dim=-1).contiguous()
        return features
