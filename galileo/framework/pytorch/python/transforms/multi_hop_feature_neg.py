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
from galileo.platform.export import export
from galileo.framework.pytorch.python.ops import PTOps as ops
from .multi_hop_feature import MultiHopFeatureTransform


@export('galileo.pytorch')
class MultiHopFeatureNegTransform(MultiHopFeatureTransform):
    r'''
    \brief transform for multi hop features with negative sampling

    This is inputs for Unsupervised graphSAGE

    \par Examples:
    \code{.py}
        >>> from galileo.pytorch import MultiHopFeatureNegTransform
        >>> transform = MultiHopFeatureNegTransform([[0],[0]],[2,3],[0],5,
                False,['feature'],5).transform
        >>> res = transform([2,4,6])
        >>> res.keys()
        dict_keys(['target', 'context', 'negative'])
        >>> res['target'].keys()
        dict_keys(['ids', 'dense'])
        >>> res['target']['dense'].shape
        torch.Size([3, 1, 9, 5])
        >>> res['context']['dense'].shape
        torch.Size([3, 1, 9, 5])
        >>> res['negative']['dense'].shape
        torch.Size([3, 5, 9, 5])
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
            dict(ids=tensor, dense=tensor, sparse=tensor, edge_weight=tensor)
        '''
        metapath = self.config['metapath']
        vertex_type = self.config['vertex_type']
        negative_num = self.config['negative_num']
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.int64)
        vertices = inputs.flatten().contiguous()
        target_features = super().transform(vertices)
        target_features = self.reshape_outputs(target_features, 1)

        context = ops.sample_neighbors(vertices,
                                       metapath[0],
                                       count=1,
                                       has_weight=False)
        if len(context) == 0:
            raise ValueError('Error sample neighbors, see logs for details')
        context = context[0].flatten().contiguous()
        context_features = super().transform(context)
        context_features = self.reshape_outputs(context_features, 1)
        size = len(context)
        negative = ops.sample_vertices(types=vertex_type,
                                       count=size * negative_num)[0]
        negative = negative.flatten().contiguous()
        negative_features = super().transform(negative)
        negative_features = self.reshape_outputs(negative_features,
                                                 negative_num)
        return dict(target=target_features,
                    context=context_features,
                    negative=negative_features)

    def reshape_outputs(self, output, size):
        if 'ids' in output:
            shp = output['ids'].shape
            output['ids'] = output['ids'].view(-1, size, shp[-1]).contiguous()
        if 'edge_weight' in output:
            shp = output['edge_weight'].shape
            output['edge_weight'] = output['edge_weight'].view(
                -1, size, shp[-1]).contiguous()
        if 'dense' in output:
            shp = output['dense'].shape
            output['dense'] = output['dense'].view(-1, size, shp[-2],
                                                   shp[-1]).contiguous()
        if 'sparse' in output:
            shp = output['sparse'].shape
            output['sparse'] = output['sparse'].view(-1, size, shp[-2],
                                                     shp[-1]).contiguous()
        return output
