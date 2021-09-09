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
from galileo.framework.python.base_transform import BaseTransform
from galileo.framework.python.utils.utils import get_fanouts_list
from galileo.platform.export import export


@export('galileo.pytorch')
class BipartiteTransform(BaseTransform):
    r'''
    \brief transform to convert bipartites

    \details
    a bipartite is a dict:
    \code{.py}
        dict(
            src=tensor,
            dst=tensor,
            src_feature=tensor,
            dst_feature=tensor,
            edge_weight=tensor,
        )
    \endcode

    \par examples
    \code{.py}
        >>> from galileo.pytorch import BipartiteTransform
        >>> bt = BipartiteTransform([2,3])
        >>> res = bt.transform(dict(ids=torch.randint(10,(4,9)),
                    feature=torch.rand(4,9,16),
                    edge_weight=torch.rand(4,9)))
        >>> len(res)
        2
        >>> res[0]['src'].shape
        torch.Size([4, 2])
        >>> res[0]['dst'].shape
        torch.Size([4, 6])
        >>> res[0]['src_feature'].shape
        torch.Size([4, 2, 16])
        >>> res[0]['dst_feature'].shape
        torch.Size([4, 6, 16])
        >>> res[0]['edge_weight'].shape
        torch.Size([4, 6])
        >>> res[1]['src'].shape
        torch.Size([4, 1])
        >>> res[1]['dst'].shape
        torch.Size([4, 2])
        >>> res[1]['src_feature'].shape
        torch.Size([4, 1, 16])
        >>> res[1]['dst_feature'].shape
        torch.Size([4, 2, 16])
        >>> res[1]['edge_weight'].shape
        torch.Size([4, 2])
    \endcode
    '''
    def __init__(self, fanouts: list, **kwargs):
        r'''
        \param fanouts number of multi hop
        '''
        assert fanouts, 'fanouts must be specified'
        config = dict(fanouts=fanouts)
        config.update(kwargs)
        super().__init__(config=config)
        self.fanouts = fanouts
        self.fanouts_list = get_fanouts_list(fanouts)

    def transform(self, inputs):
        r'''
        \param inputs
            dict(ids=tensor,feature=tensor,edge_weight=tensor)

        \return list of bipartite\n
            items in bipartites are arranged in the direction of aggregation,
            one of item:\n
            \li src -> dst are direction of edges
            \li  dst -> src are direction of aggregation
            \li  src shape: (*, fanouts_list[i-1])
            \li  dst edge_weight shape: (*, fanouts_list[i])

            may have duplicated vertices in src and dst
        '''
        fans = {}
        if 'ids' in inputs:
            fans['ids'] = torch.split(inputs['ids'], self.fanouts_list, dim=-1)
        if 'feature' in inputs:
            fans['feature'] = torch.split(inputs['feature'],
                                          self.fanouts_list,
                                          dim=-2)
        if 'edge_weight' in inputs:
            fans['edge_weight'] = torch.split(inputs['edge_weight'],
                                              self.fanouts_list,
                                              dim=-1)
        bipartites = []
        for i in reversed(range(1, len(self.fanouts_list))):
            bip = {}
            if 'ids' in fans:
                bip['src'] = fans['ids'][i - 1]
                bip['dst'] = fans['ids'][i]
            if 'feature' in fans:
                bip['src_feature'] = fans['feature'][i - 1]
                bip['dst_feature'] = fans['feature'][i]
            if 'edge_weight' in fans:
                bip['edge_weight'] = fans['edge_weight'][i]
            bipartites.append(bip)
        return bipartites
