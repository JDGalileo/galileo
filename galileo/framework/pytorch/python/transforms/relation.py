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
from galileo.framework.python.utils.utils import (
    get_fanouts_list,
    get_fanouts_indices,
)
from galileo.platform.export import export


@export('galileo.pytorch')
class RelationTransform(BaseTransform):
    r'''
    \brief transform multi hops to relation graph

    \details
    a relation graph is a dict:
    \code{.py}
        dict(
            relation_indices=tensor,
            relation_weight=tensor,
            target_indices=tensor,
        )
    \endcode

    relation_indices is a [2,E] int tensor, E is number of edges,\n
        indices of relation/edge of graph
    relation_weight is a [E,1] float tensor, weight of relation\n
    target_indices is indices of target vertices, [batch size]

    \par Examples
    \code{.py}
        >>> from galileo.pytorch import RelationTransform
        >>> # fanouts= [2,3] batch size=5 num nodes=10
        >>> ids = torch.randint(10, [5, 9])
        >>> indices = ids.unique(return_inverse=True)[1]
        >>> rt = RelationTransform([2,3])
        >>> res = rt.transform(dict(indices=indices,
                    edge_weight=tf.random.normal((5,9))))
        >>> res.keys()
        dict_keys([relation_indices', 'relation_weight', 'target_indices'])
        >>> res['relation_indices'].shape
        torch.Size([2, 40])
        >>> res['relation_weight'].shape
        torch.Size([40, 1])
        >>> res['target_indices'].shape
        torch.Size([5])
    \endcode
    '''
    def __init__(self, fanouts: list, sort_indices: bool = False, **kwargs):
        r'''
        \param fanouts number of multi hop
        \param sort_indices sort relation indices
        '''
        assert fanouts, 'fanouts must be specified'
        config = dict(fanouts=fanouts)
        config.update(kwargs)
        super().__init__(config=config)
        self.fanouts = fanouts
        self.fanouts_list = get_fanouts_list(fanouts)
        self.fanouts_dim = sum(self.fanouts_list)
        self.fanouts_indices = get_fanouts_indices(fanouts)
        self.sort_indices = sort_indices

    def transform(self, inputs):
        r'''
        \param inputs
            list or tuple or \n
            dict(indices=tensor, edge_weight=tensor)\n
            size of indices and edge_weight must be N * fanouts_dim

        \return
            dict(
                relation_indices=tensor,
                relation_weight=tensor,
                target_indices=tensor,
            )
        '''
        if isinstance(inputs, (list, tuple)):
            indices, edge_weight = inputs[:2]
        elif isinstance(inputs, dict):
            indices = inputs['indices']
            edge_weight = inputs.get('edge_weight')
        else:
            indices, edge_weight = inputs, None

        # convert indices to relation indices
        indices_t = indices.view(-1, self.fanouts_dim).t()
        if indices.dim() > 2:
            target_indices = indices_t[0].view(indices.shape[:-1])
        else:
            target_indices = indices_t[0]
        relation_indices = indices_t.index_select(
            0, torch.tensor(self.fanouts_indices))
        relation_indices = torch.cat(relation_indices.split(2), dim=1)

        if self.sort_indices:
            index = relation_indices[0].argsort()
            relation_indices = relation_indices.index_select(1, index)

        relation_weight = None
        if edge_weight is not None:
            relation_weight = edge_weight[:, 1:].reshape(-1, 1).contiguous()

        return dict(
            relation_indices=relation_indices.contiguous(),
            relation_weight=relation_weight,
            target_indices=target_indices.contiguous(),
        )
