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
from galileo.framework.python.utils.utils import get_fanouts_list
from galileo.framework.tf.python.ops import TFOps as ops
from galileo.platform.export import export


@export('galileo.tf')
class MultiHopNeighborTransform(BaseTransform):
    r'''
    \brief transform for multi hop neighbors

    \par Examples:
    \code{.py}
        #without edge weight
        >>> from galileo.tf import MultiHopNeighborTransform
        >>> transform = MultiHopNeighborTransform([[0],[0]],[2,3]).transform
        >>> res = transform([2,4])
        >>> res['ids'].shape
        TensorShape([2, 9])

        #with edge weight
        >>> transform = MultiHopNeighborTransform([[0],[0]],[2,3],True).transform
        >>> res = transform([2,4])
        >>> res['ids'].shape
        TensorShape([2, 9])
        >>> res['edge_weight'].shape
        TensorShape([2, 9])
    \endcode
    '''
    def __init__(self,
                 metapath: list,
                 fanouts: list,
                 edge_weight: bool = False,
                 **kwargs):
        r'''
        \param metapath list of list, edge types of multi hop
        \param fanouts number of multi hop
        \param edge_weight has weight or not
        '''
        assert metapath, 'metapath must be specified'
        assert fanouts, 'fanouts must be specified'
        assert len(metapath) == len(fanouts)
        config = dict(metapath=metapath,
                      fanouts=fanouts,
                      edge_weight=edge_weight)
        config.update(kwargs)
        super().__init__(config=config)
        self.fanouts_list = get_fanouts_list(fanouts)

    def sample_multi_hop(self, inputs):
        r'''
        \brief sample multi hop neighbors
        \param inputs vertices
        '''
        metapath = self.config['metapath']
        fanouts = self.config['fanouts']
        edge_weight = self.config['edge_weight']

        if not tf.is_tensor(inputs):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.int64)
        vertices = tf.reshape(inputs, [-1])
        multi_hops = ops.sample_seq_by_multi_hop(vertices=vertices,
                                                 metapath=metapath,
                                                 fanouts=fanouts,
                                                 has_weight=edge_weight)
        return multi_hops

    def transform(self, inputs):
        r'''
        \param inputs vertices
        \return dict(ids=tensor, edge_weight=tensor)\n
        may have duplicated vertices in ids
        '''
        edge_weight = self.config['edge_weight']
        multi_hops = self.sample_multi_hop(inputs)
        outputs = dict(ids=multi_hops[0])
        if edge_weight:
            outputs['edge_weight'] = multi_hops[1]
        return outputs
