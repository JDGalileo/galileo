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
from galileo.framework.pytorch.python.ops import PTOps as ops
from galileo.platform.export import export


@export('galileo.pytorch')
class EdgeNegTransform(BaseTransform):
    r'''
    \brief edge with negative sampling
    '''
    def __init__(self, vertex_type: list, negative_num: int, **kwargs):
        r'''
        \param vertex_type
        \param negative_num
        '''
        config = dict(vertex_type=vertex_type, negative_num=negative_num)
        super().__init__(config=config)

    def transform(self, inputs):
        r'''
        \param inputs edges
        \return dict(target=tensor,context=tensor,negative=tensor)
        '''
        vertex_type = self.config['vertex_type']
        negative_num = self.config['negative_num']

        target = inputs[0].view(-1, 1)
        context = inputs[1].view(-1, 1)
        size = target.shape[0]
        negative = ops.sample_vertices(types=vertex_type,
                                       count=size * negative_num)[0]
        negative = negative.view(size, negative_num)
        return {
            'target': target.contiguous(),
            'context': context.contiguous(),
            'negative': negative.contiguous()
        }
