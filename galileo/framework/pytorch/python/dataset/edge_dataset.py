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

import random
from galileo.framework.pytorch.python.dataset.base_dataset import BaseDataset
from galileo.framework.pytorch.python.ops import PTOps as ops
from galileo.platform.export import export


@export('galileo.pytorch')
class EdgeDataset(BaseDataset):
    r'''
    args:
        args from BaseDataset
        edge_types:

    Output:
        edges that contains src, dst, etypes
    '''
    def __init__(self,
                 edge_types,
                 batch_size,
                 max_id,
                 dataset_num_parallel=0,
                 zk_server=None,
                 zk_path=None,
                 world_size=1,
                 batch_num=None,
                 **kwargs):
        super().__init__(
            batch_size,
            max_id,
            dataset_num_parallel,
            zk_server,
            zk_path,
            world_size,
            batch_num,
            **kwargs,
        )
        assert edge_types
        self.edge_types = edge_types

    def batch(self):
        edge_types = self.edge_types
        if len(edge_types) > 1:
            # random select one type
            edge_types = [random.choice(edge_types)]
        src, dst, etypes = ops.sample_edges(edge_types, count=self.batch_size)
        return src, dst, etypes
