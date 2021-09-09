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

from galileo.framework.pytorch.python.dataset.base_dataset import BaseDataset
from galileo.framework.pytorch.python.ops import PTOps as ops
from galileo.platform.export import export


@export('galileo.pytorch')
class VertexDataset(BaseDataset):
    r'''
    args:
        args from BaseDataset
        vertex_type:

    Output:
        vertices
    '''
    def __init__(self,
                 vertex_type,
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
        assert vertex_type
        self.vertex_type = vertex_type

    def batch(self):
        vertices = ops.sample_vertices(types=self.vertex_type,
                                       count=self.batch_size)[0]
        return vertices
