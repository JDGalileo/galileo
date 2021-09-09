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

from abc import ABCMeta, abstractmethod
import torch
from torch.utils.data import IterableDataset
from galileo.platform.export import export


@export('galileo.pytorch')
class BaseDataset(IterableDataset, metaclass=ABCMeta):
    r'''
    base dataset for galileo engine

    args:
        batch_size: batch_size
        max_id: max vertex id in dataset
        dataset_num_parallel: =0
        zk_server: required when dataset_num_parallel>0
        zk_path: required when dataset_num_parallel>0
        world_size: world size
        batch_num: batch number, is max_id/batch_size when not set
    '''
    def __init__(
        self,
        batch_size,
        max_id,
        dataset_num_parallel=0,
        zk_server=None,
        zk_path=None,
        world_size=1,
        batch_num=None,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer '
                             f'value, but got batch_size={batch_size}')
        assert max_id > 0, 'max_id must be set > 0'
        self.batch_size = batch_size
        self.max_id = max_id

        if batch_num is None or batch_num < 0:
            batch_num = -(-max_id // batch_size)
        # data split strategy: split batch_num
        if world_size < 0:
            world_size = 1
        self.batch_num = -(-batch_num // world_size)

        self.dataset_num_parallel = dataset_num_parallel
        if dataset_num_parallel > 0:
            assert zk_server and zk_path, 'required when dataset_num_parallel>0'
        self.zk_server = zk_server
        self.zk_path = zk_path

    def __iter__(self):
        # for multi processes
        worker_info = torch.utils.data.get_worker_info()
        batch_num = self.batch_num
        if worker_info is not None:
            # split workload
            batch_num = -(-self.batch_num // worker_info.num_workers)
        for _ in range(batch_num):
            r'''
            here must create graph client when using
            multi process dataloader with multi process traning
            '''
            self._create_graph_client()
            yield self.batch()

    @abstractmethod
    def batch(self):
        r'''
        return a batch data
        '''

    def __len__(self):
        return self.batch_num

    def _create_graph_client(self):
        if self.dataset_num_parallel > 0:
            from galileo.framework.python.client import create_client
            create_client(self.zk_server, self.zk_path)
