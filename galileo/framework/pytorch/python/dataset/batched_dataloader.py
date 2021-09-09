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

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from galileo.platform.export import export


@export('galileo.pytorch')
class BatchedDataLoader(DataLoader):
    r'''
    Data loader for batched dataset

    args:
        dataset (Dataset): subclass of IterableDataset
        collate_fn (callable): collate_fn for dataset
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: ``0``)
        pin_memory (bool, optional): If ``True``, the data loader will copy
            tensors into CUDA pinned memory before returning them.
        timeout (numeric, optional): if positive, the timeout value for
            collecting a batch from workers. Should always be non-negative.
            (default: ``0``)

    NOTE: This dataloader only support dataset from Galileo, which is subclass of
        IterableDataset, and batch size must be handled by dataset.
    '''
    def __init__(self,
                 dataset,
                 collate_fn=None,
                 num_workers=0,
                 pin_memory=True,
                 timeout=0,
                 **kwargs):
        assert isinstance(dataset, IterableDataset),\
                'BatchedDataLoader only support IterableDataset'
        assert num_workers >= 0, 'num_workers option cannot be negative; '\
                'use num_workers=0 to disable multiprocessing.'
        assert timeout >= 0, 'timeout option should be non-negative'
        super().__init__(dataset,
                         batch_size=None,
                         shuffle=False,
                         sampler=None,
                         batch_sampler=None,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=False,
                         timeout=timeout,
                         worker_init_fn=None,
                         multiprocessing_context=None)

    def __len__(self):
        if self.num_workers == 0:
            return len(self.dataset)
        length = -(-len(self.dataset) // self.num_workers)
        return length * self.num_workers
