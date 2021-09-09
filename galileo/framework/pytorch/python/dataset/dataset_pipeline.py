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

from torch.utils.data import DataLoader
from galileo.framework.pytorch.python.dataset.batched_dataloader \
        import BatchedDataLoader
from galileo.framework.pytorch.python.dataset.vertex_dataset \
        import VertexDataset
from galileo.framework.pytorch.python.dataset.edge_dataset import EdgeDataset
from galileo.platform.default_values import DefaultValues
from galileo.platform.export import export


@export('galileo.pytorch')
def dataset_pipeline(base_dataset_fun, transform=None, **kwargs):
    '''
    dataset pipeline

    args:
        base_dataset_fun: callable, base dataset
        transform: callable, dataset transform
        kwargs:
            batch_size: batch size
            dataset_num_parallel: parallel number
            multiprocessing_distributed:
            args for base_dataset_fun

    return dataloader
    '''
    assert callable(base_dataset_fun), 'base_dataset_fun must set'
    dataset = base_dataset_fun(**kwargs)

    dataset_num_parallel = kwargs.get('dataset_num_parallel', 0)
    if isinstance(dataset, (VertexDataset, EdgeDataset)):
        dataloader = BatchedDataLoader(dataset,
                                       collate_fn=transform,
                                       num_workers=dataset_num_parallel,
                                       pin_memory=False)
    else:
        batch_size = kwargs.get('batch_size')
        if batch_size is None or batch_size < 1:
            raise ValueError('dataset pipeline require batch_size > 0')
        is_distributed = kwargs.get('multiprocessing_distributed', False)
        sampler = None
        if is_distributed:
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                collate_fn=transform,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=dataset_num_parallel,
                                shuffle=False,
                                drop_last=False)
    return dataloader
