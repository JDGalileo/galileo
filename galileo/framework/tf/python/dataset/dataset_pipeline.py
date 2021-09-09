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
from galileo.platform.default_values import DefaultValues
from galileo.framework.tf.python.dataset.vertex_dataset import VertexDataset
from galileo.framework.tf.python.dataset.edge_dataset import EdgeDataset
from galileo.platform.export import export


@export('galileo.tf')
def dataset_pipeline(base_dataset_fun, transform=None, **kwargs):
    r'''
    dataset pipeline

    args:
        base_dataset_fun: callable, base dataset
        transform: callable, dataset transform
        kwargs:
            batch_size: batch size
            dataset_num_parallel: parallel number
            dataset_prefetch_size:
            args for base_dataset_fun
            args for shard:
                distribution_strategy
                input_context
                num_workers
                task_id

    return dataset
    '''
    assert callable(base_dataset_fun), 'base_dataset_fun must set'
    dataset = base_dataset_fun(**kwargs)
    batched_dataset = isinstance(dataset, (VertexDataset, EdgeDataset))

    # first do shard
    # is_shard attr is added in TextLineDataset
    if not hasattr(dataset, 'is_shard') or not dataset.is_shard:
        distribution_strategy = kwargs.get('distribution_strategy')
        if distribution_strategy == 'parameter_server':
            input_context = kwargs.get('input_context')
            if input_context:
                dataset = dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
            else:
                dataset = dataset.shard(kwargs['num_workers'],
                                        kwargs['task_id'])
        else:
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = \
                    tf.data.experimental.AutoShardPolicy.DATA
            dataset = dataset.with_options(options)

    if not batched_dataset:
        batch_size = kwargs.get('batch_size')
        if batch_size is None or batch_size < 1:
            raise ValueError('dataset pipeline require batch_size > 0')
        dataset = dataset.batch(batch_size)

    if callable(transform):
        dataset_num_parallel = kwargs.get('dataset_num_parallel')
        # default dataset_num_parallel=None is processed sequentially
        dataset = dataset.map(transform,
                              num_parallel_calls=dataset_num_parallel,
                              deterministic=False)

    # default dataset_prefetch_size=None is tf.data.experimental.AUTOTUNE
    dataset_prefetch_size = kwargs.get('dataset_prefetch_size')
    dataset = dataset.prefetch(dataset_prefetch_size)
    return dataset
