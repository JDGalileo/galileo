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

import os
from galileo.platform.export import export


def get_worker_env(worker_index=0, worker_num=1):
    if 'TF_CONFIG' in os.environ:
        import tensorflow as tf
        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        cluster_spec = resolver.cluster_spec().as_dict()
        if cluster_spec:
            task_id = resolver.task_id
            num_chief = len(cluster_spec.get('chief', []))
            num_workers = (num_chief + len(cluster_spec.get('worker', [])))
            return task_id, num_workers
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        return rank, world_size
    return worker_index, worker_num


@export()
def convert(
    vertex_source_path,
    edge_source_path,
    schema_path,
    output_binary_path,
    partition_num=1,
    parallel=1,
    hdfs_addr='',
    hdfs_port=0,
    field_separator='\t',
    array_separator=',',
    worker_index=0,
    worker_num=1,
    **kwargs,
):
    import galileo.framework.pywrap.py_convertor as convertor
    conf = convertor.Config()
    conf.vertex_source_path = vertex_source_path
    conf.edge_source_path = edge_source_path
    conf.schema_path = schema_path
    conf.edge_binary_path = output_binary_path
    conf.vertex_binary_path = output_binary_path
    conf.slice_count = partition_num
    conf.worker_count = parallel
    conf.hdfs_addr = hdfs_addr
    conf.hdfs_port = hdfs_port
    conf.field_separator = field_separator
    conf.array_separator = array_separator
    worker_index, worker_num = get_worker_env(worker_index, worker_num)
    conf.process_index = worker_index
    conf.process_count = worker_num
    convertor.start_convert(conf)
