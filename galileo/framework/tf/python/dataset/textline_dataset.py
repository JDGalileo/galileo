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
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from galileo.platform.default_values import DefaultValues
from galileo.platform.log import log
from galileo.platform.export import export


@export('galileo.tf')
def TextLineDataset(input_file, input_file_num_cols, **kwargs):
    '''
    csv file dataset
    do shard, repeat, shuffle and parse to int64


    args
        input_file:  must be a dir
        input_file_num_cols: num cols of csv files
        file_pattern:
        shuffle:
        repeat:
        dataset_num_parallel:
    '''
    shuffle = kwargs.get('shuffle', False)
    repeat = kwargs.get('repeat', False)
    is_shard = False

    file_pattern = kwargs.get('file_pattern', '*.csv')
    files = tf.data.Dataset.list_files(os.path.join(input_file, file_pattern),
                                       shuffle=shuffle)
    cardinality = tf.data.experimental.cardinality(files)
    num_workers = kwargs['num_workers'] or 1
    if tf.executing_eagerly():
        num_files = cardinality.numpy()
        assert num_files > 0, f'No *.csv files in {input_file}'
        if num_files < num_workers:  # pragma: no cover
            log.warning('The number of input files is recommended to be '
                        f'greater than workers={num_workers}')
    if num_workers > 1:
        # May cause no files on one worker when num_files < num_workers
        task_id = kwargs['task_id'] or 0
        files = files.shard(num_workers, task_id)
        is_shard = True

    dataset = tf.data.TextLineDataset(files, buffer_size=1024 * 1024 * 100)
    if repeat:
        dataset = dataset.repeat()

    def parse_csv(line):  # pragma: no cover
        example_defaults = [[0]] * input_file_num_cols
        parsed_line = tf.io.decode_csv(line, example_defaults)
        line_data = tf.stack(parsed_line, axis=0)
        line_data = tf.cast(line_data, dtype=tf.int64)
        return line_data

    dataset_num_parallel = kwargs.get('dataset_num_parallel')
    dataset = dataset.map(parse_csv, num_parallel_calls=dataset_num_parallel)
    if shuffle:  # pragma: no cover
        dataset = dataset.shuffle(1000)

    dataset.is_shard = is_shard
    return dataset
