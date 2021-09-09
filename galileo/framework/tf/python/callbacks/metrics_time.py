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

import numpy as np
from timeit import default_timer
from collections import defaultdict

from galileo.platform.utils import get_time_str
from galileo.platform.export import export

import tensorflow as tf
from tensorflow.python.eager import context


@export('galileo.tf')
class MetricsTimeCallback(tf.keras.callbacks.Callback):
    r'''
    trainning time and metrics
    '''
    def __init__(self, summary_dir=None, skip_first=True):
        super().__init__()
        with context.eager_mode():
            self.summary_writer = tf.summary.create_file_writer(
                summary_dir) if summary_dir else None
        self.skip_first = skip_first
        self.global_step = 0

    def append_metrics(self, logs):
        if logs:
            for k, v in logs.items():
                if k not in ['batch', 'size']:
                    self.metrics[k].append(v)

    def on_train_begin(self, logs=None):
        self.train_begin_time = default_timer()
        self.epoch_times = []
        self.batch_times = []
        self.metrics = defaultdict(list)
        self.global_step = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin_time = default_timer()

    def on_batch_begin(self, batch, logs=None):
        self.batch_begin_time = default_timer()

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        self.batch_times.append(default_timer() - self.batch_begin_time)
        self.append_metrics(logs)
        if self.summary_writer:
            with context.eager_mode():
                with self.summary_writer.as_default():
                    tf.summary.scalar('batch_time',
                                      self.batch_times[-1],
                                      step=self.global_step)
                self.summary_writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(default_timer() - self.epoch_begin_time)
        self.append_metrics(logs)
        if self.summary_writer:
            with context.eager_mode():
                with self.summary_writer.as_default():
                    tf.summary.scalar('epoch_time',
                                      self.epoch_times[-1],
                                      step=epoch)
                self.summary_writer.flush()

    def on_train_end(self, logs=None):
        train_time = default_timer() - self.train_begin_time
        out = 'Summary:'
        if self.epoch_times:
            out += f'\n\tTotal epochs: {len(self.epoch_times)}'
            epoch_times = self.epoch_times[1:] if self.skip_first and \
                    len(self.epoch_times) > 1 else self.epoch_times
            epoch_time = get_time_str(np.mean(epoch_times))
            out += f'\n\tMean per epoch time: {epoch_time}'
        if self.batch_times:
            out += f'\n\tTotal steps: {len(self.batch_times)}'
            batch_times = self.batch_times[1:] if self.skip_first and \
                    len(self.batch_times) > 1 else self.batch_times
            batch_time = get_time_str(np.mean(batch_times))
            out += f'\n\tMean per step time: {batch_time}'
        if self.metrics:
            for k, v in self.metrics.items():
                ts = np.array(v)
                a, b, c = ts.min(), ts.mean(), ts.max()
                out += f'\n\tmin/mean/max {k}: {a:.4f}/{b:.4f}/{c:.4f}'
        out += f'\nTrain elapse {get_time_str(train_time)}'
        print(out, flush=True)
