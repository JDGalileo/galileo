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

from galileo.platform.export import export
from galileo.framework.tf.python.callbacks.metrics_time \
        import MetricsTimeCallback
from galileo.framework.tf.python.callbacks.gpu_status \
        import GPUStatusCallback
from galileo.framework.tf.python.callbacks.dump_graph \
        import DumpGraphCallback


@export('galileo.tf')
def get_callbacks(is_checkpoint=True,
                  is_summary=True,
                  is_dump_graph=True,
                  **kwargs):
    '''
    callbacks for keras

    args:
        model_dir
        is_checkpoint
        is_summary
        is_dump_graph
        gpu_status
        train_verbose
        early_stop_patience
        save_checkpoint_epochs
        tensorboard_steps
        profile_batch
        batch_num
        num_epochs
        num_gpus
        callbacks
    '''
    custom_callbacks = kwargs.get('callbacks', [])
    for cb in custom_callbacks:
        assert isinstance(
            cb, tf.keras.callbacks.Callback
        ), 'custom callback should inherit tf.keras.callbacks.Callback'
    model_dir = kwargs.get('model_dir')
    checkpoint_path = os.path.join(model_dir, 'ckpt-{epoch:04d}')
    summary_dir = os.path.join(model_dir, 'summary')
    train_verbose = kwargs.get('train_verbose', 1)
    gpu_status = kwargs.get('gpu_status')
    early_stop_patience = kwargs.get('early_stop_patience')

    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    if early_stop_patience:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor='loss',
                                             verbose=1,
                                             restore_best_weights=True,
                                             patience=early_stop_patience))
    if is_checkpoint:
        save_checkpoint_epochs = kwargs.get('save_checkpoint_epochs', 1)
        if save_checkpoint_epochs > 0:
            # don't save ckp when save_checkpoint_epochs < 1
            batch_num = kwargs.get('batch_num', 1)
            num_epochs = kwargs.get('num_epochs', 1)
            if save_checkpoint_epochs > num_epochs:
                save_checkpoint_epochs = num_epochs
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='loss',
                    mode='min',
                    save_best_only=False,
                    save_weights_only=True,
                    verbose=1,
                    save_freq=save_checkpoint_epochs * batch_num))
    if is_summary:
        tensorboard_steps = kwargs.get('tensorboard_steps', 'epoch')
        if tensorboard_steps not in ['epoch', 'batch']:
            tensorboard_steps = int(tensorboard_steps)
        profile_batch = kwargs.get('profile_batch', 0)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(summary_dir,
                                           write_graph=True,
                                           write_images=False,
                                           update_freq=tensorboard_steps,
                                           profile_batch=profile_batch))
    callbacks.append(MetricsTimeCallback(summary_dir, skip_first=True))
    if gpu_status:
        num_gpus = kwargs.get('num_gpus', 1)
        callbacks.append(GPUStatusCallback(num_gpus, summary_dir))
    if is_dump_graph:
        callbacks.append(DumpGraphCallback(summary_dir))
    callbacks.extend(custom_callbacks)
    return callbacks
