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
from galileo.framework.tf.python.hooks.gpu_status_hook import GpuStatusHook
from galileo.framework.tf.python.hooks.step_counter_hook import StepCounterHook
from galileo.framework.tf.python.hooks.elapsed_time import ElapsedTimeHook


def _get_custom_hooks(key='hooks', **kwargs):
    custom_hooks = kwargs.get(key, [])
    for hook in custom_hooks:
        assert isinstance(
            hook, tf.estimator.SessionRunHook
        ), 'custom hook should inherit tf.estimator.SessionRunHook'
    return custom_hooks


@export('galileo.tf')
def get_train_hooks(log_tensor_dict=None, **kwargs):
    r'''
    args:
        log_tensor_dict
        log_steps
        model_dir
        gpu_status
        num_gpus
        hooks
    '''
    hooks = [ElapsedTimeHook('Train')]
    log_steps = kwargs.get('log_steps')
    if log_steps and log_steps > 0:
        if log_tensor_dict:
            hooks.append(
                tf.estimator.LoggingTensorHook(log_tensor_dict,
                                               every_n_iter=log_steps))

        model_dir = kwargs.get('model_dir')
        summary_dir = os.path.join(model_dir, 'summary')
        hooks.append(
            StepCounterHook(every_n_steps=log_steps, output_dir=summary_dir))
    profile_batch = kwargs.get('profile_batch')
    if isinstance(profile_batch, int) and profile_batch > 0:
        model_dir = kwargs.get('model_dir')
        summary_dir = os.path.join(model_dir, 'summary')
        with tf.Graph().as_default():
            hooks.append(
                tf.estimator.ProfilerHook(save_steps=profile_batch,
                                          output_dir=summary_dir))
    gpu_status = kwargs.get('gpu_status')
    if gpu_status:
        num_gpus = kwargs.get('num_gpus', 1)
        hooks.append(GpuStatusHook(num_gpus))
    hooks.extend(_get_custom_hooks(**kwargs))
    return hooks


@export('galileo.tf')
def get_evaluate_hooks(**kwargs):
    hooks = [ElapsedTimeHook('Evaluate')]
    hooks.extend(_get_custom_hooks(**kwargs))
    return hooks


@export('galileo.tf')
def get_predict_hooks(**kwargs):
    hooks = [ElapsedTimeHook('Predict')]
    hooks.extend(_get_custom_hooks(**kwargs))
    return hooks
