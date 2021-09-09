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
from tensorflow.python.training.checkpoint_management import \
    update_checkpoint_state_internal
from galileo.platform.log import log
from galileo.platform.export import export


def _loss_smaller(best_eval_result, current_eval_result):
    r'''
    \brief Compares two evaluation results and returns true
    if current_eval_result is smaller.

    Both evaluation results should have the values for loss, which are
    used for comparison.

    Args:
        best_eval_result: best eval metrics.
        current_eval_result: current eval metrics.

    Returns:
        True if the loss of current_eval_result is smaller; otherwise, False.

    Raises:
    ValueError: If input eval result is None or no loss is available.
    '''
    default_key = 'loss'
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')
    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')
    return best_eval_result[default_key] > current_eval_result[default_key]


@export('galileo.tf')
class BestCheckpointsExporter(tf.estimator.Exporter):
    r'''
    \brief Export the checkpoints of the best models.

    This is not like tf.estimator.BestExporter export everytime the new
    checkpoint of model, instead just copy the best checkpoints

    \par Examples
    \code{.py}
        tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100,
            exporters=BestCheckpointsExporter())
    \endcode
    '''
    def __init__(self,
                 name='best_checkpoints',
                 compare_fn=_loss_smaller,
                 max_to_keep=5,
                 sort_metric='loss',
                 sort_reverse=False):
        self._name = name
        self._compare_fn = compare_fn
        self._max_to_keep = max_to_keep
        self._sort_metric = sort_metric
        self._sort_reverse = sort_reverse

        if not callable(self._compare_fn):
            raise ValueError('`compare_fn` must be callable.')
        if max_to_keep is not None and max_to_keep <= 0:
            raise ValueError('`max_to_keep` must be a positive number.'
                             f' Got {max_to_keep}')
        self._checkpoints = dict()
        self._best_eval_result = None

    @property
    def name(self):
        return self._name

    def _keep_max_checkpoints(self):
        if len(self._checkpoints) < self._max_to_keep:
            return
        # sort by sort_metric in eval_result
        to_be_removed = sorted(self._checkpoints.items(),
                               key=lambda x: x[1][self._sort_metric],
                               reverse=self._sort_reverse)
        # keep one to add a new checkpoint
        to_be_removed = to_be_removed[(self._max_to_keep - 1):]
        for removing in to_be_removed:
            checkpoint_path = removing[0]
            for src in tf.io.gfile.glob(fr'{checkpoint_path}*'):
                try:
                    tf.io.gfile.remove(src)
                except Exception as e:
                    # ignore this error
                    log.error(f'delete {src} error {e}')
            del self._checkpoints[checkpoint_path]

    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        if self._best_eval_result is None or self._compare_fn(
                self._best_eval_result, eval_result):
            log.info(f'export checkpoint {checkpoint_path} of '
                     f'current best model {eval_result}')
            tf.io.gfile.makedirs(export_path)
            # copy all ckp files: data, index, meta
            for src in tf.io.gfile.glob(fr'{checkpoint_path}*'):
                src_name = os.path.basename(src)
                dst = os.path.join(export_path, src_name)
                tf.io.gfile.copy(src, dst, overwrite=True)
            new_checkpoint_path = os.path.join(
                export_path, os.path.basename(checkpoint_path))
            self._best_eval_result = eval_result
            self._keep_max_checkpoints()
            self._checkpoints[new_checkpoint_path] = eval_result
            update_checkpoint_state_internal(
                export_path,
                new_checkpoint_path,
                all_model_checkpoint_paths=self._checkpoints.keys(),
                save_relative_paths=True)
        else:
            log.info(f'skipping checkpoint {checkpoint_path}')
