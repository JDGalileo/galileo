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
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.estimator import (
    Estimator,
    ModeKeys,
    TrainSpec,
    EvalSpec,
    EstimatorSpec,
)

from galileo.platform.default_values import DefaultValues
from galileo.platform.log import log
from galileo.platform.export import export
from galileo.framework.python.utils.save_embedding import save_embedding
from galileo.framework.tf.python.tf_trainer import TFTrainer
from galileo.framework.tf.python.hooks.hooks import (
    get_train_hooks,
    get_evaluate_hooks,
    get_predict_hooks,
)


@export('galileo.tf')
class EstimatorTrainer(TFTrainer):
    r'''
    \brief Trainer for tf estimator

    attention API: galileo.tf.EstimatorTrainer
    '''
    def init_model(self, **kwargs):
        super().init_model(**kwargs)
        if 1 == self.config['num_workers'] and 0 == self.config['num_ps']:
            # remove TF_CONFIG when num_workers==1 and num_ps==0
            # otherwise assert error not _is_device_list_single_worker(devices)
            os.environ['TF_CONFIG'] = '{}'
        self.model_args['is_add_metrics'] = False
        batch_num = self.run_config.get('batch_num')
        log_steps = self.run_config.get('log_steps', DefaultValues.LOG_STEPS)
        log_max_times_per_epoch = self.run_config.get(
            'log_max_times_per_epoch', DefaultValues.LOG_MAX_TIMES_PER_EPOCH)
        save_checkpoint_epochs = self.run_config.get('save_checkpoint_epochs')
        keep_checkpoint_max = self.run_config.get('keep_checkpoint_max', 5)
        if batch_num and batch_num > 0:
            # avoid too much batch log
            log_steps = min(log_steps, batch_num)
            if batch_num // log_steps > log_max_times_per_epoch:
                log_steps = batch_num // log_max_times_per_epoch
            if save_checkpoint_epochs and save_checkpoint_epochs > 0:
                save_checkpoints_steps = save_checkpoint_epochs * batch_num
        else:
            save_checkpoints_steps = None
        self.run_config['log_steps'] = log_steps
        tensorboard_steps = self.run_config.get('tensorboard_steps', 'epoch')
        if 'epoch' == tensorboard_steps:
            tensorboard_steps = batch_num if batch_num and batch_num > 0 else 100
        elif 'batch' == tensorboard_steps:
            tensorboard_steps = 1
        else:
            tensorboard_steps = int(tensorboard_steps)
        rel_model_dir = os.path.relpath(self.model_dir)
        # RunConfig will parse TF_CONFIG
        self.estimator_config = tf.estimator.RunConfig(
            model_dir=rel_model_dir,
            train_distribute=self.strategy,
            eval_distribute=self.strategy,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=keep_checkpoint_max,
            log_step_count_steps=None,
            save_summary_steps=tensorboard_steps)
        if self.inputs is not None:
            self.inputs_dict = {
                ModeKeys.TRAIN: self.inputs.train_data,
                ModeKeys.EVAL: self.inputs.evaluate_data,
                ModeKeys.PREDICT: self.inputs.predict_data,
            }

    def create_estimator(self):
        self.estimator = Estimator(self.model_fn,
                                   config=self.estimator_config,
                                   model_dir=None,
                                   params=None,
                                   warm_start_from=None)

        custom_metric_fn = self.run_config.get('custom_metric_fn')
        if callable(custom_metric_fn):
            self.estimator = tf.estimator.add_metrics(self.estimator,
                                                      custom_metric_fn)

    def model_fn(self, features, labels, mode):
        self.model = self.model_class(**self.model_args)
        r'''
        the metric_objs of model must be a function, not a `Metric` class
        for tf/estimator version 2.3.0
        '''
        if hasattr(self.model, 'metric_objs'):
            for name, mo in self.model.metric_objs.items():
                if isinstance(mo, tf.keras.metrics.Metric):
                    raise ValueError(f'metric {name} for estimator must be a '
                                     f'function, not a Metric class {mo}')
        outputs = self.model(features, training=mode == ModeKeys.TRAIN)
        if mode == ModeKeys.PREDICT:
            return EstimatorSpec(
                mode,
                predictions=outputs,
                prediction_hooks=get_predict_hooks(**self.run_config))
        loss = outputs.pop('loss')
        logits = outputs.pop('logits', None)
        if mode == ModeKeys.EVAL:
            r'''
            eval_metric_ops is dict of metric results keyed by name.
            The values of the dict can be one of the following: (1) instance of
            `Metric` class. (2) Results of calling a metric function, namely a
            `(metric_tensor, update_op)` tuple.

            when metric results is returned by model, value must be tensor
            returned by a function, not a `Metric` class for tf version 2.3.0
            '''
            eval_metric_ops = {}
            for name, o in outputs.items():
                if tf.is_tensor(o):
                    eval_metric_ops[name] = tf.compat.v1.metrics.mean(o)
            if len(eval_metric_ops) == 0:
                eval_metric_ops = None
            return EstimatorSpec(
                mode,
                loss=loss,
                predictions={'logits': logits},
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=get_evaluate_hooks(**self.run_config))
        optimizer = self.get_optimizer()
        global_step = tf.compat.v1.train.get_or_create_global_step()
        optimizer.iterations = global_step
        trainable_variables = self.model.trainable_variables
        update_ops = self.model.updates
        train_op = optimizer.get_updates(loss, trainable_variables)[0]
        if update_ops is not None and len(update_ops) > 0:
            train_op = tf.group(train_op, *update_ops)
        log_tensor_dict = dict(loss=loss, step=global_step)
        log_tensor_dict.update(outputs)
        train_hooks = get_train_hooks(log_tensor_dict, **self.run_config)
        return EstimatorSpec(
            mode,
            loss=loss,
            predictions={'logits': logits},
            train_op=train_op,
            training_hooks=train_hooks,
        )

    def get_dataset(self, mode, input_context=None):
        # args from self.config and self.run_config
        batch_size = self.run_config['batch_size']
        if self.should_dist_dataset and self.strategy is not None:
            batch_size *= self.strategy.num_replicas_in_sync
        inputs_args = dict(
            distribution_strategy=self.config['distribution_strategy'],
            num_workers=self.config['num_workers'],
            task_id=self.config['task_id'],
            batch_size=batch_size,
            max_id=self.run_config.get('max_id'),
            input_context=input_context,
        )
        if self.inputs is not None:
            self.inputs.config.update(inputs_args)
            dataset = self.inputs_dict[mode]()
        else:
            if not callable(self.run_config.get('inputs_fn')):
                raise ValueError('inputs_fn must be specified and callable'
                                 'when self.inputs is None')
            kwargs = self.run_config.copy()
            kwargs.update(inputs_args)
            kwargs['mode'] = mode
            dataset = self.run_config['inputs_fn'](**kwargs)
        if self.should_dist_dataset and self.strategy is not None:
            dataset = self.strategy.experimental_distribute_dataset(dataset)
        return dataset

    def do_train(self):
        self.create_estimator()
        max_steps = None
        num_epochs = self.run_config.get('num_epochs')
        if self.config['task_type'] != 'ps':
            batch_num = self.run_config.get('batch_num')
            assert batch_num and batch_num > 0
            max_steps = batch_num * num_epochs
            log.info(f'start train model {self.model_name}, '
                     f'epochs: {num_epochs}, steps per epoch: {batch_num}, '
                     f'all steps: {max_steps}')
        eval_hooks = self.run_config.get('eval_hooks')
        exporters = self.run_config.get('eval_exporters')
        throttle_secs = self.run_config.get('eval_throttle_secs') or 600
        estimator_hooks_fn = self.run_config.get('estimator_hooks_fn') or (
            lambda **kwargs: [])
        train_spec = TrainSpec(
            self.get_dataset,
            max_steps=max_steps,
            hooks=estimator_hooks_fn(estimator=self.estimator,
                                     **self.run_config))
        eval_spec = EvalSpec(self.get_dataset,
                             steps=None,
                             hooks=eval_hooks,
                             exporters=exporters,
                             throttle_secs=throttle_secs)
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def do_evaluate(self):
        if self.config['task_type'] == 'ps':
            log.info(f'parameter server exits when evaluate')
            return
        log.info(f'starting evaluate model {self.model_name}')
        self.estimator_config = self.estimator_config.replace(
            eval_distribute=None)
        self.create_estimator()
        outputs = self.estimator.evaluate(self.get_dataset, steps=None)
        log.info(f'evaluate output: {outputs}')
        return outputs

    def do_predict(self):
        if self.config['task_type'] == 'ps':
            log.info(f'parameter server exits when predict')
            return
        self.create_estimator()
        save_predict_dir = os.path.join(self.model_dir, 'predict_results')
        os.makedirs(save_predict_dir, exist_ok=True)
        log.info(f'starting save predict outputs to {save_predict_dir}')
        save_predict_fn = self.run_config.get('save_predict_fn')
        task_id = self.config['task_id']
        outputs = self.estimator.predict(self.get_dataset)
        ids = []
        embeddings = []
        ret_outputs = []
        for output in outputs:
            if 'ids' in output and 'embeddings' in output:
                ids.append(output['ids'])
                embeddings.append(output['embeddings'])
            ret_outputs.append(output)
        if ids and embeddings:
            embeddings = np.stack(embeddings, axis=0)
            if not callable(save_predict_fn):
                save_predict_fn = save_embedding
            save_predict_fn(ids, embeddings, save_predict_dir, task_id)
        return ret_outputs, task_id


export('galileo.tf').var('Trainer', EstimatorTrainer)
