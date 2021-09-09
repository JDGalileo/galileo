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
from tensorflow.keras import Model
from tensorflow.python.framework.errors_impl import FailedPreconditionError

from galileo.platform.default_values import DefaultValues
from galileo.platform.log import log
from galileo.platform.utils import DummyContextManager
from galileo.platform.export import export
from galileo.framework.python.utils.save_embedding import save_embedding
from galileo.framework.tf.python.tf_trainer import TFTrainer
from galileo.framework.tf.python.callbacks.callbacks import get_callbacks


@export('galileo.tf')
class KerasTrainer(TFTrainer):
    r'''
    \brief Trainer for tf keras

    \attention API: galileo.tf.KerasTrainer
    '''
    def init_model(self, **kwargs):
        distribution_strategy = self.config.get('distribution_strategy')
        if distribution_strategy == 'parameter_server':
            raise ValueError('parameter server strategy is not '
                             'supported with keras training')
        super().init_model(**kwargs)
        self.model_args['is_add_metrics'] = True
        use_eager = self.config['use_eager']
        strategy_context = (DummyContextManager
                            if self.strategy is None else self.strategy.scope)
        with strategy_context():
            if not isinstance(self.model, Model):
                self.model = self.model_class(**self.model_args)
            if not self.model._is_compiled:
                self.optimizer = self.get_optimizer()
                self.model.compile(loss=None,
                                   optimizer=self.optimizer,
                                   run_eagerly=use_eager)
            self.latest_ckp = tf.train.latest_checkpoint(self.model_dir)
            if self.latest_ckp is not None:
                self.model.load_weights(self.latest_ckp).expect_partial()
                log.info(f'loaded checkpoint from "{self.latest_ckp}"')

        # config batch_size for allreduce
        batch_size = self.run_config.get('batch_size')
        num_replicas = self.config['num_workers']
        if (self.strategy is not None
                and distribution_strategy != 'parameter_server'):
            num_replicas = self.strategy.num_replicas_in_sync
        if num_replicas > 1:
            batch_size *= num_replicas
            self.run_config['batch_size'] = batch_size

    def get_dataset(self, mode):
        # args from self.config and self.run_config
        inputs_args = dict(
            distribution_strategy=self.config['distribution_strategy'],
            num_workers=self.config['num_workers'],
            task_id=self.config['task_id'],
            batch_size=self.run_config['batch_size'],
            max_id=self.run_config.get('max_id'),
        )
        if self.inputs is not None:
            self.inputs.config.update(inputs_args)
            dataset = getattr(self.inputs, f'{mode}_data')()
        else:
            if not callable(self.run_config.get('inputs_fn')):
                raise ValueError('inputs_fn must be specified and callable'
                                 'when self.inputs is None')
            kwargs = self.run_config.copy()
            kwargs.update(inputs_args)
            kwargs['mode'] = mode
            dataset = self.run_config['inputs_fn'](**kwargs)
        return dataset

    def do_train(self):
        train_verbose = self.run_config.get('train_verbose', 1)
        batch_num = self.run_config.get('batch_num')
        start_epoch = self.run_config.get('start_epoch',
                                          DefaultValues.START_EPOCH)
        num_epochs = self.run_config.get('num_epochs',
                                         DefaultValues.NUM_EPOCHS)
        log.info(f'start train model {self.model_name}, '
                 f'epochs: {num_epochs}, steps: {batch_num}')
        dataset = self.get_dataset('train')
        try:
            self.model.fit(x=dataset,
                           initial_epoch=start_epoch,
                           epochs=start_epoch + num_epochs,
                           steps_per_epoch=batch_num,
                           verbose=train_verbose,
                           callbacks=get_callbacks(**self.run_config))
        except FailedPreconditionError as e:
            if 'Directory not empty' in e.message:
                log.warning('skip remove files error on some file systems')
            else:
                raise

    def do_evaluate(self):
        verbose = self.run_config.get('train_verbose', 1)
        log.info(f'start evaluate model {self.model_name}')
        if self.latest_ckp is None:
            log.warning(f'no checkpoint files found in {self.model_dir}, '
                        f'evaluate model may not what you want')
        dataset = self.get_dataset('evaluate')
        outputs = self.model.evaluate(x=dataset, steps=None, verbose=verbose)
        log.info(f'evaluate output: {outputs}')
        return outputs

    def do_predict(self):
        if self.latest_ckp is None:
            log.warning(f'no checkpoint files found in {self.model_dir}, '
                        f'predict model may not what you want')
        save_predict_dir = os.path.join(self.model_dir, 'predict_results')
        os.makedirs(save_predict_dir, exist_ok=True)
        log.info(f'starting save predict outputs to {save_predict_dir}')
        dataset = self.get_dataset('predict')
        outputs = self.model.predict(x=dataset)
        task_id = self.run_config.get('task_id', 0)
        if 'ids' in outputs and 'embeddings' in outputs:
            ids = outputs['ids'].squeeze()
            embeddings = outputs['embeddings'].squeeze()
            save_predict_fn = self.run_config.get('save_predict_fn')
            if not callable(save_predict_fn):
                save_predict_fn = save_embedding
            save_predict_fn(ids, embeddings, save_predict_dir, task_id)
        return outputs, task_id
