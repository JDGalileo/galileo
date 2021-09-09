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
import inspect
from abc import abstractmethod
from galileo.platform.default_values import DefaultValues
from galileo.platform.log import log
from galileo.framework.python.base_trainer import BaseTrainer
from galileo.framework.python.client import create_client

import tensorflow as tf
from tensorflow.keras import Model


class TFTrainer(BaseTrainer):
    r'''
    \brief Trainer for tf

    \note use subclasses of TFTrainer, not this

    \par internel config
        num_gpus
        task_type
        task_id
        num_chief
        num_workers
        num_ps
        is_chief

    \par internel attrs
        run_config
        strategy
        model_name
        model_args
        optimizer
        latest_ckp
        estimator_config
        should_dist_dataset
    '''
    def __init__(
        self,
        model,
        inputs=None,
        module=None,
        model_args=None,
        distribution_strategy=None,
        seed=None,
        zk_server=DefaultValues.ZK_SERVER,
        zk_path=DefaultValues.ZK_PATH,
        use_eager=False,
        soft_device_placement=True,
        log_device_placement=False,
    ):
        r'''
        \param model: instance or subclass of tf.keras.Model
        \param inputs Inputs for model
        \param module Module for trainer, use default
        \param model_args args for model
        \param distribution_strategy 'one_device', 'mirrored',
                    'multi_worker_mirrored','parameter_server'.
                    default is None
        \param seed seed for initializing training
        \param zk_server zookeeper server address
        \param zk_path zookeeper registration node name
        \param use_eager bool, use eager when debug
        \param soft_device_placement for tf.config.set_soft_device_placement
        \param log_device_placement for tf.debugging.set_log_device_placement
        '''
        if inspect.isclass(model):
            if not issubclass(model, Model):
                raise ValueError(f'{model} must be a subclass '
                                 'of tf.keras.Model')
            self.model_class = model
        else:
            if not isinstance(model, Model):
                raise ValueError(f'{model} should be instance'
                                 'of tf.keras.Model')
        super().__init__(model, inputs, module)
        self._config = {}

        tf.get_logger().setLevel('INFO')
        if seed:
            tf.random.set_seed(seed)
            log.info(f'You have chosen to seed training, seed: {seed}')
        self._config['seed'] = seed
        tf.config.run_functions_eagerly(use_eager)
        tf.config.set_soft_device_placement(soft_device_placement)
        tf.debugging.set_log_device_placement(log_device_placement)
        self.model_args = model_args or {}
        self.model_name = self.model_args.get('name', '')
        self._config['distribution_strategy'] = distribution_strategy
        self._config['zk_server'] = zk_server
        self._config['zk_path'] = zk_path
        self._config['use_eager'] = use_eager
        self._config['soft_device_placement'] = soft_device_placement
        self._config['log_device_placement'] = log_device_placement
        self.should_dist_dataset = False

        self.config_device()
        # parse TF_CONFIG
        self.parse_tf_config()
        # config distribution strategy
        self.config_dist_strategy()
        # create graph client
        self.create_graph_client()

    @abstractmethod
    def get_dataset(self, mode):
        pass

    @abstractmethod
    def do_train(self):
        pass

    @abstractmethod
    def do_evaluate(self):
        pass

    @abstractmethod
    def do_predict(self):
        pass

    def get_optimizer(self):
        optimizer_name = self.run_config.get('optimizer',
                                             DefaultValues.OPTIMIZER)
        lr = self.run_config.get('learning_rate', DefaultValues.LEARNING_RATE)
        momentum = self.run_config.get('momentum', DefaultValues.MOMENTUM)
        if optimizer_name == 'adam':
            return tf.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'momentum':
            return tf.optimizers.SGD(learning_rate=lr,
                                     momentum=momentum,
                                     nesterov=True)
        elif optimizer_name == 'sgd':
            return tf.optimizers.SGD(learning_rate=lr)
        elif optimizer_name == 'adagrad':
            return tf.optimizers.Adagrad(learning_rate=lr)
        raise ValueError('Unsupported optimizer ' + optimizer_name)

    def train(self, **kwargs):
        r'''
        \note all args are stored in run_config

        \param model_dir model dir
        \param inputs_fn inputs function, requried when self.inputs is None
        \param batch_size Mini-batch size
        \param max_id max vertex id
        \param batch_num Number of mini-batch, default is [max_id] / [batch_size]
        \param log_steps Number of steps to print log
        \param log_max_times_per_epoch log max times per epoch, default is 100
        \param start_epoch start of epoch
        \param num_epochs number epochs
        \param optimizer adam, sgd, momentum, adagrad
        \param learning_rate learning rate
        \param momentum momentum for optimizer
        \param save_checkpoint_epochs The frequency to save checkpoint per epoch
        \param gpu_status show gpu status
        \param save_predict_fn callback for save results of predict
                    save_predict_fn(ids, embeddings, dir, task_id)

        \par spacial args for tf
        \param train_verbose:
            \li 0 = silent
            \li 1 = progress bar
            \li 2 = one line per epoch
        \param tensorboard_steps update tensorboard every steps
        \param profile_batch int or pair
        \param estimator_hooks_fn estimator hooks function for early stop
            (tf.estimator.experimental.stop_if_no_decrease_hook)
        \param callbacks custom keras callbacks
        \param hooks custom estimator hooks
        \param custom_metric_fn custom estimator metric function
        \param eval_exporters instance of tf.estimator.Exporter
        \param eval_hooks for tf.estimator.EvalSpec
        \param eval_throttle_secs throttle_secs arg for tf.estimator.EvalSpec
        \param keep_checkpoint_max args for tf.estimator.RunConfig

        \par other args for input_fn
        '''
        self.init_model(**kwargs)
        return self.do_train()

    def evaluate(self, **kwargs):
        r'''
        \copydoc train()
        '''
        self.init_model(**kwargs)
        return self.do_evaluate()

    def predict(self, **kwargs):
        r'''
        \copydoc train()
        '''
        self.init_model(**kwargs)
        return self.do_predict()

    def init_model(self, **kwargs):
        r'''
        \brief init model
        \copydoc train()
        '''
        model_dir = kwargs.get('model_dir', DefaultValues.MODEL_DIR)
        os.makedirs(model_dir, exist_ok=True)
        self.run_config = kwargs
        self.run_config['num_gpus'] = self.config['num_gpus']
        self.model_dir = model_dir
        self.config_batch_num()

    def config_device(self):
        devices = tf.config.experimental.get_visible_devices('GPU')
        if devices:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                log.warning(e)
        self.config['num_gpus'] = len(devices)

    def parse_tf_config(self):
        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        cluster_spec = resolver.cluster_spec().as_dict()
        if cluster_spec:
            # TF_CONFIG exists
            task_type = resolver.task_type
            task_id = resolver.task_id
            num_chief = len(cluster_spec.get('chief', []))
            if num_chief > 0:
                assert num_chief == 1, 'Only one chief is allowed'
            if num_chief > 0 and task_type == 'worker':
                task_id += 1
            num_workers = (num_chief + len(cluster_spec.get('worker', [])))
            num_ps = len(cluster_spec.get('ps', []))
            if task_type == 'chief' or task_type == 'evaluator':
                is_chief = True
            elif (num_chief == 0 and task_type == 'worker' and task_id == 0):
                is_chief = True
            else:
                is_chief = False
        else:
            # TF_CONFIG not set
            log.warning('env TF_CONFIG is NOT set, use one worker traning')
            task_type = 'worker'
            num_chief = 0
            task_id = 0
            num_workers = 1
            num_ps = 0
            is_chief = True

        self.config['task_type'] = task_type
        self.config['task_id'] = task_id
        self.config['num_chief'] = num_chief
        self.config['num_workers'] = num_workers
        self.config['num_ps'] = num_ps
        self.config['is_chief'] = is_chief
        num_of_task_types = num_ps if task_type == 'ps' else num_workers
        log.info(f'I am {task_type}-{task_id} of {num_of_task_types}')

    def create_graph_client(self):
        # worker and chief role need to create graph client
        if self.config['task_type'] in ['worker', 'chief']:
            create_client(
                self.config.get('zk_server', DefaultValues.ZK_SERVER),
                self.config.get('zk_path', DefaultValues.ZK_PATH))

    def config_dist_strategy(self):
        distribution_strategy = self.config.get('distribution_strategy')
        if distribution_strategy is None:
            self.strategy = None
            return
        num_gpus = self.config['num_gpus']
        if distribution_strategy == 'one_device':
            strategy = tf.distribute.OneDeviceStrategy(
                '/gpu:0' if num_gpus > 0 else '/cpu:0')
        elif distribution_strategy == 'mirrored':
            strategy = tf.distribute.MirroredStrategy()
        elif distribution_strategy == 'multi_worker_mirrored':
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        elif distribution_strategy == 'parameter_server':
            strategy = tf.distribute.experimental.ParameterServerStrategy()
        elif distribution_strategy == 'central_storage':
            raise ValueError('central_storage is not availible')
        else:
            raise ValueError(f'not support distribution strategy '
                             f'{distribution_strategy}')
        self.strategy = strategy
        if distribution_strategy != 'parameter_server':
            if not strategy.extended._in_multi_worker_mode():
                if num_gpus > 0:
                    log.info(f'one worker use {num_gpus} GPU')
                    if num_gpus > 1:
                        self.should_dist_dataset = True
                else:
                    log.info(f'one worker use CPU')
            else:
                num_workers = self.config['num_workers']
                assert num_workers == strategy.extended._num_workers
                if num_gpus > 0:
                    log.info(f'{num_workers} workers use {num_gpus} '
                             'GPU per worker')
                    if num_gpus > 1:
                        self.should_dist_dataset = True
                else:
                    log.info(f'{num_workers} workers use CPU')

    def config_batch_num(self):
        batch_size = self.run_config.get('batch_size')
        if self.config['task_type'] in ['worker', 'chief']:
            batch_num = self.run_config.get('batch_num')
            if batch_num is None or batch_num < 0:
                max_id = self.run_config.get('max_id')
                assert max_id > 0, "max_id should be set > 0"
                self.run_config['batch_num'] = (max_id - 1 + batch_size -
                                                1) // batch_size
