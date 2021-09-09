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
import torch
import torch.cuda as cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.nn import Module as torchModule

from galileo.framework.python.client import create_client
from galileo.framework.python.base_trainer import BaseTrainer
from galileo.platform.default_values import DefaultValues
from galileo.platform.log import log
from galileo.platform.utils import cpu_count
from galileo.platform.export import export

from galileo.framework.pytorch.python.hooks.hooks import get_hooks
from galileo.framework.pytorch.python.utils import data_to_cuda, data_to_numpy
from galileo.framework.pytorch.python.module import Module


@export('galileo.pytorch')
class Trainer(BaseTrainer):
    r'''
    \brief pytorch trainer

    \par internal attrs:
        use_cuda
        global_rank
        local_rank
        is_master

    \attention API: galileo.pytoch.Trainer
    '''
    def __init__(
        self,
        model,
        inputs=None,
        module=None,
        multiprocessing_distributed=False,
        num_procs_per_shard=None,
        rank=0,
        world_size=1,
        dist_url=None,
        dist_backend=None,
        seed=None,
        zk_server=DefaultValues.ZK_SERVER,
        zk_path=DefaultValues.ZK_PATH,
    ):
        r'''
        \param model instance of torch.nn.Module
        \param inputs Inputs for model
        \param module Module for trainer, use default
        \param multiprocessing_distributed multi-processing distributed training
        \param num_procs_per_shard use default
        \param rank read from env RANK, use default
        \param world_size read from env WORLD_SIZE, use default
        \param dist_url use default
        \param dist_backend use default
        \param seed seed for initializing training
        \param zk_server zookeeper server address
        \param zk_path zookeeper registration node name
        '''
        if not isinstance(model, torchModule):
            raise ValueError(f'{model} should be subclass of torch.nn.Module')
        super().__init__(model, inputs, module)
        self._config = {}
        self.module = self.module or Module()

        use_cuda = cuda.is_available()
        if num_procs_per_shard is None:
            if use_cuda:
                num_procs_per_shard = cuda.device_count()
            else:
                num_procs_per_shard = cpu_count()
                log.warning(
                    'CUDA is not available, and will spawn {} processes '
                    'for parallel training if use multiprocessing distributed'.
                    format(num_procs_per_shard))
        else:
            num_procs_per_shard = int(num_procs_per_shard)
        if num_procs_per_shard < 1:
            num_procs_per_shard = 1
        self._config['num_procs_per_shard'] = num_procs_per_shard
        self._config['use_cuda'] = use_cuda

        # load envs
        rank = int(rank)
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
        if rank < 0:
            rank = 0
        self._config['rank'] = rank

        # here world_size is number of shard
        world_size = int(world_size)
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
        if world_size <= 0:
            world_size = 1
        self._config['world_size'] = world_size

        if dist_url is None:
            dist_url = 'tcp://127.0.0.1:23456'
        if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
            dist_url = None
        self._config['dist_url'] = dist_url

        # Note for multiprocessing_distributed argument:
        # 1. Auto distributed trainning and discard multiprocessing_distributed
        # argument when world_size > 1.
        # 2. Force one device trainning when world_size == 1 and
        # num_procs_per_shard == 1.
        # 3. use multiprocessing_distributed argument to decide when world_size
        # == 1 and num_procs_per_shard > 1.
        self._config['multiprocessing_distributed'] = (
            (multiprocessing_distributed and num_procs_per_shard > 1)
            or world_size > 1)

        if self._config['multiprocessing_distributed']:
            log.info(f'Use multiprocessing distributed for training, '
                     f'I am {rank + 1} of {world_size}, '
                     f'processes per worker is {num_procs_per_shard}')
            # update the world_size
            self._config['world_size'] = world_size * num_procs_per_shard
            if use_cuda:
                dist_backend = 'nccl'
                log.info('Use backend nccl for GPU distributed backend')
            else:
                dist_backend = 'gloo'
                log.info('Use backend gloo for CPU distributed backend')
        else:
            log.info('Use one worker for training')
        self._config['dist_backend'] = dist_backend

        if seed:
            torch.manual_seed(seed)
            cuda.manual_seed(seed)
            log.info(f'You have chosen to seed training, seed: {seed}')
        self._config['seed'] = seed
        self._config['zk_server'] = zk_server or DefaultValues.ZK_SERVER
        self._config['zk_path'] = zk_path or DefaultValues.ZK_PATH

    def get_optimizer(self):
        optimizer_name = self.run_config.get('optimizer',
                                             DefaultValues.OPTIMIZER)
        lr = self.run_config.get('learning_rate', DefaultValues.LEARNING_RATE)
        weight_decay = self.run_config.get('weight_decay',
                                           DefaultValues.WEIGHT_DECAY)
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(),
                                    weight_decay=weight_decay,
                                    lr=lr)
        elif optimizer_name == 'momentum':
            momentum = self.run_config.get('momentum', DefaultValues.MOMENTUM)
            return torch.optim.SGD(self.model.parameters(),
                                   weight_decay=weight_decay,
                                   lr=lr,
                                   momentum=momentum,
                                   nesterov=True)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(),
                                   weight_decay=weight_decay,
                                   lr=lr)
        raise ValueError(f'Unsupported optimizer {optimizer_name}')

    def train(self, **kwargs):
        r'''
        \param model_dir model dir
        \param inputs_fn inputs function, requried when self.inputs is None
        \param log_steps Number of steps to print log
        \param log_max_times_per_epoch log max times per epoch, default is 100
        \param start_epoch start of epoch
        \param num_epochs number epochs
        \param optimizer adam, sgd, momentum
        \param learning_rate learning rate
        \param momentum momentum for optimizer
        \param save_checkpoint_epochs The frequency to save checkpoint per epoch
        \param gpu_status bool show gpu status
        \param save_predict_fn callback for save results of predict
                save_predict_fn(ids, embeddings, dir, rank)
        \param save_best_model bool, save the best model

        \par spacial params for pytorch
        \param weight_decay weight_decay for optimizer
        \param resume file to checkpoint
        \param hooks hooks for log metrics and so on

        \par params for inputs_fn
        \param batch_size Mini-batch size
        \param max_id max vertex id
        \param batch_num Number of mini-batch, default is [max_id] / [batch_size]
        '''
        self.run(mode='train', **kwargs)

    def evaluate(self, **kwargs):
        r'''
        \copydoc train()
        '''
        self.run(mode='evaluate', **kwargs)

    def predict(self, **kwargs):
        r'''
        \copydoc train()
        '''
        self.run(mode='predict', **kwargs)

    def run(self, **kwargs):
        r'''
        \brief run train, evaluate, predict
        \param mode str, train, evaluate, predict
        \copydoc train()
        '''
        model_dir = kwargs.get('model_dir', DefaultValues.MODEL_DIR)
        os.makedirs(model_dir, exist_ok=True)
        self.run_config = kwargs
        if self.config['multiprocessing_distributed']:
            # Use torch.multiprocessing.spawn to launch distributed processes
            mp.spawn(self.run_worker,
                     nprocs=self.config['num_procs_per_shard'])
        else:
            self.run_worker(None)
        return 0

    def run_worker(self, local_rank):
        if local_rank is not None:
            if self.config['use_cuda']:
                log.info('Use GPU: {} for training'.format(local_rank))
            else:
                log.info('Use CPU for training')

        self.create_client()

        global_rank = self.config['rank']
        self.config['local_rank'] = local_rank
        if self.config['multiprocessing_distributed']:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            global_rank = (global_rank * self.config['num_procs_per_shard'] +
                           local_rank)
            dist.init_process_group(backend=self.config['dist_backend'],
                                    init_method=self.config['dist_url'],
                                    world_size=self.config['world_size'],
                                    rank=global_rank)
        self.config['global_rank'] = global_rank

        self.config['is_master'] = (
            not self.config['multiprocessing_distributed'] or
            (self.config['multiprocessing_distributed']
             and self.config['global_rank'] % self.config['world_size'] == 0))
        if self.config['use_cuda']:
            if local_rank is not None:
                # copy model to special cuda device
                cuda.set_device(local_rank)
                self.model.cuda(local_rank)
                if self.config['multiprocessing_distributed']:
                    self.model = DistributedDataParallel(
                        self.model, device_ids=[local_rank])
            else:
                self.model.cuda()
                if self.config['multiprocessing_distributed']:
                    self.model = DistributedDataParallel(self.model)
        else:
            if self.config['multiprocessing_distributed']:
                self.model = DistributedDataParallel(self.model)

        # train or evaluate or predict
        mode = self.run_config['mode']
        log.info(f'start {mode} model')
        getattr(self, f'do_{mode}')()

    def create_client(self):
        zk_server = self.config.get('zk_server')
        zk_path = self.config.get('zk_path')
        create_client(zk_server, zk_path)

    def get_dataset(self, mode):
        # args from self.config
        inputs_args = dict(
            zk_server=self.config['zk_server'],
            zk_path=self.config['zk_path'],
            world_size=self.config['world_size'],
            multiprocessing_distributed=self.
            config['multiprocessing_distributed'],
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
        dataloader = self.get_dataset('train')
        optimizer = self.get_optimizer()
        hooks = get_hooks(self, optimizer)
        hooks.on_train_begin()
        num_epochs = self.run_config.get('num_epochs',
                                         DefaultValues.NUM_EPOCHS)
        start_epoch = self.run_config.get('start_epoch',
                                          DefaultValues.START_EPOCH)
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.model.train()
            hooks.on_epoch_begin(epoch, len(dataloader))
            outputs = {}
            for subset in dataloader:
                hooks.on_batch_begin()
                if self.config['use_cuda']:
                    subset = data_to_cuda(subset, self.config['local_rank'])
                outputs = self.module.train_step(subset, self.model, optimizer)
                hooks.on_batch_end(outputs)
            hooks.on_epoch_end(outputs)
        hooks.on_train_end()

    def do_evaluate(self):
        self.run_config['batch_num'] = None  # del batch_num if specified
        dataloader = self.get_dataset('evaluate')
        hooks = get_hooks(self, optimizer=None)
        hooks.on_evaluate_begin()
        with torch.no_grad():
            self.model.eval()
            hooks.on_epoch_begin(0, len(dataloader))
            outputs = {}
            for subset in dataloader:
                hooks.on_batch_begin()
                if self.config['use_cuda']:
                    subset = data_to_cuda(subset, self.config['local_rank'])
                outputs = self.module.evaluate_step(subset, self.model)
                hooks.on_batch_end(outputs)
            hooks.on_epoch_end(outputs)
        hooks.on_evaluate_end()

    def do_predict(self):
        self.run_config['batch_num'] = None  # del batch_num if specified
        dataloader = self.get_dataset('predict')
        hooks = get_hooks(self, optimizer=None)

        def _predict_iter():
            for subset in dataloader:
                if not torch.is_tensor(subset['target']) and isinstance(
                        subset['target'], (list, tuple)):
                    subset['target'] = torch.tensor(subset['target'])
                if self.config['use_cuda']:
                    subset = data_to_cuda(subset, self.config['local_rank'])
                outputs = self.module.predict_step(subset, self.model)
                yield data_to_numpy(outputs)

        hooks.on_predict_begin()
        with torch.no_grad():
            self.model.eval()
            outputs = _predict_iter()
            hooks.on_predict_end(outputs)
