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
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from galileo.platform.log import log
from galileo.platform.export import export
from galileo.framework.pytorch.python.hooks.base import BaseHook

DEFAULT_CHECKPOINT_FILENAME = 'checkpoint.pth'


def load_checkpoint(trainer, checkpoint_file, optimizer=None):
    if not os.path.isfile(checkpoint_file):
        return False
    local_rank = trainer.config['local_rank']
    if local_rank is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        if trainer.config['use_cuda']:
            # Map model to be loaded to specified single gpu.
            loc = f'cuda:{local_rank}'
            checkpoint = torch.load(checkpoint_file, map_location=loc)
    try:
        trainer.model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        from torch.nn.parallel import DistributedDataParallel
        dist.init_process_group(backend=trainer.config['dist_backend'],
                                init_method=trainer.config['dist_url'],
                                world_size=trainer.config['world_size'],
                                rank=trainer.config['global_rank'])
        trainer.model = DistributedDataParallel(trainer.model)
        trainer.model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    trainer.run_config['start_epoch'] = start_epoch
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    log.info(f'loaded checkpoint "{checkpoint_file}" (epoch {start_epoch})')
    return True


@export('galileo.pytorch')
class CheckpointHook(BaseHook):
    r'''
    load and save checkpoint
    '''
    def __init__(self, trainer, optimizer):
        super().__init__()
        self.trainer = trainer
        self.optimizer = optimizer
        self.resume = trainer.run_config.get('resume')
        self.model_dir = trainer.run_config.get('model_dir')
        self.checkpoint_file = os.path.join(self.model_dir,
                                            DEFAULT_CHECKPOINT_FILENAME)
        self.save_checkpoint_epochs = trainer.run_config.get(
            'save_checkpoint_epochs')
        self.save_best_model = trainer.run_config.get('save_best_model')
        self.save_model = None
        self.save_loss = None
        self.epoch = 0
        self._checkpoint_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix='checkpoint-threadpool')

    def on_train_begin(self):
        if self.resume:
            if not load_checkpoint(
                    self.trainer, self.resume, optimizer=self.optimizer):
                log.warning(f'no checkpoint found at "{self.resume}"')

    def on_evaluate_begin(self):
        resume = self.resume
        if not resume:
            resume = self.checkpoint_file
        if not load_checkpoint(self.trainer, resume, optimizer=None):
            log.warning(f'no checkpoint found at "{resume}", '
                        'this may not what you want')

    def on_evaluate_end(self):
        if dist.is_initialized():
            dist.barrier()

    def on_predict_begin(self):
        self.on_evaluate_begin()

    def on_epoch_begin(self, epoch, steps):
        self.epoch = epoch

    def on_epoch_end(self, outputs):
        loss = outputs.pop('loss')
        if self.save_best_model and self.optimizer is not None:
            if self.save_model is None or self.save_loss > loss.item():
                self.save_loss = loss.item()
                self.save_model = (self.trainer.model.state_dict(),
                                   self.optimizer.state_dict())

        if ((self.epoch + 1) % self.save_checkpoint_epochs
                == 0) and self.trainer.config['is_master']:
            if not self.save_best_model and self.optimizer is not None:
                self.save_loss = loss.item()
                self.save_model = (self.trainer.model.state_dict(),
                                   self.optimizer.state_dict())
            if self.save_loss is not None and self.save_model is not None:
                self._checkpoint_executor.submit(
                    torch.save, {
                        'epoch': self.epoch + 1,
                        'loss': self.save_loss,
                        'state_dict': self.save_model[0],
                        'optimizer': self.save_model[1],
                    }, self.checkpoint_file)

    def on_train_end(self):
        self._checkpoint_executor.shutdown()
