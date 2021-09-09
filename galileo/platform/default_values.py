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

import attr


@attr.s(frozen=True)
class _DefaultValues:
    '''
    default values shared by models, datasets and train
    frozen means all values is immutable
    '''
    BACKEND = 'tf'

    # graph service config fields
    ZK_SERVER = attr.ib(default='127.0.0.1:2181')
    ZK_PATH = attr.ib(default='/galileo')

    # train config fields
    OPTIMIZER = attr.ib(default='adam')
    LEARNING_RATE = attr.ib(default=0.001)
    MOMENTUM = attr.ib(default=0.9)
    MODEL_DIR = attr.ib(default='./')
    START_EPOCH = attr.ib(default=0)
    NUM_EPOCHS = attr.ib(default=1)
    LOG_STEPS = attr.ib(default=10)
    LOG_MAX_TIMES_PER_EPOCH = attr.ib(default=100)
    GPU_STATUS = attr.ib(default=False)
    SAVE_CHECKPOINT_EPOCHS = attr.ib(default=1)

    # fields for pytorch
    WEIGHT_DECAY = attr.ib(default=0)
    MULTIPROCESSING_DISTRIBUTED = attr.ib(default=False)

    # rpc fields
    RPC_TIMEOUT_MS = attr.ib(default=-1)
    RPC_BODY_SIZE = attr.ib(default=2147483647)
    RPC_BTHREAD_CONCURRENCY = attr.ib(default=9)


DefaultValues = _DefaultValues()
