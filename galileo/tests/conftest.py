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
import pytest
import galileo
import galileo.unify
from galileo.tests.utils import (
    zk_server,
    zk_path,
)
from galileo.framework.python.client import create_client

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


@pytest.fixture(scope='session')
def prepare_tf_env():
    galileo.unify.set_backend('tf')
    create_client(zk_server, zk_path)


@pytest.fixture(scope='session')
def prepare_pytorch_env():
    galileo.unify.set_backend('pytorch')
    create_client(zk_server, zk_path)
