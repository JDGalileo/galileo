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
import sys
from galileo.tests.utils import (
    zk_server,
    zk_path,
)
from galileo.framework.python.service import start_service

shard_index, shard_count = 0, 1
if len(sys.argv) > 1:
    shard_index = int(sys.argv[1])
if len(sys.argv) > 2:
    shard_count = int(sys.argv[2])

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.realpath(os.path.join(base_dir, '../../testdata'))
start_service(data_dir,
              zk_server=zk_server,
              zk_path=zk_path,
              shard_index=shard_index,
              shard_count=shard_count,
              daemon=False)
