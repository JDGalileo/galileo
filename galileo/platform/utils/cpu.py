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


def cpu_count():
    r'''
    consider run in docker
    '''
    quota = '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'
    period = '/sys/fs/cgroup/cpu/cpu.cfs_period_us'
    share = '/sys/fs/cgroup/cpu/cpu.shares'
    avail_cpu = -1
    if os.path.isfile(quota):
        cpu_quota = int(open(quota).read().rstrip())
        if os.path.isfile(period):
            cpu_period = int(open(period).read().rstrip())
            avail_cpu = cpu_quota // cpu_period
    elif os.path.isfile(share):
        cpu_shares = int(open(share).read().rstrip())
        avail_cpu = cpu_shares // 1024
    return avail_cpu if avail_cpu > 0 else os.cpu_count()
