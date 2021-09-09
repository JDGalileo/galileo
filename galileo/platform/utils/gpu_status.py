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

from galileo.platform.log import log


def get_gpu_status(dev_id):
    r'''
    get gpu status of cuda

    return res.gpu res.memory
    '''
    try:
        import py3nvml.py3nvml as nv
        nv.nvmlInit()
        assert dev_id >= 0 and dev_id < nv.nvmlDeviceGetCount()
        handle = nv.nvmlDeviceGetHandleByIndex(dev_id)
        return nv.nvmlDeviceGetUtilizationRates(handle)
    except:
        log.warning('could not get status of cuda {}'.format(dev_id))
    return None
