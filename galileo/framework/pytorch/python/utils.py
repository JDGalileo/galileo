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

import torch
from functools import wraps


def convert(func):
    @wraps(func)
    def _convert(sequence, *args, **kwargs):
        if isinstance(sequence, torch.Tensor):
            return func(sequence, *args, **kwargs)
        elif isinstance(sequence, (list, tuple)):
            return [
                _convert(ip, *args, **kwargs) if ip is not None else None
                for ip in sequence
            ]
        elif isinstance(sequence, dict):
            return {
                key: _convert(sequence[key], *args, **kwargs)
                if sequence[key] is not None else None
                for key in sequence
            }
        else:
            raise ValueError(f'sequence {sequence} type not supported.')

    return _convert


@convert
def data_to_cuda(inputs, device):
    return inputs.cuda(device=device, non_blocking=True)


@convert
def data_to_numpy(inputs):
    return inputs.cpu().numpy()
