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
from torch.utils.data import Dataset
from galileo.platform.export import export


@export('galileo.pytorch')
class RangeDataset(Dataset):
    r'''
    range dataset

    args:
        start: int
        end: int, not including end
        step: int, 1
    '''
    def __init__(self, start, end, step=1, **kwargs):
        super().__init__()
        assert end > start
        self.start = start
        self.end = end
        self.data = torch.arange(start, end, step=step)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.end - self.start
        return self.data[idx]
